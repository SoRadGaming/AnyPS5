// LOCAL EXPERIMENT (APS5_EXPERIMENT_GDS_IN_STREAM, DRAIN-PLAN D2): DMA_DATA transfers that touch the
// GDS recorded in the stream (GdsStream.hpp). The decode first (no device), then, through the driver on
// compute queue 0x40 (queue 0x48 and queue 0 wait), with the append kernel and the GDS counter packets
// of GdsAppendConsume.cpp (SetCounters: DMA_DATA 0x40100000, ReadCounters: 0x20000000):
//   readback      an append dispatch, the counters stored between it and a second one, a readback:
//                 =1/=2 record the readback (CopyBuffer path 1 or 3) and the four stores (no targeted
//                 sync), and the second dispatch sees the stored values;
//   write-after   a readback recorded behind a slow non-GDS dispatch in a batch newer than every GDS
//                 user, then a fallback CPU store into the GDS (3 bytes: not a GPU store): the store
//                 waits for the readback's batch (judge fix 1), so the readback holds the counter;
//   wait          WAIT_REG_MEM on a readback destination from queue 0 and from queue 0x48: completes
//                 (by polling once the copy ran), no hang;
//   copy-in       a memory -> GDS copy of a dispatch's pending output, read back;
//   fallbacks     stores while the recorder is idle (reason 1: required whenever the recorder was idle
//                 after the case's sync), a readback into memory without a host import, and the 3-byte
//                 store above take today's path, counted; the decode refusals (high word, range, shape)
//                 and the store refusals' names (reasons 1-5) in the decode part (Validate rejects such
//                 packets first); reason 3 is the readback case's on a device without a host import of
//                 the GDS, reason 4 is ResolveStore's (store-shape) before WriteLabelOnGpu sees it;
//   switch off    nothing counted, nothing armed.
// Every run checks the values, and cpu-gds-access-unnoted stays 0. A device without a host import of the
// GDS records nothing: the run says so and checks the fallbacks instead. The mode expected comes from the
// environment (=1/=2 need APS5_EXPERIMENT_GDS_TARGETED_SYNC and are refused with APS5_PIPELINE_DMA).
// --behind-completions (its own process) tests reason 5 on the device instead: WriteLabelOnGpu with
// `behindCompletions` false while recorded work has a completion action (APS5_LABEL_GATE_ALL=1, which
// that mode sets: through the driver a copied writer over the GDS needs a buffer that spans the GDS
// mapping and memory the device does not import) returns 5 with nothing recorded, noted or stamped, and
// the refused bytes never land; a store that accepts the completion path is recorded and lands.
// A failure prints and leaves at once (std::_Exit): GPU work may still reference the test's memory, which
// is static or never freed, and a lost device must not hang a teardown.
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GdsStream.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <future>
#include <initializer_list>
#include <new>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace GdsStream = AgcDriver::GdsStream;
namespace Pm4 = AgcDriver::Pm4;
using GdsStream::Fallback;
using GdsStream::Op;

[[noreturn]] void Fail(const std::string& what) {
    std::fprintf(stderr, "gds in stream: %s\n", what.c_str());
    std::fflush(stderr);
    std::_Exit(1);
}

void Require(bool condition, const std::string& what) {
    if (!condition) Fail(what);
}

constexpr std::uint32_t ProducerQueue = 0x40;
constexpr std::uint32_t WaiterQueue = 0x48;
constexpr std::uint32_t Threads = 64;
constexpr std::uint32_t ActiveLanes = 40;
constexpr std::uint32_t Counters = 4;
constexpr std::uint32_t CounterOffset = 0x10;
constexpr std::uint32_t CopyInOffset = 0x40;
constexpr std::array<std::uint32_t, Counters> Starts{1000, 2000, 3000, 4000};
constexpr std::array<std::uint32_t, Counters> Before{500, 600, 700, 800};
// After SetCounters(Starts) and one append dispatch: ds_append / ds_consume over every lane, then over
// the active lanes only (GdsAppendConsume's steps).
constexpr std::array<std::uint32_t, Counters> After{Starts[0] + Threads, Starts[1] - Threads, Starts[2] + ActiveLanes, Starts[3] - ActiveLanes};
constexpr std::uint32_t Untouched = 0xdeadbeefu;

// GdsAppendConsume.cpp's append/consume kernel (wave32, 64 threads, 40 active lanes in the masked half).
alignas(256) constexpr auto AppendCode = std::to_array<std::uint32_t>({
    0x34060082,
    0x7e0c02ff, 0x5eed0001u, 0x7e0e02ff, 0x5eed0002u,
    0xbefc03ff, 0x0000ffff,
    0xd8fa0010, 0x04000000,
    0xd8f60014, 0x05000000,
    0x7da800ff, ActiveLanes,
    0xd8fa0018, 0x06000000,
    0xd8f6001c, 0x07000000,
    0xbefe04c1,
    0xbf8cc07f,
    0xe0702000, 0x80000403, 0xe0702004, 0x80000503, 0xe0702008, 0x80000603, 0xe070200c, 0x80000703,
    0xbf810000,
});
alignas(256) std::array<std::uint32_t, Threads * 4> AppendOutput{};

// SameQueueLabelWait.cpp's spin kernel: SpinIterations scalar steps, then one store; a dispatch that
// writes no GDS and keeps the GPU busy.
alignas(256) constexpr std::array<std::uint32_t, 15> SpinCode{0xf4080200u, 0xfa000000u, 0xbe840381u, 0x9304ff04u, 0x0019660du, 0x8004ff04u, 0x3c6ef35fu, 0x80828102u, 0xbf078002u, 0xbf85fff9u, 0xbf8cc07fu, 0x7e000204u, 0xe0700000u, 0x80020000u, 0xbf810000u};
constexpr std::uint32_t SpinIterations = 1u << 20u;

// A readback destination in memory the device cannot import (static, not a registered allocation).
alignas(256) std::array<std::uint32_t, Counters> Unregistered{};

// Registered guest memory (importable), never freed: GPU work may still reference it at a failure.
struct Block {
    alignas(256) std::array<std::uint32_t, Counters> readback{};
    alignas(256) std::array<std::uint32_t, Counters> afterRead{};
    alignas(256) std::array<std::uint32_t, Counters> waited{};
    alignas(256) std::array<std::uint32_t, Counters> copiedIn{};
    alignas(256) std::array<std::uint32_t, Counters> spinOutput{};
    alignas(256) std::array<std::uint32_t, Counters> spinDescriptor{};
};

Block& Memory() {
    static Block* block = [] {
        constexpr std::size_t Bytes = 65536;
        static_assert(sizeof(Block) <= Bytes);
#ifdef _WIN32
        void* memory = VirtualAlloc(nullptr, Bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
        void* memory = std::aligned_alloc(Bytes, Bytes);
#endif
        Require(memory != nullptr, "cannot allocate the guest block");
        std::memset(memory, 0, Bytes);
        GuestAllocations::Mutation().Add(memory, Bytes, true, true, true);
        return new (memory) Block();
    }();
    return *block;
}

std::uint32_t Low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address);
}

std::uint32_t High(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 32u);
}

std::uint64_t Address(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer);
}

using Words = std::vector<std::uint32_t>;

void SetRegister(Words& words, std::uint32_t offset, std::uint32_t value) {
    words.insert(words.end(), {0xc0017600u, offset, value});
}

void Dispatch(Words& words, const void* code, std::uint32_t threads, std::initializer_list<std::uint32_t> userData) {
    const auto address = Address(code);
    SetRegister(words, 0x20c, static_cast<std::uint32_t>(address >> 8u));
    SetRegister(words, 0x20d, static_cast<std::uint32_t>(address >> 40u) & 0xffu);
    SetRegister(words, 0x207, threads);
    SetRegister(words, 0x208, 1);
    SetRegister(words, 0x209, 1);
    SetRegister(words, 0x212, 0);
    SetRegister(words, 0x213, static_cast<std::uint32_t>(userData.size()) << 1u);
    std::uint32_t offset = 0x240;
    for (const auto value : userData) SetRegister(words, offset++, value);
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x8041u});
}

// A GDS user: the append kernel over AppendOutput (GdsAppendConsume's buffer descriptor).
void Append(Words& words) {
    const auto output = Address(AppendOutput.data());
    Dispatch(words, AppendCode.data(), Threads, {Low(output), (High(output) & 0xffffu) | (4u << 16u), static_cast<std::uint32_t>(AppendOutput.size()), 0x11016facu});
}

// Not a GDS user: the spin kernel, storing into the block's spinOutput.
void Spin(Words& words) {
    auto& block = Memory();
    const auto output = Address(block.spinOutput.data());
    block.spinDescriptor = {Low(output), High(output) & 0xffffu, static_cast<std::uint32_t>(sizeof(block.spinOutput)), 0x31016facu};
    const auto descriptor = Address(block.spinDescriptor.data());
    Dispatch(words, SpinCode.data(), 1, {Low(descriptor), High(descriptor), SpinIterations});
}

void DmaData(Words& words, std::uint32_t control, std::uint32_t source, std::uint32_t sourceHigh, std::uint32_t destination, std::uint32_t destinationHigh, std::uint32_t bytes) {
    words.insert(words.end(), {0xc0055000u, control, source, sourceHigh, destination, destinationHigh, bytes});
}

// Immediate -> GDS stores of the four counters (GdsAppendConsume's SetCounters).
void SetCounters(Words& words, const std::array<std::uint32_t, Counters>& values) {
    for (std::uint32_t counter = 0; counter < Counters; ++counter) DmaData(words, 0x40100000u, values[counter], 0, CounterOffset + counter * 4u, 0, 4);
}

// GDS -> memory (GdsAppendConsume's ReadCounters).
void Readback(Words& words, const void* destination, std::uint32_t offset, std::uint32_t bytes) {
    const auto address = Address(destination);
    DmaData(words, 0x20000000u, offset, 0, Low(address), High(address), bytes);
}

void WaitEqual(Words& words, const void* target, std::uint32_t value) {
    const auto address = Address(target);
    words.insert(words.end(), {0xc0053c00u, 0x13u, Low(address), High(address), value, 0xffffffffu, 0x19u});
}

void Submit(std::deque<Words>& kept, Words words, std::uint32_t queue) {
    auto& stored = kept.emplace_back(std::move(words));
    Packet packet{stored.data(), static_cast<std::uint32_t>(stored.size()), 0, {}};
    AgcDriver::Submit(&packet, queue);
}

// Every queue's submissions processed, or a failure after 20 s (a wait that never ends).
void Idle(const std::string& what) {
    auto idle = std::async(std::launch::async, [] { AgcDriverWaitIdle_nid_postfix(); });
    if (idle.wait_for(std::chrono::seconds(20)) != std::future_status::ready) Fail(what + ": the queues did not finish in 20 s (a hang)");
    try {
        idle.get();
    } catch (const std::exception& error) {
        Fail(what + ": " + error.what());
    }
}

// Through the checked guest memory path: the flush hook waits for the recorded work that writes it.
template <std::size_t Count>
std::array<std::uint32_t, Count> Read(std::uint64_t address) {
    std::array<std::uint32_t, Count> values{};
    AgcDriver::GuestMemory::Read(address, std::as_writable_bytes(std::span(values)), 4);
    return values;
}

std::array<std::uint32_t, Counters> ReadGdsCounters() {
    return Read<Counters>(Pm4::GdsAddress() + CounterOffset);
}

std::string Text(std::span<const std::uint32_t> values) {
    std::string text;
    for (const auto value : values) text += (text.empty() ? "" : " ") + std::to_string(value);
    return text;
}

void RequireValues(std::span<const std::uint32_t> actual, std::span<const std::uint32_t> expected, const std::string& what) {
    Require(std::equal(actual.begin(), actual.end(), expected.begin(), expected.end()), what + ": " + Text(actual) + ", expected " + Text(expected));
}

GdsStream::Totals Delta(const GdsStream::Totals& before) {
    auto now = GdsStream::CurrentTotals();
    for (std::size_t op = 0; op < GdsStream::Ops; ++op) now.recorded[op] -= before.recorded[op];
    for (std::size_t path = 0; path < now.copyPaths.size(); ++path) now.copyPaths[path] -= before.copyPaths[path];
    now.notes -= before.notes;
    now.idleCopies -= before.idleCopies;
    for (std::size_t reason = 0; reason < GdsStream::Fallbacks; ++reason) now.fallbacks[reason] -= before.fallbacks[reason];
    now.targetedSyncs -= before.targetedSyncs;
    now.accessChecks -= before.accessChecks;
    now.unnoted -= before.unnoted;
    now.verified -= before.verified;
    now.explained -= before.explained;
    return now;
}

std::uint64_t Recorded(const GdsStream::Totals& totals, Op op) {
    return totals.recorded[static_cast<std::size_t>(op)];
}

std::uint64_t Fell(const GdsStream::Totals& totals, Fallback reason) {
    return totals.fallbacks[static_cast<std::size_t>(reason)];
}

std::uint64_t RecordedAll(const GdsStream::Totals& totals) {
    std::uint64_t sum = 0;
    for (const auto value : totals.recorded) sum += value;
    return sum;
}

// WriteLabelOnGpu's refusals of a store into the GDS (reasons 1-5).
std::uint64_t StoreRefusals(const GdsStream::Totals& totals) {
    return Fell(totals, Fallback::StoreCpu) + Fell(totals, Fallback::StoreDrain) + Fell(totals, Fallback::StoreNoImport) + Fell(totals, Fallback::StoreUnsuitable) + Fell(totals, Fallback::StoreBehindCompletions);
}

// A copy's refusals after its decode (the GPU path's).
std::uint64_t CopyRefusals(const GdsStream::Totals& totals) {
    return Fell(totals, Fallback::CopyCompletions) + Fell(totals, Fallback::CopyNoImport) + Fell(totals, Fallback::CopyInaccessible);
}

std::uint64_t FellAll(const GdsStream::Totals& totals) {
    std::uint64_t sum = 0;
    for (const auto value : totals.fallbacks) sum += value;
    return sum;
}

std::string Describe(const GdsStream::Totals& totals) {
    std::string text = "recorded";
    for (std::size_t op = 0; op < GdsStream::Ops; ++op) text += std::string(" ") + GdsStream::OpName(static_cast<Op>(op)) + " " + std::to_string(totals.recorded[op]);
    text += "; copy paths 1/3 " + std::to_string(totals.copyPaths[1]) + "/" + std::to_string(totals.copyPaths[3]) + "; notes " + std::to_string(totals.notes) + "; idle copies " + std::to_string(totals.idleCopies) + "; fallbacks";
    for (std::size_t reason = 0; reason < GdsStream::Fallbacks; ++reason) {
        if (totals.fallbacks[reason] != 0) text += std::string(" ") + GdsStream::FallbackName(static_cast<Fallback>(reason)) + " " + std::to_string(totals.fallbacks[reason]);
    }
    text += "; targeted syncs " + std::to_string(totals.targetedSyncs) + "; access checks " + std::to_string(totals.accessChecks) + ", unnoted " + std::to_string(totals.unnoted) + "; =2 verified " + std::to_string(totals.verified) + ", explained " + std::to_string(totals.explained);
    return text;
}

// Shared by every case: the mode, and whether the device records GDS operations at all (the GDS
// imported); learned by the first readback.
struct Run {
    int mode = 0;
    bool gpuGds = false;
};

// What every recorded operation must leave in the counts: one whole-GDS note each; =2 verified each,
// with nothing to explain (nothing races in these runs); the invariant at 0.
void RequireBookkeeping(const Run& run, const GdsStream::Totals& delta, const std::string& what) {
    Require(delta.unnoted == 0, what + ": cpu-gds-access-unnoted " + std::to_string(delta.unnoted));
    if (run.mode == 0) {
        Require(RecordedAll(delta) == 0 && FellAll(delta) == 0 && delta.notes == 0 && delta.idleCopies == 0 && delta.targetedSyncs == 0 && delta.accessChecks == 0 && delta.verified == 0, what + ": counted with the switch off: " + Describe(delta));
        return;
    }
    Require(delta.notes == RecordedAll(delta), what + ": " + std::to_string(delta.notes) + " whole-GDS notes for " + std::to_string(RecordedAll(delta)) + " recorded operations");
    Require(delta.copyPaths[1] + delta.copyPaths[3] == Recorded(delta, Op::Readback) + Recorded(delta, Op::GdsCopy) + Recorded(delta, Op::CopyIn), what + ": copy paths do not match the recorded copies: " + Describe(delta));
    if (run.mode == 2) Require(delta.verified == RecordedAll(delta) && delta.explained == 0, what + ": =2 verified " + std::to_string(delta.verified) + " (explained " + std::to_string(delta.explained) + ") of " + std::to_string(RecordedAll(delta)) + " recorded operations");
    else Require(delta.verified == 0, what + ": =1 value-checked an operation");
}

int ExpectedMode() {
    const char* text = std::getenv("APS5_EXPERIMENT_GDS_IN_STREAM");
    if (text == nullptr || text[0] == '\0' || std::strcmp(text, "0") == 0) return 0;
    if (std::getenv("APS5_EXPERIMENT_GDS_TARGETED_SYNC") == nullptr || std::getenv("APS5_PIPELINE_DMA") != nullptr) return 0;
    return std::strcmp(text, "2") == 0 ? 2 : 1;
}

// The decode (no device): Pm4::DecodeGdsCopy's mapping and refusals, ResolveStore's GDS case.
void Decode() {
    using Refusal = Pm4::GdsCopyRefusal;
    const auto gds = Pm4::GdsAddress();
    alignas(16) static std::array<std::uint32_t, 8> memory{};
    const auto mem = Address(memory.data());
    const auto other = mem + 16;
    struct Case {
        const char* what;
        std::array<std::uint32_t, 7> packet;
        Refusal refusal;
        std::uint64_t source;
        std::uint64_t destination;
        std::size_t bytes;
    };
    const std::array<Case, 12> cases{{
        {"a readback", {0xc0055000u, 0x20000000u, 0x10, 0, Low(mem), High(mem), 16}, Refusal::None, gds + 0x10, mem, 16},
        {"a memory -> GDS copy", {0xc0055000u, 0x00100000u, Low(mem), High(mem), 0x20, 0, 8}, Refusal::None, mem, gds + 0x20, 8},
        {"a GDS -> GDS copy", {0xc0055000u, 0x20100000u, 0x100, 0, 0x200, 0, 64}, Refusal::None, gds + 0x100, gds + 0x200, 64},
        {"an overlapping GDS -> GDS copy", {0xc0055000u, 0x20100000u, 0x100, 0, 0x120, 0, 64}, Refusal::Shape, 0, 0, 0},
        {"a 3-byte readback", {0xc0055000u, 0x20000000u, 0x10, 0, Low(mem), High(mem), 3}, Refusal::Shape, 0, 0, 0},
        {"a GDS source with a high word", {0xc0055000u, 0x20000000u, 0x10, 1, Low(mem), High(mem), 16}, Refusal::HighWord, 0, 0, 0},
        {"a GDS destination with a high word", {0xc0055000u, 0x00100000u, Low(mem), High(mem), 0x20, 1, 8}, Refusal::HighWord, 0, 0, 0},
        {"a readback past the GDS", {0xc0055000u, 0x20000000u, 0xfff8, 0, Low(mem), High(mem), 16}, Refusal::Range, 0, 0, 0},
        {"a zero-byte readback", {0xc0055000u, 0x20000000u, 0x10, 0, Low(mem), High(mem), 0}, Refusal::NotCopy, 0, 0, 0},
        {"an immediate store into the GDS", {0xc0055000u, 0x40100000u, 7, 0, 0x10, 0, 4}, Refusal::NotCopy, 0, 0, 0},
        {"a memory -> memory copy", {0xc0055000u, 0, Low(mem), High(mem), Low(other), High(other), 4}, Refusal::NotCopy, 0, 0, 0},
        {"a GDS prefetch", {0xc0055000u, 0x20200000u, 0x10, 0, Low(mem), High(mem), 16}, Refusal::NotCopy, 0, 0, 0},
    }};
    for (const auto& item : cases) {
        auto refusal = Refusal::None;
        const auto copy = Pm4::DecodeGdsCopy(item.packet, &refusal);
        Require(refusal == item.refusal, std::string("DecodeGdsCopy: ") + item.what + " refused with " + std::to_string(static_cast<int>(refusal)) + ", expected " + std::to_string(static_cast<int>(item.refusal)));
        Require(copy.has_value() == (item.refusal == Refusal::None), std::string("DecodeGdsCopy: ") + item.what + (copy.has_value() ? " decoded" : " did not decode"));
        if (copy.has_value()) Require(copy->source == item.source && copy->destination == item.destination && copy->bytes == item.bytes, std::string("DecodeGdsCopy: ") + item.what + " decoded to the wrong ranges");
        Require(Pm4::DmaTouchesGds(item.packet) == (item.what != std::string("a memory -> memory copy")), std::string("DmaTouchesGds: ") + item.what);
    }
    AgcDriver::QueueState queue;
    const std::array<std::uint32_t, 7> store{0xc0055000u, 0x40100000u, 0x11223344u, 0, 0x40, 0, 8};
    Require(!Pm4::ResolveStore(store, queue, 65536).has_value(), "ResolveStore: a store into the GDS resolved without gdsStores");
    const auto resolved = Pm4::ResolveStore(store, queue, 65536, true);
    Require(resolved.has_value() && resolved->address == gds + 0x40 && resolved->Bytes().size() == 8, "ResolveStore: a store into the GDS did not resolve to GdsAddress() + 0x40, 8 bytes");
    std::array<std::uint32_t, 2> words{};
    std::memcpy(words.data(), resolved->Bytes().data(), 8);
    Require(words[0] == 0x11223344u && words[1] == 0x11223344u, "ResolveStore: a store into the GDS resolved to the wrong bytes");
    const std::array<std::uint32_t, 7> shortStore{0xc0055000u, 0x40100000u, 0x11223344u, 0, 0x40, 0, 3};
    Require(!Pm4::ResolveStore(shortStore, queue, 65536, true).has_value(), "ResolveStore: a 3-byte store into the GDS resolved");
    const std::array<std::uint32_t, 7> unaligned{0xc0055000u, 0x40100000u, 0x11223344u, 0, 0x42, 0, 4};
    Require(!Pm4::ResolveStore(unaligned, queue, 65536, true).has_value(), "ResolveStore: a store into the GDS at an unaligned offset resolved");
    const std::array<std::uint32_t, 7> high{0xc0055000u, 0x40100000u, 0x11223344u, 0, 0x40, 1, 4};
    Require(!Pm4::ResolveStore(high, queue, 65536, true).has_value(), "ResolveStore: a store into the GDS with a high word resolved");
    // WriteLabelOnGpu's refusals of a store into the GDS by name (any other reason counts as 4).
    const std::array<std::pair<int, Fallback>, 6> refusals{{{1, Fallback::StoreCpu}, {2, Fallback::StoreDrain}, {3, Fallback::StoreNoImport}, {4, Fallback::StoreUnsuitable}, {5, Fallback::StoreBehindCompletions}, {6, Fallback::StoreUnsuitable}}};
    for (const auto& [reason, fallback] : refusals) Require(GdsStream::StoreFallback(reason) == fallback, "StoreFallback: reason " + std::to_string(reason) + " counted as " + GdsStream::FallbackName(GdsStream::StoreFallback(reason)) + ", expected " + GdsStream::FallbackName(fallback));
    std::puts("gds in stream: decode passed");
}

// Rows 1 and 2: the readback after an append dispatch, and the counter stores between two dispatches.
void ReadbackAndStores(Run& run) {
    auto& block = Memory();
    std::deque<Words> kept;
    Words words;
    SetCounters(words, Before);
    Submit(kept, std::move(words), ProducerQueue);
    Idle("readback: the first counters");
    block.readback.fill(Untouched);
    const auto before = GdsStream::CurrentTotals();
    words = {};
    Append(words);
    SetCounters(words, Starts);
    Append(words);
    Readback(words, block.readback.data(), CounterOffset, Counters * 4);
    Submit(kept, std::move(words), ProducerQueue);
    Idle("readback");
    // Missed stores would leave Before + two dispatches.
    RequireValues(Read<Counters>(Address(block.readback.data())), After, "readback: the destination (the second dispatch saw the stores between them)");
    const auto delta = Delta(before);
    std::printf("gds in stream: readback: %s\n", Describe(delta).c_str());
    RequireBookkeeping(run, delta, "readback");
    if (run.mode == 0) return;
    run.gpuGds = Recorded(delta, Op::Readback) == 1;
    if (!run.gpuGds) {
        // No host import of the GDS (or a completion over it): today's path, counted.
        Require(CopyRefusals(delta) + delta.idleCopies == 1, "readback: neither recorded nor counted as a fallback: " + Describe(delta));
        Require(Recorded(delta, Op::Store) + StoreRefusals(delta) == 4, "readback: the four stores were neither recorded nor counted: " + Describe(delta));
        std::puts("gds in stream: the readback was not recorded on this device (no host import of the GDS?): the GPU half is not tested, the fallbacks are");
        return;
    }
    Require(Recorded(delta, Op::Store) == 4, "readback: the four stores between the dispatches were not recorded on the GPU: " + Describe(delta));
    Require(delta.targetedSyncs == 0 && FellAll(delta) == 0, "readback: a store into the GDS synced or fell back: " + Describe(delta));
}

// Row 3: write-after-read. S1 ends with GDS users in a submitted batch; S2 records a readback behind a
// slow non-GDS dispatch, so the readback's batch is newer than every GDS user, then a 3-byte store into
// the GDS that the GPU path does not take (the targeted path: SyncThrough of the GDS, then the CPU
// store). Without the whole-GDS note the store lands before the readback runs.
void WriteAfterRead(const Run& run) {
    auto& block = Memory();
    block.afterRead.fill(Untouched);
    std::deque<Words> kept;
    const auto before = GdsStream::CurrentTotals();
    Words first;
    Append(first);
    SetCounters(first, Starts);
    Append(first);
    Submit(kept, std::move(first), ProducerQueue);
    Words second;
    Spin(second);
    Readback(second, block.afterRead.data(), CounterOffset, Counters * 4);
    constexpr std::uint32_t Stored = 0x00abcdefu;
    DmaData(second, 0x40100000u, Stored, 0, CounterOffset, 0, 3);
    Submit(kept, std::move(second), ProducerQueue);
    Idle("write-after-read");
    RequireValues(Read<Counters>(Address(block.afterRead.data())), After, "write-after-read: the readback (a fallback store into the GDS landed before it ran)");
    const auto counter = ReadGdsCounters()[0];
    Require(counter == ((After[0] & 0xff000000u) | Stored), "write-after-read: GDS counter 0 is " + std::to_string(counter) + " after the 3-byte store");
    const auto delta = Delta(before);
    std::printf("gds in stream: write-after-read: %s\n", Describe(delta).c_str());
    RequireBookkeeping(run, delta, "write-after-read");
    if (run.mode == 0) return;
    // Every store into the GDS that fell back synced (S1's four too, on a device that records nothing).
    Require(Fell(delta, Fallback::StoreShape) == 1 && delta.targetedSyncs == StoreRefusals(delta) + 1, "write-after-read: the 3-byte store did not take the targeted path, counted: " + Describe(delta));
    if (run.gpuGds) {
        Require(Recorded(delta, Op::Readback) == 1, "write-after-read: the readback was not recorded: " + Describe(delta));
        Require(delta.accessChecks >= 1, "write-after-read: the targeted sync was not checked against the recorded readback");
    }
}

// Row 4: WAIT_REG_MEM on a readback destination from queue 0 and from queue 0x48, submitted before the
// producer: both complete once the copy ran (they poll), and nothing hangs.
void WaitOnReadback(const Run& run) {
    auto& block = Memory();
    block.waited.fill(Untouched);
    std::deque<Words> kept;
    const auto before = GdsStream::CurrentTotals();
    Words graphics;
    WaitEqual(graphics, block.waited.data(), After[0]);
    Submit(kept, std::move(graphics), 0);
    Words compute;
    WaitEqual(compute, block.waited.data(), After[0]);
    Submit(kept, std::move(compute), WaiterQueue);
    Words producer;
    Append(producer);
    SetCounters(producer, Starts);
    Append(producer);
    Readback(producer, block.waited.data(), CounterOffset, 4);
    Submit(kept, std::move(producer), ProducerQueue);
    Idle("wait on a readback destination");
    Require(Read<1>(Address(block.waited.data()))[0] == After[0], "wait on a readback destination: the destination does not hold the counter");
    const auto delta = Delta(before);
    std::printf("gds in stream: wait on a readback destination: %s\n", Describe(delta).c_str());
    RequireBookkeeping(run, delta, "wait on a readback destination");
    if (run.mode != 0 && run.gpuGds) Require(Recorded(delta, Op::Readback) == 1, "wait on a readback destination: the readback was not recorded: " + Describe(delta));
}

// A memory -> GDS copy of a dispatch's pending output, then (its own submission, so each step is
// counted apart) a readback of it.
void CopyIn(const Run& run) {
    auto& block = Memory();
    block.copiedIn.fill(Untouched);
    block.spinOutput.fill(0);
    std::deque<Words> kept;
    auto before = GdsStream::CurrentTotals();
    Words words;
    Spin(words);
    const auto output = Address(block.spinOutput.data());
    DmaData(words, 0x00100000u, Low(output), High(output), CopyInOffset, 0, 4);
    Submit(kept, std::move(words), ProducerQueue);
    Idle("copy-in");
    auto delta = Delta(before);
    std::printf("gds in stream: copy-in: %s\n", Describe(delta).c_str());
    RequireBookkeeping(run, delta, "copy-in");
    if (run.mode != 0) {
        // The spin output may be idle by the time the copy is decided, or written through a copied
        // buffer (a completion over it): then a store, or today's path.
        Require(Recorded(delta, Op::CopyIn) + Recorded(delta, Op::StoreFromMemory) + CopyRefusals(delta) + StoreRefusals(delta) == 1, "copy-in: the memory -> GDS copy was neither recorded nor counted: " + Describe(delta));
        if (Recorded(delta, Op::CopyIn) == 0) std::puts("gds in stream: copy-in: the copy took another path (its source was not a pending write of an import)");
    }
    before = GdsStream::CurrentTotals();
    words = {};
    Readback(words, block.copiedIn.data(), CopyInOffset, 4);
    Submit(kept, std::move(words), ProducerQueue);
    Idle("copy-in readback");
    const auto spun = Read<1>(output)[0];
    Require(spun != 0, "copy-in: the spin kernel stored nothing");
    Require(Read<1>(Pm4::GdsAddress() + CopyInOffset)[0] == spun, "copy-in: the GDS does not hold the copied word");
    Require(Read<1>(Address(block.copiedIn.data()))[0] == spun, "copy-in: the readback does not hold the copied word");
    delta = Delta(before);
    std::printf("gds in stream: copy-in readback: %s\n", Describe(delta).c_str());
    RequireBookkeeping(run, delta, "copy-in readback");
    if (run.mode != 0) Require(Recorded(delta, Op::Readback) + CopyRefusals(delta) + delta.idleCopies == 1, "copy-in readback: neither recorded nor counted: " + Describe(delta));
}

// Fallbacks: stores while the recorder is idle (reason 1) and a readback into memory without a host
// import take today's path, counted.
void Fallbacks(const Run& run) {
    std::deque<Words> kept;
    // Whether the recorder was idle after the sync: nothing records before the first store then (every
    // queue finished, and an immediate store into the GDS records nothing ahead of WriteLabelOnGpu), so
    // that store meets the idle recorder: reason 1.
    bool idleAfterSync = false;
    {
        AgcDriver::GuestMemory::TagGpuLockSite(AgcDriver::GuestMemory::GpuLockSite::Flush);
        std::lock_guard gpuLock(AgcDriver::GuestMemory::GpuMutex());
        if (auto* recorder = AgcDriver::Graphics::Recorder::Active()) {
            recorder->Sync();
            idleAfterSync = recorder->Idle();
        }
    }
    auto before = GdsStream::CurrentTotals();
    Words idle;
    SetCounters(idle, Before);
    Submit(kept, std::move(idle), ProducerQueue);
    Idle("fallbacks: stores on an idle recorder");
    RequireValues(ReadGdsCounters(), Before, "fallbacks: the counters stored on an idle recorder");
    auto delta = Delta(before);
    std::printf("gds in stream: fallbacks, idle recorder: %s\n", Describe(delta).c_str());
    RequireBookkeeping(run, delta, "fallbacks: idle recorder");
    if (run.mode != 0) {
        Require(Recorded(delta, Op::Store) + StoreRefusals(delta) == 4, "fallbacks: the four stores on an idle recorder were neither recorded nor counted: " + Describe(delta));
        Require(delta.targetedSyncs == FellAll(delta), "fallbacks: a store fallback did not take the targeted path: " + Describe(delta));
        if (idleAfterSync) Require(Fell(delta, Fallback::StoreCpu) >= 1, "fallbacks: the recorder was idle after the sync, but no store into the GDS fell back with reason 1: " + Describe(delta));
        else std::puts("gds in stream: fallbacks: the recorder was not idle after the sync, reason 1 not required");
    }

    Unregistered.fill(Untouched);
    before = GdsStream::CurrentTotals();
    Words words;
    Append(words);
    SetCounters(words, Starts);
    Append(words);
    Readback(words, Unregistered.data(), CounterOffset, Counters * 4);
    Submit(kept, std::move(words), ProducerQueue);
    Idle("fallbacks: a readback without a host import");
    RequireValues(Read<Counters>(Address(Unregistered.data())), After, "fallbacks: the readback into unimported memory");
    delta = Delta(before);
    std::printf("gds in stream: fallbacks, no import: %s\n", Describe(delta).c_str());
    RequireBookkeeping(run, delta, "fallbacks: no import");
    if (run.mode != 0) Require(Recorded(delta, Op::Readback) == 0 && CopyRefusals(delta) + delta.idleCopies == 1, "fallbacks: the readback into unimported memory was recorded or not counted: " + Describe(delta));
}

// --behind-completions: reason 5 (judge fix), on a device of its own, with APS5_LABEL_GATE_ALL=1 (main sets
// it before the first label: WriteLabelOnGpu reads it once), so every label meets the completion path
// while recorded work has a completion action. A fill makes recorded work, an empty completion action
// is added to its batch, then a store refuses the completion path (`behindCompletions` false, the GDS
// stream's store into the GDS) and another accepts it (every other caller). The refused one returns 5
// with nothing recorded, noted or stamped and never lands (its completion memcpy would land at a reap,
// after later GPU users of the GDS); the accepted one returns 5 too (the gate was taken) and lands.
int BehindCompletions() {
    namespace GuestMemory = AgcDriver::GuestMemory;
    using AgcDriver::Graphics::Recorder;
    const auto device = OpenVulkanTestDevice();
    if (!device) return VulkanTestSkipped;
    auto& block = Memory();
    block.readback.fill(0);
    // Held throughout, as the driver's label paths hold it; released only after the batch finished.
    std::lock_guard lock(GuestMemory::GpuMutex());
    const std::array<std::uint32_t, 4> pattern{0x5a5a5a5au, 0x5a5a5a5au, 0x5a5a5a5au, 0x5a5a5a5au};
    if (!device->FillBuffer(Address(block.copiedIn.data()), sizeof(block.copiedIn), pattern)) {
        std::puts("skipped, the device does not import guest memory");
        return VulkanTestSkipped;
    }
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "behind completions: no active recorder after the fill");
    Require(recorder->AfterRecordedWork([] {}), "behind completions: the completion action was not added (the fill left the recorder idle)");
    const auto refused = Address(&block.readback[1]);
    const auto accepted = Address(&block.readback[2]);
    constexpr std::uint32_t RefusedValue = 0x11223344u;
    constexpr std::uint32_t AcceptedValue = 0x55667788u;
    const auto completionLabels = Recorder::PendingCompletionLabels();
    const auto stamp = GuestMemory::NewestStamp(refused, 4);
    // A compute queue's store: no opportunistic reap first, so the completion action stays pending.
    const int reason = device->WriteLabelOnGpu(refused, std::as_bytes(std::span(&RefusedValue, 1)), 1, ProducerQueue, true, false);
    Require(reason == 5, "behind completions: a store that refuses the completion path returned " + std::to_string(reason) + ", expected 5");
    Require(!recorder->PendingWriteOverlaps(refused, 4), "behind completions: the refused store left a pending write over its range");
    Require(!recorder->PendingLabelIn(refused, 4) && !recorder->CompletionLabelIn(refused, 4), "behind completions: the refused store left a label over its range");
    Require(Recorder::PendingCompletionLabels() == completionLabels, "behind completions: the refused store added a completion label");
    Require(GuestMemory::NewestStamp(refused, 4) == stamp, "behind completions: the refused store stamped its range");
    const int acceptedReason = device->WriteLabelOnGpu(accepted, std::as_bytes(std::span(&AcceptedValue, 1)), 2, ProducerQueue, true, true);
    Require(acceptedReason == 5, "behind completions: a store that accepts the completion path returned " + std::to_string(acceptedReason) + ", expected 5 (APS5_LABEL_GATE_ALL not in effect?)");
    Require(recorder->PendingWriteOverlaps(accepted, 4), "behind completions: the accepted store left no pending write over its range");
    device->SubmitRecorded(false);
    device->WaitIdle();
    Require(block.readback[1] == 0, "behind completions: the refused store landed (" + std::to_string(block.readback[1]) + ")");
    Require(block.readback[2] == AcceptedValue, "behind completions: the accepted store holds " + std::to_string(block.readback[2]) + ", expected " + std::to_string(AcceptedValue));
    RequireValues(block.copiedIn, pattern, "behind completions: the fill");
    std::puts("gds in stream behind-completions tests passed");
    return 0;
}

}

int main(int argc, char** argv) {
    const bool behindCompletions = argc > 1 && std::strcmp(argv[1], "--behind-completions") == 0;
#ifdef _WIN32
    _putenv_s("APS5_PIPELINED_DRAWS", "1");
    if (behindCompletions) _putenv_s("APS5_LABEL_GATE_ALL", "1");
#else
    setenv("APS5_PIPELINED_DRAWS", "1", 1);
    if (behindCompletions) setenv("APS5_LABEL_GATE_ALL", "1", 1);
#endif
    try {
        if (behindCompletions) return BehindCompletions();
        Run run;
        run.mode = ExpectedMode();
        Decode();
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        // Read at the first DMA_DATA that touches the GDS: the first case reads it.
        ReadbackAndStores(run);
        Require(GdsStream::Mode() == run.mode, "the switch resolved to " + std::to_string(GdsStream::Mode()) + ", expected " + std::to_string(run.mode));
        std::printf("gds in stream: mode %d, GDS operations recorded on this device: %s\n", run.mode, run.gpuGds ? "yes" : "no");
        WriteAfterRead(run);
        WaitOnReadback(run);
        CopyIn(run);
        Fallbacks(run);
        const auto totals = GdsStream::CurrentTotals();
        std::printf("gds in stream: totals: %s\n", Describe(totals).c_str());
        Require(totals.unnoted == 0, "cpu-gds-access-unnoted " + std::to_string(totals.unnoted));
        if (run.mode == 0) Require(!GdsStream::Armed(), "the switch is off but the GDS stream was armed");
        if (run.mode != 0 && RecordedAll(totals) != 0) Require(GdsStream::Armed() && totals.accessChecks != 0, "operations were recorded but no CPU access to the GDS was checked");
        AgcDriverShutdown_nid_postfix();
        std::puts("gds in stream tests passed");
        return 0;
    } catch (const std::exception& error) {
        Fail(error.what());
    }
}
