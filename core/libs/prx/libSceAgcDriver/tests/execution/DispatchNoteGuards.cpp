// LOCAL EXPERIMENT (DRAIN-PLAN D1): queue 0's ordering guards. A dispatch that is not the first
// after a gate runs with draws still queued; the notes it makes and the guest memory it reads before
// its own drain must not run ahead of a queued commit:
//   G-msaa  (APS5_EXPERIMENT_DISPATCH_NOTE_GUARDS): TakeMultisampledSingleClear records a clear color
//           that a queued draw's commit would consume early;
//   G-vfill (the same switch): NoteMemoryValueFill reads a value a queued commit writes;
//   hidden  (APS5_EXPERIMENT_HIDDEN_WRITE_DRAIN): a queued draw that may write guest memory it does
//           not declare (page-table, GDS or image-table stores, depth HTILE: Enqueue's hiddenWrites)
//           before a queue-0 dispatch's unregistered code read or a fill's input reads.
// =1 drains (one drain, and the late reader sees the committed state), =2 and the [stream] counters
// (APS5_DEBUG_STREAM_STATS=1) only count (Drain::GuardTotals), unset counts nothing. A blocker item
// holds the committer until the test sees the guard; the dispatch under test follows a fill that
// waits for the GPU mutex the test holds, so the blocker is queued behind the gate's drain.
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/DrainStats.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/BufferFill.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using namespace AgcDriver::DriverDetail;
namespace Drain = AgcDriver::Drain;

alignas(256) constexpr auto FillCode = PatternFillKernel;
// The pattern fill kernel again at its own address, registered (RegisterFillCode) for the second
// HiddenFill run: a dispatch of a registered program reads no raw code, so the fill's input reads are
// its only guarded read.
alignas(256) std::array<std::uint32_t, PatternFillKernel.size()> RegisteredFillCode = PatternFillKernel;
alignas(256) std::array<std::uint32_t, 4> Pattern{};
alignas(256) std::array<std::uint32_t, 4> Control{};
alignas(256) std::array<std::uint32_t, 260> Output{};
alignas(256) std::array<std::uint32_t, 4> ScratchPattern{0x0f0f0f0fu, 0, 0, 0};
alignas(256) std::array<std::uint32_t, 4> ScratchControl{256, 0, 0, 0};
alignas(256) std::array<std::uint32_t, 260> Scratch{};

// RADV's multisampled comp-to-single clear (Graphics::MatchMultisampledSingleClear), padded with
// s_endpgm so the raw code read decodes whole.
alignas(256) constexpr std::array<std::uint32_t, 64> MsaaClearCode = [] {
    std::array<std::uint32_t, 64> code{};
    constexpr std::array<std::uint32_t, 22> kernel{0xbe800303u, 0xbe830382u, 0xf40c0401u, 0xfa000000u, 0xd7460000u, 0x04010609u, 0xd7460001u, 0x0405060au, 0x7e060208u, 0x7e0c020bu, 0x7e0e0280u, 0xd5690004u, 0x02000100u, 0x7e000205u, 0xd5690005u, 0x02000901u, 0xd7000001u, 0x02000c80u, 0xbf8cc07fu, 0xf0201f38u, 0x00040004u, 0xbf810000u};
    for (std::size_t i = 0; i < code.size(); ++i) code[i] = i < kernel.size() ? kernel[i] : 0xbf810000u;
    return code;
}();
// A 4x multisampled T# (the canvas descriptor of GuestTextureResource's tests), base 0x21f200000.
constexpr std::array<std::uint32_t, 8> MsaaDescriptor{0x021f2000u, 0x03800000u, 0x804a004au, 0xe1b20facu, 0x00000000u, 0x00400020u, 0x00000000u, 0x00000000u};
constexpr std::uint64_t MsaaSurface = 0x21f200000ull;
constexpr std::array<std::uint32_t, 4> MsaaColor{0xff00ffccu, 0u, 0u, 0u};

// The memory-value fill kernel (Graphics::MatchMemoryValueFill, DwordPatternFill's memory-value fill).
alignas(256) constexpr std::array<std::uint32_t, 64> ValueFillCode = [] {
    std::array<std::uint32_t, 64> code{};
    constexpr std::array<std::uint32_t, 9> kernel{0xd7460000u, 0x04010c08u, 0xf4201a82u, 0xfa000000u, 0xbf8cc07fu, 0x7e02026au, 0xe0102000u, 0x80000100u, 0xbf810000u};
    for (std::size_t i = 0; i < code.size(); ++i) code[i] = i < kernel.size() ? kernel[i] : 0xbf810000u;
    return code;
}();
alignas(256) std::array<std::uint32_t, 4> Value{};
alignas(256) std::array<std::uint32_t, 260> ValueOutput{};

struct Modes {
    int guards = 0;
    int hidden = 0;
    bool stream = false;
    bool CountsGuards() const { return guards != 0 || stream; }
    bool CountsHidden() const { return hidden != 0 || stream; }
};

void Registers(std::vector<std::uint32_t>& words, std::uint32_t first, std::span<const std::uint32_t> values) {
    words.push_back(0xc0007600u | (static_cast<std::uint32_t>(values.size()) << 16u));
    words.push_back(first);
    words.insert(words.end(), values.begin(), values.end());
}

void Program(std::vector<std::uint32_t>& words, const void* code, const std::array<std::uint32_t, 3>& shape, std::span<const std::uint32_t> userData) {
    Registers(words, 0x207, shape);
    const auto address = reinterpret_cast<std::uintptr_t>(code);
    const std::array<std::uint32_t, 2> program{static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>(address >> 40u)};
    Registers(words, 0x20c, program);
    const std::array<std::uint32_t, 1> resource{(static_cast<std::uint32_t>(userData.size()) << 1u) | (1u << 7u)};
    Registers(words, 0x213, resource);
    Registers(words, 0x240, userData);
}

std::array<std::uint32_t, 4> Descriptor(const void* pointer, std::uint32_t stride, std::uint32_t count, std::uint32_t format) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) | (stride << 16u), count, format};
}

void AppendPatternFill(std::vector<std::uint32_t>& words, const void* pattern, const void* control, void* output, const void* code = FillCode.data()) {
    std::vector<std::uint32_t> data;
    for (const auto& descriptor : {Descriptor(pattern, 4, 4, 0x14004u), Descriptor(output, 4, 256, 0x14004u), Descriptor(control, 16, 1, 0x4dfacu)}) data.insert(data.end(), descriptor.begin(), descriptor.end());
    Program(words, code, {64, 1, 1}, data);
    words.insert(words.end(), {0xc0031500u, 4, 1, 1, 0x41u | 0x8000u});
}

void AppendScratchFill(std::vector<std::uint32_t>& words, const void* code = FillCode.data()) {
    AppendPatternFill(words, ScratchPattern.data(), ScratchControl.data(), Scratch.data(), code);
}

// Holds the committer until released, then runs `then` (outside the GPU mutex) and takes the
// mutex as a commit does.
struct Blocker {
    std::atomic<bool> release{false};
    std::function<void()> then;
    ~Blocker() { release = true; }
    void Enqueue(std::vector<DrawPipeline::Range> writes, bool hidden) {
        DrawPipeline::Queue0().Enqueue([this] {
            while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (then) then();
            std::lock_guard gpuLock(AgcDriver::GuestMemory::GpuMutex());
        }, std::move(writes), 0, {}, {}, DrawPipeline::ItemKind::Other, nullptr, hidden);
    }
};

// Submits with the GPU mutex held, so the queue worker stops inside the first (gate-first) fill
// after its gate drain and input reads; the blocker is queued then, ahead of everything after it.
void SubmitBehind(Blocker& blocker, std::vector<std::uint32_t>& words, std::vector<DrawPipeline::Range> writes, bool hidden) {
    std::unique_lock gpuLock(AgcDriver::GuestMemory::GpuMutex());
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    AgcDriver::Submit(&packet, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    blocker.Enqueue(std::move(writes), hidden);
}

// Releases the blocker once `seen` holds, or after `limit` (a mode that never shows it).
void ReleaseWhen(Blocker& blocker, const std::function<bool()>& seen, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!seen() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    blocker.release = true;
}

void Finish(const char* what) {
    auto idle = std::async(std::launch::async, [] { AgcDriverWaitIdle_nid_postfix(); });
    if (idle.wait_for(std::chrono::seconds(20)) != std::future_status::ready) {
        std::fprintf(stderr, "dispatch note guards (%s): the queue worker and the draw pipeline deadlocked\n", what);
        std::fflush(stderr);
        std::_Exit(1);
    }
    idle.get();
}

void Warm() {
    std::vector<std::uint32_t> words;
    AppendScratchFill(words);
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    AgcDriver::Submit(&packet, 0);
    Finish("warm-up");
}

// The T# of the MSAA clear at an address whose high dword is 2 (the kernel's fixed high word).
std::uint32_t* MsaaDescriptorBlock() {
    constexpr std::size_t Bytes = 65536;
    for (std::uint64_t attempt = 0; attempt < 64; ++attempt) {
        const auto hint = 0x210000000ull + attempt * 0x1000000ull;
#ifdef _WIN32
        auto* block = static_cast<std::uint32_t*>(VirtualAlloc(reinterpret_cast<void*>(hint), Bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        void* mapped = mmap(reinterpret_cast<void*>(hint), Bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        auto* block = mapped == MAP_FAILED ? nullptr : static_cast<std::uint32_t*>(mapped);
#endif
        if (block != nullptr && reinterpret_cast<std::uintptr_t>(block) >> 32u == 2u && AgcDriver::GuestMemory::Accessible(block, Bytes)) return block;
    }
    return nullptr;
}

void GuardMsaa(const Modes& modes, std::uint32_t* block) {
    std::memcpy(block, MsaaDescriptor.data(), sizeof(MsaaDescriptor));
    Require(!AgcDriver::Graphics::RecordedMultisampledSingleClear(MsaaSurface).has_value(), "G-msaa: a single clear was recorded before the test");
    const auto descriptor = reinterpret_cast<std::uintptr_t>(block);
    const std::array<std::uint32_t, 9> userData{0u, 0u, static_cast<std::uint32_t>(descriptor), 8u, 8u, MsaaColor[0], MsaaColor[1], MsaaColor[2], MsaaColor[3]};
    std::vector<std::uint32_t> words;
    AppendScratchFill(words);
    Program(words, MsaaClearCode.data(), {8, 8, 1}, userData);
    words.insert(words.end(), {0xc0031500u, 1, 1, 1, 0x41u});
    const auto before = Drain::GuardTotals();
    const auto msaa = static_cast<std::size_t>(Drain::Guard::Msaa);
    std::atomic<bool> sawClear{false};
    Blocker blocker;
    blocker.then = [&] { sawClear = AgcDriver::Graphics::RecordedMultisampledSingleClear(MsaaSurface).has_value(); };
    SubmitBehind(blocker, words, {}, false);
    ReleaseWhen(blocker, [&] { return Drain::GuardTotals().hits[msaa] != before.hits[msaa]; }, std::chrono::milliseconds(modes.CountsGuards() ? 10000 : 1500));
    Finish("G-msaa");
    const auto after = Drain::GuardTotals();
    const auto hits = after.hits[msaa] - before.hits[msaa];
    const auto drains = after.drains[msaa] - before.drains[msaa];
    std::printf("dispatch note guards: G-msaa hits %llu, drains %llu, the blocker saw the clear %d\n", static_cast<unsigned long long>(hits), static_cast<unsigned long long>(drains), sawClear.load() ? 1 : 0);
    Require(hits == (modes.CountsGuards() ? 1u : 0u), "G-msaa: " + std::to_string(hits) + " guard hits for one single-clear kernel behind a queued commit");
    Require(drains == (modes.guards == 1 ? 1u : 0u), "G-msaa: " + std::to_string(drains) + " guard drains");
    if (modes.guards == 1) Require(!sawClear.load(), "G-msaa: the queued commit saw the single clear the dispatch after it records");
    const auto recorded = AgcDriver::Graphics::RecordedMultisampledSingleClear(MsaaSurface);
    Require(recorded.has_value() && *recorded == MsaaColor, "G-msaa: the single clear was not recorded");
}

void GuardValueFill(const Modes& modes) {
    constexpr std::uint32_t Old = 0x11111111u;
    constexpr std::uint32_t New = 0xc0c0c0c0u;
    Value.fill(0);
    Value[0] = Old;
    ValueOutput.fill(0xcafebabeu);
    const auto output = Descriptor(ValueOutput.data(), 4, 256, 0x00014004u);
    const auto value = Descriptor(Value.data(), 0, 16, 0x00014004u);
    const std::array<std::uint32_t, 8> userData{output[0], output[1], output[2], output[3], value[0], value[1], value[2], value[3]};
    std::vector<std::uint32_t> words;
    AppendScratchFill(words);
    Program(words, ValueFillCode.data(), {64, 1, 1}, userData);
    words.insert(words.end(), {0xc0031500u, 4, 1, 1, 0x41u});
    const auto before = Drain::GuardTotals();
    const auto notesBefore = AgcDriver::Graphics::LastNotedMemoryValueFill().notes;
    const auto fill = static_cast<std::size_t>(Drain::Guard::ValueFill);
    Blocker blocker;
    const auto valueAddress = reinterpret_cast<std::uintptr_t>(Value.data());
    blocker.then = [valueAddress] {
        const std::uint32_t written = New;
        AgcDriver::GuestMemory::Write(valueAddress, std::as_bytes(std::span(&written, 1)), 4);
    };
    SubmitBehind(blocker, words, {{valueAddress, valueAddress + 4}}, false);
    // Unguarded, the fill's own input read drains for the declared write (the value is read again
    // there): seen once a drain for it is requested.
    const auto drainsBefore = DrawPipeline::DrainRequestsByOpcode(0x15);
    ReleaseWhen(blocker, [&] { return Drain::GuardTotals().hits[fill] != before.hits[fill] || (!modes.CountsGuards() && DrawPipeline::DrainRequestsByOpcode(0x15) != drainsBefore); }, std::chrono::milliseconds(10000));
    Finish("G-vfill");
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(ValueOutput.data()), sizeof(ValueOutput));
    const auto after = Drain::GuardTotals();
    const auto hits = after.hits[fill] - before.hits[fill];
    const auto drains = after.drains[fill] - before.drains[fill];
    const auto noted = AgcDriver::Graphics::LastNotedMemoryValueFill();
    std::printf("dispatch note guards: G-vfill hits %llu, drains %llu, noted 0x%08x (%llu notes)\n", static_cast<unsigned long long>(hits), static_cast<unsigned long long>(drains), noted.value, static_cast<unsigned long long>(noted.notes - notesBefore));
    Require(hits == (modes.CountsGuards() ? 1u : 0u), "G-vfill: " + std::to_string(hits) + " guard hits for one memory-value fill behind a queued commit");
    Require(drains == (modes.guards == 1 ? 1u : 0u), "G-vfill: " + std::to_string(drains) + " guard drains");
    if (modes.guards != 0) Require(noted.notes == notesBefore + 1 && noted.address == reinterpret_cast<std::uintptr_t>(ValueOutput.data()), "G-vfill: the memory-value fill was not noted once");
    if (modes.guards == 1) Require(noted.value == New, "G-vfill: the noted fill value " + std::to_string(noted.value) + " is not the one the queued commit wrote");
    for (std::size_t i = 0; i < 256; ++i) Require(ValueOutput[i] == New, "G-vfill: the fill did not store the value the queued commit wrote, at " + std::to_string(i));
}

// A fill whose pattern a queued item rewrites at its commit (no writes declared, flagged hidden),
// behind the gate-first scratch fill. Unregistered fill code: the dispatch prologue's raw code read
// is the first guarded read and drains (=1), which leaves nothing queued for the input guard.
// Registered (both fills; RegisterFillCode): no raw code read, so drainPendingInputWrites' hidden
// branch is the guard that drains.
void HiddenFill(const Modes& modes, bool registered) {
    constexpr std::uint32_t Old = 0x5a5a5a5au;
    constexpr std::uint32_t New = 0x3c3c3c3cu;
    Output.fill(0xcafebabeu);
    Pattern = {Old, 0, 0, 0};
    Control = {256, 0, 0, 0};
    const void* code = registered ? static_cast<const void*>(RegisteredFillCode.data()) : static_cast<const void*>(FillCode.data());
    std::vector<std::uint32_t> words;
    AppendScratchFill(words, code);
    AppendPatternFill(words, Pattern.data(), Control.data(), Output.data(), code);
    const auto before = Drain::GuardTotals();
    const auto shader = static_cast<std::size_t>(Drain::Guard::HiddenShader);
    const auto inputs = static_cast<std::size_t>(Drain::Guard::HiddenFill);
    Blocker blocker;
    const auto patternAddress = reinterpret_cast<std::uintptr_t>(Pattern.data());
    blocker.then = [patternAddress] {
        const std::uint32_t written = New;
        AgcDriver::GuestMemory::Write(patternAddress, std::as_bytes(std::span(&written, 1)), 4);
    };
    // A draw that writes the pattern through a pointer it does not declare: no writes, flagged hidden.
    SubmitBehind(blocker, words, {}, true);
    const auto seen = [&] {
        const auto now = Drain::GuardTotals();
        return now.hits[shader] + now.hits[inputs] != before.hits[shader] + before.hits[inputs];
    };
    const std::string name = registered ? "hidden write (registered fill)" : "hidden write";
    ReleaseWhen(blocker, seen, std::chrono::milliseconds(modes.CountsHidden() ? 10000 : 1500));
    Finish(name.c_str());
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(Output.data()), sizeof(Output));
    const auto after = Drain::GuardTotals();
    const auto shaderHits = after.hits[shader] - before.hits[shader];
    const auto shaderDrains = after.drains[shader] - before.drains[shader];
    const auto fillHits = after.hits[inputs] - before.hits[inputs];
    const auto fillDrains = after.drains[inputs] - before.drains[inputs];
    std::printf("dispatch note guards: %s: code-read hits %llu, drains %llu; fill-input hits %llu, drains %llu; fill 0x%08x\n", name.c_str(), static_cast<unsigned long long>(shaderHits), static_cast<unsigned long long>(shaderDrains), static_cast<unsigned long long>(fillHits), static_cast<unsigned long long>(fillDrains), Output[0]);
    // The guard that meets the queued item first, and the other one.
    const auto firstHits = registered ? fillHits : shaderHits;
    const auto firstDrains = registered ? fillDrains : shaderDrains;
    const auto laterHits = registered ? shaderHits : fillHits;
    const auto laterDrains = registered ? shaderDrains : fillDrains;
    const char* firstName = registered ? "fill-input" : "code-read";
    const char* laterName = registered ? "code-read" : "fill-input";
    Require(modes.hidden == 1 ? firstHits == 1 : modes.CountsHidden() ? firstHits >= 1 : firstHits == 0, name + ": " + std::to_string(firstHits) + " " + firstName + " hits");
    Require(firstDrains == (modes.hidden == 1 ? 1u : 0u), name + ": " + std::to_string(firstDrains) + " " + firstName + " drains, expected " + (modes.hidden == 1 ? "one" : "none"));
    Require(laterDrains == 0, name + ": " + std::to_string(laterDrains) + " " + laterName + " drains after the first guard");
    // Under =1 the first guard's drain committed the item; a registered program reads no raw code.
    // Counting only, the item is released at the first hit, so a later input read may still see it.
    if (modes.hidden == 1 || !modes.CountsHidden() || registered) Require(laterHits == 0, name + ": " + std::to_string(laterHits) + " " + laterName + " hits after the first guard");
    if (modes.hidden == 1) {
        for (std::size_t i = 0; i < 256; ++i) Require(Output[i] == New, name + ": the fill read its pattern ahead of the queued hidden write, at " + std::to_string(i));
    }
}

void RegisterFillCode() {
    alignas(8) static ::Shader shader{};
    shader.file_header = 0x34333231u;
    shader.version = 0x18;
    shader.header_size = sizeof(::Shader);
    shader.shader_size = sizeof(RegisteredFillCode);
    shader.code = RegisteredFillCode.data();
    AgcDriverRegisterShader_nid_postfix(&shader);
}

// The raw code read alone, without a device (as tests/execution/QueuedRawShader.cpp): a queued item
// flagged hidden that rewrites the code word. A read outside a queue-0 dispatch prologue (draw
// decode, draw-cache validation, another queue's dispatch) is not guarded; the prologue's is.
void HiddenRawShader(const Modes& modes) {
    alignas(256) static std::array<std::uint32_t, 64> code{};
    code.fill(0xbf810000u);
    code[0] = 0x7e000200u;
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    const auto before = Drain::GuardTotals();
    const auto shader = static_cast<std::size_t>(Drain::Guard::HiddenShader);
    std::atomic<bool> started{false};
    std::atomic<bool> release{false};
    DrawPipeline::Queue0().Enqueue([&, address] {
        started.store(true, std::memory_order_release);
        while (!release.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const std::uint32_t replacement = 0x7e0002f2u;
        AgcDriver::GuestMemory::Write(address, std::as_bytes(std::span(&replacement, 1)), alignof(std::uint32_t));
    }, {}, 0, {}, {}, DrawPipeline::ItemKind::Other, nullptr, true);
    while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
    Require(DrawPipeline::Queue0().HiddenWritesQueued() == 1, "hidden raw shader: a hidden-write item was not counted while queued");
    const bool active = DrawPipeline::Active();
    DrawPipeline::Active() = true;
    std::shared_ptr<const ShaderSnapshot> unguarded;
    std::shared_ptr<const ShaderSnapshot> read;
    std::thread releaser;
    try {
        // Not a queue-0 dispatch prologue's read: nothing counted or drained, the code as it is.
        unguarded = ReadRawShader(address);
        const auto between = Drain::GuardTotals();
        Require(between.hits[shader] == before.hits[shader] && between.drains[shader] == before.drains[shader], "hidden raw shader: a read outside a queue-0 dispatch prologue was guarded");
        // Released from here when the prologue's read drains (=1): the drain waits for this item.
        releaser = std::thread([&] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(modes.hidden == 1 ? 10000 : 200);
            while (Drain::GuardTotals().drains[shader] == before.drains[shader] && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            release.store(true, std::memory_order_release);
        });
        read = ReadRawShader(address, true);
    } catch (...) {
        DrawPipeline::Active() = active;
        release.store(true, std::memory_order_release);
        if (releaser.joinable()) releaser.join();
        throw;
    }
    DrawPipeline::Active() = active;
    releaser.join();
    Require(unguarded->code.front() == 0x7e000200u, "hidden raw shader: the unguarded read did not see the code as it was");
    DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
    const auto after = Drain::GuardTotals();
    Require(DrawPipeline::Queue0().HiddenWritesQueued() == 0, "hidden raw shader: a committed hidden-write item is still counted");
    const auto hits = after.hits[shader] - before.hits[shader];
    const auto drains = after.drains[shader] - before.drains[shader];
    std::printf("dispatch note guards: hidden raw shader hits %llu, drains %llu, code 0x%08x\n", static_cast<unsigned long long>(hits), static_cast<unsigned long long>(drains), read->code.front());
    Require(hits == (modes.CountsHidden() ? 1u : 0u), "hidden raw shader: " + std::to_string(hits) + " hits");
    Require(drains == (modes.hidden == 1 ? 1u : 0u), "hidden raw shader: " + std::to_string(drains) + " drains");
    if (modes.hidden == 1) Require(read->code.front() == 0x7e0002f2u, "hidden raw shader: the raw code read ran ahead of a queued hidden write to the code");
}

// LOCAL DEBUG (DRAIN-PLAN C0): the [stream] window tag on the r71/r72 stats runs' windows (presents,
// draws committed, flush-guard queue-0 lease wait, the window before).
void WindowTags() {
    using Drain::Window;
    constexpr std::uint64_t Ms = 1'000'000;
    struct Case {
        const char* what;
        std::uint64_t presents, draws, leaseNs;
        Window previous, expected;
    };
    constexpr std::array<Case, 9> cases{{
        {"boot menu (r71-stats L2098)", 294, 7028, 0, Window::Menu, Window::Menu},
        {"menu with lease waits (r72-on-stats L42409)", 287, 14019, 0, Window::Menu, Window::Menu},
        {"no presents (loading)", 0, 5000, 0, Window::Hold, Window::Menu},
        {"menu frames then fly-in (r71-stats L52559)", 113, 39984, 897 * Ms, Window::Menu, Window::Entry},
        {"hub entry as heavy as the hold (r71-on-stats L52071)", 45, 91553, 841 * Ms, Window::Menu, Window::Entry},
        {"hub entry after a heavy window, lease waits", 50, 105401, 786 * Ms, Window::Hold, Window::Entry},
        {"hold right after the entry (r71-on-stats L53274)", 110, 238848, 0, Window::Entry, Window::Hold},
        {"hold with 72 presents and one lease wait (r71-stats L63457)", 72, 162119, 3 * Ms, Window::Entry, Window::Hold},
        {"hold with 71 presents (r72-on-stats L62862)", 71, 157919, 0, Window::Hold, Window::Hold},
    }};
    for (const auto& item : cases) {
        const auto tagged = Drain::ClassifyWindow(item.presents, item.draws, item.leaseNs, item.previous);
        Require(tagged == item.expected, std::string("window tag: ") + item.what + " is tagged " + Drain::WindowName(tagged) + ", not " + Drain::WindowName(item.expected));
    }
}

}

int main() {
#ifdef _WIN32
    _putenv_s("APS5_PIPELINED_DRAWS", "1");
#else
    setenv("APS5_PIPELINED_DRAWS", "1", 1);
#endif
    try {
        Modes modes;
        modes.guards = Drain::NoteGuards();
        modes.hidden = Drain::HiddenWriteDrain();
        modes.stream = AgcDriver::Stream::On();
        std::printf("dispatch note guards: guards=%d hidden=%d stream=%d\n", modes.guards, modes.hidden, modes.stream ? 1 : 0);
        WindowTags();
        if (DrawPipeline::Depth() == 0) return 77;
        HiddenRawShader(modes);
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        Warm();
        if (auto* block = MsaaDescriptorBlock()) {
            GuardMsaa(modes, block);
        } else {
            std::puts("dispatch note guards: G-msaa not run, no memory at a 0x2'xxxx'xxxx address");
        }
        GuardValueFill(modes);
        HiddenFill(modes, false);
        RegisterFillCode();
        HiddenFill(modes, true);
        AgcDriverShutdown_nid_postfix();
        std::puts("dispatch note guards tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
