#include "prx/libSceAgcDriver/Execution/include/GdsStream.hpp"
#include "prx/libSceAgcDriver/Execution/include/DrainStats.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/StreamStats.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

// LOCAL EXPERIMENT (APS5_EXPERIMENT_GDS_IN_STREAM, DRAIN-PLAN D2): see GdsStream.hpp.
namespace AgcDriver::GdsStream {

namespace {

using Stream::Add;
using Stream::Counter;

constexpr std::array<const char*, Ops> OpNames{"readback", "gds-copy", "copy-in", "store", "store-mem", "store-gds"};
constexpr std::array<const char*, Fallbacks> FallbackNames{"r1", "r2", "r3", "r4", "r5", "store-shape", "copy-completions", "copy-no-import", "copy-inaccessible", "copy-high-word", "copy-range", "copy-size", "copy-shape"};

// Cumulative; relaxed atomics (the report and the tests read them).
std::array<Counter, Ops> recorded{};
std::array<Counter, 4> copyPaths{};
Counter notes{};
Counter idleCopies{};
Counter readbacksOff{};
std::array<Counter, Fallbacks> fallbacks{};
Counter targetedSyncs{};
Counter accessChecks{};
Counter unnoted{};
Counter verified{};
Counter explained{};

// The invariant's state, written under GpuMutex by NoteRecorded: whether an operation was ever recorded,
// the batch the newest one went into (the open batch's serial; 0: none) and its recorder; and the
// newest such serial a check found finished (a lock-free hint that lets the guest-access check skip
// its lock until the next operation; reset when the recorder changes).
std::atomic<bool> armed{false};
std::atomic<std::uint64_t> opSerial{0};
std::atomic<const void*> opRecorder{nullptr};
std::atomic<std::uint64_t> settledThrough{0};

bool overlapsGds(std::uint64_t address, std::size_t bytes) {
    const auto gds = Pm4::GdsAddress();
    return bytes != 0 && address < gds + Pm4::GdsBytes && gds < address + bytes;
}

// Under GpuMutex.
void checkLocked(const Graphics::Recorder& recorder) {
    const auto serial = opSerial.load(std::memory_order_relaxed);
    if (serial == 0 || opRecorder.load(std::memory_order_relaxed) != &recorder) return;
    Add(accessChecks);
    const auto finished = recorder.FinishedSerial();
    if (serial <= finished) {
        if (serial > settledThrough.load(std::memory_order_relaxed)) settledThrough.store(serial, std::memory_order_release);
        return;
    }
    // The newest unfinished batch that notes a write of the GDS (the open one counts with the serial it
    // will get): at or after the operation's batch, the access waits for it.
    const auto noted = recorder.PendingWriterSerial(Pm4::GdsAddress(), Pm4::GdsBytes);
    if (noted >= serial) return;
    Add(unnoted);
    static std::atomic<int> shown{0};
    if (shown.fetch_add(1, std::memory_order_relaxed) < 8) {
        std::fprintf(stderr, "[gds-stream] cpu-gds-access-unnoted: a CPU access to the GDS while the operation recorded into batch %llu is unfinished (finished through %llu) and the newest batch noting a GDS write is %llu\n", static_cast<unsigned long long>(serial), static_cast<unsigned long long>(finished), static_cast<unsigned long long>(noted));
    }
}

unsigned long long count(std::uint64_t value) {
    return static_cast<unsigned long long>(value);
}

// Formatted with printf's checks; a piece is cut at 191 characters (as the D0 lines').
__attribute__((format(gnu_printf, 2, 3))) void append(std::string& text, const char* format, ...) {
    char item[192];
    std::va_list args;
    va_start(args, format);
    const auto size = std::vsnprintf(item, sizeof(item), format, args);
    va_end(args);
    if (size > 0) text.append(item, std::min(static_cast<std::size_t>(size), sizeof(item) - 1));
}

std::mutex reportMutex;
Totals reported{};

void maybeReportAlone() {
    static std::atomic<std::int64_t> last{0};
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto previous = last.load(std::memory_order_relaxed);
    if (previous == 0) {
        last.compare_exchange_strong(previous, now);
        return;
    }
    if (now - previous < 10000 || !last.compare_exchange_strong(previous, now)) return;
    ReportWindow("-", true);
}

void counted() {
    if (!Stream::On()) maybeReportAlone();
}

}

int mode() {
    const char* text = std::getenv("APS5_EXPERIMENT_GDS_IN_STREAM");
    if (text == nullptr || text[0] == '\0' || std::strcmp(text, "0") == 0) return 0;
    const int requested = std::strcmp(text, "2") == 0 ? 2 : 1;
    // The targeted path's own test (PacketMemory): every fallback store into the GDS takes it.
    if (std::getenv("APS5_EXPERIMENT_GDS_TARGETED_SYNC") == nullptr) {
        std::fprintf(stderr, "[gds-stream] APS5_EXPERIMENT_GDS_IN_STREAM=%d needs APS5_EXPERIMENT_GDS_TARGETED_SYNC=1 (its path is the fallback): off\n", requested);
        return 0;
    }
    // enqueueDmaPacket's own test.
    if (std::getenv("APS5_PIPELINE_DMA") != nullptr) {
        std::fprintf(stderr, "[gds-stream] APS5_EXPERIMENT_GDS_IN_STREAM=%d is refused with APS5_PIPELINE_DMA (its queue-0 DMA items can store into the GDS on the CPU ahead of queued draws): off\n", requested);
        return 0;
    }
    // R's switch read here only for the line (ReadbackCopies reads it again, at its first use).
    const char* readbacks = std::getenv("APS5_EXPERIMENT_GDS_READBACK_COPIES");
    const bool withReadbacks = readbacks != nullptr && readbacks[0] != '\0' && std::strcmp(readbacks, "0") != 0;
    std::fprintf(stderr, "[gds-stream] APS5_EXPERIMENT_GDS_IN_STREAM=%d: DMA_DATA transfers that touch the GDS are recorded in the stream, halves %s%s\n", requested, withReadbacks ? "S+copy-in+R (APS5_EXPERIMENT_GDS_READBACK_COPIES: readbacks recorded as GPU copies add no label history)" : "S+copy-in (readbacks and GDS -> GDS copies take today's path; APS5_EXPERIMENT_GDS_READBACK_COPIES=1 adds them)", requested == 2 ? ", each value-checked" : "");
    return requested;
}

bool readbackCopies() {
    const char* text = std::getenv("APS5_EXPERIMENT_GDS_READBACK_COPIES");
    if (text == nullptr || text[0] == '\0' || std::strcmp(text, "0") == 0) return false;
    if (!On()) {
        std::fprintf(stderr, "[gds-stream] APS5_EXPERIMENT_GDS_READBACK_COPIES needs APS5_EXPERIMENT_GDS_IN_STREAM (on, with its requirements): off\n");
        return false;
    }
    return true;
}

const char* OpName(Op op) {
    const auto index = static_cast<std::size_t>(op);
    return index < Ops ? OpNames[index] : "?";
}

const char* FallbackName(Fallback reason) {
    const auto index = static_cast<std::size_t>(reason);
    return index < Fallbacks ? FallbackNames[index] : "?";
}

Fallback StoreFallback(int reason) {
    switch (reason) {
        case 1: return Fallback::StoreCpu;
        case 2: return Fallback::StoreDrain;
        case 3: return Fallback::StoreNoImport;
        case 5: return Fallback::StoreBehindCompletions;
        default: return Fallback::StoreUnsuitable;
    }
}

void NoteRecorded(Graphics::Recorder& recorder, Op op, int copyPath) {
    // Judge fix 1: the whole GDS, as every GPU user of it notes it (G/ShaderResources.cpp's Gds role),
    // into the open batch that holds the operation, under the hold that recorded it.
    recorder.NotePendingWrite(Pm4::GdsAddress(), Pm4::GdsBytes);
    // Review fix 1 (submit liveness): the trigger a label store arms, so the batch goes out as today's
    // label store's would (the note above opened the batch if the record had submitted it).
    recorder.ArmLabelSubmit();
    Add(notes);
    if (opRecorder.exchange(&recorder, std::memory_order_relaxed) != &recorder) settledThrough.store(0, std::memory_order_relaxed);
    // The note opened the batch if the operation's record had submitted it: the serial is then the
    // note's batch, later than the operation's, which only delays the check's finished test.
    opSerial.store(recorder.OpenSerial(), std::memory_order_release);
    Add(recorded[static_cast<std::size_t>(op)]);
    if (copyPath > 0 && copyPath < static_cast<int>(copyPaths.size())) Add(copyPaths[static_cast<std::size_t>(copyPath)]);
    // GuestMemory calls the check through the registered counter (it links without this file).
    if (!armed.exchange(true, std::memory_order_acq_rel)) GuestMemory::SetGuestAccessCounter(&Drain::GuestAccess);
    counted();
}

void CountFallback(Fallback reason) {
    const auto index = static_cast<std::size_t>(reason);
    if (index < Fallbacks) Add(fallbacks[index]);
    counted();
}

void CountIdleCopy() {
    Add(idleCopies);
    counted();
}

void CountReadbackOff() {
    Add(readbacksOff);
    counted();
}

void CountTargetedSync() {
    Add(targetedSyncs);
}

bool Armed() {
    return armed.load(std::memory_order_relaxed);
}

void CheckTargetedSync(const Graphics::Recorder& recorder) {
    if (Armed()) checkLocked(recorder);
}

void CheckGuestAccess(std::uint64_t address, std::size_t bytes) {
    if (!overlapsGds(address, bytes)) return;
    const auto serial = opSerial.load(std::memory_order_acquire);
    if (serial == 0 || serial <= settledThrough.load(std::memory_order_acquire)) return;
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Hook);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    if (const auto* recorder = Graphics::Recorder::Active()) checkLocked(*recorder);
}

Expected ExpectCopy(std::uint64_t source, std::size_t bytes) {
    Expected expected;
    expected.source = source;
    expected.bytes.resize(bytes);
    // Before the read (its lock stamps the host writes that ended so far, which the read then sees): a
    // host write that ends during or after the read is stamped newer, so it can explain the mismatch it
    // causes.
    expected.generation = GuestMemory::TrackerGeneration();
    // Through the flush hook: under the caller's hold its SyncThrough of the source waits locked, and
    // storage results pending over the source are stored first.
    GuestMemory::Read(source, expected.bytes);
    return expected;
}

Expected ExpectStore(std::span<const std::byte> bytes) {
    Expected expected;
    expected.bytes.assign(bytes.begin(), bytes.end());
    expected.generation = GuestMemory::TrackerGeneration();
    return expected;
}

void VerifyRecorded(Graphics::Recorder& recorder, std::uint32_t queue, Op op, std::uint64_t destination, const Expected& expected) {
    const auto bytes = expected.bytes.size();
    if (bytes == 0) return;
    // Locked: the caller's hold is kept, so nothing is recorded between the operation and the compare.
    recorder.SyncThrough(destination, bytes);
    Add(verified);
    if (std::memcmp(reinterpret_cast<const void*>(destination), expected.bytes.data(), bytes) == 0) return;
    // Explained only by a CPU stamp over a page of the source or the destination since Expected: an H
    // host write, or a CPU store a collect stamped. Never by a driver stamp: the operation's own
    // MarkWritten (a GDS -> GDS copy stamps the source's 64 KiB block too), or a write-back that the
    // reaps inside the SyncThrough above ran (a stale one over the destination is the defect this
    // check exists for).
    if (GuestMemory::CpuWrittenSince(destination, bytes, expected.generation) || (expected.source != 0 && GuestMemory::CpuWrittenSince(expected.source, bytes, expected.generation))) {
        Add(explained);
        return;
    }
    std::fprintf(stderr, "[gds-stream] =2 disagreement queue 0x%x kind %s dst 0x%llx bytes %zu\n", queue, OpName(op), static_cast<unsigned long long>(destination), bytes);
    std::fflush(stderr);
    std::abort();
}

Totals CurrentTotals() {
    Totals totals;
    for (std::size_t op = 0; op < Ops; ++op) totals.recorded[op] = recorded[op].load(std::memory_order_relaxed);
    for (std::size_t path = 0; path < copyPaths.size(); ++path) totals.copyPaths[path] = copyPaths[path].load(std::memory_order_relaxed);
    totals.notes = notes.load(std::memory_order_relaxed);
    totals.idleCopies = idleCopies.load(std::memory_order_relaxed);
    totals.readbacksOff = readbacksOff.load(std::memory_order_relaxed);
    for (std::size_t reason = 0; reason < Fallbacks; ++reason) totals.fallbacks[reason] = fallbacks[reason].load(std::memory_order_relaxed);
    totals.targetedSyncs = targetedSyncs.load(std::memory_order_relaxed);
    totals.accessChecks = accessChecks.load(std::memory_order_relaxed);
    totals.unnoted = unnoted.load(std::memory_order_relaxed);
    totals.verified = verified.load(std::memory_order_relaxed);
    totals.explained = explained.load(std::memory_order_relaxed);
    return totals;
}

void ReportWindow(const char* window, bool print) {
    if (Mode() == 0) return;
    std::lock_guard lock(reportMutex);
    const auto now = CurrentTotals();
    const auto last = reported;
    reported = now;
    if (!print) return;
    std::string line;
    append(line, "[gds-stream] window=%s mode=%d halves %s recorded", window, Mode(), ReadbackCopies() ? "S+copy-in+R" : "S+copy-in");
    for (std::size_t op = 0; op < Ops; ++op) append(line, "%s %s %llu", op == 0 ? "" : ",", OpNames[op], count(now.recorded[op] - last.recorded[op]));
    append(line, "; copy paths 1/3 %llu/%llu; whole-GDS notes %llu; idle copies (read on the CPU without a wait) %llu", count(now.copyPaths[1] - last.copyPaths[1]), count(now.copyPaths[3] - last.copyPaths[3]), count(now.notes - last.notes), count(now.idleCopies - last.idleCopies));
    if (!ReadbackCopies()) append(line, "; readbacks on today's path (R off) %llu", count(now.readbacksOff - last.readbacksOff));
    line += "; fallbacks to today's path:";
    bool first = true;
    for (std::size_t reason = 0; reason < Fallbacks; ++reason) {
        const auto n = now.fallbacks[reason] - last.fallbacks[reason];
        if (n == 0) continue;
        append(line, "%s %s %llu", first ? "" : ",", FallbackNames[reason], count(n));
        first = false;
    }
    if (first) line += " none";
    append(line, "; targeted syncs %llu; cpu-gds-access checks %llu, cpu-gds-access-unnoted %llu", count(now.targetedSyncs - last.targetedSyncs), count(now.accessChecks - last.accessChecks), count(now.unnoted - last.unnoted));
    if (Verify()) append(line, "; =2 verified %llu, explained by a CPU stamp %llu", count(now.verified - last.verified), count(now.explained - last.explained));
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
}

}
