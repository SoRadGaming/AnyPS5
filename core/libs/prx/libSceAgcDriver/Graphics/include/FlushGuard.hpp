#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_FLUSHGUARD_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_FLUSHGUARD_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// LOCAL DEBUG (APS5_DEBUG_STREAM_STATS=1, PARALLEL-COMMIT-PLAN section 5, "Queue-0 GPU waits in
// dispatch preparation"): the waits of StorageTexture::FlushPending's hand-over guard (9fffceaa9:
// with another image's results pending over the range and an unfinished batch writing it or
// holding it in an address-based lease, the flush syncs the open batch, or finishes the in-flight
// batches up to that writer, before it stores). Each wait is classified by F1's predicate (the
// candidate fix that skips the wait, StorageTexture::guardWaitClass), first match in this order:
// - AliasCopy: the flush of an alias copy's destination (VulkanDevice's AliasCopy notes the copy
//   without stamping it and flushes at once); F1 leaves it as it is;
// - Lease: an unfinished batch reads a scanned range through an address-based lease (its pages are
//   marked written only when the batch's completion runs);
// - CopiedWriter: a listed copied writer (a dispatch's or a draw's write-back: a CPU store when its
//   batch's completion runs) writes a scanned range;
// - CompletionLabel: a completion label (stored by a completion action, or stored on the GPU with a
//   completion store behind a CPU write-back) or a label a worker queued lies in a scanned range;
// - CompletionMark: an unfinished batch holds a write-back completion (GuestBufferMemory::WriteBack
//   of a non-reusable build: a listed copied writer, or a fault-checked build) whose written ranges
//   (any kind: its direct and copied-back regions are stamped once more by GuestMemory::MarkWritten
//   when the completion runs) lie in a scanned range (Recorder::NoteCompletionMarks);
// - Untracked: a flushed image lies outside the write watch (its write-back compares the guest
//   bytes, compareUntracked);
// - Baseline: a pending unit of a flushed image holds a byte baseline (`original`, or the generation
//   baseline: unchangedSinceBaseline compares the bytes of its stamped blocks);
// - NoImport: a flushed image has no host import (its write-back is the CPU round trip, which reads
//   the guest bytes and waits for the recorder by itself);
// - Stamped: none of these. Every unfinished writer of the ranges stamped its blocks when it was
//   recorded, so the write-back's keep decision reads the same stamps with or without the wait, and
//   F1 skips it ("would skip").
// The scanned ranges are the 64 KiB blocks of the access, of every flushed image's surface and of
// its DCC keys (the write tracker, the keep rule and the unit shadow judge whole blocks: a completion
// stamp in the part of a stored block outside the surface, landing after the write-back's settle,
// makes the unit stale and drops it from the shadow), over the open batch and every in-flight one.
// Counters: relaxed, constant-initialized; DrawPipeline's [stream] report takes them
// (TakeFlushGuardCounts).
namespace AgcDriver::Graphics {

enum class FlushWaitClass : std::uint8_t { Stamped, AliasCopy, Lease, CopiedWriter, CompletionLabel, CompletionMark, Untracked, Baseline, NoImport, Count };
constexpr std::size_t FlushWaitClasses = static_cast<std::size_t>(FlushWaitClass::Count);
inline constexpr std::array<const char*, FlushWaitClasses> FlushWaitClassNames{"stamped", "alias-copy", "lease", "copied-writer", "completion-label", "completion-mark", "untracked", "baseline", "no-import"};
// The guard's two waits: the writer is the open batch (Recorder::Sync) or in flight (FinishUpTo).
constexpr std::size_t FlushWaitKinds = 2;
inline constexpr std::array<const char*, FlushWaitKinds> FlushWaitKindNames{"sync", "finish"};
// How the batch the guard waits for (Recorder::PendingWriterSerial over the access) matched: by a
// write note, by an address-based lease, or by both.
constexpr std::size_t FlushWriterMatches = 3;
inline constexpr std::array<const char*, FlushWriterMatches> FlushWriterMatchNames{"note", "lease", "both"};

using FlushCounter = std::atomic<std::uint64_t>;

struct FlushGuardCounts {
    // Per wait kind and class (the first condition that held): the waits, their wall time (the Sync
    // or FinishUpTo call, its submit and completions included), their GPU wait (the thread's
    // Stream::ThreadGpuWait over the call) and that GPU wait when the thread was queue 0.
    std::array<std::array<FlushCounter, FlushWaitClasses>, FlushWaitKinds> waits{};
    std::array<std::array<FlushCounter, FlushWaitClasses>, FlushWaitKinds> wallNs{};
    std::array<std::array<FlushCounter, FlushWaitClasses>, FlushWaitKinds> gpuNs{};
    std::array<std::array<FlushCounter, FlushWaitClasses>, FlushWaitKinds> queue0GpuNs{};
    // Per wait kind: how the waited-for batch matched.
    std::array<std::array<FlushCounter, FlushWriterMatches>, FlushWaitKinds> matches{};
    // Every condition that held, not only the first (Stamped: none held), over both kinds.
    std::array<FlushCounter, FlushWaitClasses> conditions{};
};

inline constinit FlushGuardCounts flushGuardCounts;

// What the report reads: the counters above, each exchanged with 0.
struct FlushGuardTotals {
    std::array<std::array<std::uint64_t, FlushWaitClasses>, FlushWaitKinds> waits{};
    std::array<std::array<std::uint64_t, FlushWaitClasses>, FlushWaitKinds> wallNs{};
    std::array<std::array<std::uint64_t, FlushWaitClasses>, FlushWaitKinds> gpuNs{};
    std::array<std::array<std::uint64_t, FlushWaitClasses>, FlushWaitKinds> queue0GpuNs{};
    std::array<std::array<std::uint64_t, FlushWriterMatches>, FlushWaitKinds> matches{};
    std::array<std::uint64_t, FlushWaitClasses> conditions{};
};

inline FlushGuardTotals TakeFlushGuardCounts() {
    auto& counts = flushGuardCounts;
    const auto take = [](FlushCounter& counter) { return counter.exchange(0, std::memory_order_relaxed); };
    FlushGuardTotals totals;
    for (std::size_t kind = 0; kind < FlushWaitKinds; ++kind) {
        for (std::size_t index = 0; index < FlushWaitClasses; ++index) {
            totals.waits[kind][index] = take(counts.waits[kind][index]);
            totals.wallNs[kind][index] = take(counts.wallNs[kind][index]);
            totals.gpuNs[kind][index] = take(counts.gpuNs[kind][index]);
            totals.queue0GpuNs[kind][index] = take(counts.queue0GpuNs[kind][index]);
        }
        for (std::size_t match = 0; match < FlushWriterMatches; ++match) totals.matches[kind][match] = take(counts.matches[kind][match]);
    }
    for (std::size_t index = 0; index < FlushWaitClasses; ++index) totals.conditions[index] = take(counts.conditions[index]);
    return totals;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_NARROW_FLUSH_SYNC, PARALLEL-COMMIT-PLAN section 5, F1): the guard
// skips its wait when F1's predicate holds (class Stamped above) and stores at once: every unfinished
// writer of the scanned blocks stamped them when it was recorded, the GPU runs it before the
// write-back recorded behind it, and no completion of those batches stores into the blocks on the
// CPU or stamps them again. Otherwise it waits as today. The switch changes the guard alone. 1 uses
// it, and counts the completion stores and marks (a copied writer's write-back via NoteWrittenBack,
// a completion label's store, a write-back completion's or a lease's MarkWritten via
// Recorder::NoteCompletionMarked) over a skipped flush's blocks that were registered before its
// write-back and land after it (must stay 0; a tagged line each, the first 8). 2 verifies it: the
// guard waits as today wherever the predicate would skip, and aborts with a [narrow-flush] line when
// a completion stored into or marked the blocks during the wait, or when a flushed image's keep
// decision (each piece of its write-back's 64 KiB blocks: stored, or left to the CPU) differs before
// and after the wait in a block no CPU store was collected in meanwhile (a difference only in blocks
// a CPU store newly stamped is counted apart: today's own race, not F1's). Unset or 0: today's guard
// alone. The CPU round-trip write-back (no import) and the alias copy's flush keep their wait
// (classes NoImport and AliasCopy).
inline int NarrowFlushSync() {
    static const int mode = [] {
        const char* text = std::getenv("APS5_EXPERIMENT_NARROW_FLUSH_SYNC");
        if (text == nullptr) return 0;
        const auto value = std::strtol(text, nullptr, 10);
        return value == 0 ? 0 : value == 2 ? 2 : 1;
    }();
    return mode;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_REFRESH_LEASE_WAIT=1, PARALLEL-COMMIT-PLAN section 5, apart from
// F1): StorageTexture::Refresh's own write-back of its pending results (writeBackLayers(stored))
// first waits, as the guard does, for an unfinished batch holding an address-based lease over the
// image's surface or keys (its keep decision would miss the lease's marks, which land at completion:
// the 9fffceaa9 hole on that path). Nothing waits there today. A behaviour change of its own (the
// lease writer's blocks are then kept for the CPU instead of taking the image's texels, and the
// image re-uploads them), so it has its own switch and is compared with control on its own. Unset or
// 0: no wait.
inline bool RefreshLeaseWait() {
    static const bool on = [] {
        const char* text = std::getenv("APS5_EXPERIMENT_REFRESH_LEASE_WAIT");
        return text != nullptr && std::strtol(text, nullptr, 10) != 0;
    }();
    return on;
}

// Relaxed, constant-initialized; the [stream] report takes them (TakeNarrowFlushCounts).
struct NarrowFlushCounts {
    // =1: guard waits skipped, and completion stores or marks behind a skipped write-back (must
    // stay 0).
    FlushCounter skipped{};
    FlushCounter storesBehindSkip{};
    // =2: waits the predicate would skip, made and checked; completion stores or marks over the
    // blocks during them and keep decisions that differed in a block without a new CPU stamp (each
    // aborts); keep decisions that differed only in blocks a CPU store stamped during the wait.
    FlushCounter verified{};
    FlushCounter storesDuringWait{};
    FlushCounter keepDiffered{};
    FlushCounter keepRaced{};
    // APS5_EXPERIMENT_REFRESH_LEASE_WAIT: Refresh's own write-back waiting for an address-based
    // lease: the waits, their wall time, their GPU wait (the thread's Stream::ThreadGpuWait over the
    // wait) and that GPU wait when the thread was queue 0.
    FlushCounter refreshLeaseWaits{};
    FlushCounter refreshLeaseNs{};
    FlushCounter refreshLeaseGpuNs{};
    FlushCounter refreshLeaseQueue0GpuNs{};
};

inline constinit NarrowFlushCounts narrowFlushCounts;

struct NarrowFlushTotals {
    std::uint64_t skipped = 0, storesBehindSkip = 0, verified = 0, storesDuringWait = 0, keepDiffered = 0, keepRaced = 0, refreshLeaseWaits = 0, refreshLeaseNs = 0, refreshLeaseGpuNs = 0, refreshLeaseQueue0GpuNs = 0;
};

inline NarrowFlushTotals TakeNarrowFlushCounts() {
    auto& counts = narrowFlushCounts;
    const auto take = [](FlushCounter& counter) { return counter.exchange(0, std::memory_order_relaxed); };
    NarrowFlushTotals totals;
    totals.skipped = take(counts.skipped);
    totals.storesBehindSkip = take(counts.storesBehindSkip);
    totals.verified = take(counts.verified);
    totals.storesDuringWait = take(counts.storesDuringWait);
    totals.keepDiffered = take(counts.keepDiffered);
    totals.keepRaced = take(counts.keepRaced);
    totals.refreshLeaseWaits = take(counts.refreshLeaseWaits);
    totals.refreshLeaseNs = take(counts.refreshLeaseNs);
    totals.refreshLeaseGpuNs = take(counts.refreshLeaseGpuNs);
    totals.refreshLeaseQueue0GpuNs = take(counts.refreshLeaseQueue0GpuNs);
    return totals;
}

[[noreturn]] inline void NarrowFlushDisagrees(const char* what, unsigned long long address, unsigned long long bytes, const char* reason) {
    std::fprintf(stderr, "[narrow-flush] APS5_EXPERIMENT_NARROW_FLUSH_SYNC=2: a guard wait F1 would skip (%s 0x%llx+0x%llx) %s\n", reason != nullptr ? reason : "?", address, bytes, what);
    std::fflush(stderr);
    std::abort();
}

}

#endif
