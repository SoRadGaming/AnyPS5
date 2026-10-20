#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_FLUSHGUARD_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_FLUSHGUARD_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

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
// - Untracked: a flushed image lies outside the write watch (its write-back compares the guest
//   bytes, compareUntracked);
// - Baseline: a pending unit of a flushed image holds a byte baseline (`original`, or the generation
//   baseline: unchangedSinceBaseline compares the bytes of its stamped blocks);
// - NoImport: a flushed image has no host import (its write-back is the CPU round trip, which reads
//   the guest bytes and waits for the recorder by itself);
// - Stamped: none of these. Every unfinished writer of the ranges stamped its blocks when it was
//   recorded, so the write-back's keep decision reads the same stamps with or without the wait, and
//   F1 skips it ("would skip").
// The scanned ranges are the access, every flushed image's surface and its DCC keys, over the open
// batch and every in-flight one. Counters: relaxed, constant-initialized; DrawPipeline's [stream]
// report takes them (TakeFlushGuardCounts).
namespace AgcDriver::Graphics {

enum class FlushWaitClass : std::uint8_t { Stamped, AliasCopy, Lease, CopiedWriter, CompletionLabel, Untracked, Baseline, NoImport, Count };
constexpr std::size_t FlushWaitClasses = static_cast<std::size_t>(FlushWaitClass::Count);
inline constexpr std::array<const char*, FlushWaitClasses> FlushWaitClassNames{"stamped", "alias-copy", "lease", "copied-writer", "completion-label", "untracked", "baseline", "no-import"};
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

}

#endif
