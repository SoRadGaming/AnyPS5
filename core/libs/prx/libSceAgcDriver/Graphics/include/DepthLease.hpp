#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEPTHLEASE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEPTHLEASE_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <cstddef>
#include <cstdint>
#include <string>

// LOCAL DEBUG (APS5_PROFILE_DRAW, GPU-OWNED-PLAN 9.5 D5-0): the DepthSurfaceView lease counter. No
// behaviour change: it counts how many DepthSurfaceView calls a lease would have answered. A call is
// eligible when its target is single-sample and not routed to a native depth owner, the GPU lock is
// held if the device has an active recorder, and the thread has a collect memo key
// (GuestMemory::CurrentCollectMemoKey; zero without APS5_DEBUG_COVERAGE=1). Its target is looked up
// in a per-thread table of 16 sets x 4 ways, keyed on the recorder, the device and every DepthTarget
// field. Each eligible call leaves its entry the state at its end (DepthLeaseMark), and the next call
// of that target would hit when, since then:
// - the target resolves to the same surface (identity), under the same memo key;
// - DepthSurfacesSerial and the un-retire serial (a surface found retired by DepthSurfaceView) held;
// - planeBinds, the clearedHtiles insert serial and mirrorEpoch held;
// - no batch finished (a reap, sync or wait: Recorder::FinishedSerial);
// - nothing stamped the depth or stencil plane since the generations NoteMemory took, nor the HTILE
//   since the generation its read proof (htileReads) held (GuestMemory::UnchangedSinceAll; an HTILE
//   with no read proof, which is not accessible, misses as stamped);
// - no write over the HTILE was noted after the newest note at that end (with no batch finished, so
//   Recorder::NewestWriteNote over the HTILE is unchanged: Recorder::WrittenAfterNote).
// Otherwise it misses, by the first of those that moved (a memo key move by its cause: the epoch,
// else the allocations, else UnwatchSerial; a stamp by its plane). Recorder::WriteGeneration is no
// condition: its moves are counted apart, for the would-hits and for the misses. A would-hit whose
// call still did work a lease would skip (an HTILE clear stored back, a cleared HTILE taken, the depth
// mirror synced) is counted too. The checks before the call's own work read state only: one tracker
// lock (which, under APS5_EXPERIMENT_HOST_WRITE_STAMPS=1, stamps the host writes queued meanwhile, as
// the call's first collect would a moment later) and the recorder under the GPU lock; at the end one
// htileReads lookup. Counted on the [proofs] line, with the counter's own time, which the [draws]
// "prepare" phase carries.
namespace AgcDriver::Graphics {

// The state an eligible call leaves (its end) and the next call of the target reads at its start.
struct DepthLeaseMark {
    // The DepthSurface the target resolved to (null at a call that created it).
    const void* surface = nullptr;
    GuestMemory::CollectMemoKey memo;
    std::uint64_t surfaces = 0;
    std::uint64_t unretired = 0;
    std::uint64_t planeBinds = 0;
    std::uint64_t clearedHtiles = 0;
    std::uint64_t mirrorEpoch = 0;
    // Recorder::FinishedSerial and Recorder::WriteNotes (0 without a recorder).
    std::uint64_t finished = 0;
    std::uint64_t writeNotes = 0;
    std::uint64_t writeGeneration = 0;
};

// Would-hit, then the misses in the order they are tested.
enum class DepthLease : std::uint8_t { WouldHit, NoEntry, Evicted, Identity, Epoch, Allocations, Unwatch, Surfaces, Unretired, PlaneBinds, ClearedHtile, MirrorEpoch, Reaped, StampedDepth, StampedStencil, StampedHtile, HtileNote, Count };

struct DepthLeaseVerdict {
    DepthLease outcome = DepthLease::WouldHit;
    bool writeGenerationMoved = false;
};

// The first condition that moved from `previous` (an end) to `now` (a start, with the identity of the
// call's own lookup). `stamped` (StampedDepth, StampedStencil, StampedHtile, or WouldHit for none) and
// `htileWritten` are read only once the memo key and the finished serial held, where the caller
// evaluates them.
DepthLeaseVerdict ClassifyDepthLease(const DepthLeaseMark& previous, const DepthLeaseMark& now, DepthLease stamped, bool htileWritten);

// What a call counted as a would-hit still did that a lease would skip (CountDepthLease's `work`).
enum DepthLeaseWork : std::uint8_t { DepthLeaseHtileStoreBack = 1, DepthLeaseClearedHtileTaken = 2, DepthLeaseMirrorSync = 4 };

// Counts one classified call (NoEntry and Evicted come from the table, the rest from the classifier).
void CountDepthLease(const DepthLeaseVerdict& verdict, std::uint8_t work);

enum DepthLeaseIneligible : std::uint8_t { DepthLeaseEligible, DepthLeaseMultisampled, DepthLeaseUnlocked, DepthLeaseNoMemoKey, DepthLeaseIneligibleCount };

// Totals since the start (the [proofs] line prints them as 10 s deltas).
struct DepthLeaseCounts {
    std::uint64_t calls = 0;
    std::uint64_t ineligible[DepthLeaseIneligibleCount]{};
    std::uint64_t outcomes[static_cast<std::size_t>(DepthLease::Count)]{};
    std::uint64_t hitsWriteGeneration = 0;
    std::uint64_t missesWriteGeneration = 0;
    std::uint64_t hitWork[3]{};
    std::uint64_t evictions = 0;
    std::uint64_t counterNs = 0;
};
DepthLeaseCounts DepthLeaseTotals();
const char* DepthLeaseName(DepthLease lease);
// The [proofs] line's part (reportProofs, every 10 s): counts since the last call.
std::string DepthLeaseReport();

}

#endif
