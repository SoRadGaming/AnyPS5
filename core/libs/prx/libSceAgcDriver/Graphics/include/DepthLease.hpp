#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEPTHLEASE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEPTHLEASE_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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
// - planeBinds, the clearedHtiles insert serial and mirrorEpoch held, and no clearedHtiles entry
//   covering the target's HTILE was left at that end (takeClearedHtile takes one entry per call, so
//   a second one set before the target's surface existed is taken by its next call, with no insert
//   in between);
// - no batch finished (a reap, sync or wait: Recorder::FinishedSerial);
// - every range has a proof: the depth and stencil planes a nonzero generation from NoteMemory, the
//   HTILE one from its read proof (htileReads), each still write-watched (a range unwatched, by an
//   import the tracker compares instead of watching, or an HTILE with no read proof because it is
//   not accessible, misses as unproven, not as stamped);
// - nothing stamped the depth or stencil plane since the generations NoteMemory took, nor the HTILE
//   since the generation its read proof held (GuestMemory::StampsHold without the drain);
// - no host write had ended unstamped over the pages of a range (APS5_EXPERIMENT_HOST_WRITE_STAMPS=1
//   queues them for the next tracker lock holder to stamp: the stamp test does not drain them, and one
//   queued over a range stamps it when drained, so with every range unstamped it misses as "host
//   writes pending"; one elsewhere stamps no range when drained, so it is no miss);
// - no write over the HTILE was noted after the newest note at that end (with no batch finished, so
//   Recorder::NewestWriteNote over the HTILE is unchanged: Recorder::WrittenAfterNote).
// Otherwise it misses, by the first of those that moved (a memo key move by its cause: the epoch,
// else the allocations, else UnwatchSerial; a stamp by its plane). Recorder::WriteGeneration is no
// condition: its moves are counted apart, for the would-hits and for the misses. A would-hit whose
// call still did work a lease would skip (an HTILE clear stored back, a cleared HTILE taken, the depth
// mirror synced) is counted too. The checks before the call's own work read state only: one tracker
// lock without the drain, not counted on [guestmem]'s tracker waits line, and under it the host-write
// queue's mutex while a write is pending (the host writes queued meanwhile stay queued for the call's
// own first collect, so the counter changes neither the tracker's drain order nor its counted locks,
// though its hold may make another lock wait; its per-range answers name the range) and the recorder
// under the GPU lock; at the end one htileReads lookup and a walk of clearedHtiles (normally empty).
// Counted on the [proofs] line, with the counter's own time, summed apart by the [draws] phase that
// carries it: "prepare" for Draw's call, "readTarget" for DrawWithRecipe's (DepthLeaseRecipeScope).
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
    // At an end only: a clearedHtiles entry still covers the target's HTILE, so its next call takes
    // it (that call misses as ClearedHtile).
    bool clearedHtileLeft = false;
};

// Would-hit, then the misses in the order they are tested.
enum class DepthLease : std::uint8_t { WouldHit, NoEntry, Evicted, Identity, Epoch, Allocations, Unwatch, Surfaces, Unretired, PlaneBinds, ClearedHtile, MirrorEpoch, Reaped, Unproven, StampedDepth, StampedStencil, StampedHtile, HostWrites, HtileNote, Count };

struct DepthLeaseVerdict {
    DepthLease outcome = DepthLease::WouldHit;
    bool writeGenerationMoved = false;
};

// The first condition that moved from `previous` (an end) to `now` (a start, with the identity of the
// call's own lookup). `stamped` (Unproven, StampedDepth, StampedStencil, StampedHtile, HostWrites, or
// WouldHit for none; the caller names a range with no proof before a stamp, and a stamp before pending
// host writes) and `htileWritten` are read only once the memo key and the finished serial held, where
// the caller evaluates them.
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
    // The counter's own time outside DepthLeaseRecipeScope (Draw's call, in the [draws] prepare
    // phase) and inside it (DrawWithRecipe's, in readTarget).
    std::uint64_t counterNs = 0;
    std::uint64_t recipeCounterNs = 0;
};
DepthLeaseCounts DepthLeaseTotals();
const char* DepthLeaseName(DepthLease lease);
// The [proofs] line's part (reportProofs, every 10 s): counts since the last call.
std::string DepthLeaseReport();

// DrawWithRecipe's DepthSurfaceView call runs inside this scope, active while profiling, so the
// counter's time there is summed apart (recipeCounterNs): DrawWithRecipe's timer books that call to
// the [draws] readTarget phase, Draw's to prepare. Inactive, it does nothing. Sets the thread's flag
// and returns the one before.
bool DepthLeaseSetInRecipe(bool inside);
class DepthLeaseRecipeScope {
public:
    explicit DepthLeaseRecipeScope(bool enabled) : active(enabled) {
        if (active) previous = DepthLeaseSetInRecipe(true);
    }
    ~DepthLeaseRecipeScope() {
        if (active) DepthLeaseSetInRecipe(previous);
    }
    DepthLeaseRecipeScope(const DepthLeaseRecipeScope&) = delete;
    DepthLeaseRecipeScope& operator=(const DepthLeaseRecipeScope&) = delete;

private:
    bool active;
    bool previous = false;
};

// LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11 D5): DepthSurfaceView leases. Read
// once: unset or 0 is today exactly (no table, no hook, D5-0 counts as before while profiling); 1 uses
// the leases; 2 verifies them: a side-effect-free prediction runs first, today's full path always runs
// and its result is used, and any difference the prediction cannot explain aborts with
// [depth-lease]. While the switch is set D5-0 counts nothing (its table and time would sit in the
// measured phase) and [proofs] prints this line instead of its line. Needs APS5_DEBUG_COVERAGE=1:
// without a collect memo key every call is ineligible. Do not combine with Q0 ticket or receipt verify
// runs: a hit skips the collects tagged WalkCaller::Depth.
//
// A call is eligible when its target is not routed to a native depth owner (so single-sample), its
// depth plane is mirrorable (D32, swizzle mode Z64KBX), the device's recorder is active and this thread
// holds the GPU lock, and the thread has a collect memo key. Its lease lives in a per-thread table of
// 16 sets x 4 ways keyed on the recorder, the device and every DepthTarget field, and holds no
// generation of any kind: the memo key its creation proved the HTILE, plane, depth and stencil ranges
// covered under; the surface with the surfaces and released serials it was found under; the HTILE
// route of today's store-back (no HTILE; not accessible; not pending: no open or in-flight write over
// it; pending: a pending fill over all of it whose word clears nothing), with the recorder's finished
// serial and newest write note read at that decision; the plane's and the HTILE's sizes (pure). A hit
// re-reads every input of the steps it skips live, under the locks those steps take (the GPU lock it
// holds, the surfaces mutex, htileReads', one tracker lock), builds the plane's not-written row from
// the surface's live oldest unit generation (so a NoteStored that lowered the units, as
// FlushDepthPlanes' does after its store, is seen), and writes the values today's steps would write:
// the HTILE's read proof (pending route), every unit and the depth and stencil memory, from the
// generations that lock reads. Every per-draw ordering step (takePlanes, the fast clear, written and
// drawnDepth, the mirror scan, the stale marks) runs live.
struct Context;
struct DepthTarget;

int DepthViewLeases();

enum class DepthViewRoute : std::uint8_t { NoHtile, NotAccessible, NotPending, Pending, Count };
// Where today's HTILE store-back returned (storeBackClearedHtile's verdict): no HTILE; not accessible;
// not pending and its read proof unchanged; not pending, read by the CPU with nothing cleared; pending,
// a fill covering it all whose word clears nothing; pending otherwise (a fill not covering it all, or
// the guest words read); pending with the pending fills switched off; an HTILE clear stored back, or a
// CPU read that cleared.
enum class HtileVerdictKind : std::uint8_t { NoHtile, NotAccessible, NotPendingProved, NotPendingRead, PendingCovered, PendingNotCovering, PendingFillsOff, StoredBack, Count };
enum class DepthViewIneligible : std::uint8_t { Eligible, Native, NotMirrorable, Unlocked, NoMemoKey, Count };
// A hit's (or prediction's) first failed check, in the order the check runs (GPU-OWNED-PLAN 11.6).
enum class DepthViewMiss : std::uint8_t { NoLease, Evicted, Epoch, Allocations, Unwatch, Reaped, HtileNote, Accessibility, Surfaces, Retired, ClearedHtile, UnitsResized, NoPlaneProof, HtileReadProofMoved, PlaneUnproven, PlaneStamped, HtileUnproven, HtileStamped, HostWritesPending, Count };
// Why a full path made no lease (GPU-OWNED-PLAN 11.8), the first that held.
enum class DepthViewRefusal : std::uint8_t { Created, StoredBackOrClearedTaken, PendingNotCovering, PendingFillsOff, MemoKeyMoved, RecorderChanged, NotCovered, Count };
// =2: the part of today's call that disagreed with the prediction first (None: verified).
enum class DepthViewPart : std::uint8_t { None, StoreBack, Identity, Sync, Refresh, Count };

template<typename T>
constexpr std::size_t DepthViewIndex(T value) {
    return static_cast<std::size_t>(value);
}

// Totals since the start, counted while the switch is set (the [proofs] line prints 10 s deltas).
struct DepthViewLeaseCounts {
    std::uint64_t calls = 0;
    std::uint64_t ineligible[DepthViewIndex(DepthViewIneligible::Count)]{};
    std::uint64_t hits = 0;
    std::uint64_t hitRoutes[DepthViewIndex(DepthViewRoute::Count)]{};
    std::uint64_t verdicts[DepthViewIndex(HtileVerdictKind::Count)]{};
    std::uint64_t created = 0;
    std::uint64_t renewed = 0;
    std::uint64_t refusals[DepthViewIndex(DepthViewRefusal::Count)]{};
    std::uint64_t misses[DepthViewIndex(DepthViewMiss::Count)]{};
    std::uint64_t evictions = 0;
    std::uint64_t predicted = 0;
    std::uint64_t verified = 0;
    // By the part that disagreed: store-back, identity, sync, refresh.
    std::uint64_t raced[4]{};
};
DepthViewLeaseCounts DepthViewLeaseTotals();
// The [proofs] line's part while the switch is set (reportProofs, every 10 s): counts since the last call.
std::string DepthViewLeaseReport();
const char* DepthViewRouteName(DepthViewRoute route);
const char* HtileVerdictName(HtileVerdictKind kind);
const char* DepthViewMissName(DepthViewMiss miss);
const char* DepthViewRefusalName(DepthViewRefusal refusal);
const char* DepthViewPartName(DepthViewPart part);

// =2 (GPU-OWNED-PLAN 11.9). What landed after the prediction, read at a part's own point of today's
// call, before that part's own actions. Recorder serials are never evidence: the GPU lock is held for
// the whole call, so no other thread moves them, and the call's own moves are its own.
// The store-back part's, read before DS's storeBackClearedHtile (the memo key) and at that function's
// decision, before any store of its own (from its verdict): its Accessible answer differs from the
// prediction's; its read-proof lookup (not pending) found another value than the prediction's; its
// collect answered a generation above the prediction's memo-hit generation. And one read at the call's
// end: the HTILE stamped since the prediction.
struct DepthViewStoreBackEvidence {
    bool memoMoved = false;
    bool accessibleDiffers = false;
    bool foundDiffers = false;
    // Printed, never an excuse (the r73 review): today's HTILE collect is a memo hit, which answers the
    // tracker generation, so it is above the prediction's after a stamp anywhere, not only on the HTILE.
    bool collectedAbove = false;
    // Read at the call's end, on a not-pending prediction whose store-back disagreed and stored nothing
    // (one tracker lock without the drain): the prediction's unchanged HTILE row, from its read proof h,
    // no longer holds, so a stamp landed on the HTILE since the prediction (stamps only rise, so one
    // before today's UnchangedSince is still there). It stands for the collect's answer as the evidence
    // of a stamp. Never read after a store-back that stored: its own store stamps the whole HTILE.
    bool htileStamped = false;
    // Only the evidence that can change today's verdict on the predicted route excuses a store-back
    // disagreement (the r73 review). Not pending, by today's verdict: a pending one (a StoredBack after
    // a pending write found included) never, as the recorder's answer, which no other thread moves while
    // this one holds the GPU lock; not accessible only Accessible (the page protection moved); a read
    // proof that failed (not pending read, or a StoredBack after the lookup) only the memo key (a walk
    // under a moved key), the lookup (forgetHtileReads moves it) or the HTILE stamped (another thread's
    // stamp fails UnchangedSince); any other never. Pending, and not accessible, only Accessible (the
    // rest of the verdict is the recorder's, and the pending fills switch, read once: a collect's value
    // is stored, never tested); no HTILE, none (htileAddress is a key field).
    bool Explains(DepthViewRoute route, HtileVerdictKind verdict, bool pending) const {
        if (route == DepthViewRoute::NotPending) {
            if (verdict == HtileVerdictKind::NotAccessible) return accessibleDiffers;
            if (verdict == HtileVerdictKind::NotPendingRead || (verdict == HtileVerdictKind::StoredBack && !pending)) return memoMoved || foundDiffers || htileStamped;
            return false;
        }
        if (route == DepthViewRoute::Pending || route == DepthViewRoute::NotAccessible) return accessibleDiffers;
        return false;
    }
};
// The identity point's (after the surfaces mutex is taken, before takeClearedHtile) and the sync
// point's (before SyncFromMemory): the memo key moved; the surfaces or released serial moved (then
// nothing of the predicted surface is read); the predicted surface retired, or its units resized or
// their oldest generation moved; a cleared HTILE entry covers the target's HTILE; the tracker
// generation (no drain) is above the prediction's. Each part is excused only by the evidence that can
// change its own facts (the r73 review; CheckDepthViewPrediction): the identity's (a cleared HTILE
// taken, the surface created, another or a retired surface found) change only through the surface
// list, retire and active, and clearedHtiles, all frozen under the surfaces mutex from the identity
// point on, so only the surfaces serials, retired and a cleared HTILE entry excuse it; the memo key,
// the units and the clock are printed there, never an excuse.
struct DepthViewEvidence {
    bool memoMoved = false;
    bool surfacesMoved = false;
    bool retired = false;
    bool oldestMoved = false;
    bool clearedHtile = false;
    bool clockMoved = false;
    bool ExplainsIdentity() const { return surfacesMoved || retired || clearedHtile; }
};
// One verified call: the prediction, today's route and refresh values (read at the call's end, under
// the surfaces mutex), and the evidence at each part's point.
struct DepthViewFacts {
    DepthViewRoute route = DepthViewRoute::NoHtile;
    bool stencil = false;
    // The not-pending route's read proof, and the generations the prediction's tracker lock read
    // (TrackerGeneration's and a memo-hit collect's answers).
    std::uint64_t h = 0;
    std::uint64_t current0 = 0;
    std::uint64_t memoCurrent0 = 0;
    // Today's store-back: its verdict and return value, and whether it found a pending write over the
    // HTILE (then it looked up no read proof).
    HtileVerdictKind verdict = HtileVerdictKind::NoHtile;
    std::uint32_t htileCleared = 0;
    bool storeBackPending = false;
    DepthViewStoreBackEvidence storeBack;
    // Today's identity: a cleared HTILE taken, the surface created, the predicted surface found, and
    // found retired.
    bool cleared = false;
    bool created = false;
    bool sameSurface = true;
    bool retiredBefore = false;
    DepthViewEvidence identity;
    // Today's sync (SyncFromMemory copied the mirror in).
    bool synced = false;
    DepthViewEvidence sync;
    // Read at the call's end, on a synced call only (a disagreement; one tracker lock without the
    // drain): the plane's not-written row from the prediction's oldest unit generation no longer holds,
    // so a stamp landed on the plane since the prediction (stamps only rise, so one before today's
    // WrittenSince is still there). A clock move excuses a sync only with it.
    bool planeStamped = false;
    // Today's refresh values: the HTILE's read proof, the lowest and highest unit generation (each
    // unit holds what WrittenUnits read at its decision), the depth and stencil memory.
    bool htileReadFound = false;
    std::uint64_t htileRead = 0;
    std::uint64_t unitsLow = 0;
    std::uint64_t unitsHigh = 0;
    std::uint64_t depthMemory = 0;
    std::uint64_t stencilMemory = 0;
};
struct DepthViewCheck {
    DepthViewPart part = DepthViewPart::None;
    bool raced = false;
    const char* reason = "verified";
};
// Pure. The parts in today's order; the first that disagrees decides: raced when the evidence read at
// its own point names an update since the prediction that can change that part (store-back:
// DepthViewStoreBackEvidence::Explains for the predicted route and today's verdict; identity:
// ExplainsIdentity), else an abort (evidence read only at a later part's point never excuses it; the
// HTILE and plane rows read at the call's end stand for their parts' decisions, as stamps only rise).
// Store-back agrees when the verdict is the predicted route's (pending: pending covered; not pending:
// not pending proved) and nothing was cleared; identity when no cleared HTILE was taken, nothing
// created, and the predicted surface was found and not retired; sync when nothing synced, and a sync
// is raced when the sync point's memo key, surfaces serials or units moved, or when the clock moved
// (at the sync point, or the units hold a generation above the prediction's: WrittenUnits read it at
// its own decision) and the plane itself was stamped since the prediction (planeStamped: a clock move
// elsewhere cannot make WrittenUnits change a unit); refresh when every value equals its prediction (pending: the read proof is
// the memo-hit generation; not pending: it is the prediction's; no HTILE or not accessible: not
// checked; every unit the TrackerGeneration answer; depth memory the memo-hit generation; stencil
// memory that too with a stencil plane, else exactly 0). A refresh value above its prediction is
// raced (each value is the generation at its own read, and on an agreed route the call makes no
// tracker update of its own); one below (or a read proof missing, units unequal, a stencil memory
// without a stencil plane) aborts.
DepthViewCheck CheckDepthViewPrediction(const DepthViewFacts& facts);

// Test hooks (LOCAL EXPERIMENT). The state of the surface DepthSurfaceView resolves `target` to (the
// first active surface of the device that sameSurface matches): its unit generations, the HTILE's
// read proof, the depth and stencil memory, the stale units and whether the plane is in the stale
// list, the pending clear, written, whether it has a mirror plane (SyncFromMemory made one) and
// retired.
struct DepthViewLeaseState {
    bool found = false;
    std::vector<std::uint64_t> units;
    bool htileRead = false;
    std::uint64_t htileReadValue = 0;
    std::uint64_t depthMemory = 0;
    std::uint64_t stencilMemory = 0;
    std::size_t staleCount = 0;
    std::size_t staleUnits = 0;
    bool stalePlaneListed = false;
    std::uint32_t pendingClear = 0;
    std::uint64_t written = 0;
    bool mirrored = false;
    bool retired = false;
};
DepthViewLeaseState DepthViewLeaseProbe(const Context& context, const DepthTarget& target);
// NoteStored(generation) over the first `units` units (all by default) of that surface, under the
// surfaces mutex, as FlushDepthPlanes' does after a plane store. False when there is no surface.
bool DepthSurfaceNoteStoredForTest(const Context& context, const DepthTarget& target, std::uint64_t generation, std::size_t units = static_cast<std::size_t>(-1));

}

#endif
