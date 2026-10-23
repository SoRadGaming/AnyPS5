#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_INPUTLEASE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_INPUTLEASE_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

// LOCAL DEBUG (APS5_PROFILE_DRAW, REVALIDATE-PLAN section 7, I0; stage S1): the draw-input lease
// counters. No behaviour change: L1's lease table (section 5.1) is filled from today's results and
// never applied while APS5_EXPERIMENT_INPUT_LEASES is unset.
// Every draw input that reaches CopyDrawInput's collect (an index range, or a planned vertex copy
// range; InPlaceDrawInput runs after the probe) is looked up in its recorder's table, which exists
// only while profiling (or the switch is set) and only under the GpuMutex (InputLeaseFor), and its
// rows are predicted:
//   P: nothing pending, flushing or live-shadowed overlaps the range at PendingSerial S: the lease's
//      S still current (memo), else StorageTexture::NothingPendingOver (query) stores the answer;
//   C: the lease's collect memo key and coverage owner are the thread's now (GuestMemory's lease
//      rule), checked against CollectCovered before today's collect (W counts a predicted hold the
//      coverage does not show: must be 0); a collect reported covered under an unchanged key fills it;
//   M: the lease's drawSnapshots entry, while its set's serial (bumped by every erase and emplace of a
//      (address, use) hashing to the set) did not move; today's lookup must find that entry (else
//      `wrong`, must be 0); a found entry fills it.
// The [input-lease] line every 10 s prints the window's counts (see reportInputLeases).
//
// LOCAL EXPERIMENT (APS5_EXPERIMENT_INPUT_LEASES, REVALIDATE-PLAN section 5, L1; stages S3 and S4): the
// rows applied. Unset or 0: today's path exactly (no table unless I0 counts; every new branch tests the
// switch's static). 1: on. 2: verify: today's path runs and its results are used; each row's hold is
// predicted, then checked against what today's path did: a self-inflicted or unexplained difference
// aborts with an [input-lease] line, one only another thread's move can explain is counted (void or
// raced, the first 8 printed). APS5_EXPERIMENT_INPUT_LEASE_ROWS (letters of PCME, default all; for A/B
// attribution) names the rows the switch applies:
//   C (5.3): a lease whose collect memo key and coverage owner are the thread's now takes
//      GuestMemory::CoveredCollectGeneration in place of CopyDrawInput's collect (the lease rule of
//      GuestMemory.hpp); =2: the key must still be the lease's just before today's collect (else
//      void) and the range covered under it (else abort);
//   M (5.4): a lease whose set serial did not move since it found its drawSnapshots entry takes that
//      entry in place of ReusableDrawSnapshot's lookup; =2: the lookup must find that very entry;
//   E (5.5): GuestMemory's speculative read gate is mirrored in a TEB slot, read in place of the
//      thread_local (GuestMemory.cpp; =2 compares both at every read);
//   P (5.2): nothing pending, flushing or live-shadowed overlaps the range at PendingSerial S (the
//      lease's S while it holds, else StorageTexture::NothingPendingOver's answer, kept): while S
//      holds (PendingRowStill), InPlaceDrawInput's vertex FlushPending returns after FlushDepthPlanes
//      (skip point (a)) and the flush hook of CopyDrawInput's FlushGpuWrites skips AccessKeptByCpu,
//      WaitForFlushing and PublishShadowsOnly (skip point (b); with HookFlushCpuBlocks its
//      FlushPending skips as (a)). Every addition to what the query sees (a pending image pushed or
//      re-marked, unit shadows made live) bumps PendingSerial atomically with it, except MarkShadowed
//      (the bump follows the registry mutex): every P skip holds the GpuMutex, and an addition made
//      without it first turns row P off for good (the kill switch, counted as U). =2: today's path runs
//      at each skip point, which must have had no effect where the row would have skipped; an effect
//      is raced only when other threads' additions moved since the prediction (removals and the
//      thread's own additions never excuse one), else an [input-lease] P line and abort; after the
//      input, while S still holds, NothingPendingOver must still answer S.
// Probes need the GpuMutex (InputLeaseFor), as every row's proof does.
namespace AgcDriver::Graphics {

// Whether the I0 counters run: APS5_PROFILE_DRAW, read once.
bool InputLeaseCounting();
// LOCAL EXPERIMENT (L1): APS5_EXPERIMENT_INPUT_LEASES, read once (the TargetLeases idiom): 0 off, 1
// on, 2 verify.
int InputLeases();
// The rows of APS5_EXPERIMENT_INPUT_LEASE_ROWS, read once.
enum class InputLeaseRow : std::uint8_t { P = 1, C = 2, M = 4, E = 8 };
// Whether the switch applies `row` (false while it is 0).
bool InputLeaseRowOn(InputLeaseRow row);
// Whether a draw input gets a lease view (InputLeaseShadow): I0 counts or the switch is set.
bool InputLeaseViewed();

// One table way's payload (REVALIDATE-PLAN 5.1): the drawSnapshots entry and the set serial it was
// found at (row M), the collect memo key and coverage owner a covered collect left (row C), the
// PendingSerial a nothing-over answer held at (row P, 0: none). `mapHeld`: row M's hold or
// prediction for the input in progress.
struct Recorder::InputLease {
    std::map<DrawSnapshotKey, DrawSnapshot>::iterator entry{};
    bool mapValid = false;
    bool mapHeld = false;
    std::uint64_t mapSerial = 0;
    GuestMemory::CollectMemoKey memo{};
    std::uint64_t owner = 0;
    std::uint64_t pendingAt = 0;
};

// 4096 sets of 4 ways (REVALIDATE-PLAN 5.1): a set header is one cache line holding the ways' keys
// (address, bytes, use), the set's map serial and the ways' recency; the payloads lie in a parallel
// array, so a probe touches two lines. A key miss takes the oldest way and resets its payload. The
// map serial is 64 bits (own: a 32-bit one could wrap back to a stale lease's copy while its iterator
// is gone; the header still fits its line).
struct Recorder::InputLeaseTable {
    static constexpr std::size_t Sets = 4096;
    static constexpr std::size_t Ways = 4;
    struct alignas(64) Set {
        std::array<std::uint64_t, Ways> address{};
        std::uint64_t mapSerial = 0;
        std::array<std::uint32_t, Ways> bytes{};
        // SnapshotUse + 1; 0: an empty way.
        std::array<std::uint8_t, Ways> use{};
        // Recency rank per way: 0 the newest, Ways - 1 the oldest (the way a key miss takes).
        std::array<std::uint8_t, Ways> age{0, 1, 2, 3};
    };
    static_assert(sizeof(Set) == 64, "an input lease set header is one cache line");
    enum class Probe : std::uint8_t { Present, Inserted, Evicted };
    std::vector<Set> sets = std::vector<Set>(Sets);
    std::vector<InputLease> leases = std::vector<InputLease>(Sets * Ways);
    static std::size_t SetOf(std::uint64_t address, SnapshotUse use);
    InputLease* Find(std::uint64_t address, std::size_t bytes, SnapshotUse use, Probe& probe);
    // Every erase and emplace of drawSnapshots bumps its (address, use)'s set (the =2 death row of row
    // M suppresses it: InputLeaseSuppressBumpsForTest).
    void Bump(std::uint64_t address, SnapshotUse use);
    std::uint64_t Serial(std::uint64_t address, SnapshotUse use) const { return sets[SetOf(address, use)].mapSerial; }
};
// Test only (REVALIDATE-PLAN 9 (q4)): while set, drawSnapshots changes without bumping a set serial.
void InputLeaseSuppressBumpsForTest(bool suppress);

// The pending registry's addition sites (REVALIDATE-PLAN 5.2): markLayersPending's push, its lent
// re-mark bump, reconcilePending's push and MarkShadowed's units going live.
enum class PendingAdditionSite : std::uint8_t { Mark, Lent, Reconcile, Shadow, Count };
// Counts one addition at its site (PendingAdditions, and the calling thread's own) and, when the
// calling thread does not hold the GpuMutex, one U for the site (row P's invariant: every addition
// runs under the GpuMutex, as every P skip does). LOCAL EXPERIMENT (L1 row P): with the switch set
// such an addition first turns row P off for good (seq_cst, before the addition lands: a skip that
// read the flag clear was decided before it); =2 prints the site. Called while InputLeaseViewed(),
// inside the site's own lock, before the push, the bump or the units going live.
void NotePendingAddition(PendingAdditionSite site);
std::uint64_t PendingAdditions();
// The calling thread's own additions (a TEB slot), which never excuse a =2 difference.
std::uint64_t OwnPendingAdditions();
// LOCAL EXPERIMENT (L1 row P): whether S proves row P now: S is not 0, the kill switch is off and
// PendingSerial is still S (seq_cst loads).
bool PendingRowStill(std::uint64_t serial);
// Whether the kill switch turned row P off (for tests).
bool PendingRowOff();
// Test only (REVALIDATE-PLAN 9 (q2)): called at =2 between a skip point's prediction (a hold) and
// today's call there; null outside tests.
void InputLeaseSetPendingProbeHookForTest(void (*hook)());

// The I0 counts since the last [input-lease] report (InputLeaseCountsNow reads them without taking
// them, for tests). While the switch is set they count its holds (=1) or predictions (=2) as
// would-holds; the shadow checks (W, M wrong, the P query) run only with the switch unset.
struct InputLeaseCounts {
    std::uint64_t indexCalls = 0, vertexCalls = 0, inPlace = 0, reused = 0, copied = 0, padding = 0;
    std::uint64_t present = 0, inserted = 0, evicted = 0, withoutGpuMutex = 0;
    std::uint64_t pMemo = 0, pQuery = 0, pPending = 0, pFlushing = 0, pShadow = 0, pMoved = 0, pNoSerial = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(PendingAdditionSite::Count)> unlocked{};
    std::uint64_t cWould = 0, cNoKey = 0, cOwner = 0, cEpoch = 0, cUnwatch = 0, cMapping = 0, cNotCovered = 0, cWrong = 0;
    std::uint64_t mWould = 0, mNoLease = 0, mMoved = 0, mWrong = 0;
    std::uint64_t held = 0;
};
InputLeaseCounts InputLeaseCountsNow();

// LOCAL EXPERIMENT (L1): the switch's cumulative counts, with or without APS5_PROFILE_DRAW (relaxed;
// =2 prints 10 s deltas on its own [input-lease] =2 line). Per row: `held` (=1: applied),
// `predicted` (=2), `verified` (=2: today's path agreed), `voided` (=2, row C: the key moved before
// today's collect, so =1 would not have skipped it either).
// Row P counts skip points: `pHeld` (=1: a skip), `pPredicted`/`pVerified`/`pRaced` (=2), the
// after-input checks (`pAfter`; `pAfterMoved`: the query saw PendingSerial move, so it proved nothing
// either way) and `pKilled` (additions that turned row P off).
struct InputLeaseExperimentCounts {
    std::uint64_t cHeld = 0, cPredicted = 0, cVerified = 0, cVoided = 0;
    std::uint64_t mHeld = 0, mPredicted = 0, mVerified = 0;
    std::uint64_t pHeld = 0, pPredicted = 0, pVerified = 0, pRaced = 0, pAfter = 0, pAfterMoved = 0, pKilled = 0;
};
InputLeaseExperimentCounts InputLeaseExperimentNow();

// One draw input's view of its lease, made before InPlaceDrawInput (or by CopyDrawInput for a direct
// caller) while InputLeaseViewed(): probes the recorder's table; I0 (the switch unset, profiling)
// predicts row P; CopyDrawInput hands it its collect (row C) and its ReusableDrawSnapshot the lease
// (row M), then its outcome. Inactive (counts nothing, no lease) without a recorder or for an empty
// range.
class InputLeaseShadow {
public:
    InputLeaseShadow(Recorder* recorder, std::uint64_t address, std::size_t bytes, Recorder::SnapshotUse use);
    ~InputLeaseShadow();
    InputLeaseShadow(const InputLeaseShadow&) = delete;
    InputLeaseShadow& operator=(const InputLeaseShadow&) = delete;
    // The input was bound in place (InPlaceDrawInput): no collect or lookup follows.
    void InPlace();
    // CopyDrawInput's collect and `registryGeneration`. Switch unset or row C off: today's
    // (GuestAllocationsGeneration, then CollectWritesCovering, which is CollectWrites), after I0's
    // prediction and its W check; a collect reported covered under the key read before it, which
    // still holds after it, fills the lease's key and owner. Row C (L1 5.3): one CurrentCollectMemoKey
    // read gives both the registry generation (its `mapping` is that same GuestAllocationsGeneration
    // read) and the lease rule's key; a lease that holds takes CoveredCollectGeneration (=1) or is
    // checked before today's collect (=2). The answer is today's on every path.
    std::uint64_t Collect(std::uint64_t address, std::size_t bytes, std::uint64_t& registryGeneration);
    // The lease for ReusableDrawSnapshot's row M (null: none).
    Recorder::InputLease* Lease() const { return lease; }
    // CopyDrawInput's outcome: a kept copy reused, or one refused for its padding (else a copy).
    void Result(bool reused, bool padding);
    // LOCAL EXPERIMENT (L1 row P): the S the row holds the input at (0: none), from the lease while it
    // holds, else the query (counted as I0's memo and query rows while profiling).
    std::uint64_t PendingAt() const { return pAt; }
    // Skip point (a), InPlaceDrawInput's vertex FlushPending: =1 passes S (FlushPending returns after
    // FlushDepthPlanes while it holds); =2 runs today's call and checks it; else today's call.
    void FlushPending(std::uint64_t address, std::size_t bytes);
    // =2: a skip point's prediction and its check against today's path there (`effect`: it stored,
    // waited, published, kept or listed an image); the after-input check uses the first one.
    struct PendingPrediction {
        bool would = false;
        std::uint64_t additions = 0;
        std::uint64_t own = 0;
    };
    PendingPrediction PredictPending();
    void VerifyPending(const PendingPrediction& prediction, bool effect, const char* point);

private:
    // I0's collect (S1): today's, with row C's prediction and W check while counting.
    std::uint64_t countedCollect(std::uint64_t address, std::size_t bytes);
    Recorder::InputLease* lease = nullptr;
    std::uint64_t address = 0;
    std::size_t bytes = 0;
    std::uint64_t pAt = 0;
    // =2: the skip points predicted, whether every one would have skipped, the first prediction's
    // additions (the after-input check's raced rule).
    std::uint32_t pPoints = 0;
    bool pAllWould = true;
    PendingPrediction pFirst{};
    bool active = false;
    bool counting = false;
    bool pHeld = false;
    bool cHeld = false;
    bool finished = false;
};

// LOCAL EXPERIMENT (L1 row P, skip point (b)): CopyDrawInput's arguments for the flush hook, set by
// this scope around its FlushGpuWrites and cleared when it ends (no hook is installed until the first
// storage image goes pending, and a later, unrelated hook must not find them). A TEB slot points at
// the scope's own copy; nothing is set without row P or an S.
class InputLeaseHookArgs {
public:
    InputLeaseHookArgs(InputLeaseShadow* shadow, std::uint64_t address, std::size_t bytes);
    ~InputLeaseHookArgs();
    InputLeaseHookArgs(const InputLeaseHookArgs&) = delete;
    InputLeaseHookArgs& operator=(const InputLeaseHookArgs&) = delete;
    struct Data {
        std::uint64_t address = 0;
        std::size_t bytes = 0;
        std::uint64_t serial = 0;
        InputLeaseShadow* shadow = nullptr;
    };

private:
    Data data{};
    bool set = false;
};
// Whether the calling thread has hook arguments set (for tests).
bool InputLeaseHookArgsSet();

// The flush hook's side (FlushForAccess, at most 12 lines there): takes and clears the arguments at
// its top (nested hooks never see them) and keeps them when they are this access's. At(), at the skip
// point: =1 the S to skip at while it holds (0: today's path); =2 predicts and returns 0. Verify(),
// after today's path at =2, checks it.
class InputLeaseHookSkip {
public:
    InputLeaseHookSkip(std::uint64_t address, std::size_t bytes);
    std::uint64_t At();
    void Verify(bool stored, bool published, bool kept, std::size_t images);

private:
    const InputLeaseHookArgs::Data* args = nullptr;
    InputLeaseShadow::PendingPrediction prediction{};
};

}

#endif
