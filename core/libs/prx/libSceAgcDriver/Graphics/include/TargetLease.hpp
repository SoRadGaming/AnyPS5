#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_TARGETLEASE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_TARGETLEASE_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

// LOCAL DEBUG (APS5_PROFILE_DRAW, GPU-OWNED-PLAN 9.4.1 D4-0): the target lease counter. No behaviour
// change: it counts how many of the draw slot loop's resident target lookups an epoch-owned lease
// (D4-B) would have answered. An eligible lookup (a tiled single-sample 2D target with one level,
// no target dumps, a recording recorder under the GPU lock, the DCC key fast path, the texture
// cache on, under 1 GiB, and a collect memo key: GuestMemory::CurrentCollectMemoKey) is looked up in
// a per-thread table of 16 sets x 4 ways, keyed on the recorder, the device and every ColorTarget
// field but the slot. It would hit when an entry exists, the entry's memo key and storage levels
// serial (StorageLevelsSerial) equal today's, and the lookup was quiet: its refresh was proved
// (RefreshesProved rose by one; a proof needs the GPU lock, so no other thread moves it), its
// collects walked nothing (ThreadCollectedBytes) and neither fast-clear materialization did any
// work. Otherwise it misses, by the first reason in that order (a memo key move by its cause: the
// epoch, else the allocations, else UnwatchSerial; a refresh not proved by StorageTexture::
// TakeTargetMiss). A quiet lookup whose memo key and levels serial held across it writes the
// entry, as D4-B's creation rule would (with D4-B's switch unset: while it is set only the leases'
// creations and hits write the table, and the counter only classifies). Counted on the [proofs]
// line. Both calls sit outside the [draws] lookup wall (refreshResidentTarget), so a stats run's
// lookup wall stays comparable with r70's, which had no counter; the draw's phase totals still
// carry the counter's cost.
//
// LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, GPU-OWNED-PLAN 9.4.3 D4-B): epoch-owned target
// leases on the same table. Unset or 0: today's code (the slot loop builds no TargetLeaseUse unless
// profiling, and then only D4-0 counts). 1: a lease answers the lookup. 2: verify: the lease's check
// runs without side effects first, today's full path always runs and its result is used, and a
// disagreement aborts with a [target-lease] line (or counts as raced when a store or a registry
// change landed in between, read before the part of today's path that disagrees ran). Run =2 with
// APS5_EXPERIMENT_PROOF_UNITS=1 and APS5_EXPERIMENT_KEY_PIN=1 (their own =2 hands their hits to
// today's path, so every lease would miss), never with the Q0 ticket verify runs (a hit skips
// targetSurface()), and never as an A/B arm.
//
// A lease is the entry plus the storage key its image is cached under, the image (the cache must still
// hold it there: another image is a replaced entry, a miss, the stricter reading of fix 2 of 9.2; the
// hit reads the image's proof live either way), the collect memo key and levels serial it was made
// under, and a copy of the CMASK's proof with the proof serial it was read at. It is made only after a full path whose slot was quiet (the refresh proved, no clear work, and
// every range of the lease already covered by the epoch coverage before it: the walk D4-0 counts,
// without its profiling), with the memo key and levels serial unchanged across the path and the key,
// the image's refresh proof held, the key found holding the image where the lookup would neither
// evict nor replace, and the surface, the image's keys, the target's keys and the CMASK covered.
// It lasts for the committer's collect epoch.
//
// A hit, inside refreshResidentTarget's try block and target lookup scope, in the lookup's order: the
// entry and its memo key and levels serial; RequireNoMultisampleTarget as cachedStorageTexture calls
// it; no stale depth plane over the target (else a miss: the lookup's FlushDepthPlanes would store
// it; the slot loop's own flush and RetireDepthSurfaces ran before, so no depth surface is the
// image); Recorder::BoundKeptBytes; then under the cache mutex the memo key, levels serial and depth
// planes again (BoundKeptBytes may submit or finish batches), the cached image (the lease's, else a
// miss), and its RefreshForLease over the image's rows, the target keys' row (the live target-key
// proof, when it would answer: ProvedCurrentDccKeys' test) and the CMASK's row (the live proof of
// that address, read under its mutex unless no proof changed since the lease's copy; a miss without
// one), then the splice and hit count; then the lookup's two Requires on the image, and the
// materializations: materializeRegisterClear only when the target keys' row was not in the lock (K1
// targets, whose range proof is never stable), materializeCmaskClear with its proof test skipped
// only when the register clear was skipped too (its code and size checks still run; after a live
// register clear, which may sync, store or flush over the CMASK past the lock, the proof runs as
// today). Every revocation is live: identity, eviction and replacement
// (the cache), levels, the epoch, UnwatchSerial and allocations (the memo key), stamps of every kind
// and host writes (StampsHold drains them under the tracker lock), the pending registry and P1's unit
// check, K1's pin (refreshProved), depth planes, copied writers and completion labels (the head of
// CompletePendingCpuWrites), kept-bytes submits (BoundKeptBytes runs) and the MSAA guards.
namespace AgcDriver::Graphics {

class Recorder;
class StorageTexture;

// LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, D4-B): what =2 predicted for one lookup, checked
// by TargetLeaseAfter against today's full path.
struct TargetLeasePrediction {
    bool hit = false;
    const StorageTexture* texture = nullptr;
    // The materializations the hit would skip (the CMASK proof only with the register clear).
    bool skipRegisterClear = false;
    bool skipCmaskProof = false;
    bool unitChecked = false;
    bool pinned = false;
    std::uint64_t pinNote = 0;
    std::uint64_t serial = 0;
    std::uint64_t pendingSerial = 0;
    std::uint64_t current = 0;
    // The rows of the check, for the race test: unchanged ones (surface, the image's keys, the target
    // keys) and the CMASK's not-stored-over one (generation 0: none).
    std::array<GuestMemory::UnchangedQuery, 3> unchanged{};
    std::size_t unchangedRows = 0;
    GuestMemory::UnchangedQuery cmask{};
    // The proofs today's materializations would leave alone: the target-key proof and the CMASK's.
    DccRangeProof targetKeys;
    bool cmaskFound = false;
    std::size_t cmaskBytes = 0;
    std::uint64_t cmaskGeneration = 0;
    // Today's lookup as it returned (TargetLeaseLookedUp), before the materializations, which a hit
    // runs live where it does not skip them: the proved count and the image's refresh proof
    // (StorageTexture::LeaseProofState's fields).
    bool lookedUp = false;
    std::uint64_t provedAfter = 0;
    std::uint64_t proofGeneration = 0;
    std::uint64_t proofPendingSerial = 0;
    std::uint64_t imageGeneration = 0;
    bool proofUnitChecked = false;
    bool layersAt = false;
    // The race test at each point where the evidence of a check is complete: after the lookup
    // returned (TargetLeasePoint::Lookup, TargetLeaseLookedUp) and right before each materialization
    // the hit skips (RegisterClear, Cmask: TargetLeaseBeforeMaterialize). Per point, whether it was
    // reached, what had landed since the prediction by then (null: nothing) and the walked count
    // there. A failure is classified at the point before its own check, so a materialization's own
    // stamps, pending marks and reaps never excuse its own disagreement. The CMASK's landed is read
    // before today's register clear (`cmaskLandedEarly`), which runs between the hit's lock and the
    // CMASK at =2 only: its stores over the CMASK abort instead of counting as raced.
    std::array<bool, 3> reached{};
    std::array<const char*, 3> landed{};
    std::array<std::uint64_t, 3> walkedAt{};
    bool cmaskLandedEarly = false;
};
enum class TargetLeasePoint : std::uint8_t { Lookup, RegisterClear, Cmask };

// One slot loop lookup: the slot loop fills the first part, TargetLeaseBefore the second, the two
// materializations their flags; TargetLeaseAfter classifies. LOCAL EXPERIMENT (D4-B): `context` and
// `profiled` (D4-0 counts only while profiling) from the slot loop; TargetLeaseHit sets `hit` with
// the materializations it proved no-ops, or (=2) the prediction, and `covered` before a full path.
struct TargetLeaseUse {
    const ColorTarget* color = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    std::size_t colorBytes = 0;
    // tileMode != Linear and the context has a detiler; DumpTargetLimit() != 0.
    bool tiled = false;
    bool dumping = false;
    const Context* context = nullptr;
    bool profiled = false;
    std::uint8_t ineligible = 0;
    const Recorder* recorder = nullptr;
    GuestMemory::CollectMemoKey memo;
    std::uint64_t levels = 0;
    std::uint64_t proved = 0;
    std::uint64_t walked = 0;
    bool registerClearWorked = false;
    bool cmaskClearWorked = false;
    bool hit = false;
    bool skipRegisterClear = false;
    bool skipCmaskProof = false;
    bool covered = false;
    TargetLeasePrediction prediction;
};

void TargetLeaseBefore(TargetLeaseUse& use);
// LOCAL EXPERIMENT (D4-B): =1, the lease's answer to the lookup (null: a miss, the full path runs); =2,
// the prediction (always null). Null at once with the switch unset or an ineligible lookup.
std::shared_ptr<StorageTexture> TargetLeaseHit(TargetLeaseUse& use);
// =2: today's lookup returned `resident` (it notes what a prediction is checked against, and the race
// test there; nothing without one).
void TargetLeaseLookedUp(TargetLeaseUse& use, StorageTexture* resident);
// =2: the race test right before today's materializeRegisterClear (`cmask` false) or
// materializeCmaskClear (true), when a prediction skips it; nothing otherwise.
void TargetLeaseBeforeMaterialize(TargetLeaseUse& use, bool cmask);
// `resident`: the image the lookup returned (null: none).
void TargetLeaseAfter(TargetLeaseUse& use, StorageTexture* resident);
// The [proofs] line's lease part (reportProofs, every 10 s): counts since the last call.
std::string TargetLeaseReport();
// LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, D4-B): the switch, read once.
int TargetLeases();
// Cumulative counts while the switch is set (relaxed): lookups and eligible ones (counted only while
// profiling), hits (of which unit-checked, pinned, and with the target keys' row missing, so
// materializeRegisterClear ran), =2's would-hits, verified and raced ones, leases made and refused,
// and misses (every reason).
struct TargetLeaseCounts {
    std::uint64_t lookups = 0;
    std::uint64_t eligible = 0;
    std::uint64_t hits = 0;
    std::uint64_t unitChecked = 0;
    std::uint64_t pinned = 0;
    std::uint64_t targetKeysLive = 0;
    std::uint64_t wouldHits = 0;
    std::uint64_t verified = 0;
    std::uint64_t raced = 0;
    std::uint64_t created = 0;
    std::uint64_t refused = 0;
    std::uint64_t misses = 0;
};
TargetLeaseCounts TargetLeaseTotals();
// Defined in Draw.cpp: the CMASK proof of `address` (cmaskProved's entry: `found`, its bytes and
// generation) with the proof serial, read under the CMASK proof mutex; the serial alone (moves with
// every proof of any address while the switch is set; fix 6 of 9.2: a hit tests its own address's
// entry and uses the serial only to skip reading it again).
struct CmaskProofEntry {
    bool found = false;
    std::size_t bytes = 0;
    std::uint64_t generation = 0;
    std::uint64_t serial = 0;
};
CmaskProofEntry CmaskProofFor(std::uint64_t address);
std::uint64_t CmaskProofSerial();
// Tests: proveCmask (a CMASK proof of `address`, as a materialization or a recorded CMASK clear takes
// it), and one draw's resident lookup of a tiled single-level 2D color target as prepareDraw's slot
// loop makes it (FlushDepthPlanes and RetireDepthSurfaces over the target, then refreshResidentTarget
// with the lease, through the lookup closure's CachedStorageSurface and Requires); `used`, when given,
// receives the lookup's lease use as it ended (default when the slot loop would build none).
void NoteCmaskProof(std::uint64_t address, std::size_t bytes, std::uint64_t generation);
std::shared_ptr<StorageTexture> TargetLeaseLookup(const Context& context, const ColorTarget& color, TargetLeaseUse* used = nullptr);

}

#endif
