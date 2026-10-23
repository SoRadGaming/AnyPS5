#include "prx/libSceAgcDriver/Graphics/include/TargetLease.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/FieldCount.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace AgcDriver::Graphics {
namespace {

// LOCAL DEBUG (APS5_PROFILE_DRAW, GPU-OWNED-PLAN 9.4.1 D4-0): why a lookup is not eligible (no memo
// key last), and what an eligible one would have been; relaxed, counted only while profiling, taken
// with exchange(0) by the [proofs] line.
enum LeaseIneligible : std::uint8_t { Eligible, IneligibleLinear, IneligibleLayered, IneligibleDumping, IneligibleUnlocked, IneligibleKeyPath, IneligibleNoCache, IneligibleLarge, IneligibleNoMemoKey, LeaseIneligibleCount };
constexpr std::array<const char*, LeaseIneligibleCount> LeaseIneligibleNames{"eligible", "linear or no detiler", "array/mips/3D/MSAA", "dumping targets", "no recording recorder or GPU lock", "DCC without the key fast path", "texture cache off", "1 GiB or more", "no memo key"};
enum LeaseOutcome : std::size_t { WouldHit, MissNoEntry, MissEvicted, MissEpoch, MissAllocations, MissUnwatch, MissLevels, MissKeyProofEmpty, MissNoProof, MissStamped, MissNotProved, MissClearWork, MissWalked, LeaseOutcomeCount };
constexpr std::array<const char*, LeaseOutcomeCount> LeaseOutcomeNames{"would-hit", "no entry", "evicted by a set conflict", "memo key moved: epoch", "allocations", "unwatch", "levels moved", "not proved: K1", "no proof", "stamped", "other", "clear work", "walked"};

struct LeaseCounts {
    std::atomic<std::uint64_t> lookups{0};
    std::array<std::atomic<std::uint64_t>, LeaseIneligibleCount> ineligible{};
    std::array<std::atomic<std::uint64_t>, LeaseOutcomeCount> outcomes{};
    std::atomic<std::uint64_t> evictions{0};
};
constinit LeaseCounts leaseCounts;

void countLease(std::atomic<std::uint64_t>& counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, GPU-OWNED-PLAN 9.4.3 D4-B): the lease's counts,
// relaxed and cumulative while the switch is set (TargetLeaseTotals; the [proofs] line prints 10 s
// deltas): events, misses by their first reason (a refusal of the lease's refresh by its verdict,
// "other pending" by OtherPending in its order), and creations refused by their first reason.
enum LeaseEvent : std::size_t { EventLookups, EventEligible, EventHits, EventUnitChecked, EventPinned, EventTargetKeysLive, EventWouldHits, EventVerified, EventRaced, EventCreated, LeaseEventCount };
enum LeaseMiss : std::size_t { LeaseMissNone, LeaseMissNoLease, LeaseMissEpoch, LeaseMissAllocations, LeaseMissUnwatch, LeaseMissLevels, LeaseMissMultisample, LeaseMissDepthPlane, LeaseMissMovedLate, LeaseMissNotCached, LeaseMissReplaced, LeaseMissCmask, LeaseMissCopiedWriter, LeaseMissNoProof, LeaseMissMoved, LeaseMissCollect, LeaseMissKeyProof, LeaseMissStamped, LeaseMissOtherOutside, LeaseMissOtherFlushing, LeaseMissOtherSameRange, LeaseMissOtherInside, LeaseMissOtherKeyRange, LeaseMissOtherDepthPlane, LeaseMissKeyPin, LeaseMissP1Verifying, LeaseMissK1Verifying, LeaseMissCount };
constexpr std::array<const char*, LeaseMissCount> LeaseMissNames{"none", "no lease", "memo key moved: epoch", "allocations", "unwatch", "levels moved", "MSAA guard (=2)", "depth plane", "moved by the kept-bytes bound", "not cached", "replaced", "CMASK moved", "CPCW would sync", "no proof", "generation/keys moved", "collect 0", "key proof", "stamped", "other pending: outside only (P1 off)", "flushing", "same range", "unit inside", "key range", "depth plane stale", "K1 pin refusal", "P1 verifying", "K1 verifying"};
static_assert(LeaseMissOtherDepthPlane - LeaseMissOtherOutside + 1 == static_cast<std::size_t>(StorageTexture::OtherPending::Count), "the lease's other-pending misses follow OtherPending");
enum LeaseRefusal : std::size_t { RefuseNotProved, RefuseClearWork, RefuseWalked, RefuseMemoMoved, RefuseLevelsMoved, RefuseNoProof, RefuseShape, RefuseKeyNotFound, RefuseNotCovered, RefuseCmaskProof, LeaseRefusalCount };
constexpr std::array<const char*, LeaseRefusalCount> LeaseRefusalNames{"not proved", "clear work", "ranges not covered before (walked)", "memo key moved", "levels moved", "no refresh proof", "image shape", "key not found or the lookup would act", "ranges not covered", "no CMASK proof"};

struct LeaseExperimentCounts {
    std::array<std::atomic<std::uint64_t>, LeaseEventCount> events{};
    std::array<std::atomic<std::uint64_t>, LeaseMissCount> misses{};
    std::array<std::atomic<std::uint64_t>, LeaseRefusalCount> refusals{};
};
constinit LeaseExperimentCounts leaseExperiment;

void countEvent(LeaseEvent event) {
    leaseExperiment.events[event].fetch_add(1, std::memory_order_relaxed);
}

// The key holds every ColorTarget field but the slot (32 members with the slot; the two VkExtent2D
// are compared by their parts). A member added by a merge, wherever it lands (a padding hole too,
// where sizeof would not move), fails the field count until it is compared here and the count is
// raised (FieldCount.hpp).
static_assert(AggregateFieldCount<ColorTarget, 32>, "a ColorTarget field was added: compare it in sameTarget (the D4-0 lease key holds every field but the slot)");
bool sameTarget(const ColorTarget& a, const ColorTarget& b) {
    return a.address == b.address && a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.format == b.format && a.bytes == b.bytes && a.componentMapping == b.componentMapping && a.tileMode == b.tileMode && a.elementBytes == b.elementBytes && a.packing == b.packing && a.dccAddress == b.dccAddress && a.dccAlphaOnMsb == b.dccAlphaOnMsb && a.dccPipeAligned == b.dccPipeAligned && a.cmaskAddress == b.cmaskAddress && a.cmaskBytes == b.cmaskBytes && a.cmaskFastClear == b.cmaskFastClear && a.surfaceAddress == b.surfaceAddress && a.surfaceExtent.width == b.surfaceExtent.width && a.surfaceExtent.height == b.surfaceExtent.height && a.mipCount == b.mipCount && a.mip == b.mip && a.mipTail == b.mipTail && a.clearWords == b.clearWords && a.depth == b.depth && a.depthSlice == b.depthSlice && a.arrayAddress == b.arrayAddress && a.arrayLastSlice == b.arrayLastSlice && a.arraySlice == b.arraySlice && a.layers == b.layers && a.exportIndex == b.exportIndex && a.uintExport == b.uintExport && a.pipeBankXor == b.pipeBankXor && a.samples == b.samples && a.dccControl == b.dccControl;
}

// 16 sets x 4 ways, the set from the target's 64 KiB block (a direct-mapped table on the address
// collided about 37% of the time: GPU-OWNED-PLAN 9.2, D4 row 11). Per thread (ThreadScratch).
constexpr std::size_t LeaseSets = 16;
constexpr std::size_t LeaseWays = 4;

struct LeaseEntry {
    bool valid = false;
    const Recorder* recorder = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    ColorTarget color{};
    GuestMemory::CollectMemoKey memo;
    std::uint64_t levels = 0;
    std::uint64_t used = 0;
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, D4-B): the lease, while `leased`, made under
    // `memo` and `levels` above: the storage key's words, the image (compared by address only: another
    // image under the key is a replaced entry), and the CMASK proof's copy with the proof serial it
    // was read at.
    bool leased = false;
    std::array<std::uint32_t, 8> words{};
    const StorageTexture* texture = nullptr;
    std::size_t cmaskBytes = 0;
    std::uint64_t cmaskGeneration = 0;
    std::uint64_t cmaskSerial = 0;
};

struct LeaseTable {
    std::array<std::array<LeaseEntry, LeaseWays>, LeaseSets> sets{};
    // Per set, fingerprints of the last keys a set conflict evicted (0 empty): a later lookup of
    // one with no entry is counted as evicted rather than new.
    std::array<std::array<std::uint64_t, LeaseWays>, LeaseSets> evicted{};
    std::array<std::uint8_t, LeaseSets> nextEvicted{};
    std::uint64_t tick = 0;
};
struct LeaseTableTag {};

std::size_t leaseSet(std::uint64_t address) {
    return static_cast<std::size_t>(((address >> 16u) * 0x9E3779B97F4A7C15ull) >> 60u);
}

std::uint64_t leaseFingerprint(const Recorder* recorder, VkDevice device, const ColorTarget& color) {
    std::uint64_t hash = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(recorder)) ^ (static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(device)) << 1u);
    for (const std::uint64_t word : {color.address, color.dccAddress, color.cmaskAddress, (std::uint64_t{color.extent.width} << 32u) | color.extent.height, static_cast<std::uint64_t>(static_cast<std::uint32_t>(color.format)) | (std::uint64_t{color.elementBytes} << 32u), (std::uint64_t{color.mip} << 32u) | color.depthSlice}) {
        hash = (hash ^ word) * 0x9E3779B97F4A7C15ull;
    }
    return hash | 1u;
}

bool takeEvicted(LeaseTable& table, std::size_t set, std::uint64_t fingerprint) {
    for (auto& evicted : table.evicted[set]) {
        if (evicted != fingerprint) continue;
        evicted = 0;
        return true;
    }
    return false;
}

LeaseEntry* findEntry(LeaseTable& table, std::size_t set, const Recorder* recorder, VkDevice device, const ColorTarget& color) {
    for (auto& way : table.sets[set]) {
        if (way.valid && way.recorder == recorder && way.device == device && sameTarget(way.color, color)) return &way;
    }
    return nullptr;
}

// A new entry: a free way of the set, else its least recently used one (a set-conflict eviction,
// whose fingerprint is kept for the "evicted" reason).
LeaseEntry& insertEntry(LeaseTable& table, std::size_t set, const Recorder* recorder, VkDevice device, const ColorTarget& color) {
    auto* entry = &table.sets[set][0];
    for (auto& way : table.sets[set]) {
        if (!way.valid) {
            entry = &way;
            break;
        }
        if (way.used < entry->used) entry = &way;
    }
    if (entry->valid) {
        countLease(leaseCounts.evictions);
        table.evicted[set][table.nextEvicted[set]++ % LeaseWays] = leaseFingerprint(entry->recorder, entry->device, entry->color);
    }
    *entry = LeaseEntry{};
    entry->valid = true;
    entry->recorder = recorder;
    entry->device = device;
    entry->color = color;
    entry->used = ++table.tick;
    return *entry;
}

// LOCAL DEBUG (APS5_PROFILE_DRAW, D4-0): the would-hit classification of one eligible lookup, and the
// entry a quiet lookup makes or renews.
void classifyLookup(TargetLeaseUse& use, bool resident) {
    countLease(leaseCounts.lookups);
    if (use.ineligible != Eligible) {
        countLease(leaseCounts.ineligible[use.ineligible]);
        return;
    }
    countLease(leaseCounts.ineligible[Eligible]);
    const auto miss = StorageTexture::TakeTargetMiss();
    const bool proved = resident && StorageTexture::RefreshesProved() == use.proved + 1;
    const bool walked = GuestMemory::ThreadCollectedBytes() != use.walked;
    const bool worked = use.registerClearWorked || use.cmaskClearWorked;
    const auto& color = *use.color;
    auto& table = ThreadScratch<LeaseTable, LeaseTableTag>();
    const auto set = leaseSet(color.address);
    auto* entry = findEntry(table, set, use.recorder, use.device, color);
    auto outcome = WouldHit;
    if (entry == nullptr) outcome = takeEvicted(table, set, leaseFingerprint(use.recorder, use.device, color)) ? MissEvicted : MissNoEntry;
    else if (entry->memo.epoch != use.memo.epoch) outcome = MissEpoch;
    else if (entry->memo.mapping != use.memo.mapping) outcome = MissAllocations;
    else if (entry->memo.unwatched != use.memo.unwatched) outcome = MissUnwatch;
    else if (entry->levels != use.levels) outcome = MissLevels;
    else if (!proved) outcome = miss == StorageTexture::TargetMiss::KeyProofEmpty ? MissKeyProofEmpty : miss == StorageTexture::TargetMiss::NoProof ? MissNoProof : miss == StorageTexture::TargetMiss::Stamped ? MissStamped : MissNotProved;
    else if (worked) outcome = MissClearWork;
    else if (walked) outcome = MissWalked;
    countLease(leaseCounts.outcomes[outcome]);
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, D4-B): while the switch is set the table is the
    // leases' own, renewed by their hits and written by their creations only, so which way a set
    // conflict replaces does not depend on APS5_PROFILE_DRAW; the counter then only classifies.
    if (TargetLeases() != 0) return;
    if (entry != nullptr) entry->used = ++table.tick;
    // Only a quiet lookup under the key it started with makes or renews an entry (fix 3 of 9.2: a
    // move during the lookup refuses the creation).
    if (!proved || walked || worked || GuestMemory::CurrentCollectMemoKey() != use.memo || StorageLevelsSerial() != use.levels) return;
    if (entry == nullptr) entry = &insertEntry(table, set, use.recorder, use.device, color);
    entry->memo = use.memo;
    entry->levels = use.levels;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, D4-B): the ranges a hit reads the stamps of without
// collecting them, for the full path collects them (refreshProved's surface and image keys, which a
// lease's image shares with the target, ProvedCurrentDccKeys' target keys and cmaskProved's CMASK):
// all covered under the current memo key, so each of those collects would be a memo hit.
bool leaseRangesCovered(const ColorTarget& color, std::size_t colorBytes) {
    if (!GuestMemory::CollectCovered(color.address, colorBytes)) return false;
    if (color.dccAddress != 0 && (!GuestMemory::CollectCovered(color.dccAddress, colorBytes / 256) || !GuestMemory::CollectCovered(color.dccAddress, DccKeyBytes(color.bytes)))) return false;
    return color.cmaskAddress == 0 || GuestMemory::CollectCovered(color.cmaskAddress, color.cmaskBytes);
}

// One hit attempt (=1) or prediction (=2): what the probe found under the cache mutex.
struct LeaseProbe {
    TargetLeaseUse* use;
    LeaseEntry* entry;
    bool apply;
    LeaseMiss miss = LeaseMissNone;
    bool held = false;
    StorageTexture* image = nullptr;
    StorageTexture::LeaseRows rows;
    // =2: the image's target-key proof when the check held (today's materializeRegisterClear must
    // leave it) and its pin's note, and the CMASK's live proof (today's materializeCmaskClear must
    // leave it).
    DccRangeProof targetKeys;
    std::uint64_t pinNote = 0;
    CmaskProofEntry cmask;
};

LeaseMiss verdictMiss(const StorageTexture::LeaseRows& rows) {
    using Verdict = StorageTexture::LeaseVerdict;
    switch (rows.verdict) {
        case Verdict::NoProof: return LeaseMissNoProof;
        case Verdict::Moved: return LeaseMissMoved;
        case Verdict::Collect: return LeaseMissCollect;
        case Verdict::KeyProof: return LeaseMissKeyProof;
        case Verdict::Stamped: return LeaseMissStamped;
        case Verdict::OtherPending: return static_cast<LeaseMiss>(LeaseMissOtherOutside + static_cast<std::size_t>(rows.other));
        case Verdict::KeyPin: return LeaseMissKeyPin;
        case Verdict::P1Verifying: return LeaseMissP1Verifying;
        case Verdict::K1Verifying: return LeaseMissK1Verifying;
        case Verdict::CopiedWriter: return LeaseMissCopiedWriter;
        default: return LeaseMissNoProof;
    }
}

[[noreturn]] void leaseCopyDisagrees(const ColorTarget& color, const LeaseEntry& entry, const CmaskProofEntry& live) {
    std::fprintf(stderr, "[target-lease] APS5_EXPERIMENT_TARGET_LEASES=2: the CMASK proof copy of target 0x%llx (CMASK 0x%llx) holds %zu bytes at generation %llu under proof serial %llu, but the live proof under the same serial is %s%zu bytes at generation %llu\n", static_cast<unsigned long long>(color.address), static_cast<unsigned long long>(color.cmaskAddress), entry.cmaskBytes, static_cast<unsigned long long>(entry.cmaskGeneration), static_cast<unsigned long long>(entry.cmaskSerial), live.found ? "" : "absent, ", live.bytes, static_cast<unsigned long long>(live.generation));
    std::fflush(stderr);
    std::abort();
}

// The CMASK's row: the lease's copy while no proof changed since it was read (=1), else the live
// proof of the address, read under its mutex outside the cache mutex (only the committer proves
// CMASKs, under the GPU mutex it holds, so none moves before the row is checked); a miss without one
// of the CMASK's bytes. =2 always reads the live proof (the copy, under an unchanged serial, must
// equal it) and keeps it for its check of today's path.
bool prepareCmaskRow(LeaseProbe& probe) {
    const auto& color = *probe.use->color;
    auto& entry = *probe.entry;
    if (color.cmaskAddress == 0) return true;
    auto bytes = entry.cmaskBytes;
    auto generation = entry.cmaskGeneration;
    if (!probe.apply || CmaskProofSerial() != entry.cmaskSerial) {
        const auto live = CmaskProofFor(color.cmaskAddress);
        if (!probe.apply) {
            probe.cmask = live;
            if (live.serial == entry.cmaskSerial && (!live.found || live.bytes != entry.cmaskBytes || live.generation != entry.cmaskGeneration)) leaseCopyDisagrees(color, entry, live);
        }
        if (!live.found || live.bytes != color.cmaskBytes) {
            probe.miss = LeaseMissCmask;
            return false;
        }
        bytes = live.bytes;
        generation = live.generation;
        if (probe.apply) {
            entry.cmaskBytes = live.bytes;
            entry.cmaskGeneration = live.generation;
            entry.cmaskSerial = live.serial;
        }
    }
    probe.rows.cmask = {color.cmaskAddress, bytes, generation};
    return true;
}

// Under the cache mutex, with the image the cache holds under the lease's key (null: none).
bool probeLease(StorageTexture* image, void* user) {
    auto& probe = *static_cast<LeaseProbe*>(user);
    const auto& use = *probe.use;
    const auto& entry = *probe.entry;
    const auto& color = *use.color;
    const auto missed = [&](LeaseMiss miss) {
        probe.miss = miss;
        return false;
    };
    probe.image = image;
    if (image == nullptr) return missed(LeaseMissNotCached);
    // A replaced entry is a miss (the stricter reading of fix 2 of 9.2): the full path runs the
    // lookup's Requires on the new image and the lease is made again over it.
    if (image != entry.texture || image->Descriptor().dccAddress != color.dccAddress) return missed(LeaseMissReplaced);
    // =2: the lookup closure's two Requires as predicates (=1's hit runs them as Requires, step 6), so no
    // prediction stands where today's closure would throw. Both are fixed when an image is made and
    // held on this one when the lease was made, so only another image fails them: replaced.
    if (!probe.apply && (!image->Attachable() || image->GuestBytes() != use.colorBytes)) return missed(LeaseMissReplaced);
    // Fix 1 of 9.2: the kept-bytes bound ran since the first test, and a submit, a finish or a
    // deferred destruction may move the memo key; the depth planes are read again with it.
    if (GuestMemory::CurrentCollectMemoKey() != entry.memo || StorageLevelsSerial() != entry.levels) return missed(LeaseMissMovedLate);
    if (DepthPlanesStaleOver(color.address, use.colorBytes)) return missed(LeaseMissDepthPlane);
    // ProvedCurrentDccKeys' answering test over the range materializeRegisterClear reads: in the lock
    // only when it would answer something other than a register clear (else the clear runs live).
    auto& rows = probe.rows;
    if (color.dccAddress != 0) {
        const auto& proof = image->TargetKeyProof();
        const auto count = DccKeyBytes(color.bytes);
        if (proof.generation != 0 && proof.address == color.dccAddress && proof.count == count && proof.keys != DccKeys::ClearRegister) rows.targetKeys = {color.dccAddress, count, proof.generation};
    }
    rows.apply = probe.apply;
    // Fix 7 of 9.2: =2's prediction leaves the queued host writes for today's path to stamp.
    rows.drain = probe.apply;
    if (image->RefreshForLease(rows) != StorageTexture::LeaseVerdict::Hit) return missed(verdictMiss(rows));
    if (!probe.apply) {
        probe.targetKeys = image->TargetKeyProof();
        probe.pinNote = image->KeyProof().pinNote;
    }
    probe.held = true;
    return true;
}

void countMiss(LeaseMiss miss) {
    leaseExperiment.misses[miss].fetch_add(1, std::memory_order_relaxed);
}

void countRefusal(LeaseRefusal refusal) {
    leaseExperiment.refusals[refusal].fetch_add(1, std::memory_order_relaxed);
}

// =2: what landed since the prediction, as the race test reads it now (null: nothing), in this order:
// a change of the pending registry; a store into the lease's rows (the draining stamp check: the
// prediction left the queued host writes queued, and today's next tracker lock stamps them); a move
// of the collect memo key or the levels serial; the end of the pin's window. The last two also follow
// a change the prediction could not see coming: today's lookup runs the kept-bytes bound after it
// (=2 skips it), whose submit, finish or deferred destruction may end the pin (a reap) or move the
// memo key (an unwatch), as fix 1 of 9.2 re-checks for =1's hit.
const char* landedSincePrediction(const TargetLeaseUse& use) {
    const auto& prediction = use.prediction;
    const auto& color = *use.color;
    if (StorageTexture::PendingSerial() != prediction.serial) return "a change of the pending registry";
    if (!GuestMemory::StampsHold(std::span(prediction.unchanged.data(), prediction.unchangedRows), std::span(&prediction.cmask, prediction.cmask.generation != 0 ? 1 : 0))) return "a store into the lease's rows";
    if (GuestMemory::CurrentCollectMemoKey() != use.memo || StorageLevelsSerial() != use.levels) return "a move of the collect memo key or the levels serial";
    if (prediction.pinned && DccKeyPinRefusal(color.dccAddress, use.colorBytes / 256, prediction.pinNote) != KeyPin::Holds) return "the end of the pin's window";
    return nullptr;
}

// =2: the race test at one point of today's path, with the walked count there (TargetLeasePrediction).
void notePoint(TargetLeaseUse& use, TargetLeasePoint point) {
    auto& prediction = use.prediction;
    const auto at = static_cast<std::size_t>(point);
    prediction.reached[at] = true;
    prediction.landed[at] = landedSincePrediction(use);
    prediction.walkedAt[at] = GuestMemory::ThreadCollectedBytes();
}

// =2: today's full path against the prediction (9.4.3 (a)-(f); (c) also without profiling, where
// ThreadCollectedBytes counts nothing: the coverage the hit's skipped collects rely on held before
// the lookup), part by part in today's order. The lookup's part ((a), (b), (c), (e)) is read as it
// returned (TargetLeaseLookedUp): a materialization the hit does not skip runs live in both and may
// refresh or rescan. Each materialization the hit skips has its own part: no work (d), no walk (c)
// and its proof unchanged (f). A failure after which something landed (landedSincePrediction), read
// at the point before the failing part ran, or K1's advance at a later reading of the tracker clock,
// is raced (the first 8 printed): a part's own stamps, pending marks and reaps come after its point,
// so they never excuse its own failure (today's lookup is read after it returned, so its own may).
// Any other aborts.
void verifyPrediction(TargetLeaseUse& use, StorageTexture* resident) {
    const auto& prediction = use.prediction;
    const auto& color = *use.color;
    const auto walkedNow = GuestMemory::ThreadCollectedBytes();
    const auto reached = [&](TargetLeasePoint point) { return prediction.reached[static_cast<std::size_t>(point)]; };
    const auto walkedAt = [&](TargetLeasePoint point) { return prediction.walkedAt[static_cast<std::size_t>(point)]; };
    const char* failed = nullptr;
    auto point = TargetLeasePoint::Lookup;
    bool clockMoved = false;
    if (!prediction.lookedUp) failed = "(a) the lookup threw or returned no image";
    else if (resident != prediction.texture) failed = "(a) the lookup returned another image";
    else if (prediction.provedAfter != use.proved + 1) failed = "(b) the lookup's refresh was not proved once";
    else if (!use.covered) failed = "(c) the lease's ranges were not all covered under its memo key before the lookup";
    else if (walkedAt(TargetLeasePoint::Lookup) != use.walked) failed = "(c) the lookup's collects walked";
    else if (prediction.proofPendingSerial != prediction.pendingSerial) failed = "(e) the refresh proof's pending serial differs";
    else if (prediction.proofUnitChecked != prediction.unitChecked) failed = "(e) the refresh proof's unit check differs";
    else if (prediction.pinned && (!prediction.layersAt || prediction.proofGeneration != prediction.imageGeneration || prediction.imageGeneration != prediction.current)) {
        failed = "(e) K1's generation advance differs";
        clockMoved = prediction.layersAt && prediction.proofGeneration == prediction.imageGeneration && prediction.imageGeneration > prediction.current;
    } else {
        // The register clear's part (skipped only on a DCC target whose target-key row held), up to the
        // CMASK's point (or now, on a target without a CMASK).
        if (prediction.skipRegisterClear && color.dccAddress != 0 && reached(TargetLeasePoint::RegisterClear)) {
            point = TargetLeasePoint::RegisterClear;
            const auto walkedEnd = reached(TargetLeasePoint::Cmask) ? walkedAt(TargetLeasePoint::Cmask) : walkedNow;
            const auto& keys = resident->TargetKeyProof();
            const auto& before = prediction.targetKeys;
            if (use.registerClearWorked) failed = "(d) the register clear a hit skips did work";
            else if (walkedEnd != walkedAt(TargetLeasePoint::RegisterClear)) failed = "(c) the register clear's collects walked";
            else if (keys.address != before.address || keys.count != before.count || keys.keys != before.keys || keys.generation != before.generation) failed = "(f) the target-key proof changed";
        }
        // The CMASK's part, to now.
        if (failed == nullptr && prediction.skipCmaskProof && reached(TargetLeasePoint::Cmask)) {
            point = TargetLeasePoint::Cmask;
            const auto cmask = CmaskProofFor(color.cmaskAddress);
            if (use.cmaskClearWorked) failed = "(d) the CMASK clear a hit proves done did work";
            else if (walkedNow != walkedAt(TargetLeasePoint::Cmask)) failed = "(c) the CMASK proof's collect walked";
            else if (cmask.found != prediction.cmaskFound || cmask.bytes != prediction.cmaskBytes || cmask.generation != prediction.cmaskGeneration) failed = "(f) the CMASK proof changed";
        }
    }
    if (failed == nullptr) {
        countEvent(EventVerified);
        return;
    }
    const auto serialNow = StorageTexture::PendingSerial();
    // The point before the failing part (read now when the lookup threw: nothing of today's path ran
    // after it).
    const auto at = static_cast<std::size_t>(point);
    const char* landed = clockMoved ? "a stamp elsewhere (K1's advance read the clock later)" : prediction.reached[at] ? prediction.landed[at] : landedSincePrediction(use);
    static std::atomic<int> reports{0};
    if (landed != nullptr) {
        countEvent(EventRaced);
        if (reports.fetch_add(1, std::memory_order_relaxed) < 8) std::fprintf(stderr, "[target-lease] slot %u target 0x%llx+0x%zx: %s after %s that landed after the lease's prediction (not a disagreement)\n", color.slot, static_cast<unsigned long long>(color.address), use.colorBytes, failed, landed);
        return;
    }
    std::fprintf(stderr, "[target-lease] APS5_EXPERIMENT_TARGET_LEASES=2: slot %u target 0x%llx+0x%zx (dcc 0x%llx, cmask 0x%llx+0x%zx): the lease predicted a hit (rows: surface gen %llu, keys gen %llu, target keys gen %llu, CMASK gen %llu; unit-checked %d, pinned %d, pending serial %llu, clock %llu; skips the register clear %d, the CMASK proof %d) but today's lookup disagrees: %s (route: image %s, refreshes proved +%lld, walked %llu bytes, register clear worked %d, CMASK clear worked %d; proof: generation %llu of the image's %llu, pending serial %llu, unit-checked %d, layers at the image's %d; pending serial now %llu)\n",
        color.slot, static_cast<unsigned long long>(color.address), use.colorBytes, static_cast<unsigned long long>(color.dccAddress), static_cast<unsigned long long>(color.cmaskAddress), color.cmaskBytes,
        static_cast<unsigned long long>(prediction.unchanged[0].generation), static_cast<unsigned long long>(color.dccAddress != 0 ? prediction.unchanged[1].generation : 0), static_cast<unsigned long long>(prediction.unchangedRows > (color.dccAddress != 0 ? 2u : 1u) ? prediction.unchanged[prediction.unchangedRows - 1].generation : 0), static_cast<unsigned long long>(prediction.cmask.generation),
        prediction.unitChecked ? 1 : 0, prediction.pinned ? 1 : 0, static_cast<unsigned long long>(prediction.pendingSerial), static_cast<unsigned long long>(prediction.current), prediction.skipRegisterClear ? 1 : 0, prediction.skipCmaskProof ? 1 : 0,
        failed, resident == prediction.texture ? "the predicted one" : resident == nullptr ? "none" : "another", static_cast<long long>(prediction.provedAfter - use.proved), static_cast<unsigned long long>(walkedNow - use.walked), use.registerClearWorked ? 1 : 0, use.cmaskClearWorked ? 1 : 0,
        static_cast<unsigned long long>(prediction.proofGeneration), static_cast<unsigned long long>(prediction.imageGeneration), static_cast<unsigned long long>(prediction.proofPendingSerial), prediction.proofUnitChecked ? 1 : 0, prediction.layersAt ? 1 : 0, static_cast<unsigned long long>(serialNow));
    std::fflush(stderr);
    std::abort();
}

// The lease after a full path whose slot was quiet (TargetLease.hpp), else none for the target.
void createLease(TargetLeaseUse& use, StorageTexture* resident) {
    const auto& color = *use.color;
    auto& table = ThreadScratch<LeaseTable, LeaseTableTag>();
    const auto set = leaseSet(color.address);
    auto* entry = findEntry(table, set, use.recorder, use.device, color);
    const auto refuse = [&](LeaseRefusal refusal) {
        countRefusal(refusal);
        if (entry != nullptr) entry->leased = false;
    };
    // Quiet: the refresh proved (a proof needs the GPU mutex, which this thread holds, so no other
    // thread moved the count), no clear work, and every range of the lease covered before the path
    // (it walked none of them: D4-0's "walked", without its profiling, so the lease's behaviour does
    // not depend on APS5_PROFILE_DRAW).
    // (Clear work first: a register clear's own refresh may move the proved count too.)
    if (resident == nullptr) return refuse(RefuseNotProved);
    if (use.registerClearWorked || use.cmaskClearWorked) return refuse(RefuseClearWork);
    if (StorageTexture::RefreshesProved() != use.proved + 1) return refuse(RefuseNotProved);
    if (!use.covered) return refuse(RefuseWalked);
    // Fix 3 of 9.2: the memo key read before the path and the levels serial read before
    // StorageLeaseKey's StorageSurface are both compared after it.
    if (GuestMemory::CurrentCollectMemoKey() != use.memo) return refuse(RefuseMemoMoved);
    const auto levels = StorageLevelsSerial();
    if (levels != use.levels) return refuse(RefuseLevelsMoved);
    if (resident->LeaseState().proofGeneration == 0) return refuse(RefuseNoProof);
    // The image and the lookup's surface span the target's bytes exactly (cachedStorageTexture's
    // depth-plane flush covers its surface's bytes, which a hit's depth-plane test must cover too),
    // and the image's keys are the target's (MetadataMoved stays false).
    const auto surface = SurfaceForTarget(color);
    if (resident->Descriptor().dccAddress != color.dccAddress || resident->GuestBytes() != use.colorBytes || TargetMemoBytes(surface) != use.colorBytes) return refuse(RefuseShape);
    std::array<std::uint32_t, 8> words{};
    if (!StorageLeaseKey(*use.context, surface, resident, words)) return refuse(RefuseKeyNotFound);
    if (GuestMemory::CurrentCollectMemoKey() != use.memo) return refuse(RefuseMemoMoved);
    if (StorageLevelsSerial() != levels) return refuse(RefuseLevelsMoved);
    if (!leaseRangesCovered(color, use.colorBytes)) return refuse(RefuseNotCovered);
    CmaskProofEntry cmask;
    if (color.cmaskAddress != 0) {
        cmask = CmaskProofFor(color.cmaskAddress);
        if (!cmask.found || cmask.bytes != color.cmaskBytes) return refuse(RefuseCmaskProof);
    }
    if (entry == nullptr) entry = &insertEntry(table, set, use.recorder, use.device, color);
    entry->used = ++table.tick;
    entry->memo = use.memo;
    entry->levels = levels;
    entry->leased = true;
    entry->words = words;
    entry->texture = resident;
    entry->cmaskBytes = cmask.bytes;
    entry->cmaskGeneration = cmask.generation;
    entry->cmaskSerial = cmask.serial;
    countEvent(EventCreated);
}

// =2's step 2: RequireNoMultisampleTarget is pure, so the prediction tests it without letting it throw.
// Where it throws there is no prediction, and today's lookup, whose first check it is, throws it
// itself (as =1's hit would at the same step).
bool multisampleGuardsAllow(std::uint64_t address) {
    try {
        RequireNoMultisampleTarget(address, 1, "a storage image access");
        return true;
    } catch (...) {
        return false;
    }
}

}

int TargetLeases() {
    static const int mode = [] {
        const char* text = std::getenv("APS5_EXPERIMENT_TARGET_LEASES");
        if (text == nullptr) return 0;
        const auto value = std::strtol(text, nullptr, 10);
        return value == 0 ? 0 : value == 2 ? 2 : 1;
    }();
    return mode;
}

void TargetLeaseBefore(TargetLeaseUse& use) {
    static const bool noCache = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    const auto& color = *use.color;
    auto* recorder = Recorder::Active();
    use.recorder = recorder;
    if (!use.tiled) use.ineligible = IneligibleLinear;
    else if (color.samples != 1 || color.arrayLastSlice != 0 || color.mipCount != 1 || color.depth != 1) use.ineligible = IneligibleLayered;
    else if (use.dumping) use.ineligible = IneligibleDumping;
    else if (!GuestMemory::GpuMutex().HeldByThisThread() || recorder == nullptr || !recorder->Recording()) use.ineligible = IneligibleUnlocked;
    else if (color.dccAddress != 0 && !KeyFastPath()) use.ineligible = IneligibleKeyPath;
    else if (noCache) use.ineligible = IneligibleNoCache;
    else if (use.colorBytes >= (1ull << 30u)) use.ineligible = IneligibleLarge;
    if (use.ineligible != Eligible) return;
    use.memo = GuestMemory::CurrentCollectMemoKey();
    if (!use.memo.Valid()) {
        use.ineligible = IneligibleNoMemoKey;
        return;
    }
    use.levels = StorageLevelsSerial();
    use.proved = StorageTexture::RefreshesProved();
    use.walked = GuestMemory::ThreadCollectedBytes();
    static_cast<void>(StorageTexture::TakeTargetMiss());
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, GPU-OWNED-PLAN 9.4.3 D4-B): a hit, steps 1-7 of
// 9.4.3 in the lookup's order (TargetLease.hpp), which renews the entry for the table's replacement;
// =2 runs the same check without side effects (RequireNoMultisampleTarget and the closure's Requires
// as predicates, no kept-bytes bound, no splice or count, refresh writes or drain) and keeps what it
// found.
std::shared_ptr<StorageTexture> TargetLeaseHit(TargetLeaseUse& use) {
    const int mode = TargetLeases();
    if (mode == 0) return nullptr;
    if (use.profiled) countEvent(EventLookups);
    if (use.ineligible != Eligible) return nullptr;
    if (use.profiled) countEvent(EventEligible);
    const auto& color = *use.color;
    auto& table = ThreadScratch<LeaseTable, LeaseTableTag>();
    auto* entry = findEntry(table, leaseSet(color.address), use.recorder, use.device, color);
    const bool apply = mode == 1;
    LeaseProbe probe{&use, entry, apply};
    std::shared_ptr<StorageTexture> texture;
    // 1. The entry, its memo key and its levels serial (TargetLeaseBefore read both just now).
    if (entry == nullptr || !entry->leased) probe.miss = LeaseMissNoLease;
    else if (entry->memo.epoch != use.memo.epoch) probe.miss = LeaseMissEpoch;
    else if (entry->memo.mapping != use.memo.mapping) probe.miss = LeaseMissAllocations;
    else if (entry->memo.unwatched != use.memo.unwatched) probe.miss = LeaseMissUnwatch;
    else if (entry->levels != use.levels) probe.miss = LeaseMissLevels;
    // 2. cachedStorageTexture's first check, with its arguments (a guestBytes of 0 checks a byte): =2
    // tests it (no prediction where it would throw), =1 calls it below.
    else if (!apply && !multisampleGuardsAllow(color.address)) probe.miss = LeaseMissMultisample;
    else {
        if (apply) RequireNoMultisampleTarget(color.address, 1, "a storage image access");
        // 3. Its FlushDepthPlanes(Overlapped) would store nothing.
        if (DepthPlanesStaleOver(color.address, use.colorBytes)) probe.miss = LeaseMissDepthPlane;
        else {
            // 4. Its kept-bytes bound, at its place.
            if (apply) {
                if (auto* recorder = Recorder::Active(); recorder != nullptr) recorder->BoundKeptBytes();
            }
            // 5. Under the cache mutex: the image, its refresh, the splice and the hit count.
            if (prepareCmaskRow(probe)) texture = RefreshLeasedStorage(*use.context, entry->words, &probeLease, &probe, apply);
        }
    }
    if (texture != nullptr) {
        // 6. The lookup closure's two Requires, on the image the cache holds (fix 2 of 9.2).
        Require(texture->Attachable(), "storage format cannot be a color attachment");
        Require(color.mipCount > 1 || color.depth > 1 || texture->GuestBytes() == use.colorBytes, "resident image layout differs from the color layout");
        // 7. The materializations the lock proved no-ops (fix 18 of 9.2: a target without its keys'
        // row, a K1 target, clears its keys live at today's place).
        use.hit = true;
        use.skipRegisterClear = color.dccAddress == 0 || probe.rows.targetKeys.generation != 0;
        use.skipCmaskProof = color.cmaskAddress != 0;
        // The way's tick, so a set conflict replaces the least recently used lease, with or without
        // APS5_PROFILE_DRAW (only creations and hits write the table while the switch is set).
        entry->used = ++table.tick;
        countEvent(EventHits);
        if (probe.rows.unitChecked) countEvent(EventUnitChecked);
        if (probe.rows.pinned) countEvent(EventPinned);
        if (!use.skipRegisterClear) countEvent(EventTargetKeysLive);
        return texture;
    }
    if (probe.held) {
        countEvent(EventWouldHits);
        auto& prediction = use.prediction;
        const auto& rows = probe.rows;
        prediction.hit = true;
        prediction.texture = probe.image;
        prediction.skipRegisterClear = color.dccAddress == 0 || rows.targetKeys.generation != 0;
        prediction.skipCmaskProof = color.cmaskAddress != 0;
        prediction.unitChecked = rows.unitChecked;
        prediction.pinned = rows.pinned;
        prediction.pinNote = probe.pinNote;
        prediction.serial = rows.serial;
        prediction.pendingSerial = rows.pendingSerial;
        prediction.current = rows.current;
        const auto row = [](const StorageTexture::LeaseRow& lease) { return GuestMemory::UnchangedQuery{lease.address, static_cast<std::size_t>(lease.bytes), lease.generation}; };
        prediction.unchanged[prediction.unchangedRows++] = row(rows.surface);
        if (color.dccAddress != 0) prediction.unchanged[prediction.unchangedRows++] = row(rows.keys);
        if (rows.targetKeys.generation != 0) prediction.unchanged[prediction.unchangedRows++] = row(rows.targetKeys);
        prediction.cmask = row(rows.cmask);
        prediction.targetKeys = probe.targetKeys;
        prediction.cmaskFound = probe.cmask.found;
        prediction.cmaskBytes = probe.cmask.bytes;
        prediction.cmaskGeneration = probe.cmask.generation;
    } else {
        countMiss(probe.miss);
    }
    // Whether the full path about to run can walk none of the lease's ranges (its creation's test).
    use.covered = leaseRangesCovered(color, use.colorBytes);
    return nullptr;
}

void TargetLeaseLookedUp(TargetLeaseUse& use, StorageTexture* resident) {
    auto& prediction = use.prediction;
    if (!prediction.hit) return;
    if (resident != nullptr) {
        const auto state = resident->LeaseState();
        prediction.lookedUp = true;
        prediction.provedAfter = StorageTexture::RefreshesProved();
        prediction.proofGeneration = state.proofGeneration;
        prediction.proofPendingSerial = state.pendingSerial;
        prediction.imageGeneration = state.generation;
        prediction.proofUnitChecked = state.unitChecked;
        prediction.layersAt = state.layersAt;
    }
    notePoint(use, TargetLeasePoint::Lookup);
}

void TargetLeaseBeforeMaterialize(TargetLeaseUse& use, bool cmask) {
    const auto& prediction = use.prediction;
    if (!prediction.hit) return;
    if (cmask ? !prediction.skipCmaskProof : (!prediction.skipRegisterClear || use.color->dccAddress == 0)) return;
    notePoint(use, cmask ? TargetLeasePoint::Cmask : TargetLeasePoint::RegisterClear);
}

void TargetLeaseAfter(TargetLeaseUse& use, StorageTexture* resident) {
    if (use.profiled) classifyLookup(use, resident != nullptr);
    // LOCAL EXPERIMENT (D4-B): a full path is checked against =2's prediction, then makes the lease.
    if (TargetLeases() == 0 || use.ineligible != Eligible || use.hit) return;
    if (use.prediction.hit) verifyPrediction(use, resident);
    createLease(use, resident);
}

TargetLeaseCounts TargetLeaseTotals() {
    const auto event = [](LeaseEvent kind) { return leaseExperiment.events[kind].load(std::memory_order_relaxed); };
    TargetLeaseCounts counts{event(EventLookups), event(EventEligible), event(EventHits), event(EventUnitChecked), event(EventPinned), event(EventTargetKeysLive), event(EventWouldHits), event(EventVerified), event(EventRaced), event(EventCreated)};
    for (const auto& refusal : leaseExperiment.refusals) counts.refused += refusal.load(std::memory_order_relaxed);
    for (const auto& miss : leaseExperiment.misses) counts.misses += miss.load(std::memory_order_relaxed);
    return counts;
}

std::string TargetLeaseReport() {
    const auto take = [](std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed)); };
    std::string line;
    // Room for the longest piece, D4-B's summary (156 fixed characters and nine counts).
    char text[512];
    const auto lookups = take(leaseCounts.lookups);
    std::snprintf(text, sizeof(text), "; target leases (D4-0): eligible %llu of %llu (ineligible by reason:", take(leaseCounts.ineligible[Eligible]), lookups);
    line += text;
    for (std::size_t i = Eligible + 1; i < LeaseIneligibleCount; ++i) {
        std::snprintf(text, sizeof(text), "%s %s %llu", i == Eligible + 1 ? "" : ",", LeaseIneligibleNames[i], take(leaseCounts.ineligible[i]));
        line += text;
    }
    std::snprintf(text, sizeof(text), "), would-hit %llu, misses by reason:", take(leaseCounts.outcomes[WouldHit]));
    line += text;
    for (std::size_t i = WouldHit + 1; i < LeaseOutcomeCount; ++i) {
        std::snprintf(text, sizeof(text), "%s %s %llu", i == WouldHit + 1 ? "" : ",", LeaseOutcomeNames[i], take(leaseCounts.outcomes[i]));
        line += text;
    }
    std::snprintf(text, sizeof(text), ", table evictions %llu", take(leaseCounts.evictions));
    line += text;
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, D4-B): the leases' counts, as 10 s deltas.
    if (const int mode = TargetLeases(); mode != 0) {
        static std::array<std::uint64_t, LeaseEventCount> reportedEvents{};
        static std::array<std::uint64_t, LeaseMissCount> reportedMisses{};
        static std::array<std::uint64_t, LeaseRefusalCount> reportedRefusals{};
        const auto delta = [](std::atomic<std::uint64_t>& counter, std::uint64_t& reported) {
            const auto now = counter.load(std::memory_order_relaxed);
            const auto value = now - reported;
            reported = now;
            return static_cast<unsigned long long>(value);
        };
        auto& events = leaseExperiment.events;
        const auto event = [&](LeaseEvent kind) { return delta(events[kind], reportedEvents[kind]); };
        const auto eligible = event(EventEligible);
        const auto all = event(EventLookups);
        const auto hits = event(EventHits);
        const auto unitChecked = event(EventUnitChecked);
        const auto pinned = event(EventPinned);
        const auto targetKeysLive = event(EventTargetKeysLive);
        const auto wouldHits = event(EventWouldHits);
        const auto created = event(EventCreated);
        std::snprintf(text, sizeof(text), "; D4-B (APS5_EXPERIMENT_TARGET_LEASES=%d): eligible %llu of %llu; hits %llu (unit-checked %llu, pinned %llu, target keys cleared live %llu); would-hits %llu; created %llu, refused by reason:", mode, eligible, all, hits, unitChecked, pinned, targetKeysLive, wouldHits, created);
        line += text;
        for (std::size_t i = 0; i < LeaseRefusalCount; ++i) {
            std::snprintf(text, sizeof(text), "%s %s %llu", i == 0 ? "" : ",", LeaseRefusalNames[i], delta(leaseExperiment.refusals[i], reportedRefusals[i]));
            line += text;
        }
        line += "; misses by reason:";
        for (std::size_t i = LeaseMissNoLease; i < LeaseMissCount; ++i) {
            std::snprintf(text, sizeof(text), "%s %s %llu", i == LeaseMissNoLease ? "" : ",", LeaseMissNames[i], delta(leaseExperiment.misses[i], reportedMisses[i]));
            line += text;
        }
        const auto verified = event(EventVerified);
        const auto raced = event(EventRaced);
        std::snprintf(text, sizeof(text), "; =2: verified %llu, raced %llu", verified, raced);
        line += text;
    }
    return line;
}

}
