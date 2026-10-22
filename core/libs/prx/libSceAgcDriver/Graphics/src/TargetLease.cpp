#include "prx/libSceAgcDriver/Graphics/include/TargetLease.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/FieldCount.hpp"
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

void TargetLeaseAfter(TargetLeaseUse& use, bool resident) {
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
    LeaseEntry* entry = nullptr;
    for (auto& way : table.sets[set]) {
        if (way.valid && way.recorder == use.recorder && way.device == use.device && sameTarget(way.color, color)) {
            entry = &way;
            break;
        }
    }
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
    if (entry != nullptr) entry->used = ++table.tick;
    // Only a quiet lookup under the key it started with makes or renews an entry (fix 3 of 9.2: a
    // move during the lookup refuses the creation).
    if (!proved || walked || worked || GuestMemory::CurrentCollectMemoKey() != use.memo || StorageLevelsSerial() != use.levels) return;
    if (entry == nullptr) {
        entry = &table.sets[set][0];
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
        entry->valid = true;
        entry->recorder = use.recorder;
        entry->device = use.device;
        entry->color = color;
        entry->used = ++table.tick;
    }
    entry->memo = use.memo;
    entry->levels = use.levels;
}

std::string TargetLeaseReport() {
    const auto take = [](std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed)); };
    std::string line;
    char text[160];
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
    return line;
}

}
