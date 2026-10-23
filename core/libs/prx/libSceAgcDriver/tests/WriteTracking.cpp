#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/HighestIndexMemo.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
#ifdef __GLIBC__
#include <malloc.h>
#endif
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace {

using namespace AgcDriver::GuestMemory;

constexpr std::size_t Block = 65536;
#ifdef _WIN32
constexpr wchar_t HostHoleMode[] = L"--host-hole";
constexpr wchar_t ResidentRemapMode[] = L"--resident-direct-remap";
#endif

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

#ifdef _WIN32
bool CheckMetadataCommit() {
    std::uintptr_t arena = 0;
    std::size_t bytes = 0;
    GuestArena::GuestArenaRange_nid_postfix(&arena, &bytes);
    if (bytes == 0 || !GuestArena::GuestArenaWriteWatched_nid_postfix()) return false;
    PROCESS_MEMORY_COUNTERS_EX before{};
    PROCESS_MEMORY_COUNTERS_EX after{};
    Require(K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&before), sizeof(before)), "cannot measure private commit before tracking");
    Require(WriteWatched(), "the arena is not write watched");
    Require(K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&after), sizeof(after)), "cannot measure private commit after tracking");
    const auto committed = after.PrivateUsage > before.PrivateUsage ? after.PrivateUsage - before.PrivateUsage : 0;
    std::cout << "write-tracker initialization: " << committed / 1024 << " KiB private commit for " << (bytes >> 30u) << " GiB guest arena\n";
    Require(committed < (64u << 20u), "write tracking committed dense generation tables for untouched guest ranges");
    const auto generation = TrackerGeneration();
    for (std::uint64_t offset = 0; offset + Block <= bytes; offset += std::uint64_t{1} << 32u) {
        Require(NewestStamp(arena + offset, Block) == 0 && UnchangedSince(arena + offset, Block, generation), "an untouched leaf has a write stamp");
    }
    PROCESS_MEMORY_COUNTERS_EX queried{};
    Require(K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&queried), sizeof(queried)), "cannot measure private commit after empty-range queries");
    Require(queried.PrivateUsage < after.PrivateUsage + (4u << 20u), "queries allocated untouched tracking leaves");
    return true;
}

void CheckSparseLeafBoundaries() {
    constexpr std::uintptr_t LeafBytes = std::uintptr_t{1} << 32;
    std::uintptr_t arena = 0;
    std::size_t arenaBytes = 0;
    GuestArena::GuestArenaRange_nid_postfix(&arena, &arenaBytes);
    const auto boundary = arena + (arenaBytes / (2 * LeafBytes)) * LeafBytes;
    constexpr std::size_t bytes = 4 * Block;
    void* memory = GuestArena::GuestArenaAllocateAtOrAbove_nid_postfix(boundary - Block, bytes, Block);
    struct Release {
        void* memory;
        ~Release() {
            GuestArena::GuestArenaReset_nid_postfix(memory, bytes);
            GuestArena::GuestArenaRelease_nid_postfix(memory, bytes);
        }
    } release{memory};
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    Require(base == boundary - Block, "the sparse tracking fixture did not straddle its 4 GiB leaf boundary");
    GuestArena::GuestArenaCommit_nid_postfix(memory, bytes, PAGE_READWRITE, bytes);
    std::memset(memory, 0x11, bytes);
    BumpCollectEpoch();
    const auto collected = CollectWrites(base, bytes);
    Require(collected != 0, "the cross-leaf range was not collected");
    MarkWritten(boundary, 4);
    Require(UnchangedSince(base, Block, collected) && !UnchangedSince(base, bytes, collected), "a driver store across a leaf boundary did not stamp exactly its block");
    Require(UnchangedSinceCollected(boundary, Block, collected), "the cross-leaf driver store became a CPU store");
    static_cast<volatile std::uint8_t*>(memory)[0] = 0x22;
    CollectWritesUncached(base, bytes);
    Require(!UnchangedSinceCollected(base, Block, collected), "a CPU store before the leaf boundary was missed");
    Require(UnchangedSinceCollected(boundary, 3 * Block, collected), "the CPU store stamped the other leaf");
    Unwatch(base, bytes);
    const auto hint = UntrackedHint(base, bytes);
    Require(hint != 0, "the cross-leaf untracked range has no hint");
    static_cast<volatile std::uint8_t*>(memory)[0] = 0x33;
    static_cast<volatile std::uint8_t*>(memory)[3 * Block] = 0x44;
    BumpCollectEpoch();
    const auto changed = UntrackedHint(base, bytes);
    Require(changed > hint, "cross-leaf CPU stores did not change their untracked hint");
    Require(UntrackedHint(base, Block) == changed && UntrackedHint(base + 3 * Block, Block) == changed, "a cross-leaf hint store was missed");
    Require(UntrackedHint(boundary, 2 * Block) == hint, "hint stores changed untouched blocks in the other leaf");
}
#endif

void* AllocateWatched(std::size_t bytes, bool crossLeaf = false) {
#ifdef _WIN32
    static_cast<void>(crossLeaf);
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, Block);
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    const std::size_t alignment = crossLeaf ? std::size_t{1} << 32 : Block;
    const auto prefix = crossLeaf ? Block : 0;
    void* raw = mmap(nullptr, bytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) throw std::runtime_error("cannot map the watched block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = ((begin + prefix + alignment - 1) & ~static_cast<std::uintptr_t>(alignment - 1)) - prefix;
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + alignment) munmap(reinterpret_cast<void*>(aligned + bytes), begin + alignment - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    Require(Watched(reinterpret_cast<std::uint64_t>(block), bytes), "the test block is not watched");
    return block;
}

void CheckSharedBlock() {
    void* memory = AllocateWatched(3 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto boundary = base + 2 * Block - 0xb800;
    const auto firstBegin = base;
    const auto secondEnd = base + 3 * Block;
    const auto sharedFirst = boundary - (base + Block);
    const auto sharedSecond = base + 2 * Block - boundary;
    std::memset(memory, 0x11, 3 * Block);

    const auto synced = CollectWrites(base, 3 * Block);
    Require(synced != 0, "the test block is not collected");

    Require(MarkWritten(firstBegin, static_cast<std::size_t>(boundary - firstBegin)) != 0, "a driver store into watched memory is not stamped");
    std::array<std::uint64_t, 2> generations{synced, synced};
    std::array<std::uint8_t, 2> changed{};
    Require(ChangedBlocks(base + Block, 2 * Block, generations, changed) && changed[0] == BlockWritten && changed[1] == BlockUnchanged, "the shared block does not read as written");
    Require(!StoredOver(boundary, static_cast<std::size_t>(sharedSecond), synced), "a store of the neighbour's bytes counts for the second surface's part of the shared block");
    Require(StoredOver(base + Block, static_cast<std::size_t>(sharedFirst), synced), "the first surface's own store is not seen over its part of the shared block");
    Require(!StoredOver(boundary, static_cast<std::size_t>(secondEnd - boundary), synced), "the second surface reads as stored over");

    const auto touched = MarkWritten(boundary + 0x100, 4);
    Require(StoredOver(boundary, static_cast<std::size_t>(sharedSecond), synced), "a store over the second surface's bytes is not seen");
    Require(!StoredOver(boundary, static_cast<std::size_t>(sharedSecond), touched), "a store older than the generation is seen");

    const auto whole = TrackerGeneration();
    MarkWritten(base + Block, Block);
    Require(StoredOver(boundary, static_cast<std::size_t>(sharedSecond), whole), "a store over the whole block is not seen");
    const auto afterWhole = TrackerGeneration();
    MarkWritten(base + Block, static_cast<std::size_t>(sharedFirst));
    Require(!StoredOver(boundary, static_cast<std::size_t>(sharedSecond), afterWhole), "a partial store after a whole-block store is taken for the whole block");

    const auto many = TrackerGeneration();
    for (int store = 0; store < 6; ++store) MarkWritten(base + Block + static_cast<std::uint64_t>(store) * 64, 4);
    Require(StoredOver(boundary, static_cast<std::size_t>(sharedSecond), many), "forgotten driver stores are taken as missing the range");

    const auto beforeCpu = CollectWrites(base, 3 * Block);
    static_cast<volatile std::uint8_t*>(memory)[Block + 8] = 0x22;
    CollectWritesUncached(base, 3 * Block);
    Require(!StoredOver(boundary, static_cast<std::size_t>(sharedSecond), beforeCpu), "a CPU store outside the shared range counts for it");

    std::array<std::uint8_t, 64> unwatched{};
    const auto outside = reinterpret_cast<std::uint64_t>(unwatched.data());
    if (!Watched(outside, unwatched.size())) Require(StoredOver(outside, unwatched.size(), TrackerGeneration()), "an unwatched range reads as not stored over");
}

void CheckStoredAllSince() {
    constexpr std::size_t bytes = 4 * Block;
    auto* memory = static_cast<std::uint8_t*>(AllocateWatched(bytes));
    struct Cleanup {
        void* memory;
        ~Cleanup() {
#ifdef _WIN32
            GuestArena::GuestArenaReset_nid_postfix(memory, bytes);
            GuestArena::GuestArenaRelease_nid_postfix(memory, bytes);
#else
            GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(memory, bytes);
            munmap(memory, bytes);
#endif
        }
    } cleanup{memory};
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, bytes);
    auto before = CollectWritesUncached(base, bytes);
    Require(before != 0, "the complete-store block is not collected");
    Require(!StoredAllSince_nid_no_patch(base, 64, before), "a range with no driver store has complete coverage");

    MarkWritten(base + Block, Block);
    Require(StoredAllSince_nid_no_patch(base + Block, Block, before), "the first whole-block driver store has no complete coverage");
    Require(StoredAllSince_nid_no_patch(base + Block + 1, Block - 2, before), "a whole-block store lost its interior coverage");
    const auto whole = TrackerGeneration();
    Require(!StoredAllSince_nid_no_patch(base + Block, Block, whole), "a whole-block store at the baseline is taken as newer");

    before = TrackerGeneration();
    static_cast<volatile std::uint8_t*>(memory)[2 * Block + 7] = 0x22;
    CollectWritesUncached(base + 2 * Block, Block);
    Require(!StoredAllSince_nid_no_patch(base + 2 * Block, Block, before), "CPU-only stores prove complete driver coverage");

    before = TrackerGeneration();
    const std::array<std::byte, 4> dword{};
    Write(base + 16, dword);
    Require(StoredAllSince_nid_no_patch(base + 16, dword.size(), before), "a tracked dword store is absent from its exact range");
    Require(!StoredAllSince_nid_no_patch(base, 64, before), "one dword store proves complete surface coverage");
    Require(!StoredAllSince_nid_no_patch(base + 15, 5, before) && !StoredAllSince_nid_no_patch(base + 16, 5, before), "a dword store covers an unstored boundary byte");
    const auto dwordStored = TrackerGeneration();
    Require(!StoredAllSince_nid_no_patch(base + 16, dword.size(), dwordStored), "a dword store at the baseline is taken as newer");
    MarkWritten(base, 64);
    Require(StoredAllSince_nid_no_patch(base, 64, before), "a complete non-block-aligned range store is absent");

    before = TrackerGeneration();
    MarkWritten(base + 512, 16);
    MarkWritten(base + 544, 16);
    Require(!StoredAllSince_nid_no_patch(base + 512, 48, before), "split driver stores prove coverage across a hole");
    MarkWritten(base + 528, 16);
    Require(StoredAllSince_nid_no_patch(base + 512, 48, before), "out-of-order split driver stores did not form complete coverage");
    Require(!StoredAllSince_nid_no_patch(base + 512, 49, before), "split coverage includes an unstored final byte");

    before = TrackerGeneration();
    MarkWritten(base + 1024 + 8, 24);
    MarkWritten(base + 1024, 16);
    Require(StoredAllSince_nid_no_patch(base + 1024, 32, before), "overlapping driver intervals did not form complete coverage");

    before = TrackerGeneration();
    MarkWritten(base + Block - 4, 8);
    Require(StoredAllSince_nid_no_patch(base + Block - 4, 8, before), "a tracked store across a block boundary lost coverage");
    Require(!StoredAllSince_nid_no_patch(base + Block - 5, 9, before) && !StoredAllSince_nid_no_patch(base + Block - 4, 9, before), "a boundary store covers an unstored adjacent byte");

    before = TrackerGeneration();
    for (unsigned piece = 0; piece < 6; ++piece) MarkWritten(base + static_cast<std::uint64_t>(piece) * 64, 4);
    Require(!StoredAllSince_nid_no_patch(base, 4, before), "dropped driver evidence proves complete coverage");
    MarkWritten(base, Block);
    Require(StoredAllSince_nid_no_patch(base, Block, before), "a new whole-block store did not replace dropped evidence");

    before = TrackerGeneration();
    MarkWritten(base, 3 * Block);
    Require(StoredAllSince_nid_no_patch(base, 3 * Block, before), "complete multi-block stores lost coverage");
    Require(StoredAllSince_nid_no_patch(base + 1, 3 * Block - 2, before) && StoredAllSince_nid_no_patch(base + 3 * Block - 1, 1, before), "complete multi-block stores lost edge bytes");
    const auto complete = TrackerGeneration();
    Require(!StoredAllSince_nid_no_patch(base, 3 * Block, complete), "stores older than the baseline prove complete coverage");

    before = TrackerGeneration();
    MarkWritten(base + 2 * Block, Block);
    const auto driverStored = TrackerGeneration();
    static_cast<volatile std::uint8_t*>(memory)[2 * Block + 7] = 0x33;
    CollectWritesUncached(base + 2 * Block, Block);
    Require(StoredAllSince_nid_no_patch(base + 2 * Block, Block, before), "a later CPU store erased the proof that the driver stored every byte");
    Require(!UnchangedSinceCollected(base + 2 * Block, Block, driverStored), "the separate CPU-write check missed a store after complete driver coverage");

    Require(!StoredAllSince_nid_no_patch(base, 0, before) && !StoredAllSince_nid_no_patch(base, 1, 0), "empty or unknown-baseline complete coverage is accepted");
    Require(!StoredAllSince_nid_no_patch(std::numeric_limits<std::uint64_t>::max(), 2, before), "overflowing complete coverage is accepted");
    Unwatch(base + 3 * Block, Block);
    Require(MarkWritten(base + 3 * Block, Block) == 0, "an unwatched whole store acquired trusted driver coverage");
    Require(!StoredAllSince_nid_no_patch(base + 3 * Block, Block, before), "unwatched stores prove complete coverage");
    Require(!StoredAllSince_nid_no_patch(base + 3 * Block - 1, 2, before), "a range entering unwatched memory has complete coverage");
    std::array<std::uint8_t, 16> outside{};
    const auto outsideAddress = reinterpret_cast<std::uint64_t>(outside.data());
    if (!Watched(outsideAddress, outside.size())) {
        MarkWritten(outsideAddress, outside.size());
        Require(!StoredAllSince_nid_no_patch(outsideAddress, outside.size(), before), "an outside untracked range has complete coverage");
    }
}

void CheckSharedBlockProofs() {
    constexpr std::size_t page = 4096;
    void* memory = AllocateWatched(2 * Block);
    auto* guest = static_cast<volatile std::uint8_t*>(memory);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto range = base + Block + 8 * page;
    constexpr std::size_t rangeBytes = 4 * page;
    std::memset(memory, 0x11, 2 * Block);
    const auto synced = CollectWrites(base, 2 * Block);
    Require(synced != 0, "the proof blocks are not collected");

    Require(MarkWritten(base + Block, 64) != 0, "a driver store into watched memory is not stamped");
    guest[Block + 2 * page + 5] = 0x22;
    CollectWritesUncached(base, 2 * Block);
    Require(UnchangedSince(range, rangeBytes, synced), "stores beside the range in its 64 KiB block count for it");
    const std::array<UnchangedQuery, 2> beside{{{range, rangeBytes, synced}, {base + Block + 4 * page, page, synced}}};
    Require(UnchangedSinceAll(beside), "a batch over ranges beside the stores reads as changed");

    const auto beforeCpu = TrackerGeneration();
    guest[Block + 9 * page + 5] = 0x33;
    CollectWritesUncached(base, 2 * Block);
    Require(!UnchangedSince(range, rangeBytes, beforeCpu), "a CPU store inside the range is not seen");
    const std::array<UnchangedQuery, 2> mixed{{{base + Block + 4 * page, page, beforeCpu}, {range, rangeBytes, beforeCpu}}};
    Require(!UnchangedSinceAll(mixed), "a batch with a range written by the CPU reads as unchanged");

    const auto beforeDriver = TrackerGeneration();
    MarkWritten(range + 100, 4);
    Require(!UnchangedSince(range, rangeBytes, beforeDriver), "a driver store inside the range is not seen");

    const auto beforeWhole = TrackerGeneration();
    MarkWritten(base + Block, Block);
    Require(!UnchangedSince(range, rangeBytes, beforeWhole), "a store over the whole block is not seen");

    const auto beforeOwn = TrackerGeneration();
    const auto stored = base + Block + 12 * page + 0x100;
    StoreOwnBytes(stored, 64, [&] {
        std::memset(reinterpret_cast<void*>(stored), 0x55, 64);
        guest[Block + 12 * page + 0x800] = 0x66;
    });
    Require(!UnchangedSince(base + Block + 12 * page + 0x800, 16, beforeOwn), "a guest store beside a driver store in its page during the store is not seen");
    Require(UnchangedSince(base + Block + 4 * page, page, beforeOwn), "a driver store counts for another page of its block");

    const auto beforeNoRecord = TrackerGeneration();
    guest[2 * page + 5] = 0x44;
    CollectWritesUncached(base, 2 * Block);
    Require(UnchangedSince(base + 8 * page, page, beforeNoRecord), "a CPU store beside the range in a block without driver stores counts for it");

    const auto beforeFirstWhole = TrackerGeneration();
    MarkWritten(base, Block);
    Require(!UnchangedSince(base + 8 * page, page, beforeFirstWhole), "a whole-block store in a block without earlier driver stores is not seen");
    MarkWritten(base, 64);
    Require(!UnchangedSince(base + 8 * page, page, beforeFirstWhole), "a partial store after a whole-block store hides it");

    const auto beforeSpan = TrackerGeneration();
    const std::vector<std::byte> span(6 * page, std::byte{0x77});
    Write(base + 2 * page, span);
    Require(!UnchangedSince(base + 4 * page, page, beforeSpan), "a store over several pages of a block is not seen");
    Require(UnchangedSince(base + 12 * page, page, beforeSpan), "a store over several pages of a block counts for the rest of it");

    const auto beforeExact = TrackerGeneration();
    const std::vector<std::byte> nearlyWhole(Block - 512, std::byte{0x5a});
    Write(base + Block + 256, nearlyWhole);
    Require(StoredAllSince_nid_no_patch(base + Block + 256, nearlyWhole.size(), beforeExact), "an own store lost its exact coverage");
    Require(!StoredAllSince_nid_no_patch(base + Block, Block, beforeExact), "the dirty pages of an own store prove its whole block stored");
    Require(!StoredAllSince_nid_no_patch(base + Block, 256, beforeExact), "the dirty first page of an own store proves the bytes before the store stored");

    const auto beforeClear = TrackerGeneration();
    MarkWritten(base + Block + 0x100, 64);
    const std::array<std::byte, 64> label{};
    for (const std::uint64_t at : {4u, 6u, 10u}) Write(base + Block + at * page + 0x200, label);
    Require(StoredAllSince_nid_no_patch(base + Block + 0x100, 64, beforeClear), "own stores elsewhere in the block forget a driver store's complete coverage");

    const auto beforeNeighbour = TrackerGeneration();
    Write(base + Block + 14 * page + 0x100, label);
    Require(!StoredOver(base + Block + 14 * page + 0x800, 16, beforeNeighbour), "a neighbour in the page of an own store reads as stored over");
    Require(!UnchangedSince(base + Block + 14 * page + 0x800, 16, beforeNeighbour), "a range in the page an own store dirtied reads as unchanged");
}

void CheckOwnStore() {
    void* memory = AllocateWatched(2 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto second = base + Block;
    std::memset(memory, 0x11, 2 * Block);
    const auto synced = CollectWrites(base, 2 * Block);
    Require(synced != 0, "the own-store block is not collected");

    std::array<std::byte, 8192> bytes{};
    bytes.fill(std::byte{0x44});
    Write(second, bytes);
    const auto stored = TrackerGeneration();
    Require(!UnchangedSince(second, bytes.size(), synced), "the driver store is not stamped");
    CollectWritesUncached(base, 2 * Block);
    Require(UnchangedSince(second, Block, stored), "a walk after a driver store reports the store again as a newer write");
    Require(UnchangedSinceCollected(second, bytes.size(), synced), "the driver store's own page faults are stamped as a CPU write");

    static_cast<volatile std::uint8_t*>(memory)[Block + 3 * 4096 + 8] = 0x22;
    const std::array<std::byte, 64> label{};
    Write(second + 3 * 4096 + 64, label);
    Require(!UnchangedSinceCollected(second + 3 * 4096, 4096, stored), "a CPU write before a driver store in its page is not stamped as one");

    const auto beforeCpu = TrackerGeneration();
    static_cast<volatile std::uint8_t*>(memory)[16] = 0x33;
    CollectWritesUncached(base, 2 * Block);
    Require(!UnchangedSince(base, 64, beforeCpu), "a CPU write after the driver store is not seen");
}

void CheckUnwatch() {
    void* memory = AllocateWatched(3 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto imported = base + Block;
    std::memset(memory, 0x11, 3 * Block);
    BumpCollectEpoch();
    const auto before = CollectWrites(base, 3 * Block);
    Require(before != 0 && UnchangedSince(imported, Block, before), "the import range was not initially watched");
    Unwatch(imported, Block);
    Require(!Watched(imported, Block) && !UnchangedSince(imported, Block, before), "unwatch kept trusting import stamps");
    Require(CollectWrites(base, 3 * Block) == 0 && CollectWritesUncached(imported, Block) == 0, "unwatch did not invalidate a cached collect");
    Require(Watched(base, Block) && Watched(base + 2 * Block, Block), "unwatch disabled unrelated memory");
    const auto beforeDriverWrite = TrackerGeneration();
    Require(MarkWritten(imported, Block) == 0, "an unwatched import still produces trusted stamps");
    Require(TrackerGeneration() > beforeDriverWrite, "an unwatched driver write did not invalidate byte comparisons");
    const auto beforeOwnWrite = TrackerGeneration();
    StoreOwnBytes(imported, 1, [&] { *reinterpret_cast<std::uint8_t*>(imported) = 0x42; });
    Require(TrackerGeneration() > beforeOwnWrite, "an unwatched CPU store did not invalidate byte comparisons");
    std::array<std::uint64_t, 1> generations{before};
    std::array<std::uint8_t, 1> changed{};
    Require(!ChangedBlocks(imported, Block, generations, changed), "an unwatched import still uses block stamps");
    Unwatch(imported, Block);
    const auto adjacent = CollectWrites(base, Block);
    Require(adjacent != 0, "unwatch broke the adjacent block's collect");
    static_cast<volatile std::uint8_t*>(memory)[0] = 0x33;
    CollectWritesUncached(base, Block);
    Require(!UnchangedSince(base, Block, adjacent), "unwatch hid a CPU edit in unrelated memory");
}

void CheckNewestStamp() {
    void* memory = AllocateWatched(4 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto first = base + Block;
    const auto second = base + 2 * Block;
    std::memset(memory, 0x11, 4 * Block);
    BumpCollectEpoch();
    CollectWrites(base, 4 * Block);
    Unwatch(first, 2 * Block);
    const auto firstBefore = NewestStamp(first, Block);
    const auto secondBefore = NewestStamp(second, Block);
    MarkWritten(second + 16, 4);
    Require(NewestStamp(second, Block) > secondBefore, "a driver store into unwatched memory did not advance its stamp");
    Require(NewestStamp(first, Block) == firstBefore, "a driver store advanced the stamp of other unwatched memory");
    const auto generation = TrackerGeneration();
    static_cast<volatile std::uint8_t*>(memory)[0] = 0x22;
    CollectWritesUncached(base, Block);
    Require(TrackerGeneration() > generation && NewestStamp(first, Block) == firstBefore, "a CPU store into watched memory advanced the stamp of unwatched memory");
    StoreOwnBytes(first, 1, [&] { *reinterpret_cast<std::uint8_t*>(first) = 0x42; });
    Require(NewestStamp(first, Block) > firstBefore, "an own store into unwatched memory did not advance its stamp");
    const auto partial = NewestStamp(base + 3 * Block - 8, Block + 8);
    static_cast<volatile std::uint8_t*>(memory)[3 * Block] = 0x33;
    CollectWritesUncached(base + 3 * Block, Block);
    Require(NewestStamp(base + 3 * Block - 8, Block + 8) > partial, "a CPU store into the watched part of a partly watched range did not advance its stamp");
}

void CheckPartlyWatchedStore() {
    void* memory = AllocateWatched(3 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto stored = base + Block;
    const auto unwatched = base + 2 * Block;
    std::memset(memory, 0x11, 3 * Block);
    BumpCollectEpoch();
    Unwatch(unwatched, Block);
    const auto before = CollectWrites(base, 2 * Block);
    Require(before != 0 && UnchangedSince(base, 2 * Block, before), "the watched blocks were not collected");
    Require(MarkWritten(unwatched - 64, 128) == 0, "a driver store into partly watched memory is trusted");
    Require(!UnchangedSince(stored, Block, before), "a driver store into partly watched memory left its watched block unchanged");
    Require(StoredOver(unwatched - 64, 64, before), "a driver store into partly watched memory is not seen over its watched bytes");
    Require(!StoredOver(stored, Block - 64, before), "a driver store into partly watched memory is seen over watched bytes it did not store");
    Require(UnchangedSince(base, Block, before), "a driver store into partly watched memory stamped a block it did not store into");
    const auto beforeOwn = TrackerGeneration();
    Require(StoreOwnBytes(unwatched - 64, 128, [&] { std::memset(reinterpret_cast<void*>(unwatched - 64), 0x22, 128); }) == 0, "an own store into partly watched memory is trusted");
    Require(!UnchangedSince(stored, Block, beforeOwn), "an own store into partly watched memory left its watched block unchanged");
    Require(UnchangedSince(base, Block, beforeOwn), "an own store into partly watched memory stamped a block it did not store into");
#ifndef _WIN32
    void* split = AllocateWatched(Block);
    const auto splitBase = reinterpret_cast<std::uint64_t>(split);
    const auto watchedHalf = splitBase + Block / 2;
    std::memset(split, 0x11, Block);
    Unwatch(splitBase, Block / 2);
    const auto beforeSplit = CollectWrites(watchedHalf, Block / 2);
    Require(beforeSplit != 0 && UnchangedSince(watchedHalf, Block / 2, beforeSplit), "the watched half of a split block was not collected");
    Require(MarkWritten(splitBase + 64, 128) == 0 && UnchangedSince(watchedHalf, Block / 2, beforeSplit), "a driver store into the unwatched half of a split block stamped its watched half");
    Require(MarkWritten(watchedHalf - 64, 128) == 0, "a driver store across the halves of a split block is trusted");
    Require(!UnchangedSince(watchedHalf, Block / 2, beforeSplit), "a driver store across the halves of a split block left its watched half unchanged");
#endif
}

#ifndef _WIN32
void CheckWatchOverlaps() {
    const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    void* memory = AllocateWatched(Block);
    const auto base = reinterpret_cast<std::uintptr_t>(memory);
    Require(GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(memory, page), "cannot unwatch the block's first page");
    Require(GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(reinterpret_cast<void*>(base + 3 * page), Block - 3 * page), "cannot unwatch the block's last pages");
    Require(!GuestWriteWatch::GuestWriteWatchCovers_nid_postfix(base, Block), "a block with unwatched pages is covered");
    Require(GuestWriteWatch::GuestWriteWatchOverlaps_nid_postfix(base, Block), "a block with watched pages inside does not overlap the watch");
    Require(GuestWriteWatch::GuestWriteWatchOverlaps_nid_postfix(base + page - 1, 2), "a range ending in a watched page does not overlap the watch");
    Require(GuestWriteWatch::GuestWriteWatchOverlaps_nid_postfix(base + 3 * page - 1, 2), "a range starting in a watched page does not overlap the watch");
    Require(!GuestWriteWatch::GuestWriteWatchOverlaps_nid_postfix(base, page), "an unwatched page before the watched ones overlaps the watch");
    Require(!GuestWriteWatch::GuestWriteWatchOverlaps_nid_postfix(base + 3 * page, Block - 3 * page), "unwatched pages after the watched ones overlap the watch");
    Require(!GuestWriteWatch::GuestWriteWatchOverlaps_nid_postfix(base + page, 0), "an empty range overlaps the watch");
}

void CheckWatchedMiddleStore() {
    const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    void* memory = AllocateWatched(Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto middle = base + page;
    std::memset(memory, 0x11, Block);
    Unwatch(base, page);
    Unwatch(base + 3 * page, Block - 3 * page);
    const auto before = CollectWrites(middle, 2 * page);
    Require(before != 0 && UnchangedSince(middle, 2 * page, before), "the watched middle of a block was not collected");
    Require(MarkWritten(base + 3 * page + 64, 128) == 0 && UnchangedSince(middle, 2 * page, before), "a driver store into an unwatched end of a block stamped its watched middle");
    Require(MarkWritten(base + page / 2, 3 * page) == 0, "a driver store around the watched middle of a block is trusted");
    Require(!UnchangedSince(middle, 2 * page, before), "a driver store around the watched middle of a block left it unchanged");
    Require(StoredOver(middle, 2 * page, before), "a driver store around the watched middle of a block is not seen over it");
    const auto beforeOwn = TrackerGeneration();
    Require(StoreOwnBytes(base + page / 2, 3 * page, [&] { std::memset(reinterpret_cast<void*>(base + page / 2), 0x22, 3 * page); }) == 0, "an own store around the watched middle of a block is trusted");
    Require(!UnchangedSince(middle, 2 * page, beforeOwn), "an own store around the watched middle of a block left it unchanged");
}
#endif

void CheckCoarseRuns() {
    constexpr std::size_t bytes = 192 * Block;
    void* memory = AllocateWatched(bytes);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, bytes);
    BumpCollectEpoch();
    const auto collected = CollectWrites(base, bytes);
    Require(collected != 0 && UnchangedSince(base, bytes, collected), "a range nothing wrote after its collect reports a change");
    const auto stored = MarkWritten(base + bytes / 2, 4);
    Require(stored > collected && !UnchangedSince(base, bytes, collected), "a driver store inside whole 4 MiB runs of the range is not seen");
    Require(UnchangedSince(base, bytes / 2, collected), "a driver store after a range made it changed");
    Require(UnchangedSince(base, bytes, stored), "a range checked at its newest stamp reports a change");
    static_cast<volatile std::uint8_t*>(memory)[bytes - 1] = 0x22;
    CollectWritesUncached(base + bytes - Block, Block);
    Require(!UnchangedSince(base, bytes, stored), "a CPU store into the last block of the range is not seen");
}

#ifdef _WIN32
void CheckDeferredSharedImports() {
    constexpr std::size_t bytes = 4 * Block;
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_RESERVE, 0, bytes, nullptr);
    Require(section != nullptr, "cannot create shared import backing");
    auto* memory = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, Block));
    Require(memory != nullptr, "cannot allocate shared import address");
    GuestArena::GuestArenaMap_nid_postfix(memory, bytes, section, 0, PAGE_READWRITE);
    const auto base = reinterpret_cast<std::uintptr_t>(memory);
    const auto protection = [&](std::size_t offset) {
        MEMORY_BASIC_INFORMATION info{};
        Require(VirtualQuery(memory + offset, &info, sizeof(info)) == sizeof(info), "cannot query shared import protection");
        return info.Protect;
    };
    Require(ImportWatched(base, bytes, [] { return true; }), "fresh shared import failed");
    Require(protection(0) == PAGE_READWRITE && protection(bytes - 1) == PAGE_READWRITE, "fresh import armed shared pages");
    const auto gpu = MarkWritten(base + 4096, 4);
    Require(gpu != 0 && protection(0) == PAGE_READONLY && protection(Block) == PAGE_READWRITE, "GPU write did not activate only its block");
    CollectWritesUncached(base, Block);
    Require(!WrittenSince(base + 4096, 4, gpu) && !StoredOver(base + 4096, 4, gpu), "first collect overwrote a GPU generation");
    static_cast<volatile std::uint8_t*>(memory)[4096] = 0x44;
    CollectWritesUncached(base, Block);
    Require(WrittenSince(base + 4096, 4, gpu) && StoredOver(base + 4096, 4, gpu), "CPU store after GPU write was missed");

    const auto second = base + Block;
    const auto before = CollectWritesUncached(second, 4096);
    Require(MarkWritten(second, 4) != 0, "GPU write next to fresh pages failed");
    Require(UnchangedSinceCollected(second, 4, before), "fresh neighbors were stamped as new CPU writes over a collected page");
    Require(StoredOver(second + 8 * 4096, 4096, before), "fresh neighbor history was lost");
    const std::array<std::uint64_t, 1> generations{before};
    std::array<std::uint8_t, 1> changed{};
    std::array<std::uint8_t, 1> cpu{};
    Require(ChangedBlocks(second, Block, generations, changed, cpu) && changed[0] == BlockWritten && cpu[0] == 0, "fresh collection set the block's CPU stamp");
    const auto generation = TrackerGeneration();
    Require(MarkWritten(second, 4) == generation + 1, "an already collected GPU block advanced twice");

    const auto third = base + 2 * Block;
    const auto cached = CollectWritesUncached(third, Block);
    Require(ImportWatched(third, Block, [&] {
        static_cast<volatile std::uint8_t*>(memory)[2 * Block] = 0x55;
        return true;
    }), "tracked import window failed");
    Require(!UnchangedSince(third, Block, cached) && WrittenSince(third, Block, cached), "a store of new bytes within a tracked import window was missed");
    const auto afterImport = CollectWritesUncached(third, Block);
    static_cast<volatile std::uint8_t*>(memory)[2 * Block] = 0x66;
    Require(ImportWatched(third, Block, [] { return true; }) && WrittenSince(third, Block, afterImport), "pre-import CPU store was not collected");

    GuestArena::GuestArenaReset_nid_postfix(memory + 3 * Block, 16384);
    const auto edge = base + 3 * Block + 16384;
    Require(ImportWatched(edge, Block - 16384, [] { return true; }), "unaligned shared import failed");
    const auto edgeGpu = MarkWritten(edge, 4);
    Require(edgeGpu != 0, "fresh collection rejected an unmapped block neighbor");
    CollectWritesUncached(edge, Block - 16384);
    Require(!WrittenSince(edge, 4, edgeGpu), "edge block collection overwrote GPU results");
    GuestArena::GuestArenaReset_nid_postfix(memory, bytes);
    GuestArena::GuestArenaRelease_nid_postfix(memory, bytes);
    CloseHandle(section);
}


void CheckPrivateMappingReuse() {
    void* memory = AllocateWatched(3 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, 3 * Block);
    BumpCollectEpoch();
    const auto before = CollectWrites(base, 3 * Block);
    Require(before != 0, "the original backing was not collected");
    Unwatch(base + Block, Block);
    Require(!Watched(base + Block, Block), "the imported block stayed watched");
    GuestArena::GuestArenaCommit_nid_postfix(memory, 3 * Block, PAGE_READWRITE, 3 * Block);
    Require(!Watched(base + Block, Block), "a protection-only commit restored an imported block");
    const auto neighbour = CollectWrites(base, Block);
    Require(neighbour != 0, "the neighbouring block was not collected before remapping");
    GuestArena::GuestArenaReset_nid_postfix(memory, 3 * Block);
    GuestArena::GuestArenaRelease_nid_postfix(memory, 3 * Block);
    auto* replacement = GuestArena::GuestArenaAllocateAtOrAbove_nid_postfix(base, 3 * Block, Block);
    Require(replacement == memory, "the released arena address was not reused");
    GuestArena::GuestArenaCommit_nid_postfix(replacement, 3 * Block, PAGE_READWRITE, 3 * Block);
    Require(Watched(base, 3 * Block), "fresh backing at an excluded address is not watched");
    Require(!UnchangedSince(base + Block, Block, before), "fresh backing reused an old import generation");
    Require(!UnchangedSince(base, Block, neighbour), "fresh backing reused an adjacent block's old generation");
    const auto after = CollectWrites(base, 3 * Block);
    Require(after > before && UnchangedSince(base, 3 * Block, after), "fresh backing was not collected");
    static_cast<volatile std::uint8_t*>(replacement)[Block + 8] = 0x22;
    CollectWritesUncached(base, 3 * Block);
    Require(!UnchangedSince(base + Block, Block, after), "a CPU edit in fresh backing was missed");
    Unwatch(base + Block, Block);
    Require(!Watched(base + Block, Block) && CollectWrites(base + Block, Block) == 0, "a second import of fresh backing stayed watched");
    GuestArena::GuestArenaReset_nid_postfix(replacement, 3 * Block);
    GuestArena::GuestArenaRelease_nid_postfix(replacement, 3 * Block);
}

void CheckRewatch() {
    void* memory = GuestArena::GuestArenaAllocate_nid_postfix(4 * Block, Block);
    GuestArena::GuestArenaCommit_nid_postfix(memory, 4 * Block, PAGE_READWRITE, 16384);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    Require(Watched(base, 4 * Block), "the rewatch test block is not watched");
    auto* bytes = static_cast<volatile std::uint8_t*>(memory);
    const auto imported = base + Block;
    std::memset(memory, 0x11, 4 * Block);
    BumpCollectEpoch();
    const auto before = CollectWrites(base, 4 * Block);
    Require(before != 0, "the test range was not collected");
    Unwatch(imported, 2 * Block);
    const auto unwatched = UnwatchSerial();
    Require(!Watched(imported, Block) && UntrackedHint(imported, Block) != 0, "the unwatched block does not read as untracked");
    bytes[Block + 8] = 0x22;
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> partial{{{imported + 4096, imported + Block}}};
    Rewatch(partial);
    Require(!Watched(imported, Block) && UnwatchSerial() == unwatched, "part of a block returned to the write watch");
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> first{{{imported, imported + Block}}};
    Rewatch(first);
    Require(Watched(imported, Block) && !Watched(imported + Block, Block), "rewatching did not return exactly its block to the write watch");
    Require(UnwatchSerial() != unwatched, "rewatching did not invalidate what was cached while the block was unwatched");
    Require(UntrackedHint(imported, Block) == 0, "a rewatched block still reads as untracked");
    Require(!UnchangedSince(imported, Block, before) && StoredOver(imported, Block, before), "a block returned to the write watch reads as unchanged since before it left");
    Require(!UnchangedSinceCollected(imported, Block, before), "a CPU store made while the block was unwatched reads as no CPU store");
    Require(UnchangedSince(base, Block, before) && UnchangedSince(base + 3 * Block, Block, before), "rewatching stamped the blocks around it");
    BumpCollectEpoch();
    const auto after = CollectWrites(imported, Block);
    Require(after != 0 && UnchangedSince(imported, Block, after), "a rewatched block is not collected");
    bytes[Block + 16] = 0x33;
    CollectWritesUncached(imported, Block);
    Require(!UnchangedSince(imported, Block, after), "a CPU store after rewatching was missed");
    const auto serial = UnwatchSerial();
    Rewatch(first);
    Require(UnwatchSerial() == serial, "rewatching a watched block changed what it covers");
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> uncommitted{{{base + 2 * Block, base + 3 * Block}}};
    GuestArena::GuestArenaReset_nid_postfix(static_cast<std::uint8_t*>(memory) + 2 * Block, 16384);
    Rewatch(uncommitted);
    Require(!Watched(base + 2 * Block, Block), "a block with uncommitted pages returned to the write watch");
    GuestArena::GuestArenaReset_nid_postfix(memory, 4 * Block);
    GuestArena::GuestArenaRelease_nid_postfix(memory, 4 * Block);
}

void CheckResidentDirectMappingReuse() {
    constexpr std::size_t bytes = 5 * Block;
    auto* memory = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, Block));
    Require(memory != nullptr, "cannot reserve the resident remap fixture");
    struct Cleanup {
        std::uint8_t* memory;
        std::array<HANDLE, 3> sections{};
        ~Cleanup() {
            GuestArena::GuestArenaReset_nid_postfix(memory, bytes);
            GuestArena::GuestArenaRelease_nid_postfix(memory, bytes);
            for (const auto section : sections) {
                if (section == nullptr) continue;
                GuestArena::GuestArenaForgetSection_nid_postfix(section);
                CloseHandle(section);
            }
        }
    } cleanup{memory};
    GuestArena::GuestArenaCommit_nid_postfix(memory, 3 * Block, PAGE_READWRITE, 3 * Block);
    GuestArena::GuestArenaCommit_nid_postfix(memory + 3 * Block, 2 * Block, PAGE_READWRITE, 2 * Block);
    Require(Watched(reinterpret_cast<std::uint64_t>(memory), bytes), "the resident remap fixture is not watched");
    for (auto& section : cleanup.sections) {
        section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, 3 * Block, nullptr);
        Require(section != nullptr, "cannot create the resident remap backing");
    }
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto nativeType = [&](std::size_t offset) {
        MEMORY_BASIC_INFORMATION info{};
        Require(VirtualQuery(memory + offset, &info, sizeof(info)) == sizeof(info), "cannot query the resident remap backing");
        return info.Type;
    };
    for (std::size_t use = 0; use < 2; ++use) {
        BumpCollectEpoch();
        const auto before = CollectWrites(base, 3 * Block);
        Require(before != 0, "the backing before a resident remap was not collected");
        Unwatch(base + Block, Block);
        Require(!Watched(base + Block, Block), "the resident remap exclusion was not established");
        const auto outside = CollectWrites(base + 3 * Block, 2 * Block);
        const auto mapping = GuestArena::GuestArenaCommitGeneration_nid_postfix();
        GuestArena::GuestArenaMapDirect_nid_postfix(memory, 3 * Block, cleanup.sections[use], 0, PAGE_READWRITE);
        Require(nativeType(0) == MEM_PRIVATE && nativeType(2 * Block) == MEM_PRIVATE, "the resident remap did not create private memory");
        Require(Watched(base, 3 * Block), "a resident direct remap retained an old import exclusion");
        Require(GuestArena::GuestArenaCommitGeneration_nid_postfix() > mapping, "a resident direct remap reused the mapping generation");
        Require(!UnchangedSince(base, Block, before) && !UnchangedSince(base + 2 * Block, Block, before), "a resident direct remap reused old neighbour stamps");
        Require(UnchangedSince(base + 3 * Block, 2 * Block, outside), "a resident direct remap invalidated memory outside its range");
        const auto after = CollectWritesUncached(base, 3 * Block);
        Require(after > before && UnchangedSince(base, 3 * Block, after), "the resident remap could not establish a current generation");
        memory[Block + 9] = static_cast<std::uint8_t>(0x51 + use);
        CollectWritesUncached(base, 3 * Block);
        Require(!UnchangedSince(base + Block, Block, after), "a CPU store after a resident direct remap was missed");
        std::cout << "resident direct remap " << use << ": private, watched, new mapping generation, CPU store collected\n";
    }
    Unwatch(base, 3 * Block);
    GuestArena::GuestArenaMap_nid_postfix(memory, 3 * Block, cleanup.sections[0], 0, PAGE_READWRITE);
    Require(nativeType(0) == MEM_MAPPED, "the shared remap did not create mapped memory");
    for (std::size_t block = 0; block < 3; ++block) Require(!Watched(base + block * Block, Block), "an ordinary shared remap restored a private import exclusion");
    GuestArena::GuestArenaMapDirect_nid_postfix(memory + 3 * Block, 2 * Block, cleanup.sections[2], 0, PAGE_READWRITE);
    GuestArena::GuestArenaMapDirect_nid_postfix(memory, 3 * Block, cleanup.sections[2], 0, PAGE_READWRITE);
    Require(nativeType(0) == MEM_MAPPED && nativeType(Block) == MEM_MAPPED && nativeType(2 * Block) == MEM_PRIVATE, "the mixed direct remap did not establish shared and private spans");
    Require(!Watched(base, Block) && !Watched(base + Block, Block) && Watched(base + 2 * Block, Block), "the mixed direct remap restored an exclusion outside its new private span");
    std::cout << "resident direct remap controls: shared exclusions retained, only the new private span restored\n";
    Unwatch(base, 3 * Block);
    bool refused = false;
    try {
        GuestArena::GuestArenaMapDirect_nid_postfix(memory, 3 * Block, cleanup.sections[1], 0, 0);
    } catch (const std::system_error& error) {
        Require(error.code().value() == ERROR_INVALID_PARAMETER, "the failed resident remap reported an unexpected error");
        refused = true;
    }
    std::uintptr_t createdBegin = 0;
    std::uintptr_t createdEnd = 0;
    Require(refused && GuestArena::GuestArenaResidentSpan_nid_postfix(base, 1, &createdBegin, &createdEnd), "the failing protection change did not leave a new private span");
    Require(createdBegin == base && createdEnd <= base + 3 * Block && Watched(base, createdEnd - base), "a failed resident remap did not report its completed private span");
    for (auto block = createdEnd; block < base + 3 * Block; block += Block) Require(!Watched(block, Block), "a failed resident remap restored an uncreated span");
    std::cout << "resident direct remap failure: completed private span watched, original mapping error retained\n";
}

void CheckUntrackedHint() {
    constexpr std::size_t bytes = 4 * Block;
    auto* memory = static_cast<volatile std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(bytes, Block));
    GuestArena::GuestArenaCommit_nid_postfix(const_cast<std::uint8_t*>(memory), bytes, PAGE_READWRITE, 16384);
    struct Cleanup {
        void* memory;
        ~Cleanup() {
            GuestArena::GuestArenaReset_nid_postfix(memory, bytes);
            GuestArena::GuestArenaRelease_nid_postfix(memory, bytes);
        }
    } cleanup{const_cast<std::uint8_t*>(memory)};
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto hinted = base + Block;
    constexpr std::size_t hintedBytes = 2 * Block;
    for (std::size_t offset = 0; offset < bytes; offset += 4096) memory[offset] = 0x11;
    BumpCollectEpoch();
    Require(CollectWrites(base, bytes) != 0 && UntrackedHint(base, bytes) == 0, "watched memory has an untracked hint");
    Unwatch(hinted, hintedBytes);
    const auto stamp = NewestStamp(hinted, hintedBytes);
    const auto first = UntrackedHint(hinted, hintedBytes);
    Require(first != 0, "memory that left the watch has no untracked hint");
    Require(UntrackedHint(base, bytes) == 0 && UntrackedHint(hinted - 4096, 8192) == 0, "a range with watched blocks has an untracked hint");
    std::array<std::uint8_t, 64> host{};
    Require(UntrackedHint(reinterpret_cast<std::uint64_t>(host.data()), host.size()) == 0, "memory outside the arena has an untracked hint");
    Require(!Watched(hinted, hintedBytes) && CollectWrites(hinted, hintedBytes) == 0 && !UnchangedSince(hinted, hintedBytes, TrackerGeneration()), "a hinted range reads as tracked");
    std::array<std::uint64_t, 2> generations{first, first};
    std::array<std::uint8_t, 2> changed{};
    Require(!ChangedBlocks(hinted, hintedBytes, generations, changed), "a hinted range uses block stamps");
    for (unsigned epoch = 0; epoch < 8; ++epoch) {
        BumpCollectEpoch();
        Require(UntrackedHint(hinted, hintedBytes) == first, "an untracked hint moved without a store");
    }
    memory[Block + 3 * 16384 + 8] ^= 1;
    Require(UntrackedHint(hinted, hintedBytes) == first, "an untracked hint was walked twice in one epoch");
    BumpCollectEpoch();
    const auto stored = UntrackedHint(hinted, hintedBytes);
    Require(stored > first, "a CPU store into a hinted range did not move its hint");
    Require(UntrackedHint(hinted + Block, Block) == first, "a CPU store moved the hint of another block");
    BumpCollectEpoch();
    Require(UntrackedHint(hinted, hintedBytes) == stored, "an untracked hint moved again without a store");
    memory[Block + 8] = memory[Block + 8];
    BumpCollectEpoch();
    const auto rewritten = UntrackedHint(hinted, hintedBytes);
    Require(rewritten > stored, "a store of the bytes already there did not move the hint");
    std::thread([&] { memory[2 * Block + 5 * 4096] ^= 1; }).join();
    BumpCollectEpoch();
    const auto threaded = UntrackedHint(hinted, hintedBytes);
    Require(threaded > rewritten, "a store from another thread did not move the hint");
    Require(UntrackedHint(hinted + Block, Block) == threaded && UntrackedHint(hinted, Block) == rewritten, "a walk stamped a store into the wrong block");
    Require(NewestStamp(hinted, hintedBytes) == stamp, "untracked hint walks advanced the block stamps");
    Unwatch(hinted, hintedBytes);
    BumpCollectEpoch();
    Require(UntrackedHint(hinted, hintedBytes) == threaded, "unwatching a hinted range again moved its hint");
    Unwatch(base, Block);
    const auto own = UntrackedHint(base, Block);
    Require(own > threaded && UntrackedHint(hinted, hintedBytes) == threaded, "a range leaving the watch did not start a hint of its own");
    GuestArena::GuestArenaReset_nid_postfix(const_cast<std::uint8_t*>(memory) + Block, hintedBytes);
    GuestArena::GuestArenaCommit_nid_postfix(const_cast<std::uint8_t*>(memory) + Block, hintedBytes, PAGE_READWRITE, 16384);
    BumpCollectEpoch();
    const auto fresh = CollectWrites(hinted, hintedBytes);
    Require(Watched(hinted, hintedBytes) && fresh != 0 && UnchangedSince(hinted, hintedBytes, fresh), "fresh backing of a hinted range is not watched");
    memory[Block + 8] = 0x5a;
    BumpCollectEpoch();
    Require(UntrackedHint(hinted, hintedBytes) == 0, "watched fresh backing has an untracked hint");
    CollectWritesUncached(hinted, hintedBytes);
    Require(!UnchangedSince(hinted, Block, fresh), "an untracked hint walk took a CPU store from watched memory");
    Unwatch(hinted, hintedBytes);
    Require(UntrackedHint(hinted, hintedBytes) > own, "fresh backing leaving the watch did not start a new hint");
}

std::uint64_t ChildHostHole() {
    std::uintptr_t arena = 0;
    std::size_t arenaBytes = 0;
    GuestArena::GuestArenaRange_nid_postfix(&arena, &arenaBytes);
    return (arena + arenaBytes / 2) & ~(std::uint64_t{Block} - 1);
}

void CheckHostHoleInChild() {
    std::array<wchar_t, 32768> executable{};
    const DWORD length = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
    Require(length != 0 && length < executable.size(), "cannot name the test executable");
    const std::wstring command = L"\"" + std::wstring(executable.data(), length) + L"\" " + HostHoleMode;
    std::vector<wchar_t> commandLine(command.begin(), command.end());
    commandLine.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION process{};
    Require(CreateProcessW(executable.data(), commandLine.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr, &startup, &process) != 0, "cannot create the host-hole child");
    const bool reserved = VirtualAllocEx(process.hProcess, reinterpret_cast<void*>(ChildHostHole()), Block, MEM_RESERVE, PAGE_NOACCESS) != nullptr;
    const bool resumed = reserved && ResumeThread(process.hThread) != static_cast<DWORD>(-1);
    const bool exited = resumed && WaitForSingleObject(process.hProcess, 60000) == WAIT_OBJECT_0;
    if (!exited) {
        TerminateProcess(process.hProcess, 1);
        WaitForSingleObject(process.hProcess, INFINITE);
    }
    DWORD exitCode = 1;
    const bool reported = GetExitCodeProcess(process.hProcess, &exitCode) != 0;
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    Require(reserved, "cannot reserve a host range in the child's arena interval");
    Require(resumed, "cannot resume the host-hole child");
    Require(exited, "the host-hole child did not exit");
    Require(reported && exitCode == 0, "the host-hole child failed");
}

void CheckHostHole() {
    constexpr std::size_t bytes = 64;
    const auto hole = ChildHostHole();
    Require(GuestArena::GuestArenaHostRegionOverlaps_nid_postfix(hole, bytes), "the range reserved before the arena is not an arena host region");
    const auto before = TrackerGeneration();
    Require(!Watched(hole, bytes), "an arena host hole reads as watched");
    Require(CollectWrites(hole, bytes) == 0 && CollectWritesUncached(hole, bytes) == 0, "an arena host hole was collected");
    auto* watched = static_cast<std::uint8_t*>(AllocateWatched(Block, true));
    struct Cleanup {
        void* memory;
        ~Cleanup() { GuestArena::GuestArenaRelease_nid_postfix(memory, Block); }
    } cleanup{watched};
    const auto base = reinterpret_cast<std::uint64_t>(watched);
    for (unsigned epoch = 0; epoch < 2; ++epoch) {
        BumpCollectEpoch();
        watched[7] ^= 1;
        Require(CollectWrites(base + 7, 1) != 0, "a warm host-hole check could not collect its watched range");
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            Require(CollectWrites(base + 7, 1) != 0, "a warm host-hole check lost its watched range");
            Require(CollectWrites(hole, bytes) == 0, "a repeated host-hole query hit a collect memo");
            Require(CollectWrites(hole + 7, 1) == 0, "a page-rounded host-hole query hit a collect memo");
        }
    }
    Require(StoredOver(hole, bytes, before), "an arena host hole reads as not stored over");
    Require(!UnchangedSince(hole, bytes, before), "an arena host hole reads as unchanged");
    const std::array queries{UnchangedQuery{hole, bytes, before}};
    Require(!UnchangedSinceAll(queries), "an arena host hole reads as unchanged in a batch");
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, D4-B): StampsHold's rows over a host region fail
    // as UnchangedSinceAll and StoredOver do.
    Require(!StampsHold(queries, {}) && !StampsHold({}, queries), "an arena host hole reads as unchanged or not stored over in StampsHold");
    Require(!UnchangedSinceCollected(hole, bytes, before), "an arena host hole reads as unchanged since a collect");
    Require(MarkWritten(hole, bytes) == 0, "a driver store into an arena host hole is stamped");
    Require(StoreOwnBytes(hole, bytes, [] {}) == 0, "an own store into an arena host hole is stamped");
    Require(!WrittenSince(hole, bytes, before), "an arena host hole reads its stamps as written");
    std::array<std::uint64_t, 1> generations{before};
    std::array<std::uint8_t, 1> changed{};
    std::array<std::uint8_t, 1> cpu{};
    Require(!ChangedBlocks(hole, bytes, generations, changed, cpu) && changed[0] == BlockWritten && cpu[0] == 1, "an arena host hole uses block stamps");
    bool imported = false;
    Require(ImportWatched(hole, bytes, [&] {
        imported = true;
        return true;
    }) && imported, "an import into an arena host hole did not run");
    Require(!Watched(hole, bytes), "an import made an arena host hole watched");
}

std::uint64_t failedCollectBegin = 0;
std::uint64_t failedCollectEnd = 0;
std::uint64_t unwatchHookBegin = 0;
std::uint64_t unwatchHookEnd = 0;

void RecordUnwatch(std::uint64_t address, std::size_t bytes) {
    unwatchHookBegin = address;
    unwatchHookEnd = address + bytes;
}

bool FailFreshCollect(std::uintptr_t address, std::size_t bytes, void (*)(void*, std::uintptr_t, std::uintptr_t), void*) {
    failedCollectBegin = address;
    failedCollectEnd = address + bytes;
    return false;
}

void CheckFailedCollectUnwatch() {
    void* memory = AllocateWatched(3 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, 3 * Block);
    Require(Watched(base + Block, Block), "the probe block was not watched");
    SetFreshCollectHook(&FailFreshCollect);
    SetUnwatchHook(&RecordUnwatch);
    const auto stored = MarkWritten(base + Block + 0x8000, 4);
    SetFreshCollectHook(nullptr);
    SetUnwatchHook(nullptr);
    Require(stored == 0, "a failed fresh collect produced trusted stamps");
    Require(!Watched(base + Block, Block), "a failed collect left its block watched");
    Require(Watched(base, Block) && Watched(base + 2 * Block, Block), "a failed collect unwatched unrelated blocks");
    Require(failedCollectBegin == base + Block && failedCollectEnd == base + 2 * Block, "the failed fresh collect did not cover its block");
    Require(unwatchHookBegin <= base + Block && unwatchHookEnd >= base + 2 * Block, "a failed collect did not run the unwatch hook over its block");
}
#endif

void CheckVersionRanges() {
    constexpr std::size_t count = 80;
    constexpr auto bytes = count * Block;
    auto* memory = static_cast<std::uint8_t*>(AllocateWatched(bytes, true));
    struct Cleanup {
        void* memory;
        ~Cleanup() {
#ifdef _WIN32
            GuestArena::GuestArenaReset_nid_postfix(memory, bytes);
            GuestArena::GuestArenaRelease_nid_postfix(memory, bytes);
#else
            GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(memory, bytes);
            munmap(memory, bytes);
#endif
        }
    } cleanup{memory};
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    Require(UnchangedSince(base, bytes, TrackerGeneration()), "a new range has newer version stamps");
    std::memset(memory, 17, bytes);
    const auto initial = CollectWritesUncached(base, bytes);
    Require(initial != 0, "version ranges could not be collected");
    struct Write { std::size_t block; std::uint64_t generation; std::size_t begin; std::size_t end; };
    std::vector<Write> history;
    std::vector<std::uint64_t> checkpoints{initial};
    std::uint32_t random = 0x61387a29;
    const auto next = [&] { random = random * 1664525u + 1013904223u; return random; };
    const auto blocksUnchanged = [&](const UnchangedQuery& query) {
        for (const auto& write : history) {
            const auto begin = base + write.block * Block;
            if (write.generation > query.generation && begin < query.address + query.bytes && query.address < begin + Block) return false;
        }
        return true;
    };
    const auto bytesUnchanged = [&](const UnchangedQuery& query) {
        for (const auto& write : history) {
            const auto begin = base + write.block * Block + write.begin;
            const auto end = base + write.block * Block + write.end;
            if (write.generation > query.generation && begin < query.address + query.bytes && query.address < end) return false;
        }
        return true;
    };
    for (unsigned step = 0; step < 512; ++step) {
        const auto block = step < 3 ? step : next() % count;
        std::uint64_t generation;
        Write write{block, 0, 13, 17};
        if (step % 3 == 0) {
            memory[block * Block + 9] ^= 1;
            generation = CollectWritesUncached(base + block * Block, Block);
            write.begin = 0;
            write.end = 4096;
        } else if (step % 7 == 1) {
            generation = MarkWritten(base + block * Block, Block);
            write.begin = 0;
            write.end = Block;
        } else generation = MarkWritten(base + block * Block + 13, 4);
        Require(generation > checkpoints.back(), "version range write did not advance the tracker");
        write.generation = generation;
        history.push_back(write);
        checkpoints.push_back(generation);
        for (unsigned check = 0; check < 16; ++check) {
            std::array<UnchangedQuery, 3> queries;
            bool all = true;
            for (auto& query : queries) {
                const auto offset = next() % bytes;
                query = {base + offset, 1 + next() % (bytes - offset), checkpoints[next() % checkpoints.size()]};
                const auto same = UnchangedSince(query.address, query.bytes, query.generation);
                Require(!same || bytesUnchanged(query), "version range reads as unchanged over a newer write in it");
                Require(same || !blocksUnchanged(query), "version range reads as changed while its blocks are unchanged");
                all = all && same;
            }
            Require(UnchangedSinceAll(queries) == all, "batched version ranges disagree with the single ranges");
        }
    }
    Require(!UnchangedSince(base, bytes, 0), "an unknown generation was accepted");
    Require(UnchangedSince(base, bytes, checkpoints.back()), "the latest checkpoint was rejected");
}

void CheckCollectBoundaries() {
    constexpr std::size_t bytes = 3 * Block;
    auto* memory = static_cast<std::uint8_t*>(AllocateWatched(bytes));
    struct Cleanup {
        void* memory;
        ~Cleanup() {
#ifdef _WIN32
            GuestArena::GuestArenaReset_nid_postfix(memory, bytes);
            GuestArena::GuestArenaRelease_nid_postfix(memory, bytes);
#else
            Unwatch(reinterpret_cast<std::uint64_t>(memory), bytes);
            munmap(memory, bytes);
#endif
        }
    } cleanup{memory};
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 17, bytes);
    bool unboundedObserved = false;
    std::thread observer([&] {
        const auto before = CollectWrites(base + 7, 1);
        memory[7] = 19;
        const auto after = CollectWrites(base + 7, 1);
        unboundedObserved = before != 0 && after > before && !UnchangedSince(base + 7, 1, before);
    });
    observer.join();
    Require(unboundedObserved, "a thread without collect epochs reused a stale observation");
    for (const auto offset : {7u, 4096u + 3u, static_cast<unsigned>(Block - 8), static_cast<unsigned>(Block + 3)}) {
        BumpCollectEpoch();
        const auto before = CollectWrites(base + offset, 16);
        Require(before != 0, "an epoch lost watched coverage");
        memory[offset] ^= 1;
        BumpCollectEpoch();
        Require(CollectWrites(base + offset, 16) > before && !UnchangedSince(base + offset, 16, before), "a new epoch missed a CPU write");
        const auto observed = TrackerGeneration();
        memory[offset + 1] ^= 1;
        Require(CollectWritesUncached(base + offset, 16) > observed && !UnchangedSince(base + offset, 16, observed), "an uncached observation reused an epoch result");
        const auto stored = MarkWritten(base + offset + 2, 1);
        Require(CollectWrites(base + offset, 16) >= stored && !UnchangedSince(base + offset, 16, observed), "a memoized observation hid a driver write");
    }
#ifndef _WIN32
    BumpCollectEpoch();
    Require(CollectWrites(base, 1) != 0, "initial partial-block observation failed");
    Unwatch(base + 4096, 4096);
    Require(CollectWrites(base + 4096, 1) == 0, "an unwatched neighbor retained an epoch observation");
    BumpCollectEpoch();
    const auto before = CollectWrites(base, 16);
    Require(before != 0, "an unwatched neighbor prevented collecting a watched prefix");
    memory[0] ^= 1;
    BumpCollectEpoch();
    Require(CollectWrites(base, 16) > before && !UnchangedSince(base, 16, before), "partial-block collection missed a CPU write");
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(memory + 4096, 4096);
    Require(CollectWrites(base + 4096, 1) != 0, "a re-registered neighbor stayed unwatched");
#endif
}
}

#ifdef _WIN32
void CheckWidenedCollectBesideUncommittedPages() {
    constexpr std::size_t reserved = std::size_t{1} << 30;
    constexpr std::size_t committed = Block - 4 * 4096;
    void* range = GuestArena::GuestArenaAllocate_nid_postfix(reserved, Block);
    struct Cleanup {
        void* range;
        ~Cleanup() {
            GuestArena::GuestArenaReset_nid_postfix(range, reserved);
            GuestArena::GuestArenaRelease_nid_postfix(range, reserved);
        }
    } cleanup{range};
    void* memory = static_cast<std::uint8_t*>(range) + reserved - Block;
    MEMORY_BASIC_INFORMATION tail{};
    Require(VirtualQuery(static_cast<std::uint8_t*>(memory) + committed, &tail, sizeof(tail)) == sizeof(tail) && tail.State == MEM_RESERVE, "the test block's tail is not a placeholder");
    GuestArena::GuestArenaCommit_nid_postfix(memory, committed, PAGE_READWRITE, committed);
    Require(VirtualQuery(static_cast<std::uint8_t*>(memory) + committed, &tail, sizeof(tail)) == sizeof(tail) && tail.State == MEM_RESERVE, "committing the test block's head committed its tail");
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    auto* bytes = static_cast<volatile std::uint8_t*>(memory);
    std::memset(memory, 17, committed);
    BumpCollectEpoch();
    const auto before = CollectWrites(base + 4096, 64);
    Require(before != 0, "a collect beside uncommitted pages of its block failed");
    bytes[4096 + 8] = 19;
    BumpCollectEpoch();
    Require(CollectWrites(base + 4096, 64) > before && !UnchangedSince(base + 4096, 64, before), "a collect beside uncommitted pages missed a CPU write");
}
#endif

void CheckImportWindow() {
    void* memory = AllocateWatched(3 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    auto* bytes = static_cast<volatile std::uint8_t*>(memory);
    std::memset(memory, 0x11, 3 * Block);
    const auto synced = CollectWritesUncached(base, 3 * Block);
    Require(synced != 0, "the import window block is not collected");
    Require(ImportWatched(base, 3 * Block, [&] {
        bytes[0x100] = 0x11;
        bytes[Block + 0x2000] = 0x22;
        return true;
    }), "the import window import failed");
    Require(Watched(base, 3 * Block), "an import window left its range unwatched");
    const std::array<std::uint64_t, 3> generations{synced, synced, synced};
    std::array<std::uint8_t, 3> changed{};
    std::array<std::uint8_t, 3> cpu{};
    Require(ChangedBlocks(base, 3 * Block, generations, changed, cpu), "the import window block is not tracked");
    Require(changed[0] == BlockUnchanged && UnchangedSince(base, Block, synced), "a page an import window resolved without changing its bytes reads as written");
    Require(changed[1] == BlockWritten && WrittenSince(base + Block, Block, synced) && cpu[1] == 1, "a store of new bytes within an import window was missed");
    Require(changed[2] == BlockUnchanged && cpu[2] == 0, "a block the import window did not touch reads as written");
    Require(cpu[0] == 1, "a page an import window resolved no longer counts as possibly touched by the CPU");
    const auto after = CollectWritesUncached(base, 3 * Block);
    bytes[0x100] = 0x33;
    static_cast<void>(CollectWritesUncached(base, 3 * Block));
    Require(!UnchangedSince(base, Block, after), "a store after the import window was missed");
}

void CheckImportWindowOtherThread() {
    void* memory = AllocateWatched(2 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    auto* bytes = static_cast<volatile std::uint8_t*>(memory);
    std::memset(memory, 0x11, 2 * Block);
    const auto synced = CollectWritesUncached(base, 2 * Block);
    Require(synced != 0, "the other-thread import window block is not collected");
    Require(ImportWatched(base, 2 * Block, [&] {
        std::thread store([&] {
            bytes[0x100] = 0x11;
            bytes[Block + 0x100] = 0x77;
        });
        store.join();
        return true;
    }), "the other-thread import window import failed");
    const std::array<std::uint64_t, 2> generations{synced, synced};
    std::array<std::uint8_t, 2> changed{};
    std::array<std::uint8_t, 2> cpu{};
    Require(ChangedBlocks(base, 2 * Block, generations, changed, cpu), "the other-thread import window block is not tracked");
    Require(changed[0] == BlockUnchanged && cpu[0] == 1, "a same-value store from another thread within an import window lost its possible CPU touch");
    Require(changed[1] == BlockWritten && WrittenSince(base + Block, Block, synced) && cpu[1] == 1, "a store of new bytes from another thread within an import window was missed");
}

void CheckImportWindowZeroPages() {
    void* memory = AllocateWatched(5 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    auto* bytes = static_cast<volatile std::uint8_t*>(memory);
    auto* plain = static_cast<std::uint8_t*>(memory);
    std::memset(plain, 0x11, Block);
    std::memset(plain + Block, 0, 2 * Block);
    std::memset(plain + 3 * Block, 0x44, Block);
    std::memset(plain + 4 * Block, 0x66, Block);
    const auto synced = CollectWritesUncached(base, 5 * Block);
    Require(synced != 0, "the zero page import window block is not collected");
    Require(ImportWatched(base, 5 * Block, [&] {
        bytes[0x100] = 0x11;
        bytes[Block + 0x100] = 0;
        bytes[2 * Block + 0x2000] = 0x22;
        bytes[3 * Block + 0x3000] = 0x44;
        bytes[4 * Block + 0x100] = 0x55;
        return true;
    }), "the zero page import window import failed");
    const std::array<std::uint64_t, 5> generations{synced, synced, synced, synced, synced};
    std::array<std::uint8_t, 5> changed{};
    std::array<std::uint8_t, 5> cpu{};
    Require(ChangedBlocks(base, 5 * Block, generations, changed, cpu), "the zero page import window block is not tracked");
    Require(changed[0] == BlockUnchanged && cpu[0] == 1, "a page before the zero pages of an import window reads as written");
    Require(changed[1] == BlockUnchanged && UnchangedSince(base + Block, Block, synced), "a zero page an import window resolved without changing its bytes reads as written");
    Require(changed[2] == BlockWritten && WrittenSince(base + 2 * Block, Block, synced), "a store of new bytes to a zero page within an import window was missed");
    Require(changed[3] == BlockUnchanged && UnchangedSince(base + 3 * Block, Block, synced), "a page after the zero pages of an import window was compared with other bytes");
    Require(changed[4] == BlockWritten && WrittenSince(base + 4 * Block, Block, synced), "a store of new bytes after the zero pages of an import window was missed");
}

void CheckImportWindowOverUnwatchedBlock() {
    void* memory = AllocateWatched(4 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    auto* bytes = static_cast<volatile std::uint8_t*>(memory);
    std::memset(memory, 0x11, 4 * Block);
    Unwatch(base + Block, Block);
    const auto synced = CollectWritesUncached(base + 2 * Block, 2 * Block);
    const auto syncedFirst = CollectWritesUncached(base, Block);
    Require(synced != 0 && syncedFirst != 0, "the watched blocks around the unwatched one are not collected");
    bytes[0x500] = 0x44;
    Require(ImportWatched(base, 4 * Block, [&] {
        bytes[0x100] = 0x11;
        bytes[2 * Block + 0x100] = 0x11;
        bytes[3 * Block + 0x2000] = 0x22;
        return true;
    }), "the import over an unwatched block failed");
    Require(!Watched(base + Block, Block), "an import window made an unwatched block watched");
    Require(Watched(base, Block) && Watched(base + 2 * Block, 2 * Block), "an import window over an unwatched block took the watched blocks out of the write watch");
    const std::array<std::uint64_t, 2> generations{synced, synced};
    std::array<std::uint8_t, 2> changed{};
    std::array<std::uint8_t, 2> cpu{};
    Require(ChangedBlocks(base + 2 * Block, 2 * Block, generations, changed, cpu), "the watched blocks after an unwatched block are not tracked");
    Require(changed[0] == BlockUnchanged && UnchangedSince(base + 2 * Block, Block, synced), "a page an import window over an unwatched block resolved without changing its bytes reads as written");
    Require(changed[1] == BlockWritten && WrittenSince(base + 3 * Block, Block, synced), "a store of new bytes within an import window over an unwatched block was missed");
    Require(WrittenSince(base, Block, syncedFirst), "a store before an import window over an unwatched block was missed");
    const auto after = CollectWritesUncached(base, Block);
    bytes[0x100] = 0x33;
    static_cast<void>(CollectWritesUncached(base, Block));
    Require(after != 0 && !UnchangedSince(base, Block, after), "a store after an import window over an unwatched block was missed");
}

void CheckImportWindowFootprint() {
#ifdef __GLIBC__
    constexpr std::size_t bytes = std::size_t{64} << 20;
    void* memory = AllocateWatched(bytes);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x5a, bytes);
    Require(CollectWritesUncached(base, bytes) != 0, "the footprint import window block is not collected");
    const auto heapBytes = [] {
        const auto info = mallinfo2();
        return info.uordblks + info.hblkhd;
    };
    const auto before = heapBytes();
    std::size_t during = 0;
    Require(ImportWatched(base, bytes, [&] {
        during = heapBytes();
        return true;
    }), "the footprint import window import failed");
    Require(Watched(base, bytes), "the footprint import window left its range unwatched");
    const auto held = during > before ? during - before : 0;
    Require(held <= bytes / 64, "an import window over " + std::to_string(bytes >> 20) + " MiB held " + std::to_string(held >> 20) + " MiB of host memory during the import: it copies the range instead of hashing its pages");
    auto* plain = static_cast<volatile std::uint8_t*>(memory);
    const auto synced = CollectWritesUncached(base, bytes);
    Require(ImportWatched(base, bytes, [&] {
        plain[0x100] = 0x5a;
        plain[Block + 0x100] = 0x5b;
        return true;
    }), "the footprint import window re-import failed");
    Require(UnchangedSince(base, Block, synced), "a page a hashed import window resolved without changing its bytes reads as written");
    Require(WrittenSince(base + Block, Block, synced), "a store of new bytes within a hashed import window was missed");
#endif
}

void ProtectPage(std::uint64_t address, bool readable) {
#ifdef _WIN32
    DWORD previous = 0;
    Require(VirtualProtect(reinterpret_cast<void*>(address), 4096, readable ? PAGE_READWRITE : PAGE_NOACCESS, &previous) != 0, "cannot change the protection of a test page");
#else
    Require(mprotect(reinterpret_cast<void*>(address), 4096, readable ? PROT_READ | PROT_WRITE : PROT_NONE) == 0, "cannot change the protection of a test page");
#endif
    GuestAllocations::GuestAllocationsInvalidate_nid_postfix(static_cast<std::uintptr_t>(address), 4096);
}

void CheckStoreAcrossImports() {
    void* memory = AllocateWatched(3 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto boundary = base + 2 * Block;
    std::memset(memory, 0x11, 3 * Block);
    Require(ImportWatched(base, 2 * Block, [] { return true; }) && Watched(base, 2 * Block), "an import that stays watched left the watch");
    Require(ImportWatched(boundary, Block, [] { return true; }, false) && !Watched(boundary, Block), "an import that leaves the watch stayed watched");
    const auto before = CollectWrites(base, 2 * Block);
    Require(before != 0 && UnchangedSince(base, 2 * Block, before), "the watched import was not collected");
    Require(MarkWritten(boundary - 64, 128) == 0, "a GPU write across a watched import and an unwatched one is trusted");
    Require(!UnchangedSince(base + Block, Block, before), "a GPU write across a watched import and an unwatched one left the watched import's last block unchanged");
    Require(StoredOver(boundary - 64, 64, before) && !StoredOver(base + Block, Block - 64, before), "a GPU write across a watched import and an unwatched one is not seen over exactly its watched bytes");
    Require(UnchangedSince(base, Block, before), "a GPU write across a watched import and an unwatched one stamped a block it did not write");
    const auto beforeOwn = TrackerGeneration();
    Require(StoreOwnBytes(boundary - 64, 128, [&] { std::memset(reinterpret_cast<void*>(boundary - 64), 0x22, 128); }) == 0, "an own store across a watched import and an unwatched one is trusted");
    Require(!UnchangedSince(base + Block, Block, beforeOwn) && UnchangedSince(base, Block, beforeOwn), "an own store across a watched import and an unwatched one did not stamp exactly the watched block it wrote");
}

void CheckUnwatchedImport() {
    void* memory = AllocateWatched(3 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    auto* bytes = static_cast<volatile std::uint8_t*>(memory);
    std::memset(memory, 0x11, 3 * Block);
    const auto synced = CollectWritesUncached(base, 3 * Block);
    Require(synced != 0, "the unwatched import block is not collected");
    const auto imported = base + Block / 2;
    const std::size_t importedBytes = Block + Block / 2;
    const auto unreadable = base + Block + 0x3000;
    ProtectPage(unreadable, false);
    bool threw = false;
    bool ran = false;
    try {
        static_cast<void>(ImportWatched(imported, importedBytes, [&] {
            ran = true;
            return true;
        }));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    Require(threw && !ran, "an import that stays watched did not read its range before the import");
    threw = false;
    try {
        Require(ImportWatched(imported, importedBytes, [&] {
            bytes[Block / 2 + 0x100] = 0x22;
            ran = true;
            return true;
        }, false), "an import that leaves the watch failed");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    Require(!threw && ran, "an import that leaves the watch read its range");
    ProtectPage(unreadable, true);
    Require(!Watched(imported, importedBytes) && !Watched(imported, 4096) && !Watched(base + Block, Block), "an import that leaves the watch is still watched");
    Require(Watched(base + 2 * Block, Block), "an import that leaves the watch unwatched a block outside its range");
    Require(UnchangedSince(base + 2 * Block, Block, synced), "an import that leaves the watch stamped a block outside its range");
#ifndef _WIN32
    Require(Watched(base, Block / 2), "an import that leaves the watch unwatched the pages before its range");
    const std::array<std::uint64_t, 1> generations{synced};
    std::array<std::uint8_t, 1> changed{};
    std::array<std::uint8_t, 1> cpu{};
    Require(ChangedBlocks(base, Block / 2, generations, changed, cpu), "the pages before an import that leaves the watch are not tracked");
    Require(changed[0] == BlockUnchanged && UnchangedSince(base, Block / 2, synced) && !WrittenSince(base, Block / 2, synced), "a store to a page that left the watch reads as a store to the watched pages of its block");
    Require(cpu[0] == 1, "a page resolved while an import left the watch no longer counts as possibly touched by the CPU");
    const auto after = CollectWritesUncached(base, Block / 2);
    bytes[0x100] = 0x33;
    static_cast<void>(CollectWritesUncached(base, Block / 2));
    Require(after != 0 && !UnchangedSince(base, Block / 2, after), "a store beside an import that left the watch was missed");
#endif
}

class PartlyCommitted {
public:
    static constexpr std::size_t Bytes = 4 * Block;

    PartlyCommitted() {
#ifdef _WIN32
        memory = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocateAtOrAbove_nid_postfix(GuestArena::GuestArenaHighestFree_nid_no_patch(Bytes, Block), Bytes, Block));
#else
        void* raw = mmap(nullptr, Bytes + Block, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        Require(raw != MAP_FAILED, "cannot reserve the partly committed range");
        const auto begin = reinterpret_cast<std::uintptr_t>(raw);
        const auto aligned = (begin + Block - 1) & ~static_cast<std::uintptr_t>(Block - 1);
        if (aligned != begin) munmap(raw, aligned - begin);
        if (aligned + Bytes != begin + Bytes + Block) munmap(reinterpret_cast<void*>(aligned + Bytes), begin + Block - aligned);
        memory = reinterpret_cast<std::uint8_t*>(aligned);
#endif
        Require(memory != nullptr, "cannot reserve the partly committed range");
    }

    ~PartlyCommitted() {
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(memory, Bytes);
        GuestAllocations::GuestAllocationsInvalidate_nid_postfix(Address(), Bytes);
        GuestArena::GuestArenaRelease_nid_postfix(memory, Bytes);
#else
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(memory, Bytes);
        munmap(memory, Bytes);
        GuestAllocations::GuestAllocationsInvalidate_nid_postfix(Address(), Bytes);
#endif
    }

    PartlyCommitted(const PartlyCommitted&) = delete;
    PartlyCommitted& operator=(const PartlyCommitted&) = delete;

    void Commit(std::size_t block) {
#ifdef _WIN32
        GuestArena::GuestArenaCommit_nid_postfix(memory + block * Block, Block, PAGE_READWRITE, Block);
#else
        Require(mmap(memory + block * Block, Block, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) != MAP_FAILED, "cannot commit a block of the partly committed range");
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(memory + block * Block, Block);
#endif
        GuestAllocations::GuestAllocationsInvalidate_nid_postfix(Address(block), Block);
    }

    void Decommit(std::size_t block) {
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(memory + block * Block, Block);
#else
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(memory + block * Block, Block);
        Require(mmap(memory + block * Block, Block, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0) != MAP_FAILED, "cannot decommit a block of the partly committed range");
#endif
        GuestAllocations::GuestAllocationsInvalidate_nid_postfix(Address(block), Block);
    }

    std::uint8_t* Data() const { return memory; }
    std::uint64_t Address(std::size_t block = 0) const { return reinterpret_cast<std::uint64_t>(memory) + block * Block; }

private:
    std::uint8_t* memory = nullptr;
};

using Runs = std::vector<std::pair<std::uint64_t, std::uint64_t>>;

std::uint64_t CollectCommitted(const PartlyCommitted& range, Runs& runs) {
    BumpCollectEpoch();
    return CollectCommittedWrites(range.Address(), PartlyCommitted::Bytes, runs);
}

void CheckCommittedRanges() {
    PartlyCommitted range;
    const auto base = range.Address();
    range.Commit(0);
    std::memset(range.Data(), 0x11, Block);
    Require(CollectWrites(base, PartlyCommitted::Bytes) == 0, "a partly committed range was collected as a whole");
    Runs runs;
    const Runs first{{base, base + Block}};
    const auto synced = CollectCommitted(range, runs);
    Require(synced != 0 && runs == first, "the committed part of a partly committed range was not collected");
    Require(UnchangedCommittedSince(runs, synced), "a collected committed part reads as changed");

    range.Data()[8] = 0x22;
    const auto written = CollectCommitted(range, runs);
    Require(written > synced && runs == first && !UnchangedCommittedSince(runs, synced) && UnchangedCommittedSince(runs, written), "a CPU write to the committed part was missed");
    Require(MarkWritten(base + 16, 4) != 0 && !UnchangedCommittedSince(first, written), "a driver store into the committed part was missed");
    Require(UnchangedCommittedSince(first, CollectCommitted(range, runs)), "a collect after a driver store reads the committed part as changed");

    const auto beforeCommit = CollectCommitted(range, runs);
    range.Commit(2);
    const Runs both{{base, base + Block}, {range.Address(2), range.Address(3)}};
    const auto grown = CollectCommitted(range, runs);
    Require(grown != 0 && runs == both, "a block committed inside the range is not described");
    Require(!UnchangedCommittedSince(runs, beforeCommit), "a block committed inside the range reads as unchanged since before the commit");
    Require(UnchangedCommittedSince(runs, grown), "the committed parts read as changed after their collect");

    range.Decommit(2);
    range.Commit(2);
    const auto recommitted = CollectCommitted(range, runs);
    Require(recommitted != 0 && runs == both && !UnchangedCommittedSince(runs, grown), "a block decommitted and committed again reads as unchanged");

    range.Decommit(2);
    Require(CollectCommitted(range, runs) != 0 && runs == first, "a decommitted block is still described as committed");

#ifdef _WIN32
    HANDLE firstSection = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_RESERVE, 0, Block, nullptr);
    HANDLE secondSection = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_RESERVE, 0, Block, nullptr);
    struct Sections {
        HANDLE first;
        HANDLE second;
        ~Sections() {
            if (first != nullptr) CloseHandle(first);
            if (second != nullptr) CloseHandle(second);
        }
    } sections{firstSection, secondSection};
    Require(firstSection != nullptr && secondSection != nullptr, "cannot create the shared backings of the partly committed range");
    const auto beforeMap = CollectCommitted(range, runs);
    GuestArena::GuestArenaMap_nid_postfix(range.Data() + 3 * Block, Block, firstSection, 0, PAGE_READWRITE);
    GuestAllocations::GuestAllocationsInvalidate_nid_postfix(range.Address(3), Block);
    const Runs viewed{{base, base + Block}, {range.Address(3), range.Address(4)}};
    const auto mapped = CollectCommitted(range, runs);
    Require(mapped != 0 && runs == viewed, "a view mapped inside the range is not described");
    Require(!UnchangedCommittedSince(runs, beforeMap), "a view mapped inside the range reads as unchanged since before the mapping");
    Require(UnchangedCommittedSince(runs, mapped), "a collected view reads as changed");
    GuestArena::GuestArenaReset_nid_postfix(range.Data() + 3 * Block, Block);
    GuestArena::GuestArenaMap_nid_postfix(range.Data() + 3 * Block, Block, secondSection, 0, PAGE_READWRITE);
    GuestAllocations::GuestAllocationsInvalidate_nid_postfix(range.Address(3), Block);
    Require(CollectCommitted(range, runs) != 0 && runs == viewed && !UnchangedCommittedSince(runs, mapped), "a view replaced inside the range reads as unchanged");
    GuestArena::GuestArenaReset_nid_postfix(range.Data() + 3 * Block, Block);
    GuestAllocations::GuestAllocationsInvalidate_nid_postfix(range.Address(3), Block);
    Require(CollectCommitted(range, runs) != 0 && runs == first, "an unmapped view is still described as committed");
#endif

    range.Decommit(0);
    Require(CollectCommitted(range, runs) != 0 && runs.empty() && UnchangedCommittedSince(runs, TrackerGeneration()), "a range with no committed page did not collect as empty");
    Require(!UnchangedCommittedSince(first, 0), "an unknown generation was accepted for the committed parts");
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_REPLAY_INPUTS): a draw recipe's highest-index memo answers only
// while its range is unchanged since the collect made before its scan, under the same guest
// allocations: a driver store over the range, a CPU store seen by a new epoch's collect and a
// registry change each make it miss, a store beside the range's block does not.
void CheckHighestIndexMemo() {
    constexpr std::size_t bytes = 2 * Block;
    auto* memory = static_cast<std::uint8_t*>(AllocateWatched(bytes));
    struct Cleanup {
        void* memory;
        ~Cleanup() {
#ifdef _WIN32
            GuestArena::GuestArenaReset_nid_postfix(memory, bytes);
            GuestArena::GuestArenaRelease_nid_postfix(memory, bytes);
#else
            GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(memory, bytes);
            munmap(memory, bytes);
#endif
        }
    } cleanup{memory};
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    constexpr std::size_t count = 600;
    constexpr std::size_t rangeBytes = count * sizeof(std::uint16_t);
    const auto address = base + 256;
    for (std::size_t index = 0; index < count; ++index) {
        const auto value = static_cast<std::uint16_t>(index % 100);
        std::memcpy(memory + 256 + index * sizeof(value), &value, sizeof(value));
    }
    AgcDriver::Graphics::HighestIndexMemo memo;
    // As a replay asks (Draw.cpp's replayHighestIndex): the registry generation, then the collect.
    std::optional<std::uint32_t> found;
    const auto find = [&](bool skipRestart = false) {
        const auto registry = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        const auto generation = CollectWrites(address, rangeBytes);
        return memo.Find(address, rangeBytes, 2, skipRestart, generation, registry, found);
    };
    const auto keep = [&](std::uint32_t highest) {
        const auto registry = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        const auto generation = CollectWrites(address, rangeBytes);
        Require(generation != 0, "the index range is not collected");
        memo.Keep(address, rangeBytes, 2, false, generation, registry, highest);
    };
    BumpCollectEpoch();
    Require(!find(), "an empty memo answered");
    keep(99);
    Require(find() && found == 99u, "an unchanged index range missed the memo");
    Require(!find(true), "a range scanned without the restart rule answered for it");

    MarkWritten(address + 64, 4);
    Require(!find(), "a driver store over the index range kept the memo");
    keep(99);
    Require(find() && found == 99u, "a range kept after a driver store missed the memo");

    const std::uint16_t raised = 700;
    std::memcpy(memory + 256 + 7 * sizeof(raised), &raised, sizeof(raised));
    BumpCollectEpoch();
    Require(!find(), "a CPU store seen by a new epoch's collect kept the memo");
    keep(700);
    Require(find() && found == 700u, "a range kept after a CPU store missed the memo");

    MarkWritten(base + Block + 64, 4);
    Require(find() && found == 700u, "a driver store beside the index range's block dropped the memo");

    GuestAllocations::GuestAllocationsInvalidate_nid_postfix(static_cast<std::uintptr_t>(address), rangeBytes);
    Require(!find(), "a guest allocation change kept the memo");
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_LAZY_GENERATION): with the switch a walk that finds nothing
// written leaves the tracker generation where it was, without it every walk moves it; either way a
// walk that stamps returns the next generation, and only the blocks it stamped read as changed
// since the generation before it.
void CheckLazyGeneration() {
    const bool lazy = std::getenv("APS5_EXPERIMENT_LAZY_GENERATION") != nullptr;
    constexpr std::size_t bytes = 2 * Block;
    auto* memory = static_cast<std::uint8_t*>(AllocateWatched(bytes));
    struct Cleanup {
        void* memory;
        ~Cleanup() {
#ifdef _WIN32
            GuestArena::GuestArenaReset_nid_postfix(memory, bytes);
            GuestArena::GuestArenaRelease_nid_postfix(memory, bytes);
#else
            GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(memory, bytes);
            munmap(memory, bytes);
#endif
        }
    } cleanup{memory};
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, bytes);
    const auto synced = CollectWritesUncached(base, bytes);
    Require(synced != 0 && UnchangedSince(base, bytes, synced), "the test block is not collected");

    const auto clean = CollectWritesUncached(base, bytes);
    Require(lazy ? clean == synced : clean == synced + 1, lazy ? "a walk that found nothing written moved the tracker generation" : "a walk that found nothing written did not move the tracker generation");
    Require(TrackerGeneration() == clean && UnchangedSince(base, bytes, clean), "a clean walk's generation is not current");

    static_cast<volatile std::uint8_t*>(memory)[8] = 0x22;
    const auto stamped = CollectWritesUncached(base, bytes);
    Require(stamped == clean + 1, "a walk that stamped did not return the next generation");
    Require(!UnchangedSince(base, Block, clean) && UnchangedSince(base, bytes, stamped), "a stamping walk's block does not read as changed since the generation before it");
    Require(UnchangedSince(base + Block, Block, clean), "a stamping walk stamped a block it did not find written");
}

void CheckAdoptLatestEpoch() {
    std::uint64_t bumped = 0;
    std::thread([&] {
        BumpCollectEpoch();
        bumped = CollectEpoch();
    }).join();
    Require(bumped != 0, "a bumped thread has no collect epoch");
    std::uint64_t before = 1;
    std::uint64_t adopted = 0;
    bool keptOwn = false;
    std::thread([&] {
        before = CollectEpoch();
        AdoptLatestCollectEpoch();
        adopted = CollectEpoch();
        BumpCollectEpoch();
        const auto own = CollectEpoch();
        AdoptLatestCollectEpoch();
        keptOwn = CollectEpoch() == own;
    }).join();
    Require(before == 0, "a thread that never bumped had a collect epoch");
    Require(adopted == bumped, "a thread that never bumped did not take the newest bumped collect epoch");
    Require(keptOwn, "adopting the newest collect epoch moved a thread off its own");
}

// LOCAL DEBUG (GPU-OWNED-PLAN 9.4.1 D4-0): the collect memo key a target lease would hold is zero
// without the epoch coverage (APS5_DEBUG_COVERAGE=1) or on a thread that never bumped its epoch, and
// otherwise moves with exactly one of its parts at a time: the epoch at a bump, UnwatchSerial at an
// unwatch, the allocation generation at a registry change.
void CheckCollectMemoKey() {
    const bool coverage = std::getenv("APS5_DEBUG_COVERAGE") != nullptr && std::getenv("APS5_NO_COLLECT_MEMO") == nullptr && std::getenv("APS5_SHARED_COLLECT_MEMO") == nullptr;
    CollectMemoKey unbumped{1, 1, 1};
    std::thread([&] { unbumped = CurrentCollectMemoKey(); }).join();
    Require(!unbumped.Valid() && unbumped == CollectMemoKey{}, "a thread that never bumped its collect epoch has a collect memo key");
    BumpCollectEpoch();
    const auto first = CurrentCollectMemoKey();
    if (!coverage) {
        Require(!first.Valid() && first == CollectMemoKey{}, "a collect memo key without the epoch coverage (APS5_DEBUG_COVERAGE)");
        return;
    }
    Require(first.Valid() && first.epoch == ThreadCollectEpoch() && first.unwatched == UnwatchSerial() && first.mapping == GuestAllocations::GuestAllocationsGeneration_nid_postfix(), "the collect memo key is not the thread's epoch, UnwatchSerial and allocation generation");
    Require(CurrentCollectMemoKey() == first, "the collect memo key moved without a bump, an unwatch or an allocation change");
    void* memory = AllocateWatched(2 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, 2 * Block);
    BumpCollectEpoch();
    const auto bumped = CurrentCollectMemoKey();
    Require(bumped.epoch != first.epoch && bumped.epoch == ThreadCollectEpoch() && bumped.unwatched == UnwatchSerial() && bumped.mapping == GuestAllocations::GuestAllocationsGeneration_nid_postfix(), "an epoch bump did not move the collect memo key's epoch");
    Require(CollectWrites(base, 2 * Block) != 0 && CurrentCollectMemoKey() == bumped, "a collect moved the collect memo key");
    Unwatch(base + Block, Block);
    const auto unwatched = CurrentCollectMemoKey();
    Require(unwatched.epoch == bumped.epoch && unwatched.unwatched != bumped.unwatched && unwatched.mapping == bumped.mapping, "an unwatch did not move the collect memo key's UnwatchSerial alone");
    GuestAllocations::GuestAllocationsInvalidate_nid_postfix(static_cast<std::uintptr_t>(base), 4096);
    const auto remapped = CurrentCollectMemoKey();
    Require(remapped.epoch == unwatched.epoch && remapped.unwatched == unwatched.unwatched && remapped.mapping != unwatched.mapping, "an allocation change did not move the collect memo key's allocation generation alone");
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_HOST_WRITE_STAMPS, PARALLEL-COMMIT-PLAN H): the driver's hooks
// installed by hand (the switch only chooses which one device init installs), and what they see.
void NoteHostWriteHook(std::uintptr_t address, std::size_t bytes) {
    NoteHostWrite(address, bytes);
}

void CountHostWriteHook(std::uintptr_t address, std::size_t bytes) {
    CountHostWrite(address, bytes);
}

struct HostWriteHookScope {
    explicit HostWriteHookScope(void (*hook)(std::uintptr_t, std::size_t)) { GuestArena::GuestArenaSetHostWriteHook_nid_no_patch(hook); }
    ~HostWriteHookScope() { GuestArena::GuestArenaSetHostWriteHook_nid_no_patch(nullptr); }
    HostWriteHookScope(const HostWriteHookScope&) = delete;
    HostWriteHookScope& operator=(const HostWriteHookScope&) = delete;
};

// One HLE-style host write: the bytes are stored between the arena's begin and end of the write,
// and the hook runs at its end.
void HostWriteBytes(std::uint8_t* destination, std::size_t bytes, std::uint8_t value) {
    const GuestArena::HostWrite write(destination, bytes);
    Require(write.Open(), "the host write did not open");
    std::memset(destination, value, bytes);
}

struct WatchedRelease {
    void* memory;
    std::size_t bytes;
    ~WatchedRelease() {
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(memory, bytes);
        GuestArena::GuestArenaRelease_nid_postfix(memory, bytes);
#else
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(memory, bytes);
        munmap(memory, bytes);
#endif
    }
};

// Without the hook a write through GuestArena::HostWrite reads as unchanged until the next walk of
// its pages (the window a memo or a receipt keeps open); with NoteHostWrite UnchangedSince fails for
// every check after the write ended, page exact and as a CPU store (not the driver's own), and the
// next walk does not report the pages again (the stamp consumed their write watch). Writes queued
// together share one generation, and a lock-free memo hit stamps the queue before it answers. The
// count-only hook stamps nothing.
void CheckHostWriteStamps() {
    constexpr std::size_t bytes = 2 * Block;
    auto* memory = static_cast<std::uint8_t*>(AllocateWatched(bytes));
    const WatchedRelease release{memory, bytes};
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, bytes);
    const auto synced = CollectWritesUncached(base, bytes);
    Require(synced != 0 && UnchangedSince(base, bytes, synced), "the host-write block is not collected");

    HostWriteBytes(memory + 64, 32, 0x22);
    Require(UnchangedSince(base, Block, synced), "a host write without the hook was stamped before a walk of its pages");
    CollectWritesUncached(base, bytes);
    Require(!UnchangedSince(base, Block, synced), "a walk after a host write did not see it");

    {
        const HostWriteHookScope hook(&NoteHostWriteHook);
        TakeHostWriteStats();
        const auto before = TrackerGeneration();
        HostWriteBytes(memory + Block + 128, 32, 0x33);
        Require(TrackerGeneration() > before, "a host write did not move the tracker generation");
        Require(!UnchangedSince(base + Block + 128, 32, before), "a write through GuestArena::HostWrite was not stamped when it ended");
        Require(UnchangedSince(base, Block, before), "a host write stamped a block it did not write");
        Require(UnchangedSince(base + Block + 8 * 4096, 4096, before), "a host write's stamp is not page exact");
        const std::array<std::uint64_t, 1> generations{before};
        std::array<std::uint8_t, 1> changed{};
        std::array<std::uint8_t, 1> cpu{};
        Require(ChangedBlocks(base + Block, Block, generations, changed, cpu) && changed[0] == BlockWritten && cpu[0] == 1, "a host write is not stamped as a CPU store");
        const auto stats = TakeHostWriteStats();
        Require(stats.writes == 1 && stats.bytes == 32 && stats.outside == 0 && stats.drains == 1 && stats.unconsumed == 0, "the stamped host write was not counted");

        const auto stamped = TrackerGeneration();
        CollectWritesUncached(base, bytes);
        Require(UnchangedSince(base + Block + 128, 32, stamped), "the walk after a stamped host write reported its page again (the stamp did not consume its write watch)");

        // Two writes queued before the next tracker lock: one new generation stamps both.
        TakeHostWriteStats();
        const auto burst = TrackerGeneration();
        HostWriteBytes(memory + 4 * 4096, 32, 0x55);
        HostWriteBytes(memory + Block + 12 * 4096, 32, 0x56);
        Require(TrackerGeneration() == burst + 1, "two host writes queued together did not share one new generation");
        Require(!UnchangedSince(base + 4 * 4096, 32, burst) && !UnchangedSince(base + Block + 12 * 4096, 32, burst), "a host write of a burst was not stamped");
        const auto burstStats = TakeHostWriteStats();
        Require(burstStats.writes == 2 && burstStats.drains == 1, "a burst of host writes was not stamped in one drain");

        // A memo hit (lock-free) after a host write answers with a generation its stamp does not exceed.
        BumpCollectEpoch();
        const auto walked = CollectWrites(base, bytes);
        HostWriteBytes(memory + 6 * 4096, 32, 0x57);
        const auto hit = CollectWrites(base, Block);
        Require(walked != 0 && hit > walked && !UnchangedSince(base + 6 * 4096, 32, walked) && UnchangedSince(base + 6 * 4096, 32, hit), "a memo hit after a host write answered before its stamp");
    }

    {
        const HostWriteHookScope hook(&CountHostWriteHook);
        TakeHostWriteStats();
        const auto counted = TrackerGeneration();
        HostWriteBytes(memory + 256, 16, 0x44);
        Require(TrackerGeneration() == counted && UnchangedSince(base, Block, counted), "the count-only hook stamped a host write");
        const auto stats = TakeHostWriteStats();
        Require(stats.writes == 1 && stats.bytes == 16 && stats.covered == 0 && stats.coveredOlder == 0 && stats.ringBusy == 0, "the count-only hook did not count the write, or found a receipt where none was published");
    }
}

#ifdef _WIN32
// A host write over a block whose first and last pages are unwatched (they kept an import's
// exclusion) and whose middle is watched (fresh backing): the stamp is page by page, so the watched
// middle fails UnchangedSince (the block part's first and last byte alone would skip the block).
void CheckHostWriteWatchedMiddle() {
    constexpr std::size_t granule = 16384;
    void* memory = GuestArena::GuestArenaAllocate_nid_postfix(Block, Block);
    GuestArena::GuestArenaCommit_nid_postfix(memory, Block, PAGE_READWRITE, granule);
    const WatchedRelease release{memory, Block};
    auto* bytes = static_cast<std::uint8_t*>(memory);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, Block);
    GuestArena::GuestArenaReset_nid_postfix(bytes + granule, 2 * granule);
    Unwatch(base, Block);
    GuestArena::GuestArenaCommit_nid_postfix(bytes + granule, 2 * granule, PAGE_READWRITE, granule);
    Require(Watched(base + granule, 2 * granule) && !Watched(base, granule) && !Watched(base + 3 * granule, granule), "only the middle of the host-write block is watched");
    std::memset(bytes + granule, 0x11, 2 * granule);
    const auto synced = CollectWritesUncached(base + granule, 2 * granule);
    Require(synced != 0 && UnchangedSince(base + granule, 2 * granule, synced), "the watched middle of the host-write block was not collected");
    const HostWriteHookScope hook(&NoteHostWriteHook);
    HostWriteBytes(bytes + granule / 2, 3 * granule, 0x22);
    Require(TrackerGeneration() > synced && !UnchangedSince(base + granule, 2 * granule, synced), "a host write over a block's watched middle between unwatched ends was not stamped");
}
#endif

// LOCAL EXPERIMENT (APS5_EXPERIMENT_HOST_WRITE_STAMPS=2; ctest agc_write_tracking_host_write_count, whose
// environment turns queue 0's walk receipts on): the count-only hook against the receipt ring, this
// thread publishing as queue 0 does. A host write inside a range walked for the newest token counts
// as covered and one outside every receipt not at all; once a newer token holds the newest slot
// elsewhere, the first range counts as an older token's; after an Unwatch moves the serial no slot
// matches.
bool CheckHostWriteCount() {
    if (Q0Receipts() == 0) {
        std::cout << "host-write count: needs APS5_EXPERIMENT_Q0_RECEIPTS=1 and its prerequisites, skipped\n";
        return false;
    }
    constexpr std::size_t bytes = 3 * Block;
    auto* memory = static_cast<std::uint8_t*>(AllocateWatched(bytes));
    const WatchedRelease release{memory, bytes};
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, bytes);
    const HostWriteHookScope hook(&CountHostWriteHook);
    const auto walk = [](std::uint64_t token, std::uint64_t first, std::uint64_t stop) {
        BumpCollectEpoch();
        SetReceiptOwner(token);
        ReceiptWalks walks;
        const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> planned{{{first, stop}}};
        PlanReceiptWalks(planned, walks);
        Require(WalkReceipts(token, walks, std::numeric_limits<std::uint64_t>::max(), [](const void*) { return false; }, nullptr, false) && walks.Done(), "the receipt walk did not finish");
    };
    constexpr std::uint64_t token = 1000;
    walk(token, base, base + Block);
    TakeHostWriteStats();
    HostWriteBytes(memory + 128, 32, 0x22);
    auto stats = TakeHostWriteStats();
    Require(stats.writes == 1 && stats.covered == 1 && stats.coveredOlder == 0 && stats.ringBusy == 0, "a host write inside the newest token's receipt was not counted as covered");
    HostWriteBytes(memory + Block + 128, 32, 0x23);
    stats = TakeHostWriteStats();
    Require(stats.writes == 1 && stats.covered == 0 && stats.coveredOlder == 0, "a host write outside every receipt was counted");
    walk(token + 1, base + Block, base + 2 * Block);
    HostWriteBytes(memory + 256, 16, 0x24);
    stats = TakeHostWriteStats();
    Require(stats.covered == 0 && stats.coveredOlder == 1, "a host write inside an older token's receipt was not counted as older");
    HostWriteBytes(memory + Block + 256, 16, 0x25);
    stats = TakeHostWriteStats();
    Require(stats.covered == 1 && stats.coveredOlder == 0, "a host write inside the newer token's receipt was not counted as covered");
    Unwatch(base + 2 * Block, Block);
    HostWriteBytes(memory + 512, 16, 0x26);
    HostWriteBytes(memory + Block + 512, 16, 0x27);
    stats = TakeHostWriteStats();
    Require(stats.writes == 2 && stats.covered == 0 && stats.coveredOlder == 0, "a host write was counted under a receipt of an older UnwatchSerial");
    std::cout << "host-write count over walk receipts passed\n";
    return true;
}

// A host write outside the arena (host memory) stamps nothing and leaves the generation alone.
void CheckHostWriteOutsideArena() {
    std::vector<std::uint8_t> host(2 * 4096, 0);
    const auto address = reinterpret_cast<std::uint64_t>(host.data());
    if (GuestArena::GuestArenaContains_nid_postfix(host.data(), host.size()) || Watched(address, host.size())) {
        std::cout << "host-write stamps outside the arena: the host buffer is inside it, skipped\n";
        return;
    }
    const HostWriteHookScope hook(&NoteHostWriteHook);
    TakeHostWriteStats();
    const auto before = TrackerGeneration();
    const auto newest = NewestStamp(address, host.size());
    HostWriteBytes(host.data(), host.size(), 0x55);
    Require(TrackerGeneration() == before && NewestStamp(address, host.size()) == newest, "a host write outside the arena was stamped");
    const auto stats = TakeHostWriteStats();
    Require(stats.writes == 1 && stats.outside == 1, "a host write outside the arena was not counted as outside it");
}

// No lock inversion: the hook runs after the arena released the mappings' mutex (WindowsMappings;
// the write watch's on macOS) and NoteHostWrite takes no GpuMutex. From inside the hook this thread
// takes the mappings' mutex (Windows) and walks the written range (the tracker lock, then the
// mappings' mutex: the order every collect uses), while another thread holds GpuMutex for the whole
// write; on Windows also over a shared view, whose end walks the views under that mutex. Either
// inversion hangs, and the watchdog fails the test instead.
std::atomic<std::uint32_t> lockOrderHookCalls{0};

void LockOrderHook(std::uintptr_t address, std::size_t bytes) {
#ifdef _WIN32
    std::uint32_t protection = 0;
    static_cast<void>(GuestArena::GuestArenaProtection_nid_postfix(address, &protection));
#endif
    NoteHostWrite(address, bytes);
    CollectWritesUncached(address, bytes);
    lockOrderHookCalls.fetch_add(1, std::memory_order_relaxed);
}

void CheckHostWriteLockOrder() {
    constexpr std::size_t bytes = Block;
    auto* memory = static_cast<std::uint8_t*>(AllocateWatched(bytes));
    const WatchedRelease release{memory, bytes};
    std::memset(memory, 0x11, bytes);
    std::atomic<bool> finished{false};
    std::atomic<bool> held{false};
    std::atomic<bool> unlock{false};
    std::thread watchdog([&] {
        for (int tick = 0; tick < 3000 && !finished.load(); ++tick) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if (finished.load()) return;
        std::cerr << "write tracking test failed: a host write's stamp hook hung (a lock inversion with the mappings' mutex or GpuMutex)\n";
        std::_Exit(1);
    });
    std::thread holder([&] {
        const std::lock_guard gpu(GpuMutex());
        held.store(true);
        while (!unlock.load()) std::this_thread::yield();
    });
    struct Join {
        std::atomic<bool>& unlock;
        std::atomic<bool>& finished;
        std::thread& holder;
        std::thread& watchdog;
        ~Join() {
            unlock.store(true);
            holder.join();
            finished.store(true);
            watchdog.join();
        }
    } join{unlock, finished, holder, watchdog};
    while (!held.load()) std::this_thread::yield();
    lockOrderHookCalls.store(0);
    std::uint32_t expected = 0;
    {
        const HostWriteHookScope hook(&LockOrderHook);
        HostWriteBytes(memory + 512, 64, 0x66);
        ++expected;
#ifdef _WIN32
        HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_RESERVE, 0, Block, nullptr);
        Require(section != nullptr, "cannot create the host write's shared backing");
        auto* shared = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(Block, Block));
        Require(shared != nullptr, "cannot allocate the host write's shared address");
        struct SharedRelease {
            std::uint8_t* shared;
            HANDLE section;
            bool mapped = false;
            ~SharedRelease() {
                if (mapped) GuestArena::GuestArenaReset_nid_postfix(shared, Block);
                GuestArena::GuestArenaRelease_nid_postfix(shared, Block);
                CloseHandle(section);
            }
        } sharedRelease{shared, section};
        GuestArena::GuestArenaMap_nid_postfix(shared, Block, section, 0, PAGE_READWRITE);
        sharedRelease.mapped = true;
        HostWriteBytes(shared + 4096 + 32, 64, 0x77);
        ++expected;
#endif
    }
    Require(lockOrderHookCalls.load() == expected, "the host-write hook was not called at every host write's end");
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, GPU-OWNED-PLAN 9.4.3 D4-B): StampsHold is
// UnchangedSinceAll over its unchanged rows and !StoredOver over its not-stored-over rows, under one
// lock. Compared on random rows (shared edge blocks included) over driver stores at sub-block
// precision and walked CPU stores, and on unwatched, generation-0 and empty rows (the host-region row
// is CheckHostHole's); `current` is the tracker generation; without the drain a queued host write
// stays unstamped, which the draining form then sees. LOCAL DEBUG (APS5_PROFILE_DRAW, GPU-OWNED-PLAN
// 9.5 D5-0): with a report, the answer is the same and each row's is UnchangedSince's or
// StoredOver's (unproven where the row has no proof: unwatched, generation 0 or empty); without the
// drain the queued host write is reported pending (with its row held), and with it not (its row
// stamped).
void CheckStampsHold() {
    constexpr std::size_t bytes = 4 * Block;
    auto* memory = static_cast<std::uint8_t*>(AllocateWatched(bytes));
    const WatchedRelease release{memory, bytes};
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, bytes);
    std::vector<std::uint64_t> generations{CollectWritesUncached(base, bytes)};
    Require(generations[0] != 0, "the StampsHold block is not collected");
    std::uint64_t state = 0x9E3779B97F4A7C15ull;
    const auto next = [&](std::uint64_t bound) {
        state ^= state << 13u;
        state ^= state >> 7u;
        state ^= state << 17u;
        return state % bound;
    };
    for (int store = 0; store < 32; ++store) {
        const auto at = next(bytes - 1);
        if (store % 5 == 4) {
            memory[at] ^= 0x5a;
            generations.push_back(CollectWritesUncached(base, bytes));
            continue;
        }
        const auto length = 1 + next(std::min<std::uint64_t>(bytes - at, 3 * 4096));
        generations.push_back(MarkWritten(base + at, static_cast<std::size_t>(length)));
    }
    generations.push_back(TrackerGeneration());
    const auto row = [&] {
        const auto at = next(bytes - 1);
        const auto length = 1 + next(std::min<std::uint64_t>(bytes - at, 2 * Block));
        return UnchangedQuery{base + at, static_cast<std::size_t>(length), generations[next(generations.size())]};
    };
    for (int round = 0; round < 512; ++round) {
        const std::array rows{row(), row(), row()};
        const std::span<const UnchangedQuery> unchanged(rows.data(), 1 + next(3));
        const std::span<const UnchangedQuery> stored(rows.data() + 2, 1);
        const bool unchangedAll = UnchangedSinceAll(unchanged);
        const bool notStoredOver = !StoredOver(stored[0].address, stored[0].bytes, stored[0].generation);
        Require(StampsHold(unchanged, {}) == unchangedAll, "StampsHold's unchanged rows disagree with UnchangedSinceAll");
        Require(StampsHold({}, stored) == notStoredOver, "StampsHold's not-stored-over row disagrees with StoredOver");
        Require(StampsHold(unchanged, stored) == (unchangedAll && notStoredOver), "StampsHold's rows together disagree with UnchangedSinceAll and StoredOver");
        std::array<StampRow, 4> answers{};
        StampsHoldReport report{answers};
        Require(StampsHold(unchanged, stored, round % 2 == 0, nullptr, &report) == (unchangedAll && notStoredOver) && !report.hostWritesPending, "StampsHold with a report answers otherwise than without, or reports host writes pending");
        for (std::size_t i = 0; i < unchanged.size(); ++i) {
            const auto& query = unchanged[i];
            const auto expected = query.generation == 0 || query.bytes == 0 || !Watched(query.address, query.bytes) ? StampRow::Unproven : UnchangedSince(query.address, query.bytes, query.generation) ? StampRow::Held : StampRow::Stamped;
            Require(answers[i] == expected, "StampsHold's report of an unchanged row disagrees with UnchangedSince");
        }
        Require(answers[unchanged.size()] == (notStoredOver ? StampRow::Held : StampRow::Stamped), "StampsHold's report of a not-stored-over row disagrees with StoredOver");
    }
    std::array<std::uint8_t, 64> unwatched{};
    const auto outside = reinterpret_cast<std::uint64_t>(unwatched.data());
    if (!Watched(outside, unwatched.size())) {
        const std::array rows{UnchangedQuery{outside, unwatched.size(), TrackerGeneration()}};
        Require(!StampsHold(rows, {}) && !UnchangedSinceAll(rows), "an unwatched row holds in StampsHold");
        Require(!StampsHold({}, rows) && StoredOver(outside, unwatched.size(), rows[0].generation), "an unwatched row reads as not stored over in StampsHold");
        std::array<StampRow, 2> answers{};
        StampsHoldReport report{answers};
        Require(!StampsHold(rows, rows, true, nullptr, &report) && answers[0] == StampRow::Unproven && answers[1] == StampRow::Unproven, "an unwatched row is not reported unproven");
    }
    const std::array zero{UnchangedQuery{base, 64, 0}};
    Require(!StampsHold(zero, {}) && !StampsHold({}, zero) && StoredOver(base, 64, 0), "a generation-0 row holds in StampsHold");
    const std::array empty{UnchangedQuery{base, 0, TrackerGeneration()}};
    Require(!StampsHold(empty, {}) && !StampsHold({}, empty) && StoredOver(base, 0, empty[0].generation), "an empty row holds in StampsHold");
    {
        std::array<StampRow, 4> answers{};
        StampsHoldReport report{answers};
        const std::array held{UnchangedQuery{base, 64, TrackerGeneration()}};
        Require(!StampsHold(zero, held, true, nullptr, &report) && answers[0] == StampRow::Unproven && answers[1] == StampRow::Held, "a generation-0 row is not reported unproven, or a held row after it is not tested");
        Require(!StampsHold(empty, empty, false, nullptr, &report) && answers[0] == StampRow::Unproven && answers[1] == StampRow::Unproven, "an empty row is not reported unproven");
    }
    std::uint64_t current = 0;
    Require(StampsHold({}, {}, true, &current) && current == TrackerGeneration(), "StampsHold without rows did not hold, or read another generation than the tracker's");
    {
        const HostWriteHookScope hook(&NoteHostWriteHook);
        const auto before = TrackerGeneration();
        const std::array rows{UnchangedQuery{base + 2 * Block, 4096, before}};
        HostWriteBytes(memory + 2 * Block + 100, 16, 0x66);
        std::uint64_t seen = 0;
        Require(StampsHold(rows, {}, false, &seen) && seen == before, "a StampsHold without the drain stamped a queued host write");
        std::array<StampRow, 1> answer{};
        StampsHoldReport report{answer};
        Require(StampsHold(rows, {}, false, nullptr, &report) && report.hostWritesPending && answer[0] == StampRow::Held, "a StampsHold without the drain did not report the queued host write pending (its row held)");
        Require(!StampsHold(rows, {}), "the draining StampsHold missed a queued host write");
        Require(!StampsHold(rows, {}, false, nullptr, &report) && !report.hostWritesPending && answer[0] == StampRow::Stamped, "the stamped host write was still reported pending, or its row not stamped");
    }
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, GPU-OWNED-PLAN 9.4.3 D4-B): CollectCovered holds
// exactly when the next collect of the range walks nothing (ThreadCollectedBytes, counted under
// APS5_PROFILE_DRAW), coverage only grows under one collect memo key, and a bump, an unwatch and an
// allocation change each drop it; always false without the epoch coverage (APS5_DEBUG_COVERAGE=1) or
// on a thread that never bumped its epoch.
void CheckCollectCovered() {
    const bool coverage = std::getenv("APS5_DEBUG_COVERAGE") != nullptr && std::getenv("APS5_NO_COLLECT_MEMO") == nullptr && std::getenv("APS5_SHARED_COLLECT_MEMO") == nullptr;
    const bool counted = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    constexpr std::size_t bytes = 3 * Block;
    auto* memory = static_cast<std::uint8_t*>(AllocateWatched(bytes));
    const WatchedRelease release{memory, bytes};
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, bytes);
    bool unbumped = true;
    std::thread([&] { unbumped = CollectCovered(base, Block); }).join();
    Require(!unbumped, "a thread that never bumped its collect epoch has collect coverage");
    BumpCollectEpoch();
    if (!coverage) {
        Require(CollectWrites(base, Block) != 0 && !CollectCovered(base, Block), "collect coverage without the epoch coverage (APS5_DEBUG_COVERAGE)");
        return;
    }
    // One collect: it walks exactly when the range was not covered, and covers it.
    const auto collect = [&](std::uint64_t address, std::size_t size, bool covered, const char* what) {
        Require(CollectCovered(address, size) == covered, what);
        const auto walked = ThreadCollectedBytes();
        Require(CollectWrites(address, size) != 0, what);
        if (counted) Require((ThreadCollectedBytes() != walked) == !covered, what);
        Require(CollectCovered(address, size), what);
    };
    collect(base, 2 * Block, false, "the first collect of an epoch found its range covered, did not walk it, or left it uncovered");
    collect(base, 2 * Block, true, "a second collect of a range in its epoch found it uncovered or walked it");
    Require(CollectCovered(base + 100, 50) && CollectCovered(base + Block - 10, 20), "a part of a covered range is not covered");
    collect(base + 2 * Block, Block, false, "another range of the epoch was covered before its collect");
    Require(CollectCovered(base, 2 * Block) && CollectCovered(base, bytes), "coverage shrank under one collect memo key");
    BumpCollectEpoch();
    collect(base, 2 * Block, false, "coverage outlived an epoch bump");
    Unwatch(base + Block, Block);
    collect(base, Block, false, "coverage outlived an unwatch");
    GuestAllocations::GuestAllocationsInvalidate_nid_postfix(static_cast<std::uintptr_t>(base), 4096);
    collect(base, Block, false, "coverage outlived an allocation change");
    Require(!CollectCovered(base, 0), "an empty range reads as covered");
}

int main(int argc, char** argv) {
    try {
#ifdef _WIN32
        if (std::wstring_view(GetCommandLineW()).ends_with(L"--metadata-commit")) {
            return CheckMetadataCommit() ? 0 : 77;
        }
#endif
        if (!WriteWatched()) {
            std::cout << "no write watching: skipped\n";
            return 77;
        }
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_HOST_WRITE_STAMPS=2): in its own receipts environment.
        if (argc > 1 && std::string_view(argv[1]) == "--host-write-count") return CheckHostWriteCount() ? 0 : 77;
        // LOCAL DEBUG (GPU-OWNED-PLAN 9.4.1 D4-0): the collect memo key alone, in the epoch coverage's
        // environment (the full run checks it stays zero without).
        if (argc > 1 && std::string_view(argv[1]) == "--collect-memo-key") {
            CheckCollectMemoKey();
            std::cout << "collect memo key tests passed\n";
            return 0;
        }
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_TARGET_LEASES, GPU-OWNED-PLAN 9.4.3 D4-B): StampsHold and the
        // collect coverage alone, in the epoch coverage's environment with the walks counted (the full
        // run checks the coverage stays false without).
        if (argc > 1 && std::string_view(argv[1]) == "--stamps-hold") {
            CheckStampsHold();
            CheckCollectCovered();
            std::cout << "StampsHold and collect coverage tests passed\n";
            return 0;
        }
#ifdef _WIN32
        if (std::wstring_view(GetCommandLineW()).ends_with(HostHoleMode)) {
            CheckHostHole();
            return 0;
        }
        if (std::wstring_view(GetCommandLineW()).ends_with(ResidentRemapMode)) {
            CheckResidentDirectMappingReuse();
            return 0;
        }
#endif
        CheckSharedBlock();
        CheckStoredAllSince();
        CheckSharedBlockProofs();
        CheckOwnStore();
        CheckUnwatch();
        CheckNewestStamp();
        CheckPartlyWatchedStore();
#ifndef _WIN32
        CheckWatchOverlaps();
        CheckWatchedMiddleStore();
#endif
        CheckCoarseRuns();
#ifdef _WIN32
        CheckSparseLeafBoundaries();
        CheckDeferredSharedImports();
        CheckPrivateMappingReuse();
        CheckRewatch();
        CheckUntrackedHint();
        CheckHostHoleInChild();
        CheckFailedCollectUnwatch();
#endif
        CheckVersionRanges();
        CheckCollectBoundaries();
#ifdef _WIN32
        CheckWidenedCollectBesideUncommittedPages();
#endif
        CheckImportWindow();
        CheckImportWindowZeroPages();
        CheckImportWindowOtherThread();
        CheckImportWindowOverUnwatchedBlock();
        CheckStoreAcrossImports();
        CheckImportWindowFootprint();
        CheckUnwatchedImport();
        CheckCommittedRanges();
        CheckHighestIndexMemo();
        CheckLazyGeneration();
        CheckAdoptLatestEpoch();
        CheckCollectMemoKey();
        CheckStampsHold();
        CheckCollectCovered();
        CheckHostWriteStamps();
#ifdef _WIN32
        CheckHostWriteWatchedMiddle();
#endif
        CheckHostWriteOutsideArena();
        CheckHostWriteLockOrder();
    } catch (const std::exception& error) {
        std::cerr << "write tracking test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "write tracking tests passed\n";
    return 0;
}
