#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_HIGHESTINDEXMEMO_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_HIGHESTINDEXMEMO_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace AgcDriver::Graphics {

// LOCAL EXPERIMENT (APS5_EXPERIMENT_REPLAY_INPUTS, COMMITTER-PLAN S6): the highest index of the last
// few in-place index ranges one draw recipe replayed, by the rule Recorder::KnownHighestIndex keeps
// for the ranges of 4096 indices and more. The caller collects the range first (CollectWrites, after
// reading the registry generation) and passes what it got; an entry answers while its range is
// unchanged since the collect made before its scan and the guest allocations did not change, so a
// GPU or driver store (MarkWritten), a CPU store seen by a later collect and a remapping all make it
// miss. A few entries, because one recipe serves draws of several index ranges.
class HighestIndexMemo {
public:
    // Whether an entry for the range answers; `highest` then receives its answer (empty: every
    // index was a restart). A stale entry is dropped.
    bool Find(std::uint64_t address, std::size_t bytes, std::uint32_t indexSize, bool skipRestart, std::uint64_t generation, std::uint64_t registryGeneration, std::optional<std::uint32_t>& highest) {
        if (generation == 0) return false;
        for (auto& entry : entries) {
            if (entry.generation == 0 || entry.address != address || entry.bytes != bytes || entry.indexSize != indexSize || entry.skipRestart != skipRestart) continue;
            if (entry.registryGeneration != registryGeneration || !GuestMemory::UnchangedSince(address, bytes, entry.generation)) {
                entry = {};
                return false;
            }
            highest = entry.highest;
            return true;
        }
        return false;
    }

    // Remembers the scan of a range collected at `generation` (0, an unwatched range, is not kept).
    void Keep(std::uint64_t address, std::size_t bytes, std::uint32_t indexSize, bool skipRestart, std::uint64_t generation, std::uint64_t registryGeneration, std::optional<std::uint32_t> highest) {
        if (generation == 0) return;
        Entry* slot = nullptr;
        for (auto& entry : entries) {
            if (entry.address == address && entry.bytes == bytes && entry.indexSize == indexSize && entry.skipRestart == skipRestart) slot = &entry;
        }
        if (slot == nullptr) slot = &entries[next++ % entries.size()];
        *slot = {address, bytes, indexSize, skipRestart, generation, registryGeneration, highest};
    }

private:
    struct Entry {
        std::uint64_t address = 0;
        std::size_t bytes = 0;
        std::uint32_t indexSize = 0;
        bool skipRestart = false;
        std::uint64_t generation = 0;
        std::uint64_t registryGeneration = 0;
        std::optional<std::uint32_t> highest;
    };
    std::array<Entry, 4> entries{};
    std::size_t next = 0;
};

}

#endif
