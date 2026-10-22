#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_TARGETLEASE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_TARGETLEASE_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include <cstddef>
#include <cstdint>
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
// entry, as D4-B's creation rule would. Counted on the [proofs] line. Both calls sit outside the
// [draws] lookup wall (refreshResidentTarget), so a stats run's lookup wall stays comparable with
// r70's, which had no counter; the draw's phase totals still carry the counter's cost.
namespace AgcDriver::Graphics {

class Recorder;

// One slot loop lookup: the slot loop fills the first part, TargetLeaseBefore the second, the two
// materializations their flags; TargetLeaseAfter classifies.
struct TargetLeaseUse {
    const ColorTarget* color = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    std::size_t colorBytes = 0;
    // tileMode != Linear and the context has a detiler; DumpTargetLimit() != 0.
    bool tiled = false;
    bool dumping = false;
    std::uint8_t ineligible = 0;
    const Recorder* recorder = nullptr;
    GuestMemory::CollectMemoKey memo;
    std::uint64_t levels = 0;
    std::uint64_t proved = 0;
    std::uint64_t walked = 0;
    bool registerClearWorked = false;
    bool cmaskClearWorked = false;
};

void TargetLeaseBefore(TargetLeaseUse& use);
void TargetLeaseAfter(TargetLeaseUse& use, bool resident);
// The [proofs] line's lease part (reportProofs, every 10 s): counts since the last call.
std::string TargetLeaseReport();

}

#endif
