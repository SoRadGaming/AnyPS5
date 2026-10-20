#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_LOCALCUTS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_LOCALCUTS_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// LOCAL EXPERIMENT (APS5_EXPERIMENT_LOCAL_CUTS, PARALLEL-COMMIT-PLAN C1): three cuts of per-draw
// work the committer repeats for an answer it already has, each in place:
// - Committed: Driver::commitDraw's SubmitDue learns that the commit completed from a flag set on
//   its two returns, instead of asking std::uncaught_exceptions() on every commit;
// - UnsavedLayers: StorageTexture::markLayersPending's unsaved-layer list lives in the thread's
//   scratch and is filled only when some layer of the range can be unsaved, instead of a new
//   std::vector<bool> per call;
// - DeviceReads: recordDraw copies a reusable draw template's DeviceReads list, kept from its first
//   recorded draw, instead of walking the template's regions on every draw. The cut stays only if
//   =2 shows the list never differs between draws of one template; otherwise it is dropped.
// 1 uses the cuts. 2 verifies them: today's answer is computed beside each cut and used, and the
// first difference aborts with a [local-cuts] line (as TicketDisagrees does). Unset runs today's
// code alone.
namespace AgcDriver::Graphics {

// The switch, read once per process (one static for every translation unit of the module).
inline int LocalCuts() {
    static const int mode = [] {
        const char* text = std::getenv("APS5_EXPERIMENT_LOCAL_CUTS");
        if (text == nullptr) return 0;
        return std::strtol(text, nullptr, 10) == 2 ? 2 : 1;
    }();
    return mode;
}

enum class LocalCut : std::size_t { Committed, UnsavedLayers, DeviceReads, Count };
constexpr std::size_t LocalCutCount = static_cast<std::size_t>(LocalCut::Count);
inline constexpr std::array<const char*, LocalCutCount> LocalCutNames{"committed flag", "unsaved layers", "device reads"};

// =2 only (relaxed, constant-initialized; the [stream] line takes them with exchange(0)): the checks
// each cut passed (commits; markLayersPending calls; draws of a template whose list was kept
// earlier), the calls that filled the unsaved-layer list, the templates whose list was kept, and the
// disagreements (the first one aborts).
struct LocalCutCounts {
    std::array<std::atomic<std::uint64_t>, LocalCutCount> checked{};
    std::atomic<std::uint64_t> layersFilled{};
    std::atomic<std::uint64_t> readsKept{};
    std::atomic<std::uint64_t> disagreed{};
};

inline constinit LocalCutCounts localCutCounts;

inline void NoteLocalCutChecked(LocalCut cut) {
    localCutCounts.checked[static_cast<std::size_t>(cut)].fetch_add(1, std::memory_order_relaxed);
}

[[noreturn]] inline void LocalCutDisagrees(LocalCut cut, const char* what) {
    localCutCounts.disagreed.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "[local-cuts] APS5_EXPERIMENT_LOCAL_CUTS=2: the %s cut disagrees with today's code: %s\n", LocalCutNames[static_cast<std::size_t>(cut)], what);
    std::fflush(stderr);
    std::abort();
}

}

#endif
