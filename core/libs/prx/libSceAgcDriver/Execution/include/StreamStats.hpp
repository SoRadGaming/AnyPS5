#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_STREAMSTATS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_STREAMSTATS_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

// LOCAL DEBUG (APS5_DEBUG_STREAM_STATS=1, PARALLEL-COMMIT-PLAN S0): the counters of the [stream]
// line that are kept outside DrawPipeline (the committer's idle attribution, the drain sites and the
// lead times stay under the pipeline mutex there): the shadow dispatch classifier (Dispatch.cpp),
// the presenter's second hold (Presentation.cpp), the submit split (Recorder::Submit), the compare
// histogram (CompareSpansFrom), the tracker lock and GpuMutex waits per thread role (GuestMemory).
// Relaxed atomics, constant-initialized; DrawPipeline's report takes them with exchange(0). Nothing
// is counted while GuestMemory::StreamStats() is 0.
namespace AgcDriver::Stream {

using Counter = std::atomic<std::uint64_t>;
constexpr std::size_t Roles = static_cast<std::size_t>(GuestMemory::StreamRole::Count);
constexpr std::size_t Sites = static_cast<std::size_t>(GuestMemory::GpuLockSite::Count);
inline constexpr std::array<const char*, Roles> RoleNames{"other", "committer", "queue0", "compute", "presenter", "helper"};

// The switch, read once per process (one static for every translation unit of the module), so a
// check off the switch is a load and a predictable branch, not a call into GuestMemory.
inline bool On() {
    static const bool on = GuestMemory::StreamStats() != 0;
    return on;
}

inline std::uint64_t NowNs() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline void Add(Counter& counter, std::uint64_t value = 1) {
    counter.fetch_add(value, std::memory_order_relaxed);
}

inline std::uint64_t Take(Counter& counter) {
    return counter.exchange(0, std::memory_order_relaxed);
}

// The shadow classifier's classes: what A1 (stream dispatch items) would do with a queue-0 dispatch,
// first match wins in this order: Fill (the fill HLE returned before the dispatch drain), Ahead (a
// dispatch-ahead adoption), Indirect, Labels (deferred labels queued), Copy (a copy kernel), Miss (no
// dispatch-cache hit, or a hit that cannot skip its validation: no cache, stamp validation, data-hit
// verification), Unbounded* (a hit whose write list is unbounded, by the first role found in the
// order Gds, Fault, Pagetable, Bda (bdaAbiVersion), Image (a storage image without a full
// descriptor)), Reads (a bounded hit whose reads overlap a queued item's writes: A1 drains through
// the newest overlapping writer), NoRecipe (a bounded, clear hit without a recipe: an A1 item),
// Item (today's APS5_PIPELINE_DISPATCH item). Unclassified: a drain no dispatch claimed.
enum class DispatchClass : std::uint8_t { Unclassified, Item, NoRecipe, Reads, UnboundedGds, UnboundedFault, UnboundedPagetable, UnboundedBda, UnboundedImage, Indirect, Copy, Fill, Miss, Labels, Ahead, Count };
constexpr std::size_t DispatchClasses = static_cast<std::size_t>(DispatchClass::Count);
inline constexpr std::array<const char*, DispatchClasses> DispatchClassNames{"unclassified", "item", "no-recipe", "reads", "unbounded-gds", "unbounded-fault", "unbounded-pagetable", "unbounded-bda", "unbounded-image", "indirect", "copy", "fill", "miss", "labels", "ahead"};
constexpr bool ItemEligible(DispatchClass value) {
    return value == DispatchClass::Item || value == DispatchClass::NoRecipe || value == DispatchClass::Reads;
}
// Reads: the items queued behind the newest overlapping writer when the dispatch arrived (what a
// partial drain would keep queued): 0, 1, 2-3, 4-7, 8+.
constexpr std::size_t BehindBuckets = 5;
inline constexpr std::array<const char*, BehindBuckets> BehindNames{"0", "1", "2-3", "4-7", "8+"};
constexpr std::size_t BehindBucket(std::size_t behind) {
    return behind == 0 ? 0 : behind == 1 ? 1 : behind < 4 ? 2 : behind < 8 ? 3 : 4;
}

// CompareSpansFrom: span count (1, 2-16, more) by total bytes (< 256 KiB, < 1 MiB, more), and how
// it ran: on the pool, pool busy (or stopped, so serial), serial (one span or below the threshold).
constexpr std::size_t CompareSpanBuckets = 3;
constexpr std::size_t CompareByteBuckets = 3;
constexpr std::size_t CompareModes = 3;
inline constexpr std::array<const char*, CompareSpanBuckets> CompareSpanNames{"1", "2-16", "17+"};
inline constexpr std::array<const char*, CompareByteBuckets> CompareByteNames{"<256K", "<1M", ">=1M"};
inline constexpr std::array<const char*, CompareModes> CompareModeNames{"pool", "pool busy", "serial"};

struct Counters {
    std::array<Counter, DispatchClasses> dispatches{};
    std::array<Counter, BehindBuckets> readsBehind{};
    // Queue-0 dispatches left unclassified because APS5_PIPELINE_DISPATCH is on (the classifier
    // shadows A1 against today's drains, which that switch changes).
    Counter unshadowed{};
    // Presenter: the second hold (Driver::Present), from the GpuMutex request to the release: its
    // lock wait, PresentDisplayBuffer (or PresentClear) and the in-hold QueuePresent; the third
    // hold's QueuePresent (a synchronous flip's or one without slots) apart.
    Counter presents{};
    Counter holdNs{};
    Counter holdLockWaitNs{};
    Counter presentBufferNs{};
    Counter queuePresentNs{};
    Counter latePresents{};
    Counter lateQueuePresentNs{};
    // Recorder::Submit with an open batch, by the submitting thread's role and the site tagged for
    // its hold: the whole call, vkEndCommandBuffer and vkQueueSubmit.
    std::array<std::array<Counter, Sites>, Roles> submits{};
    std::array<std::array<Counter, Sites>, Roles> submitNs{};
    std::array<std::array<Counter, Sites>, Roles> endNs{};
    std::array<std::array<Counter, Sites>, Roles> queueSubmitNs{};
    std::array<std::array<Counter, CompareByteBuckets>, CompareSpanBuckets> compares{};
    std::array<std::array<Counter, CompareByteBuckets>, CompareSpanBuckets> compareNs{};
    std::array<std::array<Counter, CompareByteBuckets>, CompareSpanBuckets> compareCommitterNs{};
    std::array<Counter, CompareModes> compareModes{};
    std::array<Counter, CompareModes> compareModeNs{};
    // The write tracker's mutex (lockTracker), per role: acquisitions, the ones that found it held,
    // and their wait.
    std::array<Counter, Roles> trackerAcquisitions{};
    std::array<Counter, Roles> trackerWaits{};
    std::array<Counter, Roles> trackerWaitNs{};
    // Contended GpuMutex lock()s, per waiter role by the role of the holder seen when the wait
    // began (the outermost acquisition publishes it).
    std::array<std::array<Counter, Roles>, Roles> gpuWaits{};
    std::array<std::array<Counter, Roles>, Roles> gpuWaitNs{};
};

inline constinit Counters counters;

}

#endif
