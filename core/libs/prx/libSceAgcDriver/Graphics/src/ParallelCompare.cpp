#include "prx/libSceAgcDriver/Graphics/include/ParallelCompare.hpp"
#include "prx/libSceAgcDriver/Execution/include/StreamStats.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace AgcDriver::Graphics {
namespace {

std::atomic<unsigned> runningHelpers{0};

class ComparePool {
public:
    explicit ComparePool(unsigned count) {
        stopping = count == 0;
        for (unsigned i = 0; i < count; ++i) {
            workers.emplace_back([this] { work(); });
            runningHelpers.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    bool Run(std::span<const CompareSpan> spans, std::span<std::uint8_t> equal) {
        std::unique_lock caller(callers, std::try_to_lock);
        if (!caller.owns_lock()) return false;
        {
            std::unique_lock lock(mutex);
            if (stopping) return false;
            finished.wait(lock, [&] { return working == 0; });
            jobSpans = spans;
            jobEqual = equal;
            next.store(0, std::memory_order_relaxed);
            done.store(0, std::memory_order_relaxed);
            ++generation;
        }
        wake.notify_all();
        drain(spans, equal);
        std::unique_lock lock(mutex);
        finished.wait(lock, [&] { return working == 0 && done.load(std::memory_order_acquire) == spans.size(); });
        jobSpans = {};
        jobEqual = {};
        return true;
    }

    void Stop() {
        std::vector<std::thread> joined;
        {
            std::lock_guard caller(callers);
            std::lock_guard lock(mutex);
            stopping = true;
            joined.swap(workers);
        }
        wake.notify_all();
        for (auto& worker : joined) {
            if (worker.get_id() == std::this_thread::get_id()) worker.detach();
            else worker.join();
        }
    }

private:
    void drain(std::span<const CompareSpan> spans, std::span<std::uint8_t> equal) {
        std::size_t compared = 0;
        for (auto index = next.fetch_add(1, std::memory_order_relaxed); index < spans.size(); index = next.fetch_add(1, std::memory_order_relaxed)) {
            const auto& span = spans[index];
            equal[index] = std::memcmp(span.first, span.second, span.bytes) == 0 ? 1 : 0;
            ++compared;
        }
        if (compared != 0) done.fetch_add(compared, std::memory_order_acq_rel);
    }

    void work() {
        std::uint64_t seen = 0;
        std::unique_lock lock(mutex);
        for (;;) {
            wake.wait(lock, [&] { return generation != seen || stopping; });
            if (stopping) break;
            seen = generation;
            const auto spans = jobSpans;
            const auto equal = jobEqual;
            ++working;
            lock.unlock();
            drain(spans, equal);
            lock.lock();
            --working;
            if (working == 0) finished.notify_all();
        }
        runningHelpers.fetch_sub(1, std::memory_order_acq_rel);
    }

    std::mutex callers;
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable finished;
    std::uint64_t generation = 0;
    unsigned working = 0;
    std::span<const CompareSpan> jobSpans;
    std::span<std::uint8_t> jobEqual;
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> done{0};
    std::vector<std::thread> workers;
    bool stopping = false;
};

std::mutex& poolLifecycle() {
    static std::mutex lifecycle;
    return lifecycle;
}

bool& poolStopped() {
    static bool stopped = false;
    return stopped;
}

std::atomic<ComparePool*>& createdPool() {
    static std::atomic<ComparePool*> created{nullptr};
    return created;
}

ComparePool* sharedPool() {
    static ComparePool* const pool = [] {
        std::lock_guard lifecycle(poolLifecycle());
        auto* created = new ComparePool(poolStopped() ? 0u : CompareHelpers());
        createdPool().store(created, std::memory_order_release);
        return created;
    }();
    return pool;
}

}

unsigned CompareHelpers() {
    // LOCAL EXPERIMENT (APS5_COMPARE_HELPERS=N, PARALLEL-COMMIT-PLAN B3a): the pool size, 1-15. The
    // hub's large snapshot compares (17+ spans of 64 KiB blocks, >= 1 MiB) run on the pool while the
    // committer drains its share and waits, ~3.4% of the committer at 2 helpers (16 logical processors / 8).
    static const unsigned helpers = [] {
        if (const char* text = std::getenv("APS5_COMPARE_HELPERS"); text != nullptr && text[0] != '\0') return std::clamp(static_cast<unsigned>(std::strtoul(text, nullptr, 10)), 1u, 15u);
        return std::clamp(std::thread::hardware_concurrency() / 8u, 1u, 4u);
    }();
    return helpers;
}

namespace {

// LOCAL DEBUG (APS5_DEBUG_STREAM_STATS, PARALLEL-COMMIT-PLAN S0): one CompareSpansFrom call in the
// [stream] compare histogram (Stream::Counters): its span count and total bytes, how it ran, its
// time, and that time again when the committer made the call.
void noteStreamCompare(std::size_t spans, std::size_t total, std::size_t mode, std::uint64_t startNs) {
    const auto ns = Stream::NowNs() - startNs;
    const std::size_t spanBucket = spans <= 1 ? 0 : spans <= 16 ? 1 : 2;
    const std::size_t byteBucket = total < (std::size_t{256} << 10u) ? 0 : total < (std::size_t{1} << 20u) ? 1 : 2;
    auto& counters = Stream::counters;
    Stream::Add(counters.compares[spanBucket][byteBucket]);
    Stream::Add(counters.compareNs[spanBucket][byteBucket], ns);
    if (GuestMemory::CurrentStreamRole() == GuestMemory::StreamRole::Committer) Stream::Add(counters.compareCommitterNs[spanBucket][byteBucket], ns);
    Stream::Add(counters.compareModes[mode]);
    Stream::Add(counters.compareModeNs[mode], ns);
}

}

void CompareSpansFrom(std::span<const CompareSpan> spans, std::span<std::uint8_t> equal, std::size_t parallelBytes) {
    const bool stream = Stream::On();
    const auto startNs = stream ? Stream::NowNs() : 0;
    std::size_t total = 0;
    for (const auto& span : spans) total += span.bytes;
    // The [stream] mode: 0 on the pool, 1 the pool was busy, 2 serial.
    std::size_t mode = 2;
    if (spans.size() > 1 && total >= parallelBytes) {
        if (sharedPool()->Run(spans, equal)) {
            if (stream) noteStreamCompare(spans.size(), total, 0, startNs);
            return;
        }
        mode = 1;
    }
    for (std::size_t index = 0; index < spans.size(); ++index) equal[index] = std::memcmp(spans[index].first, spans[index].second, spans[index].bytes) == 0 ? 1 : 0;
    if (stream) noteStreamCompare(spans.size(), total, mode, startNs);
}

void StopCompareHelpers() {
    ComparePool* pool = nullptr;
    {
        std::lock_guard lifecycle(poolLifecycle());
        poolStopped() = true;
        pool = createdPool().load(std::memory_order_acquire);
    }
    if (pool != nullptr) pool->Stop();
}

unsigned CompareHelpersRunning() {
    return runningHelpers.load(std::memory_order_acquire);
}

void CompareSpans(std::span<const CompareSpan> spans, std::span<std::uint8_t> equal) {
    CompareSpansFrom(spans, equal, std::size_t{1} << 20u);
}

}
