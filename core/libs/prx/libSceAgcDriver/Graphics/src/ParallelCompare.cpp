#include "prx/libSceAgcDriver/Graphics/include/ParallelCompare.hpp"
#include "prx/libSceAgcDriver/Execution/include/StreamStats.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace AgcDriver::Graphics {
namespace {

std::atomic<unsigned> runningHelpers{0};

// LOCAL EXPERIMENT (APS5_EXPERIMENT_SNAPSHOT_HASH, GPU-OWNED-PLAN L1-1): the hash memo's totals
// (relaxed; [snapshot-stats] takes them with exchange(0)). Compares add theirs once per drain.
struct SnapshotHashTotals {
    std::atomic<std::uint64_t> decided{0};
    std::atomic<std::uint64_t> decidedEqual{0};
    std::atomic<std::uint64_t> learned{0};
    std::atomic<std::uint64_t> rehashed{0};
    std::atomic<std::uint64_t> audited{0};
    std::atomic<std::uint64_t> anomalies{0};
    std::atomic<std::uint64_t> falseEqual{0};
    std::atomic<std::uint64_t> stale{0};
};

constinit SnapshotHashTotals snapshotHashTotals;

[[noreturn]] void snapshotHashDisagrees(const CompareSpan& span, const char* what) {
    std::fprintf(stderr, "[snapshot-hash] APS5_EXPERIMENT_SNAPSHOT_HASH=3: %s (snapshot bytes at %p, guest block 0x%llx, %zu bytes)\n", what, span.first, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(span.second)), span.bytes);
    std::fflush(stderr);
    std::abort();
}

// A span with a hash memo: an unknown memo is learned from S when memcmp finds the block equal (S
// was just read, so the hash reads it from L2); a known one decides from guest memory alone. Under
// =2/=3 memcmp runs beside every memo verdict and its verdict is used. When they disagree, a memo
// that no longer describes S is stale (S was written behind it) and is repaired from S; otherwise
// the memo held a differing block equal through a lane collision (false-equal). Each span writes
// only its own memo, and Run's acquire of `done` orders the helpers' writes before the caller's
// reads.
[[gnu::noinline]] std::uint8_t compareHashed(const CompareSpan& span, SnapshotHashCounts& counts) {
    auto& memo = *span.hash;
    if (memo.known == 0) {
        if (std::memcmp(span.first, span.second, span.bytes) != 0) return 0;
        memo.lanes = HashLanes(span.first, span.bytes);
        memo.known = 1;
        ++counts.learned;
        return 1;
    }
    ++counts.decided;
    const bool hashEqual = HashLanes(span.second, span.bytes) == memo.lanes;
    if (const int mode = SnapshotHashMode(); mode >= 2) {
        const bool bytesEqual = std::memcmp(span.first, span.second, span.bytes) == 0;
        if (hashEqual != bytesEqual) {
            const auto lanes = HashLanes(span.first, span.bytes);
            const bool stale = lanes != memo.lanes;
            if (stale) {
                ++counts.stale;
                memo.lanes = lanes;
            } else {
                ++counts.falseEqual;
            }
            if (mode == 3) snapshotHashDisagrees(span, stale ? "a stale memo (S changed behind it) disagrees with memcmp" : "the memo held a differing block equal (a lane collision)");
            return bytesEqual ? 1 : 0;
        }
    }
    if (!hashEqual) return 0;
    ++counts.decidedEqual;
    return 3;
}

// One span's verdict: 1 equal by memcmp, 3 equal by the span's hash memo, 0 differs.
inline std::uint8_t compareOne(const CompareSpan& span, SnapshotHashCounts& counts) {
    if (span.hash == nullptr) return std::memcmp(span.first, span.second, span.bytes) == 0 ? 1 : 0;
    return compareHashed(span, counts);
}

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
        SnapshotHashCounts counts;
        for (auto index = next.fetch_add(1, std::memory_order_relaxed); index < spans.size(); index = next.fetch_add(1, std::memory_order_relaxed)) {
            equal[index] = compareOne(spans[index], counts);
            ++compared;
        }
        if ((counts.decided | counts.learned) != 0) NoteSnapshotHashCounts(counts);
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
    SnapshotHashCounts counts;
    for (std::size_t index = 0; index < spans.size(); ++index) equal[index] = compareOne(spans[index], counts);
    if ((counts.decided | counts.learned) != 0) NoteSnapshotHashCounts(counts);
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

std::array<std::uint64_t, 4> HashLanes(const void* data, std::size_t bytes) {
    // HashPage's seeds and round (GuestMemory.cpp); a span's length is fixed per (snapshot, block),
    // so the zero padding of the tail word cannot alias.
    constexpr std::uint64_t prime1 = 0x9E3779B185EBCA87ull;
    constexpr std::uint64_t prime2 = 0xC2B2AE3D27D4EB4Full;
    const auto round = [](std::uint64_t accumulator, std::uint64_t word) { return std::rotl(accumulator + word * prime2, 31) * prime1; };
    const auto* source = static_cast<const std::byte*>(data);
    // The stripe loop keeps the lanes and words in scalars (GCC kept an array of each in memory).
    const auto load = [](const std::byte* at) {
        std::uint64_t word;
        std::memcpy(&word, at, sizeof(word));
        return word;
    };
    std::uint64_t lane0 = prime1 + prime2, lane1 = prime2, lane2 = 0, lane3 = 0 - prime1;
    std::size_t offset = 0;
    for (; bytes - offset >= 4 * sizeof(std::uint64_t); offset += 4 * sizeof(std::uint64_t)) {
        lane0 = round(lane0, load(source + offset));
        lane1 = round(lane1, load(source + offset + 8));
        lane2 = round(lane2, load(source + offset + 16));
        lane3 = round(lane3, load(source + offset + 24));
    }
    std::array<std::uint64_t, 4> lanes{lane0, lane1, lane2, lane3};
    std::size_t lane = 0;
    for (; bytes - offset >= sizeof(std::uint64_t); offset += sizeof(std::uint64_t), ++lane) {
        std::uint64_t word;
        std::memcpy(&word, source + offset, sizeof(word));
        lanes[lane] = round(lanes[lane], word);
    }
    if (offset != bytes) {
        std::uint64_t word = 0;
        std::memcpy(&word, source + offset, bytes - offset);
        lanes[lane] = round(lanes[lane], word);
    }
    return lanes;
}

int SnapshotHashMode() {
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_SNAPSHOT_HASH, GPU-OWNED-PLAN L1-1): read once per process.
    static const int mode = [] {
        const char* text = std::getenv("APS5_EXPERIMENT_SNAPSHOT_HASH");
        if (text == nullptr) return 0;
        return static_cast<int>(std::clamp(std::strtol(text, nullptr, 10), 0l, 3l));
    }();
    return mode;
}

void NoteSnapshotHashCounts(const SnapshotHashCounts& counts) {
    const auto add = [](std::atomic<std::uint64_t>& total, std::uint64_t value) {
        if (value != 0) total.fetch_add(value, std::memory_order_relaxed);
    };
    auto& totals = snapshotHashTotals;
    add(totals.decided, counts.decided);
    add(totals.decidedEqual, counts.decidedEqual);
    add(totals.learned, counts.learned);
    add(totals.rehashed, counts.rehashed);
    add(totals.audited, counts.audited);
    add(totals.anomalies, counts.anomalies);
    add(totals.falseEqual, counts.falseEqual);
    add(totals.stale, counts.stale);
}

SnapshotHashCounts TakeSnapshotHashCounts() {
    const auto take = [](std::atomic<std::uint64_t>& total) { return total.exchange(0, std::memory_order_relaxed); };
    auto& totals = snapshotHashTotals;
    return {take(totals.decided), take(totals.decidedEqual), take(totals.learned), take(totals.rehashed), take(totals.audited), take(totals.anomalies), take(totals.falseEqual), take(totals.stale)};
}

}
