#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/PrepHelpers.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/StreamStats.hpp"
#include "prx/libSceAgcDriver/Graphics/include/FlushGuard.hpp"
#include "prx/libSceAgcDriver/Graphics/include/LocalCuts.hpp"
#include "prx/libSceAgcDriver/Graphics/include/QueueGate.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::array<const char*, static_cast<std::size_t>(DrawPipeline::DrainReason::Count)> DrainNames{"packet", "flush", "labels", "capture", "indirect", "submission"};

std::atomic<unsigned> runningCommitters{0};

std::uint64_t elapsedNs(std::chrono::steady_clock::time_point since) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
}

// LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): the [stream] line's helpers. StreamTotals is what the
// report takes from Stream::counters (each counter exchanged with 0).
static_assert(Stream::DispatchClasses < 16, "DrawPipeline::StreamClassSlots holds every dispatch class and the later-drains slot");

template<std::size_t N>
std::array<std::uint64_t, N> takeCounters(std::array<Stream::Counter, N>& counters) {
    std::array<std::uint64_t, N> values{};
    for (std::size_t i = 0; i < N; ++i) values[i] = Stream::Take(counters[i]);
    return values;
}

template<std::size_t N, std::size_t M>
std::array<std::array<std::uint64_t, M>, N> takeCounters(std::array<std::array<Stream::Counter, M>, N>& counters) {
    std::array<std::array<std::uint64_t, M>, N> values{};
    for (std::size_t i = 0; i < N; ++i) values[i] = takeCounters(counters[i]);
    return values;
}

struct StreamTotals {
    std::array<std::uint64_t, Stream::DispatchClasses> dispatches{};
    std::array<std::uint64_t, Stream::BehindBuckets> readsBehind{};
    std::uint64_t unshadowed = 0;
    std::uint64_t shadowNs = 0;
    std::uint64_t bdaRoleMismatches = 0;
    std::uint64_t presents = 0;
    std::uint64_t holdNs = 0;
    std::uint64_t holdLockWaitNs = 0;
    std::uint64_t presentBufferNs = 0;
    std::uint64_t queuePresentNs = 0;
    std::uint64_t latePresents = 0;
    std::uint64_t lateQueuePresentNs = 0;
    std::uint64_t issuedPresents = 0;
    std::uint64_t issuePresentNs = 0;
    std::uint64_t submitterSubmits = 0;
    std::uint64_t submitterSubmitNs = 0;
    std::uint64_t waitedAwaits = 0;
    std::uint64_t awaitNs = 0;
    std::uint64_t waitedDrains = 0;
    std::uint64_t drainNs = 0;
    std::uint64_t unissuedPolls = 0;
    std::array<std::array<std::uint64_t, Stream::Sites>, Stream::Roles> submits{};
    std::array<std::array<std::uint64_t, Stream::Sites>, Stream::Roles> submitNs{};
    std::array<std::array<std::uint64_t, Stream::Sites>, Stream::Roles> endNs{};
    std::array<std::array<std::uint64_t, Stream::Sites>, Stream::Roles> queueSubmitNs{};
    std::array<std::array<std::uint64_t, Stream::CompareByteBuckets>, Stream::CompareSpanBuckets> compares{};
    std::array<std::array<std::uint64_t, Stream::CompareByteBuckets>, Stream::CompareSpanBuckets> compareNs{};
    std::array<std::array<std::uint64_t, Stream::CompareByteBuckets>, Stream::CompareSpanBuckets> compareCommitterNs{};
    std::array<std::uint64_t, Stream::CompareModes> compareModes{};
    std::array<std::uint64_t, Stream::CompareModes> compareModeNs{};
    std::array<std::uint64_t, Stream::Roles> trackerAcquisitions{};
    std::array<std::uint64_t, Stream::Roles> trackerWaits{};
    std::array<std::uint64_t, Stream::Roles> trackerWaitNs{};
    std::array<std::array<std::uint64_t, Stream::Roles>, Stream::Roles> gpuWaits{};
    std::array<std::array<std::uint64_t, Stream::Roles>, Stream::Roles> gpuWaitNs{};
};

StreamTotals takeStreamTotals() {
    auto& counters = Stream::counters;
    StreamTotals totals;
    totals.dispatches = takeCounters(counters.dispatches);
    totals.readsBehind = takeCounters(counters.readsBehind);
    totals.unshadowed = Stream::Take(counters.unshadowed);
    totals.shadowNs = Stream::Take(counters.shadowNs);
    totals.bdaRoleMismatches = Stream::Take(counters.bdaRoleMismatches);
    totals.presents = Stream::Take(counters.presents);
    totals.holdNs = Stream::Take(counters.holdNs);
    totals.holdLockWaitNs = Stream::Take(counters.holdLockWaitNs);
    totals.presentBufferNs = Stream::Take(counters.presentBufferNs);
    totals.queuePresentNs = Stream::Take(counters.queuePresentNs);
    totals.latePresents = Stream::Take(counters.latePresents);
    totals.lateQueuePresentNs = Stream::Take(counters.lateQueuePresentNs);
    totals.issuedPresents = Stream::Take(counters.issuedPresents);
    totals.issuePresentNs = Stream::Take(counters.issuePresentNs);
    totals.submitterSubmits = Stream::Take(counters.submitterSubmits);
    totals.submitterSubmitNs = Stream::Take(counters.submitterSubmitNs);
    totals.waitedAwaits = Stream::Take(counters.waitedAwaits);
    totals.awaitNs = Stream::Take(counters.awaitNs);
    totals.waitedDrains = Stream::Take(counters.waitedDrains);
    totals.drainNs = Stream::Take(counters.drainNs);
    totals.unissuedPolls = Stream::Take(counters.unissuedPolls);
    totals.submits = takeCounters(counters.submits);
    totals.submitNs = takeCounters(counters.submitNs);
    totals.endNs = takeCounters(counters.endNs);
    totals.queueSubmitNs = takeCounters(counters.queueSubmitNs);
    totals.compares = takeCounters(counters.compares);
    totals.compareNs = takeCounters(counters.compareNs);
    totals.compareCommitterNs = takeCounters(counters.compareCommitterNs);
    totals.compareModes = takeCounters(counters.compareModes);
    totals.compareModeNs = takeCounters(counters.compareModeNs);
    totals.trackerAcquisitions = takeCounters(counters.trackerAcquisitions);
    totals.trackerWaits = takeCounters(counters.trackerWaits);
    totals.trackerWaitNs = takeCounters(counters.trackerWaitNs);
    totals.gpuWaits = takeCounters(counters.gpuWaits);
    totals.gpuWaitNs = takeCounters(counters.gpuWaitNs);
    return totals;
}

double streamMs(std::uint64_t ns) {
    return static_cast<double>(ns) / 1e6;
}

double streamPercent(std::uint64_t part, std::uint64_t whole) {
    return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

const char* streamFileName(const char* path) {
    if (path == nullptr) return "other";
    const char* name = path;
    for (const char* at = path; *at != '\0'; ++at) {
        if (*at == '/' || *at == '\\') name = at + 1;
    }
    return name;
}

// Formatted with printf's checks; a piece is cut at 191 characters.
__attribute__((format(gnu_printf, 2, 3))) void appendStream(std::string& text, const char* format, ...) {
    char item[192];
    std::va_list args;
    va_start(args, format);
    const auto size = std::vsnprintf(item, sizeof(item), format, args);
    va_end(args);
    if (size > 0) text.append(item, std::min(static_cast<std::size_t>(size), sizeof(item) - 1));
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_ROOM_WAKE=1): a producer waiting for room is woken once the
// committer brought the queue down to half its depth instead of after every pop, and a drain only
// once the queue is empty, so the committer pays one wake per several commits. =2 also spins the
// producer for the hand-off window before it blocks: a slot usually frees within a commit or two.
int roomWake() {
    static const int mode = [] {
        const char* text = std::getenv("APS5_EXPERIMENT_ROOM_WAKE");
        if (text == nullptr) return 0;
        return std::strtol(text, nullptr, 10) == 2 ? 2 : 1;
    }();
    return mode;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_COMMIT_RUNS=1): when the front item is a draw or a label, the
// committer takes the GPU mutex once and commits the draws and labels behind it under the same
// hold, up to 8 commits (=N for N >= 2: up to N) or about CommitRunWindow. It ends the run early
// when another thread waits for the mutex (GpuMutexType::Waiters), when the next item is anything
// else, and when the queue is empty, so it never waits for work while it holds the mutex. Each
// commit still runs its own closure and takes the mutex itself, which now nests, and each item is
// popped as soon as it finished, as before. 0 when off.
std::size_t commitRunItems() {
    static const std::size_t items = [] {
        const char* text = std::getenv("APS5_EXPERIMENT_COMMIT_RUNS");
        if (text == nullptr) return std::size_t{0};
        const auto value = std::strtoull(text, nullptr, 10);
        return value >= 2 ? static_cast<std::size_t>(std::min<unsigned long long>(value, 64)) : std::size_t{8};
    }();
    return items;
}

constexpr std::chrono::microseconds CommitRunWindow{150};

struct CurrentRunTag {};

// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_RECEIPTS): set on the committer thread (run), the only thread
// that follows tokens as a receipt follower; a closing Enqueue's inline commit on queue 0 follows
// the token without taking that role (its bump only ends queue 0's owner epoch).
struct CommitterThreadTag {};
using CommitterThread = HostThreadSlot<bool, CommitterThreadTag>;

// LOCAL EXPERIMENT (APS5_EXPERIMENT_PREP_HELPERS, PARALLEL-COMMIT-PLAN B0): the push time of the item
// the committer is committing (CurrentItemEnqueuedNs), set by commitFront around each item's commit.
struct CurrentItemEnqueuedTag {};
using CurrentItemEnqueued = HostThreadSlot<std::uint64_t, CurrentItemEnqueuedTag>;

// How a commit ended: `failed` for an exception (a failed commit in the report), `shutdown` for
// the process shutting down, which is not reported to the driver.
struct CommitOutcome {
    std::exception_ptr fatal;
    bool failed = false;
    bool shutdown = false;
};

// The bytes a label appended to an item stores (APS5_EXPERIMENT_LABEL_COALESCE); an end-of-pipe
// label that stores nothing is the empty range at 0, which overlaps nothing.
DrawPipeline::Range labelRange(const DrawPipeline::LabelPart& label) {
    if (label.bytes.empty()) return {0, 0};
    return {label.address, label.address + label.bytes.size()};
}

template<typename TWork>
CommitOutcome guarded(TWork&& work) {
    CommitOutcome outcome;
    try {
        work();
    } catch (const std::exception&) {
        outcome.failed = true;
        outcome.fatal = std::current_exception();
    } catch (const ProcessShutdown&) {
        outcome.shutdown = true;
        outcome.fatal = std::current_exception();
    } catch (...) {
        outcome.fatal = std::current_exception();
    }
    return outcome;
}

}

// The committer's state for one run, reachable from the commits inside it through a thread slot.
struct DrawPipeline::CommitRun {
    std::size_t left = 0;
    std::chrono::steady_clock::time_point until{};
    // Taken before the run waited for the GPU mutex: the first commit's time includes that wait,
    // as a commit's time included its own lock wait.
    std::chrono::steady_clock::time_point started{};
    // The check the commits left to the run's end (Driver::submitDueAfterRun), run once.
    void (*deferred)() = nullptr;
    bool labelReaped = false;
    std::exception_ptr unreported;
};

DrawPipeline& DrawPipeline::Queue0() {
    static DrawPipeline* pipeline = new DrawPipeline();
    return *pipeline;
}

std::size_t DrawPipeline::Depth() {
    static const std::size_t depth = [] {
        const char* text = std::getenv("APS5_PIPELINED_DRAWS");
        if (std::getenv("APS5_LOCKED_DRAW_PREPARE") != nullptr || std::getenv("APS5_DRAW_DRAIN") != nullptr || std::getenv("APS5_DRAIN_ALL") != nullptr || std::getenv("APS5_NO_WORDWISE_CAPTURE") != nullptr) return std::size_t{0};
        if (text == nullptr) return std::size_t{8};
        const auto value = std::strtoull(text, nullptr, 10);
        return value == 0 ? std::size_t{0} : value == 1 ? std::size_t{8} : static_cast<std::size_t>(std::min<unsigned long long>(value, 256));
    }();
    return depth;
}

std::size_t DrawPipeline::QueuedItems() {
    auto& pipeline = Queue0();
    std::lock_guard lock(pipeline.mutex);
    return pipeline.items.size();
}

bool& DrawPipeline::Active() {
    static thread_local bool active = false;
    return active;
}

std::atomic<std::uint64_t>& DrawPipeline::EpochToken() {
    static std::atomic<std::uint64_t> token{1};
    return token;
}

void DrawPipeline::FollowEpoch(std::uint64_t token) {
    struct FollowedTag {};
    using Followed = HostThreadSlot<std::uint64_t, FollowedTag>;
    if (Followed::Get() == token) return;
    Followed::Set(token);
    GuestMemory::BumpCollectEpoch();
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_RECEIPTS, COMMITTER-SPLIT-PLAN Q3): the committer follows
    // the token in the epoch it just took; receipts of that token serve it while that epoch lasts.
    if (GuestMemory::Q0Receipts() != 0 && CommitterThread::Get()) GuestMemory::SetReceiptFollower(token);
}

unsigned DrawPipeline::Committers() {
    return runningCommitters.load(std::memory_order_acquire);
}

void DrawPipeline::rethrowFailure() {
    if (failure == nullptr) return;
    auto error = std::exchange(failure, nullptr);
    std::rethrow_exception(error);
}

std::uint64_t DrawPipeline::Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes, ItemKind kind, const Prepare* prepare) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr || std::getenv("APS5_PIPELINE_REPORT") != nullptr;
    const bool streamOn = Stream::On();
    std::unique_lock lock(mutex);
    rethrowFailure();
    if (closing) {
        lock.unlock();
        commit();
        return 0;
    }
    // LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS): the depth this item is enqueued behind and its room
    // wait, for the [q0prep] line; the depth also rides with the item for the committer's walks.
    // Full depth is its own count (and the budget's flag), as Depth() may exceed the last bucket.
    const auto enqueueDepth = items.size();
    const bool full = enqueueDepth >= Depth();
    const bool prepStatsOn = GuestMemory::Q0PrepStats() != 0;
    if (prepStatsOn && kind == ItemKind::Draw) {
        ++prepStats.draws;
        ++prepStats.drawDepths[std::min(enqueueDepth, PrepDepthBuckets - 1)];
        if (full) ++prepStats.fullDraws;
    } else if (prepStatsOn && kind == ItemKind::Label) {
        ++prepStats.labels;
    }
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_TICKET): the producer's preparation (see Prepare), only
    // when the queue is deep enough for the committer to stay busy meanwhile and never while the
    // producer holds the GPU mutex.
    if (prepare != nullptr && prepare->run != nullptr) {
        if (enqueueDepth < prepare->pre) {
            if (prepare->skipped != nullptr) prepare->skipped(prepare->self, PrepareSkip::Shallow);
        } else if (GuestMemory::GpuMutex().HeldByThisThread()) {
            if (prepare->skipped != nullptr) prepare->skipped(prepare->self, PrepareSkip::Locked);
        } else {
            const PrepareBudget budget{this, enqueueDepth, full, prepare->bail};
            lock.unlock();
            {
                const GuestMemory::PrepareScope scope;
                prepare->run(prepare->self, budget);
            }
            lock.lock();
            rethrowFailure();
            if (closing) {
                lock.unlock();
                commit();
                return 0;
            }
        }
    }
    if (items.size() >= Depth()) {
        if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline full while the enqueuing thread holds the GPU lock");
        const auto start = std::chrono::steady_clock::now();
        if (roomWake() == 0) {
            ++idleWaiters;
            idle.wait(lock, [&] { return items.size() < Depth(); });
            --idleWaiters;
        } else {
            if (roomWake() == 2) {
                // Spun without the lock, as Drain spins; the wait below checks the queue again.
                lock.unlock();
                SpinUntil(HandoffSpinWindow(), [&] { return outstanding.load(std::memory_order_acquire) < Depth(); });
                lock.lock();
            }
            ++roomWaiters;
            room.wait(lock, [&] { return items.size() < Depth(); });
            --roomWaiters;
        }
        const auto waitedNs = elapsedNs(start);
        fullWaitNs += waitedNs;
        if (prepStatsOn && kind == ItemKind::Draw) {
            ++prepStats.drawWaits;
            prepStats.drawWaitNs += waitedNs;
        } else if (prepStatsOn && kind == ItemKind::Label) {
            ++prepStats.labelWaits;
            prepStats.labelWaitNs += waitedNs;
        }
        rethrowFailure();
    }
    depthSum += items.size();
    ++enqueued;
    // LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): the push time and depth (a draw's lead) end an open drain
    // episode and the gate writers kept for it. LOCAL EXPERIMENT (APS5_EXPERIMENT_PREP_HELPERS): the
    // push time is also the committer's CurrentItemEnqueuedNs, and the seq orders the helper jobs.
    const auto pushedAtNs = streamOn || PrepHelpers::On() ? Stream::NowNs() : 0;
    const auto pushDepth = streamOn ? static_cast<std::uint32_t>(items.size()) : 0;
    const auto seq = ++lastSeq;
    items.push_back({std::move(commit), std::move(writes), labelAddress, std::move(labelBytes), kind, {}, static_cast<std::uint32_t>(enqueueDepth), full, pushedAtNs, pushDepth, seq});
    if (streamOn) {
        closeStreamEpisode(pushedAtNs);
        streamGateEpisode.store(0, std::memory_order_relaxed);
        streamGateSerial = 0;
    }
    outstanding.fetch_add(1, std::memory_order_release);
    if (!thread.joinable()) {
        thread = std::thread([this] { run(); });
        runningCommitters.fetch_add(1, std::memory_order_acq_rel);
    }
    if (committerWaiting) wake.notify_one();
    if (profile) {
        const auto now = std::chrono::steady_clock::now();
        if (now - lastReport >= std::chrono::seconds(10)) report(now);
    }
    // Last: it releases the pipeline mutex before it prints.
    if (streamOn) reportStream(lock);
    return seq;
}

void DrawPipeline::report(std::chrono::steady_clock::time_point now) {
    lastReport = now;
    std::string drainText;
    for (std::size_t i = 0; i < drains.size(); ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu (%.1f ms)", DrainNames[i], static_cast<unsigned long long>(drains[i]), static_cast<double>(drainWaitNs[i]) / 1e6);
        drainText += item;
    }
    drainText += "; packet drains by opcode:";
    for (int shown = 0; shown < 5; ++shown) {
        const auto top = std::max_element(drainOpcodes.begin(), drainOpcodes.begin() + 0x100);
        if (*top == 0) break;
        char item[32];
        std::snprintf(item, sizeof(item), " 0x%02x x%llu", static_cast<unsigned>(top - drainOpcodes.begin()), static_cast<unsigned long long>(*top));
        drainText += item;
        *top = 0;
    }
    drainOpcodes = {};
    const auto perCommit = commits == 0 ? 0.0 : static_cast<double>(commitNs) / 1e3 / static_cast<double>(commits);
    const auto perDraw = enqueued == 0 ? 0.0 : static_cast<double>(fullWaitNs) / 1e3 / static_cast<double>(enqueued);
    const auto queued = enqueued == 0 ? 0.0 : static_cast<double>(depthSum) / static_cast<double>(enqueued);
    if (runs != 0) {
        char item[96];
        std::snprintf(item, sizeof(item), "; %llu commit runs of %.1f commits on average", static_cast<unsigned long long>(runs), static_cast<double>(runCommits) / static_cast<double>(runs));
        drainText += item;
    }
    if (labelsAppended != 0) {
        char item[80];
        std::snprintf(item, sizeof(item), "; %llu labels appended to queued commits", static_cast<unsigned long long>(labelsAppended));
        drainText += item;
    }
    std::fprintf(stderr, "[draw] pipelined (10 s, depth %zu): %llu commits at %.1f us each on the committer, %llu failed; %llu enqueued behind %.2f on average, waiting %.1f us per draw for room; drains:%s\n", Depth(), static_cast<unsigned long long>(commits), perCommit, static_cast<unsigned long long>(commitErrors), static_cast<unsigned long long>(enqueued), queued, perDraw, drainText.c_str());
    commits = commitNs = commitErrors = enqueued = fullWaitNs = depthSum = runs = runCommits = labelsAppended = 0;
    drains = {};
    drainWaitNs = {};
}

DrawPipeline::PrepStats DrawPipeline::TakePrepStats() {
    std::lock_guard lock(mutex);
    return std::exchange(prepStats, PrepStats{});
}

void DrawPipeline::run() {
    StartCommitterSampler();
    PerformanceTrace::BindThread("draw committer");
    if (GuestMemory::Q0Receipts() != 0) CommitterThread::Set(true);
    // LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): the loop's idle (top to the item taken; a wait when the
    // queue was empty at the top) and busy (the front's commit or run) spans; a busy span is added
    // under the mutex at the next iteration.
    const bool streamOn = Stream::On();
    if (streamOn) GuestMemory::SetStreamRole(GuestMemory::StreamRole::Committer);
    std::uint64_t busyNs = 0;
    std::size_t busyKind = 0;
    for (;;) {
        Item* front = nullptr;
        const auto idleStartNs = streamOn ? Stream::NowNs() : 0;
        const bool emptyAtTop = streamOn && outstanding.load(std::memory_order_acquire) == 0;
        if (streamOn) streamIdleSinceNs.store(idleStartNs, std::memory_order_relaxed);
        SpinUntil(HandoffSpinWindow(), [&] { return outstanding.load(std::memory_order_acquire) != 0; });
        {
            std::unique_lock lock(mutex);
            committerWaiting = true;
            wake.wait(lock, [&] { return !items.empty() || closing; });
            committerWaiting = false;
            if (streamOn) {
                const auto foundNs = Stream::NowNs();
                streamIdleSinceNs.store(0, std::memory_order_relaxed);
                streamCounts.idleNs += foundNs - idleStartNs;
                if (emptyAtTop) ++streamCounts.idleWaits;
                streamCounts.busyNs[busyKind] += std::exchange(busyNs, 0);
                // The wake after the push that ended a drain episode belongs to that drain.
                if (streamWake.pending) fileStreamWake(foundNs, idleStartNs);
            }
            if (items.empty()) break;
            front = stopped ? nullptr : &items.front();
        }
        const auto busyStartNs = streamOn ? Stream::NowNs() : 0;
        busyKind = front != nullptr ? static_cast<std::size_t>(front->kind) : 0;
        if (front != nullptr && front->kind != ItemKind::Other && commitRunItems() != 0) commitRun(front);
        else commitFront(front, nullptr);
        if (streamOn) busyNs = Stream::NowNs() - busyStartNs;
    }
    runningCommitters.fetch_sub(1, std::memory_order_acq_rel);
}

// Commits the front item (nothing once the pipeline stopped), then pops and retires it. Inside a
// run it returns the item the run continues with, chosen under the pipeline mutex with the pop.
DrawPipeline::Item* DrawPipeline::commitFront(Item* front, CommitRun* run) {
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_PREP_HELPERS, PARALLEL-COMMIT-PLAN B0): the commit of this
    // item starts now, for every item kind: a helper still building its ticket stops between parts,
    // and one that has not taken its job yet drops it.
    if (front != nullptr) committingSeq.store(front->seq, std::memory_order_release);
    const auto start = run != nullptr && run->started != std::chrono::steady_clock::time_point{} ? std::exchange(run->started, {}) : std::chrono::steady_clock::now();
    auto outcome = guarded([&] {
        if (front == nullptr) return;
        PerformanceTimer itemTiming("Committer.Item");
        // LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS): the committer's walks inside a draw's commit are
        // filed under the depth the draw was enqueued behind.
        struct ItemDepth {
            bool set;
            explicit ItemDepth(const Item& item) : set(GuestMemory::Q0PrepStats() != 0 && item.kind == ItemKind::Draw) {
                if (set) GuestMemory::SetWalkItemDepth(item.enqueueDepth + 1, item.full);
            }
            ~ItemDepth() {
                if (set) GuestMemory::SetWalkItemDepth(0);
            }
        } itemDepth{*front};
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_PREP_HELPERS): the item's push time for its commit
        // (CurrentItemEnqueuedNs), with helpers only.
        struct ItemEnqueued {
            bool set;
            explicit ItemEnqueued(const Item& item) : set(PrepHelpers::On() && item.enqueuedAtNs != 0) {
                if (set) CurrentItemEnqueued::Set(item.enqueuedAtNs);
            }
            ~ItemEnqueued() {
                if (set) CurrentItemEnqueued::Set(0);
            }
        } itemEnqueued{*front};
        if (front->labels.empty()) {
            front->commit();
            return;
        }
        // Labels appended to the item (APS5_EXPERIMENT_LABEL_COALESCE) run right after its commit,
        // in the order they were queued, inside one GPU-mutex hold with it; a commit or label that
        // throws skips the labels after it, as a stopped pipeline skips label items.
        GuestMemory::TagGpuLockSite(front->kind == ItemKind::Label ? GuestMemory::GpuLockSite::Label : GuestMemory::GpuLockSite::Draw);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        front->commit();
        const auto commitLabel = labelCommit.load(std::memory_order_acquire);
        for (const auto& label : front->labels) commitLabel(label);
    });
    const auto finished = std::chrono::steady_clock::now();
    auto spent = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(finished - start).count());
    if (run == nullptr && outcome.fatal != nullptr && !stopped) {
        stopped = true;
        if (!outcome.shutdown) Driver::Get().ReportFailure(outcome.fatal);
    }
    Item done;
    Item* next = nullptr;
    {
        std::unique_lock lock(mutex);
        if (run != nullptr) {
            ++runCommits;
            // The next item stays in place when the front is popped: the producer only appends and
            // only the committer pops.
            if (outcome.fatal == nullptr && --run->left != 0 && finished < run->until && items.size() >= 2 && items[1].kind != ItemKind::Other && GuestMemory::GpuMutex().Waiters() == 0) {
                next = &items[1];
            } else {
                ++runs;
                // The checks the run's commits left to its end belong to its last commit: they run
                // before its item is popped, so a drain returns only after them, and their failure
                // is that commit's failure.
                if (outcome.fatal == nullptr && run->deferred != nullptr) {
                    lock.unlock();
                    outcome = guarded(run->deferred);
                    spent = elapsedNs(start);
                    lock.lock();
                }
            }
            // Reported once the run gave up the GPU mutex (commitRun).
            if (outcome.fatal != nullptr && !stopped) {
                stopped = true;
                if (!outcome.shutdown) run->unreported = outcome.fatal;
            }
        }
        // LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): the [draw] commit time over the [stream] window, and a
        // committed draw's lead from its push to this commit's start, by the depth its push found.
        if (Stream::On()) {
            ++streamCounts.commits;
            streamCounts.commitNs += spent;
            const auto& item = items.front();
            const auto startNs = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count());
            if (front != nullptr && item.kind == ItemKind::Draw && item.enqueuedAtNs != 0 && startNs >= item.enqueuedAtNs) {
                const auto leadNs = startNs - item.enqueuedAtNs;
                constexpr std::array<std::uint64_t, LeadBuckets - 1> bounds{5000, 10000, 20000, 50000, 100000, 200000, 500000};
                const auto bucket = static_cast<std::size_t>(std::upper_bound(bounds.begin(), bounds.end(), leadNs) - bounds.begin());
                const auto depthBucket = std::min<std::size_t>(item.pushDepth, LeadDepthBuckets - 1);
                ++streamCounts.leadDraws;
                if (leadNs >= 20000) ++streamCounts.leadAtLeast20;
                ++streamCounts.leads[bucket];
                ++streamCounts.leadDepths[depthBucket];
                streamCounts.leadDepthNs[depthBucket] += leadNs;
            }
        }
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_PREP_HELPERS): a stopped pipeline pops without committing;
        // the pop still moves CommittingSeq past the item, so its helper job is dropped as stale.
        committingSeq.store(items.front().seq, std::memory_order_release);
        done = std::move(items.front());
        items.pop_front();
        ++commits;
        commitNs += spent;
        if (outcome.failed) ++commitErrors;
        if (outcome.fatal != nullptr && failure == nullptr) failure = outcome.fatal;
        outstanding.fetch_sub(1, std::memory_order_release);
        if (roomWake() == 0) {
            if (idleWaiters != 0) idle.notify_all();
        } else {
            // Only drains wait on `idle` here. Every queued item is popped in turn (a stopped
            // committer pops without committing), so while a producer waits for room the queue
            // passes half depth and that pop wakes it.
            if (idleWaiters != 0 && items.empty()) idle.notify_all();
            if (roomWaiters != 0 && items.size() <= Depth() / 2) room.notify_all();
        }
    }
    retire(std::move(done));
    return next;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_COMMIT_RUNS=1): one GPU-mutex hold for a run of draws and
// labels starting at `front` (see commitRunItems). A failure ends the run; it is reported to the
// driver after the hold, since the report takes the driver's own locks to fail outputs and flips.
void DrawPipeline::commitRun(Item* front) {
    using CurrentRun = HostThreadSlot<CommitRun*, CurrentRunTag>;
    CommitRun run;
    run.left = commitRunItems();
    run.started = std::chrono::steady_clock::now();
    {
        GuestMemory::TagGpuLockSite(front->kind == ItemKind::Label ? GuestMemory::GpuLockSite::Label : GuestMemory::GpuLockSite::Draw);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        run.until = std::chrono::steady_clock::now() + CommitRunWindow;
        struct Current {
            explicit Current(CommitRun& run) { CurrentRun::Set(&run); }
            ~Current() { CurrentRun::Set(nullptr); }
        } current{run};
        while (front != nullptr) front = commitFront(front, &run);
    }
    if (run.unreported != nullptr) Driver::Get().ReportFailure(run.unreported);
}

bool DrawPipeline::DeferToRunEnd(void (*check)()) {
    if (commitRunItems() == 0) return false;
    using CurrentRun = HostThreadSlot<CommitRun*, CurrentRunTag>;
    auto* run = CurrentRun::Get();
    if (run == nullptr) return false;
    run->deferred = check;
    return true;
}

bool DrawPipeline::LabelReapDue() {
    if (commitRunItems() == 0) return true;
    using CurrentRun = HostThreadSlot<CommitRun*, CurrentRunTag>;
    auto* run = CurrentRun::Get();
    return run == nullptr || !std::exchange(run->labelReaped, true);
}

void DrawPipeline::retire(Item&& item) {
    static const bool reaperEnabled = std::getenv("APS5_NO_COMMIT_REAPER") == nullptr;
    if (!reaperEnabled) {
        const Item dropped = std::move(item);
        return;
    }
    constexpr std::size_t ReapBatch = 64;
    std::lock_guard lock(retireMutex);
    retired.push_back(std::move(item));
    if (!reaper.joinable()) reaper = std::thread([this] { reap(); });
    if (retired.size() >= ReapBatch && reaperWaiting) reapWake.notify_one();
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_PREP_HELPERS, PARALLEL-COMMIT-PLAN B0 fix N1): a helper's pin of a
// pipelined draw's closure; the reaper drops it with the retired items, so the closure is never
// destroyed on a helper. Always through the reaper, also under APS5_NO_COMMIT_REAPER (which only
// keeps the items on the committer). The pin is taken by reference and moves only into the list:
// push_back has the strong guarantee (a shared_ptr's move never throws), so on a failure the pin is
// still the caller's, and it is leaked here rather than dropped on the caller, which may be a
// helper (a second failure, of that allocation, ends the process: noexcept). Without a reaper
// thread the pin waits on the list for the one a later retire starts.
void DrawPipeline::RetirePin(std::shared_ptr<void>&& pin) noexcept {
    if (pin == nullptr) return;
    constexpr std::size_t ReapBatch = 64;
    try {
        std::lock_guard lock(retireMutex);
        retiredPins.push_back(std::move(pin));
        if (!reaper.joinable()) reaper = std::thread([this] { reap(); });
        if (retired.size() + retiredPins.size() >= ReapBatch && reaperWaiting) reapWake.notify_one();
    } catch (...) {
        if (pin != nullptr) static_cast<void>(new std::shared_ptr<void>(std::move(pin)));
    }
}

std::uint64_t DrawPipeline::CurrentItemEnqueuedNs() {
    return CurrentItemEnqueued::Get();
}

void DrawPipeline::reap() {
    for (;;) {
        std::deque<Item> batch;
        std::vector<std::shared_ptr<void>> pins;
        {
            std::unique_lock lock(retireMutex);
            reaperWaiting = true;
            reapWake.wait_for(lock, std::chrono::milliseconds(2), [&] { return retired.size() + retiredPins.size() >= 64; });
            reaperWaiting = false;
            if (retired.empty() && retiredPins.empty()) continue;
            batch.swap(retired);
            pins.swap(retiredPins);
        }
        batch.clear();
        pins.clear();
    }
}

void DrawPipeline::Stop() {
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_PREP_HELPERS, PARALLEL-COMMIT-PLAN B0): the helpers stop first,
    // while the committer still runs (they never wait for it); the commits after this close their
    // slots empty and run today's code.
    PrepHelpers::Stop();
    std::thread committer;
    {
        std::lock_guard lock(mutex);
        closing = true;
        committer = std::move(thread);
    }
    wake.notify_all();
    if (!committer.joinable()) return;
    if (committer.get_id() == std::this_thread::get_id()) committer.detach();
    else committer.join();
}

void DrawPipeline::Drain(DrainReason reason, std::uint32_t opcode, std::source_location site) {
    drainRequests[std::min<std::uint32_t>(opcode, 0x100)].fetch_add(1, std::memory_order_relaxed);
    if (!Busy()) {
        // LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): nothing queued, but A1 would drain a queued dispatch
        // item here, so this cuts an open eligible episode. Not a dispatch's own drain (its class
        // decides, NoteDrainClass) nor a flush (PacketFlush notes the flushes that would drain).
        if (Stream::On() && reason != DrainReason::Flush && !(reason == DrainReason::Packet && opcode == 0x15)) NoteWouldDrain(reason, opcode);
        return;
    }
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline drain under the GPU lock (reason " + std::to_string(static_cast<unsigned>(reason)) + ", opcode " + std::to_string(opcode) + ")");
    const auto start = std::chrono::steady_clock::now();
    SpinUntil(HandoffSpinWindow() * 2, [&] { return !Busy(); });
    std::unique_lock lock(mutex);
    // LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): a drain that found items queued opens an episode.
    const bool streamOn = Stream::On();
    if (streamOn) openStreamEpisode(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count()), reason, opcode, site);
    ++idleWaiters;
    idle.wait(lock, [&] { return items.empty(); });
    --idleWaiters;
    const auto index = static_cast<std::size_t>(reason);
    ++drains[index];
    ++drainOpcodes[std::min<std::uint32_t>(opcode, 0x100)];
    const auto waitedNs = elapsedNs(start);
    drainWaitNs[index] += waitedNs;
    if (streamOn && streamEpisode.serial != 0) {
        streamEpisode.returnNs = Stream::NowNs();
        streamEpisode.waitNs = waitedNs;
        if (streamEpisode.thread == std::this_thread::get_id()) streamEpisode.returnGpuWaitNs = Stream::ThreadGpuWait::Get();
    }
    rethrowFailure();
}

std::uint64_t DrawPipeline::DrainRequestsByOpcode(std::uint32_t opcode) {
    return Queue0().drainRequests[std::min<std::uint32_t>(opcode, 0x100)].load(std::memory_order_relaxed);
}

// LOCAL DEBUG (APS5_DEBUG_STREAM_STATS, PARALLEL-COMMIT-PLAN S0): drain episodes (see the header).
// Only queue 0 drains and enqueues in the driver, so one episode is open at a time; a drain that
// finds one still open (a test's other producer) closes it first.
void DrawPipeline::openStreamEpisode(std::uint64_t startNs, DrainReason reason, std::uint32_t opcode, const std::source_location& site) {
    if (streamEpisode.serial != 0) closeStreamEpisode(startNs);
    streamEpisode = StreamEpisode{};
    streamEpisode.serial = ++streamEpisodes;
    streamEpisode.startNs = startNs;
    streamEpisode.file = site.file_name();
    streamEpisode.line = site.line();
    streamEpisode.opcode = opcode;
    streamEpisode.reason = reason;
    streamEpisode.thread = std::this_thread::get_id();
    streamEpisode.gate = streamGateSerial == streamEpisode.serial;
    streamGateSerial = 0;
    streamEpisodeSerial.store(streamEpisode.serial, std::memory_order_relaxed);
    streamCuttable.store(false, std::memory_order_relaxed);
}

void DrawPipeline::closeStreamEpisode(std::uint64_t nowNs) {
    auto& episode = streamEpisode;
    if (episode.serial == 0) return;
    // The committer's idle inside the episode: from the later of the episode's start and the
    // committer's last look for work (0 when it is committing) to now.
    const auto idleSince = streamIdleSinceNs.load(std::memory_order_relaxed);
    const auto from = std::max(idleSince, episode.startNs);
    const auto idleNs = idleSince != 0 && nowNs > from ? nowNs - from : 0;
    const auto serialNs = episode.returnNs != 0 && nowNs > episode.returnNs ? nowNs - episode.returnNs : 0;
    // A cut splits the idle: before it the episode's class, from it on the cut's slot. Queue 0's
    // GPU waits since the drain's return (the queue is empty, so the committer idles through all
    // of them) split the same way, read on the draining thread only; each part's waits and shadow
    // work are capped at its idle, and they do not overlap (the shadow never waits for the GPU).
    const bool cut = episode.cutNs != 0;
    const auto cutIdleNs = cut && idleNs != 0 ? nowNs - std::clamp(episode.cutNs, from, nowNs) : 0;
    const auto ownIdleNs = idleNs - cutIdleNs;
    std::uint64_t ownGpuWaitNs = 0, cutGpuWaitNs = 0;
    if (episode.returnNs != 0 && episode.thread == std::this_thread::get_id()) {
        const auto nowWaitNs = std::max(Stream::ThreadGpuWait::Get(), episode.returnGpuWaitNs);
        const auto cutWaitNs = cut ? std::clamp(episode.cutGpuWaitNs, episode.returnGpuWaitNs, nowWaitNs) : nowWaitNs;
        ownGpuWaitNs = std::min(cutWaitNs - episode.returnGpuWaitNs, ownIdleNs);
        cutGpuWaitNs = std::min(nowWaitNs - cutWaitNs, cutIdleNs);
    }
    // The site's slot: by call site, reason and opcode; the last slot takes the sites that find
    // every other slot taken.
    auto& sites = streamCounts.sites;
    const auto matches = [&](const StreamSite& site) { return site.line == episode.line && site.opcode == episode.opcode && site.reason == episode.reason && (site.file == episode.file || std::strcmp(site.file, episode.file) == 0); };
    std::size_t slot = 0;
    while (slot + 1 < sites.size() && sites[slot].file != nullptr && !matches(sites[slot])) ++slot;
    if (sites[slot].file == nullptr && slot + 1 < sites.size()) {
        sites[slot].file = episode.file;
        sites[slot].line = episode.line;
        sites[slot].opcode = episode.opcode;
        sites[slot].reason = episode.reason;
    }
    auto& site = sites[slot];
    ++site.episodes;
    site.waitNs += episode.waitNs;
    site.idleNs += idleNs;
    site.serialNs += serialNs;
    ++streamCounts.episodes;
    streamCounts.drainIdleNs += idleNs;
    streamCounts.serialNs += serialNs;
    const auto ownSlot = std::min<std::size_t>(episode.dispatchClass, LaterDrainSlot - 1);
    ++streamCounts.classEpisodes[ownSlot];
    streamCounts.classIdleNs[ownSlot] += ownIdleNs;
    streamCounts.classGpuWaitNs[ownSlot] += ownGpuWaitNs;
    streamCounts.classShadowNs[ownSlot] += std::min(episode.shadowNs, ownIdleNs - ownGpuWaitNs);
    if (cut) {
        streamCounts.classIdleNs[episode.cutSlot] += cutIdleNs;
        streamCounts.classGpuWaitNs[episode.cutSlot] += cutGpuWaitNs;
        streamCounts.classShadowNs[episode.cutSlot] += std::min(episode.cutShadowNs, cutIdleNs - cutGpuWaitNs);
        streamCounts.cutIdleNs += cutIdleNs;
    }
    // The committer, idle at this push, takes the item after its wake (fileStreamWake).
    streamWake = idleNs != 0 ? StreamWake{true, slot, cut ? episode.cutSlot : ownSlot, cut, nowNs} : StreamWake{};
    episode = StreamEpisode{};
    streamEpisodeSerial.store(0, std::memory_order_relaxed);
    streamCuttable.store(false, std::memory_order_relaxed);
}

// The open episode's cut, at `at` (or now) and never before its drain returned, into class slot
// `slot`. Under the mutex, with an episode open that can still be cut.
void DrawPipeline::cutStreamEpisode(std::size_t slot, StreamPoint at) {
    auto& episode = streamEpisode;
    episode.cutNs = std::max(at.ns != 0 ? at.ns : Stream::NowNs(), episode.returnNs);
    if (at.ns != 0) episode.cutGpuWaitNs = at.gpuWaitNs;
    else if (episode.thread == std::this_thread::get_id()) episode.cutGpuWaitNs = Stream::ThreadGpuWait::Get();
    episode.cutSlot = slot;
    streamCuttable.store(false, std::memory_order_relaxed);
}

// The committer's wake after the push that closed an episode, from the later of the push and its
// loop's top to the item taken: filed under that episode's site and the class slot its idle ended
// in. Under the mutex.
void DrawPipeline::fileStreamWake(std::uint64_t foundNs, std::uint64_t idleStartNs) {
    const auto wake = std::exchange(streamWake, StreamWake{});
    const auto since = std::max(wake.pushedAtNs, idleStartNs);
    if (foundNs <= since) return;
    const auto ns = foundNs - since;
    streamCounts.sites[wake.site].idleNs += ns;
    streamCounts.classIdleNs[wake.classSlot] += ns;
    if (wake.cut) streamCounts.cutIdleNs += ns;
    streamCounts.drainIdleNs += ns;
    streamCounts.wakeNs += ns;
}

void DrawPipeline::copyQueuedWriters(std::vector<QueuedWriter>& writers) const {
    for (std::size_t position = 0; position < items.size(); ++position) {
        const auto& item = items[position];
        for (const auto& [begin, end] : item.writes) writers.push_back({begin, end, static_cast<std::uint32_t>(position)});
        for (const auto& label : item.labels) {
            const auto [begin, end] = labelRange(label);
            if (end > begin) writers.push_back({begin, end, static_cast<std::uint32_t>(position)});
        }
    }
}

std::size_t DrawPipeline::SnapshotWriters(std::vector<QueuedWriter>& writers, std::uint64_t episode) {
    writers.clear();
    // Nothing queued and no gate writers kept for this episode: nothing to copy, no lock.
    if (!Busy() && (episode == 0 || streamGateEpisode.load(std::memory_order_relaxed) != episode)) return 0;
    std::lock_guard lock(mutex);
    if (episode != 0 && streamGateEpisode.load(std::memory_order_relaxed) == episode && streamEpisode.serial == episode) {
        streamGateEpisode.store(0, std::memory_order_relaxed);
        writers.swap(streamGateWriters);
        return streamGateQueued;
    }
    copyQueuedWriters(writers);
    return items.size();
}

// Right before queue 0's gate drain for a dispatch (PacketExecution): the episode that drain opens
// is marked a gate episode, and takes these writers (a push first drops both).
void DrawPipeline::NoteGateDrain() {
    std::lock_guard lock(mutex);
    streamGateWriters.clear();
    copyQueuedWriters(streamGateWriters);
    streamGateQueued = items.size();
    streamGateSerial = streamEpisodes + 1;
    streamGateEpisode.store(streamGateSerial, std::memory_order_relaxed);
}

void DrawPipeline::NoteDrainClass(std::uint64_t before, std::uint8_t dispatchClass, std::uint64_t shadowNs, StreamPoint drain) {
    // No episode open: the committer was not starved by a drain (producer slow, or a drain's
    // episode already ended at a push of this dispatch's own).
    if (streamEpisodeSerial.load(std::memory_order_relaxed) == 0) return;
    std::lock_guard lock(mutex);
    auto& episode = streamEpisode;
    if (episode.serial == 0) return;
    auto value = static_cast<Stream::DispatchClass>(dispatchClass);
    // Opened before this dispatch, by a drain made for it: the gate drain (A1 skips it, so this
    // dispatch's class decides), or its 0x16 packet's own drain (A1 keeps it: indirect).
    const bool indirectDrain = !episode.gate && episode.opcode == 0x16;
    if (episode.serial != before || (episode.dispatches == 0 && (episode.gate || indirectDrain))) {
        if (episode.serial == before && indirectDrain) value = Stream::DispatchClass::Indirect;
        episode.dispatchClass = static_cast<std::uint8_t>(value);
        streamCuttable.store(Stream::ItemEligible(value), std::memory_order_relaxed);
    } else if (streamCuttable.load(std::memory_order_relaxed) && !Stream::ItemEligible(value)) {
        // A later dispatch A1 would drain for: the cut. Its own shadow work came before it.
        episode.shadowNs += std::exchange(shadowNs, 0);
        ++streamCounts.dispatchCuts;
        cutStreamEpisode(dispatchClass, drain);
    }
    (episode.cutNs != 0 ? episode.cutShadowNs : episode.shadowNs) += shadowNs;
    ++episode.dispatches;
}

void DrawPipeline::NoteWouldDrain(DrainReason reason, std::uint32_t opcode) {
    if (!streamCuttable.load(std::memory_order_relaxed)) return;
    std::lock_guard lock(mutex);
    if (streamEpisode.serial == 0 || !streamCuttable.load(std::memory_order_relaxed)) return;
    ++streamCounts.drainCuts[static_cast<std::size_t>(reason)];
    if (reason == DrainReason::Packet) ++streamCounts.packetCuts[std::min<std::uint32_t>(opcode, 0x100)];
    cutStreamEpisode(LaterDrainSlot, {});
}

// The [stream] line, its sections in the order of PARALLEL-COMMIT-PLAN S0's items (counters in
// StreamStats.hpp and above). The first call starts the first window (what was counted before is
// dropped); later calls print every 10 s, after the pipeline mutex is released.
void DrawPipeline::reportStream(std::unique_lock<std::mutex>& lock) {
    const auto now = std::chrono::steady_clock::now();
    if (streamLastReport == std::chrono::steady_clock::time_point{}) {
        streamLastReport = now;
        streamCounts = StreamCounts{};
        streamWake = StreamWake{};
        lock.unlock();
        static_cast<void>(takeStreamTotals());
        static_cast<void>(Graphics::TakeFlushGuardCounts());
        static_cast<void>(Graphics::TakeNarrowFlushCounts());
        if (PrepHelpers::On()) static_cast<void>(PrepHelpers::TakeWindow(PrepHelpers::Report::Stream));
        return;
    }
    if (now - streamLastReport < std::chrono::seconds(10)) return;
    const auto wallNs = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - streamLastReport).count());
    streamLastReport = now;
    const auto counts = std::exchange(streamCounts, StreamCounts{});
    // A wake still to file names a site slot of the window just taken.
    streamWake = StreamWake{};
    lock.unlock();
    const auto totals = takeStreamTotals();
    const auto count = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    std::string line;
    // Cells of one section are separated by commas, sections by semicolons.
    bool first = true;
    const auto cell = [&] {
        if (!first) line += ',';
        first = false;
    };
    // 1. The committer: busy (by its front's kind) against the [draw] pipelined commit time of the
    // same window (the self-check: within 5%), idle split into drain episodes and producer slow,
    // and busy + idle against the wall.
    const auto busyNs = counts.busyNs[0] + counts.busyNs[1] + counts.busyNs[2];
    const auto drainIdleNs = std::min(counts.drainIdleNs, counts.idleNs);
    const double busyDifference = counts.commitNs == 0 ? 0.0 : 100.0 * (static_cast<double>(busyNs) - static_cast<double>(counts.commitNs)) / static_cast<double>(counts.commitNs);
    const bool busyMatches = counts.commitNs != 0 && busyDifference <= 5.0 && busyDifference >= -5.0;
    appendStream(line, "[stream] (10 s, %.2f s measured) committer: busy %.1f ms (draws %.1f, labels %.1f, other %.1f)", static_cast<double>(wallNs) / 1e9, streamMs(busyNs), streamMs(counts.busyNs[static_cast<std::size_t>(ItemKind::Draw)]), streamMs(counts.busyNs[static_cast<std::size_t>(ItemKind::Label)]), streamMs(counts.busyNs[static_cast<std::size_t>(ItemKind::Other)]));
    appendStream(line, " vs [draw] pipelined commit time %.1f ms in %llu commits: %+.1f%%, %s", streamMs(counts.commitNs), count(counts.commits), busyDifference, busyMatches ? "within 5%" : "OFF BY MORE THAN 5%");
    appendStream(line, "; idle %.1f ms (%llu waits on an empty queue): drains %.1f ms (%.1f ms of it waking after the push)", streamMs(counts.idleNs), count(counts.idleWaits), streamMs(drainIdleNs), streamMs(counts.wakeNs));
    appendStream(line, ", producer slow %.1f ms; busy + idle %.1f ms of %.1f ms", streamMs(counts.idleNs - drainIdleNs), streamMs(busyNs + counts.idleNs), streamMs(wallNs));
    // Drain episodes, queue 0's serial phases, and the sites with the most committer idle.
    appendStream(line, "; drain episodes %llu, serial phases %.1f ms (%.1f us each); sites (episodes, drain wait ms, committer idle ms, serial ms):", count(counts.episodes), streamMs(counts.serialNs), counts.episodes == 0 ? 0.0 : static_cast<double>(counts.serialNs) / 1e3 / static_cast<double>(counts.episodes));
    std::array<std::size_t, StreamSiteSlots> order{};
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return counts.sites[a].idleNs != counts.sites[b].idleNs ? counts.sites[a].idleNs > counts.sites[b].idleNs : counts.sites[a].episodes > counts.sites[b].episodes; });
    first = true;
    for (std::size_t shown = 0; shown < 10; ++shown) {
        const auto& site = counts.sites[order[shown]];
        if (site.episodes == 0) continue;
        cell();
        if (site.file == nullptr) {
            // The overflow slot, in the cells' shape: one name token, reason and opcode "-".
            appendStream(line, " other-sites - - %llu %.1f %.1f %.1f", count(site.episodes), streamMs(site.waitNs), streamMs(site.idleNs), streamMs(site.serialNs));
            continue;
        }
        char opcode[8];
        if (site.opcode >= 0x100) std::snprintf(opcode, sizeof(opcode), "-");
        else std::snprintf(opcode, sizeof(opcode), "0x%02x", site.opcode);
        appendStream(line, " %s:%u %s %s %llu %.1f %.1f %.1f", streamFileName(site.file), site.line, DrainNames[static_cast<std::size_t>(site.reason)], opcode, count(site.episodes), streamMs(site.waitNs), streamMs(site.idleNs), streamMs(site.serialNs));
    }
    // 2. The shadow dispatch classifier: per class the dispatches, the drain episodes their own
    // drains opened, the committer idle filed under the class (its episodes up to a cut, and the
    // cuts it made) and queue 0's GPU waits inside that idle; later-drains is the idle of cuts at
    // a drain or a packet. The reads class by the items queued behind the newest overlapping
    // writer. The A1 gate: item-eligible idle against all drain idle, then net of the queue-0 GPU
    // waits in it (A1 moves them onto the committer) and of the shadow's own work.
    appendStream(line, "; queue-0 dispatches, A1 shadow (class dispatches/episodes/idle ms/queue-0 GPU wait ms):");
    first = true;
    std::uint64_t eligibleNs = 0, eligibleGpuWaitNs = 0, eligibleShadowNs = 0;
    for (std::size_t index = 0; index < Stream::DispatchClasses; ++index) {
        if (Stream::ItemEligible(static_cast<Stream::DispatchClass>(index))) {
            eligibleNs += counts.classIdleNs[index];
            eligibleGpuWaitNs += counts.classGpuWaitNs[index];
            eligibleShadowNs += counts.classShadowNs[index];
        }
        if (totals.dispatches[index] == 0 && counts.classEpisodes[index] == 0 && counts.classIdleNs[index] == 0) continue;
        cell();
        appendStream(line, " %s %llu/%llu/%.1f/%.1f", Stream::DispatchClassNames[index], count(totals.dispatches[index]), count(counts.classEpisodes[index]), streamMs(counts.classIdleNs[index]), streamMs(counts.classGpuWaitNs[index]));
    }
    std::uint64_t drainCuts = 0;
    for (const auto cuts : counts.drainCuts) drainCuts += cuts;
    if (counts.classIdleNs[LaterDrainSlot] != 0 || drainCuts != 0) {
        cell();
        appendStream(line, " later-drains 0/%llu/%.1f/%.1f", count(drainCuts), streamMs(counts.classIdleNs[LaterDrainSlot]), streamMs(counts.classGpuWaitNs[LaterDrainSlot]));
    }
    if (totals.unshadowed != 0) appendStream(line, " (%llu not classified: APS5_PIPELINE_DISPATCH is on)", count(totals.unshadowed));
    if (totals.bdaRoleMismatches != 0) appendStream(line, " (%llu unbounded hits with a BDA ABI but no fault buffer, or the reverse)", count(totals.bdaRoleMismatches));
    appendStream(line, "; reads by items behind the newest writer:");
    first = true;
    for (std::size_t bucket = 0; bucket < Stream::BehindBuckets; ++bucket) {
        cell();
        appendStream(line, " %s %llu", Stream::BehindNames[bucket], count(totals.readsBehind[bucket]));
    }
    const auto netGpuNs = eligibleNs - std::min(eligibleNs, eligibleGpuWaitNs);
    const auto netNs = netGpuNs - std::min(netGpuNs, eligibleShadowNs);
    appendStream(line, "; A1 gate: item-eligible idle %.1f ms = %.1f%% of drain idle", streamMs(eligibleNs), streamPercent(eligibleNs, drainIdleNs));
    appendStream(line, ", net of its queue-0 GPU waits (%.1f ms) %.1f%%, and of the shadow's work in it (%.1f ms) %.1f%%", streamMs(eligibleGpuWaitNs), streamPercent(netGpuNs, drainIdleNs), streamMs(eligibleShadowNs), streamPercent(netNs, drainIdleNs));
    appendStream(line, "; shadow work %.1f ms; eligible episodes cut where A1 would still drain (%.1f ms filed there; cause n): dispatch %llu", streamMs(totals.shadowNs), streamMs(counts.cutIdleNs), count(counts.dispatchCuts));
    for (std::size_t reason = 0; reason < counts.drainCuts.size(); ++reason) {
        if (counts.drainCuts[reason] == 0 || reason == static_cast<std::size_t>(DrainReason::Packet)) continue;
        appendStream(line, ", %s %llu", DrainNames[reason], count(counts.drainCuts[reason]));
    }
    for (std::size_t opcode = 0; opcode < counts.packetCuts.size(); ++opcode) {
        if (counts.packetCuts[opcode] == 0) continue;
        if (opcode >= 0x100) appendStream(line, ", packet-- %llu", count(counts.packetCuts[opcode]));
        else appendStream(line, ", packet-0x%02zx %llu", opcode, count(counts.packetCuts[opcode]));
    }
    // 5. Draw leads, push to commit start, and the depth pushed behind (draws, mean lead).
    static constexpr std::array<const char*, LeadBuckets> leadNames{"<5us", "<10us", "<20us", "<50us", "<100us", "<200us", "<500us", ">=500us"};
    appendStream(line, "; draw leads %llu (>=20us %.1f%%):", count(counts.leadDraws), streamPercent(counts.leadAtLeast20, counts.leadDraws));
    first = true;
    for (std::size_t bucket = 0; bucket < LeadBuckets; ++bucket) {
        cell();
        appendStream(line, " %s %llu", leadNames[bucket], count(counts.leads[bucket]));
    }
    appendStream(line, "; by depth pushed behind (draws/mean lead us):");
    first = true;
    for (std::size_t bucket = 0; bucket < LeadDepthBuckets; ++bucket) {
        if (counts.leadDepths[bucket] == 0) continue;
        cell();
        appendStream(line, " %zu%s %llu/%.1f", bucket, bucket + 1 == LeadDepthBuckets ? "+" : "", count(counts.leadDepths[bucket]), static_cast<double>(counts.leadDepthNs[bucket]) / 1e3 / static_cast<double>(counts.leadDepths[bucket]));
    }
    // 3. The presenter's second hold and its parts; the late (third-hold) presents apart.
    appendStream(line, "; presenter: %llu presents, lock wait %.1f ms, second hold %.1f ms: PresentDisplayBuffer %.1f ms, QueuePresent %.1f ms (%.0f%% of the hold); late QueuePresent %llu, %.1f ms", count(totals.presents), streamMs(totals.holdLockWaitNs), streamMs(totals.holdNs), streamMs(totals.presentBufferNs), streamMs(totals.queuePresentNs), streamPercent(totals.queuePresentNs, totals.holdNs), count(totals.latePresents), streamMs(totals.lateQueuePresentNs));
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_PRESENT_QUEUE=1, PARALLEL-COMMIT-PLAN A3): the presents
    // issued after the hold (their PreparePresent is the QueuePresent above).
    if (Graphics::QueueGate::PresentQueueMode() != 0) appendStream(line, "; after the hold: IssuePresent %llu, %.1f ms", count(totals.issuedPresents), streamMs(totals.issuePresentNs));
    // Contended GpuMutex waits by waiter and holder role (counted under the holder seen first, the
    // time split over every holder the wait sat behind), and the committer's wait behind the
    // presenter per present (the A3 gate).
    const auto committer = static_cast<std::size_t>(GuestMemory::StreamRole::Committer);
    const auto presenter = static_cast<std::size_t>(GuestMemory::StreamRole::Presenter);
    appendStream(line, "; GpuMutex waits, committer behind the presenter %.2f ms per present (waiter<holder waits begun behind it/ms behind it):", totals.presents == 0 ? 0.0 : streamMs(totals.gpuWaitNs[committer][presenter]) / static_cast<double>(totals.presents));
    first = true;
    for (std::size_t waiter = 0; waiter < Stream::Roles; ++waiter) {
        for (std::size_t holder = 0; holder < Stream::Roles; ++holder) {
            if (totals.gpuWaits[waiter][holder] == 0 && totals.gpuWaitNs[waiter][holder] == 0) continue;
            cell();
            appendStream(line, " %s<%s %llu/%.1f", Stream::RoleNames[waiter], Stream::RoleNames[holder], count(totals.gpuWaits[waiter][holder]), streamMs(totals.gpuWaitNs[waiter][holder]));
        }
    }
    // 4. Recorder::Submit by role and hold site, and the committer's vkQueueSubmit share (the A4
    // gate).
    std::uint64_t committerSubmitNs = 0, committerQueueSubmitNs = 0;
    for (std::size_t site = 0; site < Stream::Sites; ++site) {
        committerSubmitNs += totals.submitNs[committer][site];
        committerQueueSubmitNs += totals.queueSubmitNs[committer][site];
    }
    appendStream(line, "; Recorder::Submit, committer vkQueueSubmit %.1f ms = %.0f%% of its %.1f ms (role/site n, Submit ms, vkEndCommandBuffer ms, vkQueueSubmit ms):", streamMs(committerQueueSubmitNs), streamPercent(committerQueueSubmitNs, committerSubmitNs), streamMs(committerSubmitNs));
    first = true;
    for (std::size_t role = 0; role < Stream::Roles; ++role) {
        for (std::size_t site = 0; site < Stream::Sites; ++site) {
            if (totals.submits[role][site] == 0) continue;
            cell();
            appendStream(line, " %s/%s %llu %.1f %.1f %.1f", Stream::RoleNames[role], GuestMemory::GpuLockSiteName(static_cast<GuestMemory::GpuLockSite>(site)), count(totals.submits[role][site]), streamMs(totals.submitNs[role][site]), streamMs(totals.endNs[role][site]), streamMs(totals.queueSubmitNs[role][site]));
        }
    }
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_SUBMIT_THREAD, PARALLEL-COMMIT-PLAN A4): the vkQueueSubmit
    // column above is the seal then; the submitter's own submits, and the waits on it.
    if (Graphics::QueueGate::SubmitThreadMode() != 0) {
        appendStream(line, "; submitter (APS5_EXPERIMENT_SUBMIT_THREAD=%d): vkQueueSubmit %llu, %.1f ms; awaits that waited %llu, %.1f ms; drains that waited %llu, %.1f ms", Graphics::QueueGate::SubmitThreadMode(), count(totals.submitterSubmits), streamMs(totals.submitterSubmitNs), count(totals.waitedAwaits), streamMs(totals.awaitNs), count(totals.waitedDrains), streamMs(totals.drainNs));
        appendStream(line, ", polls of unissued batches %llu", count(totals.unissuedPolls));
    }
    // 6. CompareSpansFrom by span count and total bytes, and by how it ran.
    appendStream(line, "; CompareSpansFrom (spans/bytes n, ms, committer ms):");
    first = true;
    for (std::size_t spans = 0; spans < Stream::CompareSpanBuckets; ++spans) {
        for (std::size_t bytes = 0; bytes < Stream::CompareByteBuckets; ++bytes) {
            if (totals.compares[spans][bytes] == 0) continue;
            cell();
            appendStream(line, " %s/%s %llu %.1f %.1f", Stream::CompareSpanNames[spans], Stream::CompareByteNames[bytes], count(totals.compares[spans][bytes]), streamMs(totals.compareNs[spans][bytes]), streamMs(totals.compareCommitterNs[spans][bytes]));
        }
    }
    appendStream(line, "; by mode (n, ms):");
    first = true;
    for (std::size_t mode = 0; mode < Stream::CompareModes; ++mode) {
        cell();
        appendStream(line, " %s %llu %.1f", Stream::CompareModeNames[mode], count(totals.compareModes[mode]), streamMs(totals.compareModeNs[mode]));
    }
    // 7. The write tracker's lock per role.
    appendStream(line, "; tracker lock (role acquisitions/waits, wait ms):");
    first = true;
    for (std::size_t role = 0; role < Stream::Roles; ++role) {
        if (totals.trackerAcquisitions[role] == 0) continue;
        cell();
        appendStream(line, " %s %llu/%llu %.1f", Stream::RoleNames[role], count(totals.trackerAcquisitions[role]), count(totals.trackerWaits[role]), streamMs(totals.trackerWaitNs[role]));
    }
    // 8. LOCAL EXPERIMENT (APS5_EXPERIMENT_LOCAL_CUTS=2, PARALLEL-COMMIT-PLAN C1): the checks each
    // local cut passed, so a verify run shows the cuts were exercised.
    if (Graphics::LocalCuts() == 2) {
        auto& cuts = Graphics::localCutCounts;
        const auto checked = [&](Graphics::LocalCut cut) { return count(Stream::Take(cuts.checked[static_cast<std::size_t>(cut)])); };
        // Two pieces, so neither reaches appendStream's 191-character cut.
        appendStream(line, "; local cuts verified: committed flag %llu commits, unsaved layers %llu calls (%llu lists filled)", checked(Graphics::LocalCut::Committed), checked(Graphics::LocalCut::UnsavedLayers), count(Stream::Take(cuts.layersFilled)));
        appendStream(line, ", device reads %llu template draws (%llu lists kept), disagreements %llu", checked(Graphics::LocalCut::DeviceReads), count(Stream::Take(cuts.readsKept)), count(Stream::Take(cuts.disagreed)));
    }
    // 9. LOCAL EXPERIMENT (APS5_EXPERIMENT_PREP_HELPERS, PARALLEL-COMMIT-PLAN B0): the helper slots as
    // the committer closed them (with a ticket, or empty: today's code ran), the tickets a commit
    // that started first left unused (and their parts), and the leads: a used ticket's from ready to
    // its commit's start, and the closed slots' items' from the push to the commit's start.
    if (PrepHelpers::On()) {
        const auto prep = PrepHelpers::TakeWindow(PrepHelpers::Report::Stream);
        using Counter = PrepHelpers::Counter;
        const auto get = [&](Counter which) { return count(PrepHelpers::Get(prep, which)); };
        const auto averageUs = [&](Counter ns, Counter n) { return PrepHelpers::Get(prep, n) == 0 ? 0.0 : static_cast<double>(PrepHelpers::Get(prep, ns)) / 1e3 / static_cast<double>(PrepHelpers::Get(prep, n)); };
        const auto closedReady = PrepHelpers::Get(prep, Counter::ClosedReady);
        appendStream(line, "; prep helpers (%u): jobs %llu, slots closed ready %llu, empty %llu (%.1f%% ready), tickets wasted %llu (%llu parts)", PrepHelpers::Threads(), get(Counter::Published), count(closedReady), get(Counter::ClosedEmpty), streamPercent(closedReady, closedReady + PrepHelpers::Get(prep, Counter::ClosedEmpty)), get(Counter::Wasted), get(Counter::WastedParts));
        appendStream(line, ", lead ready to commit %.1f us, push to commit %.1f us ready, %.1f us empty", averageUs(Counter::ReadyLeadNs, Counter::TimedReady), averageUs(Counter::ReadyItemLeadNs, Counter::TimedReady), averageUs(Counter::EmptyItemLeadNs, Counter::TimedEmpty));
    }
    // 10. LOCAL EXPERIMENT (PARALLEL-COMMIT-PLAN A3): the queue gate, cumulative: operations checked,
    // device-wide waits and violations; presents by APS5_EXPERIMENT_PRESENT_QUEUE arm and outcome
    // (success/suboptimal/out of date/failed).
    {
        const auto gate = Graphics::QueueGate::CountsSoFar();
        if (gate.ops != 0 || gate.deviceWaits != 0) appendStream(line, "; queue gate: %llu ops, %llu device-wide waits, violations %llu unheld/%llu overlapped/%llu misordered/%llu ordering", count(gate.ops), count(gate.deviceWaits), count(gate.unheld), count(gate.overlapped), count(gate.misordered), count(gate.ordering));
        if (gate.seals != 0) appendStream(line, ", %llu seals, %llu failed submits", count(gate.seals), count(gate.submitFailures));
        appendStream(line, "; presents by arm (ok/suboptimal/out of date/failed):");
        for (std::size_t arm = 0; arm < gate.presents.size(); ++arm) {
            const auto& outcomes = gate.presents[arm];
            if (outcomes[0] + outcomes[1] + outcomes[2] + outcomes[3] == 0) continue;
            appendStream(line, " %zu %llu/%llu/%llu/%llu", arm, count(outcomes[0]), count(outcomes[1]), count(outcomes[2]), count(outcomes[3]));
        }
    }
    // 11. LOCAL DEBUG (PARALLEL-COMMIT-PLAN section 5, FlushGuard.hpp): StorageTexture::FlushPending's
    // hand-over guard waits, by wait (sync: the writer is the open batch; finish: in flight) and by
    // F1's class (the first blocking condition; stamped: F1 would skip the wait), with their wall,
    // GPU-wait and queue-0 GPU-wait ms; would-skip summed over both waits; how the waited-for batch
    // matched; and every condition that held per wait (no-import is looked for only when nothing
    // else held).
    {
        const auto guard = Graphics::TakeFlushGuardCounts();
        appendStream(line, "; flush guard waits (wait/class n, wall ms, GPU ms, queue-0 GPU ms):");
        first = true;
        std::uint64_t skipWaits = 0, skipWallNs = 0, skipGpuNs = 0, skipQueue0Ns = 0;
        for (std::size_t kind = 0; kind < Graphics::FlushWaitKinds; ++kind) {
            for (std::size_t index = 0; index < Graphics::FlushWaitClasses; ++index) {
                if (guard.waits[kind][index] == 0) continue;
                cell();
                appendStream(line, " %s/%s %llu %.1f %.1f %.1f", Graphics::FlushWaitKindNames[kind], Graphics::FlushWaitClassNames[index], count(guard.waits[kind][index]), streamMs(guard.wallNs[kind][index]), streamMs(guard.gpuNs[kind][index]), streamMs(guard.queue0GpuNs[kind][index]));
            }
            const auto stamped = static_cast<std::size_t>(Graphics::FlushWaitClass::Stamped);
            skipWaits += guard.waits[kind][stamped];
            skipWallNs += guard.wallNs[kind][stamped];
            skipGpuNs += guard.gpuNs[kind][stamped];
            skipQueue0Ns += guard.queue0GpuNs[kind][stamped];
        }
        std::uint64_t allGpuNs = 0, allQueue0Ns = 0;
        for (std::size_t kind = 0; kind < Graphics::FlushWaitKinds; ++kind) {
            for (std::size_t index = 0; index < Graphics::FlushWaitClasses; ++index) {
                allGpuNs += guard.gpuNs[kind][index];
                allQueue0Ns += guard.queue0GpuNs[kind][index];
            }
        }
        appendStream(line, "; F1 would skip %llu waits, %.1f ms wall, %.1f ms GPU (%.1f%%), %.1f ms queue-0 GPU (%.1f%%)", count(skipWaits), streamMs(skipWallNs), streamMs(skipGpuNs), streamPercent(skipGpuNs, allGpuNs), streamMs(skipQueue0Ns), streamPercent(skipQueue0Ns, allQueue0Ns));
        appendStream(line, "; waited-for batch matched by (note/lease/both): sync %llu/%llu/%llu, finish %llu/%llu/%llu", count(guard.matches[0][0]), count(guard.matches[0][1]), count(guard.matches[0][2]), count(guard.matches[1][0]), count(guard.matches[1][1]), count(guard.matches[1][2]));
        appendStream(line, "; conditions held (n):");
        first = true;
        for (std::size_t index = 0; index < Graphics::FlushWaitClasses; ++index) {
            cell();
            appendStream(line, " %s %llu", index == 0 ? "none" : Graphics::FlushWaitClassNames[index], count(guard.conditions[index]));
        }
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_NARROW_FLUSH_SYNC, F1): the waits it skipped (=1; its
        // would-skip above then counts no waits), the completion stores that landed behind a
        // skipped write-back (must be 0), Refresh's lease waits, and =2's checks.
        if (const int narrowMode = Graphics::NarrowFlushSync(); narrowMode != 0) {
            const auto narrow = Graphics::TakeNarrowFlushCounts();
            appendStream(line, "; F1 (APS5_EXPERIMENT_NARROW_FLUSH_SYNC=%d): skipped %llu, completion stores behind a skipped write-back %llu (must be 0)", narrowMode, count(narrow.skipped), count(narrow.storesBehindSkip));
            appendStream(line, ", refresh lease waits %llu, %.1f ms", count(narrow.refreshLeaseWaits), streamMs(narrow.refreshLeaseNs));
            if (narrowMode == 2) appendStream(line, ", verified %llu: completion stores during the wait %llu, keep decisions differed %llu, raced by a CPU store %llu", count(narrow.verified), count(narrow.storesDuringWait), count(narrow.keepDiffered), count(narrow.keepRaced));
        }
    }
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
}

std::optional<std::uint64_t> DrawPipeline::PendingLabel(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || bytes > 8) return std::nullopt;
    const auto end = address + bytes;
    std::lock_guard lock(mutex);
    for (auto item = items.rbegin(); item != items.rend(); ++item) {
        // Appended labels run after the item's own commit, the newest last: newest first, they are
        // the item's newest writes.
        for (auto label = item->labels.rbegin(); label != item->labels.rend(); ++label) {
            const auto [begin, limit] = labelRange(*label);
            if (address >= limit || begin >= end) continue;
            if (address < begin || end > limit) return std::nullopt;
            std::uint64_t value = 0;
            std::memcpy(&value, label->bytes.data() + (address - begin), bytes);
            return value;
        }
        const bool overlaps = std::any_of(item->writes.begin(), item->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
        if (!overlaps) continue;
        if (item->labelBytes.empty() || address < item->labelAddress || end > item->labelAddress + item->labelBytes.size()) return std::nullopt;
        std::uint64_t value = 0;
        std::memcpy(&value, item->labelBytes.data() + (address - item->labelAddress), bytes);
        return value;
    }
    return std::nullopt;
}

bool DrawPipeline::OverlapsAny(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    if (ranges.empty()) return false;
    auto low = ranges.front().first;
    auto high = ranges.front().second;
    for (const auto& [begin, end] : ranges) {
        low = std::min(low, begin);
        high = std::max(high, end);
    }
    const auto overlaps = [&](std::uint64_t begin, std::uint64_t limit) {
        if (begin >= high || limit <= low) return false;
        return std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return range.first < limit && begin < range.second; });
    };
    std::lock_guard lock(mutex);
    for (const auto& item : items) {
        for (const auto& [begin, limit] : item.writes) {
            if (overlaps(begin, limit)) return true;
        }
        for (const auto& label : item.labels) {
            const auto [begin, limit] = labelRange(label);
            if (overlaps(begin, limit)) return true;
        }
    }
    return false;
}

bool DrawPipeline::Overlaps(std::uint64_t address, std::size_t bytes) {
    const auto end = address + bytes;
    std::lock_guard lock(mutex);
    for (const auto& item : items) {
        for (const auto& [begin, limit] : item.writes) {
            if (address < limit && begin < end) return true;
        }
        for (const auto& label : item.labels) {
            const auto [begin, limit] = labelRange(label);
            if (address < limit && begin < end) return true;
        }
    }
    return false;
}

bool DrawPipeline::AppendLabel(LabelPart& label) {
    std::lock_guard lock(mutex);
    rethrowFailure();
    // With two or more items queued the newest is not the front, the only item the committer ever
    // starts, so the label runs right after the newest item's commit and before anything queued
    // later, as an item of its own would.
    if (closing || items.size() < 2 || labelCommit.load(std::memory_order_relaxed) == nullptr) return false;
    auto& newest = items.back();
    if (newest.kind == ItemKind::Other || newest.labels.size() >= MaxAppendedLabels) return false;
    newest.labels.push_back(std::move(label));
    ++labelsAppended;
    return true;
}

}
