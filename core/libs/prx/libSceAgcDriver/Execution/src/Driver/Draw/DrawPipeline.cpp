#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include <algorithm>
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
}

unsigned DrawPipeline::Committers() {
    return runningCommitters.load(std::memory_order_acquire);
}

void DrawPipeline::rethrowFailure() {
    if (failure == nullptr) return;
    auto error = std::exchange(failure, nullptr);
    std::rethrow_exception(error);
}

void DrawPipeline::Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes, ItemKind kind) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr || std::getenv("APS5_PIPELINE_REPORT") != nullptr;
    std::unique_lock lock(mutex);
    rethrowFailure();
    if (closing) {
        lock.unlock();
        commit();
        return;
    }
    // LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS): the depth this item is enqueued behind and its room
    // wait, for the [q0prep] line; the depth also rides with the item for the committer's walks.
    const auto enqueueDepth = items.size();
    const bool prepStatsOn = GuestMemory::Q0PrepStats() != 0;
    if (prepStatsOn && kind == ItemKind::Draw) {
        ++prepStats.draws;
        ++prepStats.drawDepths[std::min(enqueueDepth, PrepDepthBuckets - 1)];
    } else if (prepStatsOn && kind == ItemKind::Label) {
        ++prepStats.labels;
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
    items.push_back({std::move(commit), std::move(writes), labelAddress, std::move(labelBytes), kind, {}, static_cast<std::uint32_t>(enqueueDepth)});
    outstanding.fetch_add(1, std::memory_order_release);
    if (!thread.joinable()) {
        thread = std::thread([this] { run(); });
        runningCommitters.fetch_add(1, std::memory_order_acq_rel);
    }
    if (committerWaiting) wake.notify_one();
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport >= std::chrono::seconds(10)) report(now);
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
    for (;;) {
        Item* front = nullptr;
        SpinUntil(HandoffSpinWindow(), [&] { return outstanding.load(std::memory_order_acquire) != 0; });
        {
            std::unique_lock lock(mutex);
            committerWaiting = true;
            wake.wait(lock, [&] { return !items.empty() || closing; });
            committerWaiting = false;
            if (items.empty()) break;
            front = stopped ? nullptr : &items.front();
        }
        if (front != nullptr && front->kind != ItemKind::Other && commitRunItems() != 0) commitRun(front);
        else commitFront(front, nullptr);
    }
    runningCommitters.fetch_sub(1, std::memory_order_acq_rel);
}

// Commits the front item (nothing once the pipeline stopped), then pops and retires it. Inside a
// run it returns the item the run continues with, chosen under the pipeline mutex with the pop.
DrawPipeline::Item* DrawPipeline::commitFront(Item* front, CommitRun* run) {
    const auto start = run != nullptr && run->started != std::chrono::steady_clock::time_point{} ? std::exchange(run->started, {}) : std::chrono::steady_clock::now();
    auto outcome = guarded([&] {
        if (front == nullptr) return;
        PerformanceTimer itemTiming("Committer.Item");
        // LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS): the committer's walks inside a draw's commit are
        // filed under the depth the draw was enqueued behind.
        struct ItemDepth {
            bool set;
            explicit ItemDepth(const Item& item) : set(GuestMemory::Q0PrepStats() != 0 && item.kind == ItemKind::Draw) {
                if (set) GuestMemory::SetWalkItemDepth(item.enqueueDepth + 1);
            }
            ~ItemDepth() {
                if (set) GuestMemory::SetWalkItemDepth(0);
            }
        } itemDepth{*front};
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

void DrawPipeline::reap() {
    for (;;) {
        std::deque<Item> batch;
        {
            std::unique_lock lock(retireMutex);
            reaperWaiting = true;
            reapWake.wait_for(lock, std::chrono::milliseconds(2), [&] { return retired.size() >= 64; });
            reaperWaiting = false;
            if (retired.empty()) continue;
            batch.swap(retired);
        }
        batch.clear();
    }
}

void DrawPipeline::Stop() {
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

void DrawPipeline::Drain(DrainReason reason, std::uint32_t opcode) {
    drainRequests[std::min<std::uint32_t>(opcode, 0x100)].fetch_add(1, std::memory_order_relaxed);
    if (!Busy()) return;
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline drain under the GPU lock (reason " + std::to_string(static_cast<unsigned>(reason)) + ", opcode " + std::to_string(opcode) + ")");
    const auto start = std::chrono::steady_clock::now();
    SpinUntil(HandoffSpinWindow() * 2, [&] { return !Busy(); });
    std::unique_lock lock(mutex);
    ++idleWaiters;
    idle.wait(lock, [&] { return items.empty(); });
    --idleWaiters;
    const auto index = static_cast<std::size_t>(reason);
    ++drains[index];
    ++drainOpcodes[std::min<std::uint32_t>(opcode, 0x100)];
    drainWaitNs[index] += elapsedNs(start);
    rethrowFailure();
}

std::uint64_t DrawPipeline::DrainRequestsByOpcode(std::uint32_t opcode) {
    return Queue0().drainRequests[std::min<std::uint32_t>(opcode, 0x100)].load(std::memory_order_relaxed);
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
