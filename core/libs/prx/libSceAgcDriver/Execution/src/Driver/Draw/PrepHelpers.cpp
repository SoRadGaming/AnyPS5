#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/PrepHelpers.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/StreamStats.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawTicket.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <utility>
#include <vector>

// LOCAL EXPERIMENT (APS5_EXPERIMENT_PREP_HELPERS=N, PARALLEL-COMMIT-PLAN B0): the prep helpers (see
// PrepHelpers.hpp). One PrepQueue for queue 0's pipeline, created at the first use and never
// destroyed (its threads are joined by Stop; one still running at exit is not destroyed with it).
namespace AgcDriver::DriverDetail {

namespace {

constexpr unsigned MostHelpers = 16;

// Counters, each on its own cache line: queue 0, the helpers and the committer add to them at once.
struct alignas(64) PaddedCounter {
    std::atomic<std::uint64_t> value{0};
};
constinit std::array<PaddedCounter, PrepHelpers::Counters> prepCounters{};

void add(PrepHelpers::Counter which, std::uint64_t value = 1) {
    prepCounters[static_cast<std::size_t>(which)].value.fetch_add(value, std::memory_order_relaxed);
}

struct PrepQueue {
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<PrepJob> jobs;
    // The queue's size, for the helpers' spin before they lock (written under the mutex).
    std::atomic<std::size_t> queued{0};
    std::vector<std::thread> threads;
    std::uint32_t waiting = 0;
    bool started = false;
    bool stopped = false;
    bool frozen = false;
    // Read by the helpers' budgets without the mutex.
    std::atomic<bool> stopping{false};
    std::atomic<unsigned> running{0};
};

PrepQueue& prepQueue() {
    static auto* queue = new PrepQueue();
    return *queue;
}

// At most the pipeline's depth plus 8 jobs: a job further back would wait for many commits.
std::size_t prepCapacity() {
    static const std::size_t capacity = DrawPipeline::Depth() + 8;
    return capacity;
}

// APS5_EXPERIMENT_PREP_HELPERS_STRESS: a random delay, on any helper thread (a shared xorshift
// state; its races only make it more random). One time in four a spin of up to about 200 us, one
// time in sixteen a 1 ms sleep instead.
void stressDelay() {
    static std::atomic<std::uint32_t> state{0x9e3779b9u};
    auto value = state.load(std::memory_order_relaxed);
    value ^= value << 13u;
    value ^= value >> 17u;
    value ^= value << 5u;
    state.store(value, std::memory_order_relaxed);
    if ((value & 15u) == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } else if ((value & 3u) == 0) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds((value >> 8u) % 200u);
        while (std::chrono::steady_clock::now() < until) SpinPause();
    }
}

// Hands a pin to the reaper; should that fail (no memory), the pin is leaked rather than dropped
// here, since this may be a helper.
void retirePin(std::shared_ptr<void>&& pin) {
    if (pin == nullptr) return;
    try {
        DrawPipeline::Queue0().RetirePin(std::move(pin));
    } catch (...) {
        if (pin != nullptr) static_cast<void>(new std::shared_ptr<void>(std::move(pin)));
    }
}

// One job on a helper: build its ticket unless its commit started, offer it to the slot, retire the
// pin. Every exception ends here: a helper must not die, and a failure only costs the ticket.
void work(PrepJob& job) {
    struct Retire {
        std::shared_ptr<void>& pin;
        ~Retire() { retirePin(std::move(pin)); }
    } retire{job.pin};
    using Counter = PrepHelpers::Counter;
    const HelperBudget budget{&DrawPipeline::Queue0(), job.seq, job.depth};
    if (PrepHelpers::Stress()) stressDelay();
    if (budget.Stop()) {
        add(PrepHelpers::Stopping() ? Counter::AtStop : Counter::Stale);
        return;
    }
    const bool timed = PrepHelpers::Timed();
    const auto pickupNs = timed ? Stream::NowNs() : 0;
    add(Counter::Builds);
    if (timed && job.enqueuedAtNs != 0 && pickupNs > job.enqueuedAtNs) add(Counter::PickupNs, pickupNs - job.enqueuedAtNs);
    std::unique_ptr<Graphics::DrawTicket> built;
    try {
        built = job.build(job.inputs, budget);
    } catch (...) {
        built.reset();
    }
    if (budget.cut) add(Counter::Cut);
    if (built == nullptr) {
        add(Counter::Empty);
        return;
    }
    if (PrepHelpers::Stress()) stressDelay();
    const auto readyNs = timed ? Stream::NowNs() : 0;
    if (timed) {
        job.slots->readyAtNs.store(readyNs, std::memory_order_relaxed);
        add(Counter::BuildNs, readyNs - pickupNs);
    }
    // The single transfer: release pairs with the commit's acquire exchange.
    Graphics::DrawTicket* expected = nullptr;
    if (job.slots->cheap.compare_exchange_strong(expected, built.get(), std::memory_order_release, std::memory_order_relaxed)) {
        built.release();
        add(Counter::Ready);
        if (timed && job.enqueuedAtNs != 0 && readyNs > job.enqueuedAtNs) add(Counter::PushToReadyNs, readyNs - job.enqueuedAtNs);
        return;
    }
    // The commit started first and closed the slot: ~DrawTicket hands every strong reference the
    // ticket holds to the ticket release list (the committer settles it under GpuMutex). Counted
    // once that is done.
    const auto parts = static_cast<std::uint64_t>(std::popcount(built->parts));
    built.reset();
    add(Counter::WastedParts, parts);
    add(Counter::Wasted);
}

void helperMain(unsigned index) {
    char name[16];
    std::snprintf(name, sizeof(name), "agcprep%u", index);
    PerformanceTrace::BindThread(name);
    if (Stream::On()) GuestMemory::SetStreamRole(GuestMemory::StreamRole::Helper);
    GuestMemory::SetPrepHelperThread(true);
    auto& queue = prepQueue();
    {
        // Helpers never take GpuMutex (an abort under APS5_ASSERT_GPU_LOCK) for their whole life.
        const GuestMemory::PrepareScope scope;
        const auto& pipeline = DrawPipeline::Queue0();
        std::vector<std::shared_ptr<void>> stale;
        for (;;) {
            SpinUntil(HandoffSpinWindow(), [&] { return queue.queued.load(std::memory_order_acquire) != 0 || queue.stopping.load(std::memory_order_acquire); });
            PrepJob job;
            bool taken = false;
            {
                std::unique_lock lock(queue.mutex);
                ++queue.waiting;
                queue.wake.wait(lock, [&] { return queue.stopped || (!queue.frozen && !queue.jobs.empty()); });
                --queue.waiting;
                if (queue.stopped) break;
                // The oldest job whose commit has not started; the ones before it are stale.
                const auto committing = pipeline.CommittingSeq();
                while (!queue.jobs.empty() && queue.jobs.front().seq <= committing) {
                    stale.push_back(std::move(queue.jobs.front().pin));
                    queue.jobs.pop_front();
                }
                if (!queue.jobs.empty()) {
                    job = std::move(queue.jobs.front());
                    queue.jobs.pop_front();
                    taken = true;
                }
                queue.queued.store(queue.jobs.size(), std::memory_order_relaxed);
            }
            if (!stale.empty()) {
                add(PrepHelpers::Counter::Stale, stale.size());
                for (auto& pin : stale) retirePin(std::move(pin));
                stale.clear();
            }
            if (taken) work(job);
        }
    }
    GuestMemory::SetPrepHelperThread(false);
    queue.running.fetch_sub(1, std::memory_order_acq_rel);
}

// Under the queue's mutex.
void startLocked(PrepQueue& queue) {
    if (queue.started || queue.stopped) return;
    queue.started = true;
    const auto count = PrepHelpers::Threads();
    for (unsigned index = 0; index < count; ++index) {
        queue.threads.emplace_back([index] { helperMain(index); });
        queue.running.fetch_add(1, std::memory_order_acq_rel);
    }
}

}

bool HelperBudget::helperStop() const {
    if (PrepHelpers::Stress()) stressDelay();
    const bool stop = pipeline->CommittingSeq() >= seq || PrepHelpers::Stopping();
    if (stop) cut = true;
    return stop;
}

PrepSlots::~PrepSlots() {
    if (PrepHelpers::On()) GuestMemory::NoteDestroyedOnHelper(GuestMemory::HelperDestroyed::Closure);
    // A ticket the commit never took (the item was popped without a commit): ~DrawTicket hands its
    // references to the release list, as a never-committed ticket did before.
    auto* left = cheap.load(std::memory_order_acquire);
    if (left != nullptr && left != ClosedSentinel()) delete left;
}

Graphics::DrawTicket* PrepSlots::takeClaimed() {
    using Counter = PrepHelpers::Counter;
    // The single transfer: acquire pairs with the helper's release compare-exchange; a helper that
    // finishes later finds the slot closed.
    auto* got = cheap.exchange(ClosedSentinel(), std::memory_order_acq_rel);
    const bool ready = got != nullptr && got != ClosedSentinel();
    if (ready) taken.reset(got);
    add(ready ? Counter::ClosedReady : Counter::ClosedEmpty);
    if (PrepHelpers::Timed()) {
        const auto nowNs = Stream::NowNs();
        const auto itemNs = DrawPipeline::CurrentItemEnqueuedNs();
        const bool item = itemNs != 0 && nowNs >= itemNs;
        if (ready) {
            const auto readyNs = readyAtNs.load(std::memory_order_relaxed);
            if (item && readyNs != 0 && nowNs >= readyNs) {
                const auto leadNs = nowNs - readyNs;
                add(leadNs < 5000 ? Counter::ReadyLeadUnder5 : leadNs < 20000 ? Counter::ReadyLeadUnder20 : leadNs < 100000 ? Counter::ReadyLeadUnder100 : Counter::ReadyLeadOver100);
                add(Counter::ReadyLeadNs, leadNs);
                add(Counter::ReadyItemLeadNs, nowNs - itemNs);
                add(Counter::TimedReady);
            }
        } else if (item) {
            add(Counter::EmptyItemLeadNs, nowNs - itemNs);
            add(Counter::TimedEmpty);
        }
    }
    return taken.get();
}

namespace PrepHelpers {

unsigned Threads() {
    static const unsigned threads = [] {
        const char* text = std::getenv("APS5_EXPERIMENT_PREP_HELPERS");
        if (text == nullptr) return 0u;
        const auto value = *text == '\0' ? 3l : std::strtol(text, nullptr, 10);
        if (value <= 0) return 0u;
        if (Graphics::Q0Ticket() == 0) {
            std::fprintf(stderr, "[q0prep] APS5_EXPERIMENT_PREP_HELPERS needs APS5_EXPERIMENT_Q0_TICKET: no prep helpers\n");
            return 0u;
        }
        return static_cast<unsigned>(std::min<long>(value, MostHelpers));
    }();
    return threads;
}

bool Stress() {
    static const bool stress = On() && std::getenv("APS5_EXPERIMENT_PREP_HELPERS_STRESS") != nullptr;
    return stress;
}

bool Timed() {
    static const bool timed = On() && (GuestMemory::Q0PrepStats() != 0 || Stream::On());
    return timed;
}

bool Stopping() {
    return prepQueue().stopping.load(std::memory_order_acquire);
}

void Start() {
    if (!On()) return;
    auto& queue = prepQueue();
    std::lock_guard lock(queue.mutex);
    startLocked(queue);
}

void Publish(PrepJob job) {
    using Counter = PrepHelpers::Counter;
    // Already committing (or committed): no helper could beat the commit. The caller holds another
    // reference to the closure, so the pin is never its last one here.
    if (DrawPipeline::Queue0().CommittingSeq() >= job.seq) {
        add(Counter::StaleAtPublish);
        return;
    }
    auto& queue = prepQueue();
    std::shared_ptr<void> dropped;
    {
        std::lock_guard lock(queue.mutex);
        if (queue.stopped) {
            add(Counter::AfterStop);
            dropped = std::move(job.pin);
        } else {
            startLocked(queue);
            if (queue.jobs.size() >= prepCapacity()) {
                add(Counter::Overflowed);
                dropped = std::move(queue.jobs.front().pin);
                queue.jobs.pop_front();
            }
            queue.jobs.push_back(std::move(job));
            queue.queued.store(queue.jobs.size(), std::memory_order_release);
            add(Counter::Published);
            if (queue.waiting != 0 && !queue.frozen) queue.wake.notify_one();
        }
    }
    retirePin(std::move(dropped));
}

void Stop() {
    if (!On()) return;
    auto& queue = prepQueue();
    std::vector<std::thread> threads;
    std::deque<PrepJob> left;
    {
        std::lock_guard lock(queue.mutex);
        queue.stopped = true;
        queue.stopping.store(true, std::memory_order_release);
        threads.swap(queue.threads);
        left.swap(queue.jobs);
        queue.queued.store(0, std::memory_order_relaxed);
        add(Counter::AtStop, left.size());
    }
    queue.wake.notify_all();
    for (auto& thread : threads) {
        if (thread.get_id() == std::this_thread::get_id()) thread.detach();
        else thread.join();
    }
    for (auto& job : left) retirePin(std::move(job.pin));
}

unsigned Running() {
    return prepQueue().running.load(std::memory_order_acquire);
}

void Freeze(bool frozen) {
    auto& queue = prepQueue();
    {
        std::lock_guard lock(queue.mutex);
        queue.frozen = frozen;
    }
    if (!frozen) queue.wake.notify_all();
}

Counts Totals() {
    Counts counts{};
    for (std::size_t index = 0; index < Counters; ++index) counts[index] = prepCounters[index].value.load(std::memory_order_relaxed);
    return counts;
}

Counts TakeWindow(Report report) {
    // Each report's last totals (never destroyed: a report may run late in the exit).
    struct Windows {
        std::mutex mutex;
        std::array<Counts, static_cast<std::size_t>(Report::Count)> last{};
    };
    static auto* windows = new Windows();
    const auto now = Totals();
    std::lock_guard lock(windows->mutex);
    auto& before = windows->last[static_cast<std::size_t>(report)];
    Counts window{};
    for (std::size_t index = 0; index < Counters; ++index) window[index] = now[index] - before[index];
    before = now;
    return window;
}

}

}
