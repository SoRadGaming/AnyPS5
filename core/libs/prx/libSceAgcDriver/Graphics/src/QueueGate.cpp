#include "prx/libSceAgcDriver/Graphics/include/QueueGate.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/StreamStats.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <system_error>

namespace AgcDriver::Graphics {
namespace {

constexpr std::array<const char*, QueueGate::Queues> QueueNames{"graphics", "present", "settle", "sparse"};

// Process-wide counters (QueueGate::Counts), relaxed: reported, never acted on.
std::atomic<std::uint64_t> opsChecked{0};
std::atomic<std::uint64_t> deviceWaits{0};
std::atomic<std::uint64_t> unheldOps{0};
std::atomic<std::uint64_t> overlappedOps{0};
std::atomic<std::uint64_t> misorderedLocks{0};
std::atomic<std::uint64_t> orderingViolations{0};
std::atomic<std::uint64_t> sealCount{0};
std::atomic<std::uint64_t> waitedAwaits{0};
std::atomic<std::uint64_t> waitedDrains{0};
std::atomic<std::uint64_t> unissuedPolls{0};
std::atomic<std::uint64_t> submitFailures{0};
std::array<std::array<std::atomic<std::uint64_t>, QueueGate::PresentOutcomes>, 3> presentOutcomes{};
std::atomic<unsigned> runningSubmitters{0};

bool assertGpuLock() {
    static const bool enabled = std::getenv("APS5_ASSERT_GPU_LOCK") != nullptr;
    return enabled;
}

int switchLevel(const char* name) {
    const char* text = std::getenv(name);
    if (text == nullptr) return 0;
    return std::strtol(text, nullptr, 10) == 2 ? 2 : 1;
}

std::uint8_t queueBit(QueueGate::Queue queue) {
    return static_cast<std::uint8_t>(1u << static_cast<unsigned>(queue));
}

unsigned long long count(std::uint64_t value) {
    return static_cast<unsigned long long>(value);
}

}

const char* QueueGate::QueueName(Queue queue) {
    return QueueNames[static_cast<std::size_t>(queue)];
}

int QueueGate::PresentQueueMode() {
    static const int mode = switchLevel("APS5_EXPERIMENT_PRESENT_QUEUE");
    return mode;
}

int QueueGate::SubmitThreadMode() {
    static const int mode = switchLevel("APS5_EXPERIMENT_SUBMIT_THREAD");
    return mode;
}

bool QueueGate::SubmitPools() {
    static const bool pools = std::getenv("APS5_EXPERIMENT_SUBMIT_POOLS") != nullptr;
    return pools;
}

QueueGate::Modes QueueGate::SwitchModes() {
    Modes modes;
    modes.presentQueue = PresentQueueMode();
    modes.submitThread = SubmitThreadMode();
    modes.submitPools = SubmitPools();
    return modes;
}

QueueGate::QueueGate(Modes modes) : modes(modes), gated(modes.presentQueue != 0 || modes.submitThread != 0) {}

QueueGate::~QueueGate() {
    StopSubmitter();
    if (!gated) return;
    const auto counts = CountsSoFar();
    std::fprintf(stderr, "[queue] gate (APS5_EXPERIMENT_PRESENT_QUEUE=%d, present queue %s; APS5_EXPERIMENT_SUBMIT_THREAD=%d%s): %llu queue operations checked, %llu device-wide waits; violations: %llu without the queue's mutex, %llu overlapping a device-wide wait, %llu out of the lock order, %llu submit ordering\n", modes.presentQueue,
        presentQueue != VK_NULL_HANDLE ? "own" : "shared with graphics", modes.submitThread, OwnPools() ? ", own pools" : "", count(counts.ops), count(counts.deviceWaits), count(counts.unheld), count(counts.overlapped), count(counts.misordered), count(counts.ordering));
    if (Threaded()) std::fprintf(stderr, "[queue] submitter: %llu seals (serial %llu issued of %llu), %llu awaits and %llu drains waited for it, %llu polls found a batch unissued, %llu submits failed\n", count(counts.seals), count(SubmittedSerial()), count(SealedSerial()), count(counts.waitedAwaits), count(counts.waitedDrains), count(counts.unissuedPolls), count(counts.submitFailures));
}

void QueueGate::Attach(VkDevice attachedDevice, VkQueue graphics, PFN_vkQueueSubmit submit, PFN_vkDeviceWaitIdle waitIdle) {
    device = attachedDevice;
    graphicsQueue = graphics;
    queueSubmit = submit;
    deviceWaitIdle = waitIdle;
}

QueueGate::Hold::Hold(std::mutex* mutex, QueueGate* gate, Queue queue) : mutex(mutex), gate(gate), queue(queue) {
    if (mutex == nullptr) return;
    // The fixed order, checked before the lock: a thread holding a later queue's mutex (or this
    // one: a std::mutex is not recursive) must not wait for this one.
    if (gate != nullptr) {
        const auto held = GuestMemory::QueueMutexesHeld();
        if ((held >> static_cast<unsigned>(queue)) != 0) gate->violation(misorderedLocks, "queue mutex taken out of the fixed order (graphics, present, settle, sparse) or twice", queue, "Lock");
    }
    mutex->lock();
    if (gate != nullptr) gate->noteHeld(queue);
}

QueueGate::Hold& QueueGate::Hold::operator=(Hold&& other) noexcept {
    if (this != &other) {
        release();
        mutex = other.mutex;
        gate = other.gate;
        queue = other.queue;
        other.mutex = nullptr;
    }
    return *this;
}

void QueueGate::Hold::release() noexcept {
    if (mutex == nullptr) return;
    if (gate != nullptr) gate->noteReleased(queue);
    mutex->unlock();
    mutex = nullptr;
}

QueueGate::Hold QueueGate::Lock(Queue queue) {
    if (!gated) {
        // Off the switches only the settle and sparse queues have a mutex (as before the gate).
        if (queue == Queue::Settle || queue == Queue::Sparse) return Hold(&Mutex(queue), nullptr, queue);
        return Hold();
    }
    return Hold(&Mutex(queue), this, queue);
}

QueueGate::Hold QueueGate::LockForeign(const char* what) {
    if (!gated) return Hold();
    Drain();
    auto hold = Lock(Queue::Graphics);
    // Its caller holds GpuMutex, so nothing was sealed since the drain.
    if (modes.submitThread == 2 && !failed.load(std::memory_order_acquire) && SubmittedSerial() != SealedSerial()) violation(orderingViolations, "foreign submit with seals not yet issued", Queue::Graphics, what);
    return hold;
}

QueueGate::AllHold::AllHold(QueueGate* gate) : gate(gate) {
    if (gate == nullptr) return;
    if (GuestMemory::QueueMutexesHeld() != 0) gate->violation(misorderedLocks, "device-wide wait by a thread that holds a queue mutex", Queue::Graphics, "AllQueues");
    for (std::size_t index = 0; index < Queues; ++index) {
        gate->mutexes[index].lock();
        gate->noteHeld(static_cast<Queue>(index));
    }
    gate->deviceWide.store(CurrentThreadToken(), std::memory_order_release);
    deviceWaits.fetch_add(1, std::memory_order_relaxed);
}

void QueueGate::AllHold::release() noexcept {
    if (gate == nullptr) return;
    gate->deviceWide.store(nullptr, std::memory_order_release);
    for (std::size_t index = Queues; index-- > 0;) {
        gate->noteReleased(static_cast<Queue>(index));
        gate->mutexes[index].unlock();
    }
    gate = nullptr;
}

QueueGate::AllHold QueueGate::AllQueues() {
    if (!gated) return AllHold();
    // Drained before any mutex is taken: the submitter needs the graphics mutex to issue (a thread
    // holding a queue mutex here would deadlock; AllHold checks that, Drain too).
    Drain();
    return AllHold(this);
}

void QueueGate::CheckOp(Queue queue, const char* what) {
    if (!gated) return;
    opsChecked.fetch_add(1, std::memory_order_relaxed);
    const auto* self = CurrentThreadToken();
    if (holders[static_cast<std::size_t>(queue)].load(std::memory_order_acquire) != self) violation(unheldOps, "queue operation without its queue's mutex", queue, what);
    if (const auto* wide = deviceWide.load(std::memory_order_acquire); wide != nullptr && wide != self) violation(overlappedOps, "queue operation overlapping another thread's device-wide wait", queue, what);
}

bool QueueGate::fatal() const {
    return modes.presentQueue == 2 || modes.submitThread == 2 || assertGpuLock();
}

void QueueGate::violation(std::atomic<std::uint64_t>& counter, const char* kind, Queue queue, const char* what) {
    counter.fetch_add(1, std::memory_order_relaxed);
    if (!fatal()) return;
    std::fprintf(stderr, "[queue] ASSERT: %s: %s on the %s queue (thread tagged 0x%x)\n", kind, what, QueueName(queue), GuestMemory::GpuLockThreadTag());
    std::fflush(stderr);
    std::abort();
}

void QueueGate::noteHeld(Queue queue) {
    holders[static_cast<std::size_t>(queue)].store(CurrentThreadToken(), std::memory_order_release);
    GuestMemory::SetQueueMutexesHeld(static_cast<std::uint8_t>(GuestMemory::QueueMutexesHeld() | queueBit(queue)));
}

void QueueGate::noteReleased(Queue queue) noexcept {
    holders[static_cast<std::size_t>(queue)].store(nullptr, std::memory_order_release);
    GuestMemory::SetQueueMutexesHeld(static_cast<std::uint8_t>(GuestMemory::QueueMutexesHeld() & ~queueBit(queue)));
}

bool QueueGate::onSubmitter() const {
    return submitterToken.load(std::memory_order_acquire) == CurrentThreadToken();
}

void QueueGate::Seal(const Sealed& sealed) {
    rethrowStored();
    if (sealed.serial <= sealedSerial.load(std::memory_order_relaxed)) violation(orderingViolations, "seal whose serial is not above the previous one's", Queue::Graphics, "Seal");
    sealCount.fetch_add(1, std::memory_order_relaxed);
    bool issueHere = false;
    {
        std::lock_guard lock(fifoMutex);
        if (!submitter.joinable()) {
            try {
                submitter = std::thread([this] { runSubmitter(); });
                // Counted here, not by the thread, so it holds once Seal returned.
                runningSubmitters.fetch_add(1, std::memory_order_acq_rel);
            } catch (const std::system_error& error) {
                static std::atomic<bool> reported{false};
                if (!reported.exchange(true)) std::fprintf(stderr, "[queue] the submit thread could not be started (%s); batches are submitted by their sealer\n", error.what());
            }
        }
        if (submitter.joinable()) {
            fifo.push_back(sealed);
            sealedSerial.store(sealed.serial, std::memory_order_release);
            if (sleeping) work.notify_one();
        } else {
            sealedSerial.store(sealed.serial, std::memory_order_release);
            issueHere = true;
        }
    }
    if (issueHere) {
        issue(sealed);
        rethrowStored();
        return;
    }
    // =2: the sealer waits until its batch was issued, as it did when it submitted itself.
    if (modes.submitThread == 2) AwaitSubmitted(sealed.serial);
}

void QueueGate::AwaitSubmitted(std::uint64_t serial) {
    if (!Threaded()) return;
    const auto target = std::min(serial, SealedSerial());
    if (!awaitSerial(target, false)) rethrowStored();
}

void QueueGate::Drain() {
    if (!Threaded() || onSubmitter()) return;
    static_cast<void>(awaitSerial(SealedSerial(), true));
}

bool QueueGate::awaitSerial(std::uint64_t target, bool drain) {
    const auto reached = [&] { return submittedSerial.load(std::memory_order_acquire) >= target || failed.load(std::memory_order_acquire); };
    if (submittedSerial.load(std::memory_order_acquire) >= target) return true;
    if (!reached()) {
        // The submitter takes the graphics mutex to issue: waiting for it under a queue mutex would
        // deadlock.
        if (GuestMemory::QueueMutexesHeld() != 0) violation(misorderedLocks, "wait for the submitter by a thread that holds a queue mutex", Queue::Graphics, drain ? "Drain" : "AwaitSubmitted");
        (drain ? waitedDrains : waitedAwaits).fetch_add(1, std::memory_order_relaxed);
        const bool stream = Stream::On();
        const auto startNs = stream ? Stream::NowNs() : 0;
        if (!SpinUntil(HandoffSpinWindow(), reached)) {
            std::unique_lock lock(fifoMutex);
            ++waiters;
            issuedWake.wait(lock, reached);
            --waiters;
        }
        if (stream) {
            auto& counters = Stream::counters;
            Stream::Add(drain ? counters.waitedDrains : counters.waitedAwaits);
            Stream::Add(drain ? counters.drainNs : counters.awaitNs, Stream::NowNs() - startNs);
        }
    }
    return submittedSerial.load(std::memory_order_acquire) >= target;
}

void QueueGate::rethrowStored() {
    if (!failed.load(std::memory_order_acquire)) return;
    std::exception_ptr error;
    {
        std::lock_guard lock(fifoMutex);
        error = failure;
    }
    if (error != nullptr) std::rethrow_exception(error);
    throw std::runtime_error("AGC graphics: vkQueueSubmit recorder failed on the submit thread");
}

bool QueueGate::PollIssued(const std::atomic<bool>& issued, const char* what) {
    if (issued.load(std::memory_order_acquire)) return true;
    unissuedPolls.fetch_add(1, std::memory_order_relaxed);
    if (Stream::On()) Stream::Add(Stream::counters.unissuedPolls);
    if (modes.submitThread == 2 && !failed.load(std::memory_order_acquire)) violation(orderingViolations, "fence polled or waited before its submit", Queue::Graphics, what);
    return false;
}

void QueueGate::NoteFenceReset(VkFence fence) {
    if (modes.submitThread != 2) return;
    bool once = false;
    {
        std::lock_guard lock(fenceMutex);
        once = issuedFences.erase(fence) == 1;
    }
    if (!once && !failed.load(std::memory_order_acquire)) violation(orderingViolations, "sealed batch's fence reset without exactly one submit", Queue::Graphics, "vkResetFences");
}

void QueueGate::runSubmitter() {
    PerformanceTrace::BindThread("agcsubmit");
    // Never GuestMemory::GpuMutex on this thread: a lock or try_lock aborts under
    // APS5_ASSERT_GPU_LOCK=1, as on queue 0 while it prepares a ticket.
    const GuestMemory::PrepareScope scope;
    submitterToken.store(CurrentThreadToken(), std::memory_order_release);
    std::unique_lock lock(fifoMutex);
    for (;;) {
        if (fifo.empty() || paused) {
            if (stopping && fifo.empty()) break;
            // A seal usually follows within the spin window (the committer seals about every 0.8 ms
            // in the hub); otherwise sleep until one is pushed.
            lock.unlock();
            SpinUntil(HandoffSpinWindow(), [&] { return SealedSerial() > SubmittedSerial() || stopRequested.load(std::memory_order_acquire); });
            lock.lock();
            sleeping = true;
            work.wait(lock, [&] { return (!fifo.empty() && !paused) || stopping; });
            sleeping = false;
            continue;
        }
        const auto sealed = fifo.front();
        fifo.pop_front();
        lock.unlock();
        issue(sealed);
        lock.lock();
        if (waiters != 0) issuedWake.notify_all();
    }
    submitterToken.store(nullptr, std::memory_order_release);
    runningSubmitters.fetch_sub(1, std::memory_order_acq_rel);
}

void QueueGate::issue(const Sealed& sealed) {
    // =2: the FIFO is serial order.
    if (modes.submitThread == 2 && sealed.serial <= SubmittedSerial()) violation(orderingViolations, "batch issued out of serial order", Queue::Graphics, "vkQueueSubmit (submitter)");
    VkSubmitInfo submission{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &sealed.commands;
    // The timeline reaches the batch's serial when it completes (Recorder::WaitSerial).
    VkTimelineSemaphoreSubmitInfoKHR timelineInfo{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR};
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &sealed.serial;
    if (sealed.timeline != VK_NULL_HANDLE) {
        submission.pNext = &timelineInfo;
        submission.signalSemaphoreCount = 1;
        submission.pSignalSemaphores = &sealed.timeline;
    }
    const bool stream = Stream::On();
    const auto startNs = stream ? Stream::NowNs() : 0;
    VkResult result = VK_SUCCESS;
    bool injected = false;
    {
        const auto hold = Lock(Queue::Graphics);
        CheckOp(Queue::Graphics, "vkQueueSubmit (submitter)");
        if (const auto injection = injectedFailure.exchange(VK_SUCCESS, std::memory_order_acq_rel); injection != VK_SUCCESS) {
            result = static_cast<VkResult>(injection);
            injected = true;
        } else {
            result = queueSubmit(graphicsQueue, 1, &submission, sealed.fence);
        }
    }
    if (stream) {
        Stream::Add(Stream::counters.submitterSubmits);
        Stream::Add(Stream::counters.submitterSubmitNs, Stream::NowNs() - startNs);
    }
    if (result != VK_SUCCESS) {
        fail(result, injected);
        return;
    }
    if (modes.submitThread == 2) {
        bool first = false;
        {
            std::lock_guard lock(fenceMutex);
            first = issuedFences.insert(sealed.fence).second;
        }
        if (!first) violation(orderingViolations, "fence submitted twice before its reset", Queue::Graphics, "vkQueueSubmit (submitter)");
    }
    sealed.issued->store(true, std::memory_order_release);
    submittedSerial.store(sealed.serial, std::memory_order_release);
}

void QueueGate::fail(VkResult result, bool injected) {
    submitFailures.fetch_add(1, std::memory_order_relaxed);
    if (result == VK_ERROR_DEVICE_LOST) {
        // Every queue's mutex around the device-wide wait, as Recorder::Submit's device-lost path;
        // not drained first (this thread is the FIFO's only consumer).
        VkResult idle = VK_SUCCESS;
        {
            const AllHold queues(gated ? this : nullptr);
            if (deviceWaitIdle != nullptr) idle = deviceWaitIdle(device);
        }
        if (idle != VK_SUCCESS && idle != VK_ERROR_DEVICE_LOST) std::terminate();
        if (!injected) ReportDeviceLost();
    }
    std::lock_guard lock(fifoMutex);
    failure = std::make_exception_ptr(std::runtime_error(std::string("AGC graphics: vkQueueSubmit recorder: Vulkan result ") + std::to_string(result) + (injected ? " (injected)" : "")));
    // The batches behind the failed one are never issued: their awaits throw, their polls stay
    // unsignaled.
    fifo.clear();
    failed.store(true, std::memory_order_release);
    issuedWake.notify_all();
}

void QueueGate::StopSubmitter() {
    std::thread joined;
    {
        std::lock_guard lock(fifoMutex);
        if (!submitter.joinable()) return;
        stopping = true;
        paused = false;
        stopRequested.store(true, std::memory_order_release);
        joined = std::move(submitter);
    }
    work.notify_all();
    // The loop issues what is left in the FIFO before it returns.
    if (joined.get_id() == std::this_thread::get_id()) joined.detach();
    else joined.join();
    std::lock_guard lock(fifoMutex);
    stopping = false;
    stopRequested.store(false, std::memory_order_release);
}

void QueueGate::PauseSubmitter(bool pause) {
    {
        std::lock_guard lock(fifoMutex);
        paused = pause;
    }
    if (!pause) work.notify_all();
}

void QueueGate::InjectSubmitFailure(VkResult result) {
    injectedFailure.store(result, std::memory_order_release);
}

unsigned QueueGate::RunningSubmitters() {
    return runningSubmitters.load(std::memory_order_acquire);
}

QueueGate::Counts QueueGate::CountsSoFar() {
    Counts counts;
    counts.ops = opsChecked.load(std::memory_order_relaxed);
    counts.deviceWaits = deviceWaits.load(std::memory_order_relaxed);
    counts.unheld = unheldOps.load(std::memory_order_relaxed);
    counts.overlapped = overlappedOps.load(std::memory_order_relaxed);
    counts.misordered = misorderedLocks.load(std::memory_order_relaxed);
    counts.ordering = orderingViolations.load(std::memory_order_relaxed);
    counts.seals = sealCount.load(std::memory_order_relaxed);
    counts.waitedAwaits = waitedAwaits.load(std::memory_order_relaxed);
    counts.waitedDrains = waitedDrains.load(std::memory_order_relaxed);
    counts.unissuedPolls = unissuedPolls.load(std::memory_order_relaxed);
    counts.submitFailures = submitFailures.load(std::memory_order_relaxed);
    for (std::size_t arm = 0; arm < counts.presents.size(); ++arm) {
        for (std::size_t outcome = 0; outcome < PresentOutcomes; ++outcome) counts.presents[arm][outcome] = presentOutcomes[arm][outcome].load(std::memory_order_relaxed);
    }
    return counts;
}

bool QueueGate::NotePresentResult(VkResult result) {
    NotePresent(result);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        presentOutOfDate.store(true, std::memory_order_release);
        return true;
    }
    return result == VK_SUCCESS;
}

void QueueGate::NotePresent(VkResult result) {
    const auto outcome = result == VK_SUCCESS ? 0 : result == VK_SUBOPTIMAL_KHR ? 1 : result == VK_ERROR_OUT_OF_DATE_KHR ? 2 : 3;
    presentOutcomes[static_cast<std::size_t>(PresentQueueMode())][static_cast<std::size_t>(outcome)].fetch_add(1, std::memory_order_relaxed);
}

}
