#include "prx/libSceAgcDriver/Graphics/include/QueueGate.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include <cstdio>
#include <cstdlib>

namespace AgcDriver::Graphics {
namespace {

constexpr std::array<const char*, QueueGate::Queues> QueueNames{"graphics", "present", "settle", "sparse"};

// Process-wide counters (QueueGate::Counts), relaxed: reported, never acted on.
std::atomic<std::uint64_t> opsChecked{0};
std::atomic<std::uint64_t> deviceWaits{0};
std::atomic<std::uint64_t> unheldOps{0};
std::atomic<std::uint64_t> overlappedOps{0};
std::atomic<std::uint64_t> misorderedLocks{0};
std::array<std::array<std::atomic<std::uint64_t>, QueueGate::PresentOutcomes>, 3> presentOutcomes{};

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

}

const char* QueueGate::QueueName(Queue queue) {
    return QueueNames[static_cast<std::size_t>(queue)];
}

int QueueGate::PresentQueueMode() {
    static const int mode = switchLevel("APS5_EXPERIMENT_PRESENT_QUEUE");
    return mode;
}

QueueGate::Modes QueueGate::SwitchModes() {
    Modes modes;
    modes.presentQueue = PresentQueueMode();
    return modes;
}

QueueGate::QueueGate(Modes modes) : modes(modes), gated(modes.presentQueue != 0) {}

QueueGate::~QueueGate() {
    if (!gated) return;
    const auto counts = CountsSoFar();
    std::fprintf(stderr, "[queue] gate (APS5_EXPERIMENT_PRESENT_QUEUE=%d, present queue %s): %llu queue operations checked, %llu device-wide waits; violations: %llu without the queue's mutex, %llu overlapping a device-wide wait, %llu queue mutexes out of order\n", modes.presentQueue,
        presentQueue != VK_NULL_HANDLE ? "own" : "shared with graphics", static_cast<unsigned long long>(counts.ops), static_cast<unsigned long long>(counts.deviceWaits), static_cast<unsigned long long>(counts.unheld), static_cast<unsigned long long>(counts.overlapped), static_cast<unsigned long long>(counts.misordered));
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
        // Off the switch only the settle and sparse queues have a mutex (as before the gate).
        if (queue == Queue::Settle || queue == Queue::Sparse) return Hold(&Mutex(queue), nullptr, queue);
        return Hold();
    }
    return Hold(&Mutex(queue), this, queue);
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
    return gated ? AllHold(this) : AllHold();
}

void QueueGate::CheckOp(Queue queue, const char* what) {
    if (!gated) return;
    opsChecked.fetch_add(1, std::memory_order_relaxed);
    const auto* self = CurrentThreadToken();
    if (holders[static_cast<std::size_t>(queue)].load(std::memory_order_acquire) != self) violation(unheldOps, "queue operation without its queue's mutex", queue, what);
    if (const auto* wide = deviceWide.load(std::memory_order_acquire); wide != nullptr && wide != self) violation(overlappedOps, "queue operation overlapping another thread's device-wide wait", queue, what);
}

bool QueueGate::fatal() const {
    return modes.presentQueue == 2 || assertGpuLock();
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

QueueGate::Counts QueueGate::CountsSoFar() {
    Counts counts;
    counts.ops = opsChecked.load(std::memory_order_relaxed);
    counts.deviceWaits = deviceWaits.load(std::memory_order_relaxed);
    counts.unheld = unheldOps.load(std::memory_order_relaxed);
    counts.overlapped = overlappedOps.load(std::memory_order_relaxed);
    counts.misordered = misorderedLocks.load(std::memory_order_relaxed);
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
