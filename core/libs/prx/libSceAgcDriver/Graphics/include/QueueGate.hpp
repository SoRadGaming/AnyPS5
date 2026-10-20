#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_QUEUEGATE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_QUEUEGATE_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace AgcDriver::Graphics {

// LOCAL EXPERIMENT (PARALLEL-COMMIT-PLAN A3, APS5_EXPERIMENT_PRESENT_QUEUE): the external
// synchronization of the device's VkQueues, one mutex per queue: graphics (index 0 of the graphics
// family: recorded batches, command batches, the presentation blit, the import probe, a sparse bind
// without its own queue), present (a further queue of the same family for vkQueuePresentKHR, which
// the presenter then issues after it released GuestMemory::GpuMutex), settle and sparse (the import
// settle queue and the sparse-binding queue; their mutexes were VulkanDevice's own before and are
// the gate's now, the same std::mutex). A device-wide wait (vkDeviceWaitIdle) needs every queue
// externally synchronized, so it takes every mutex, in the fixed order graphics, present, settle,
// sparse (AllQueues). The lock order is GpuMutex before any queue mutex, never the reverse: a queue
// mutex is held only around its queue operation (a fence wait follows outside it), and nothing
// inside one takes another queue's mutex or GpuMutex.
//
// Off (the switch unset) the gate is inert and every site runs as before: graphics and present
// operations under GpuMutex alone, device-wide waits under it, the settle and sparse queues under
// their mutexes. On (=1 or =2) every queue operation takes its queue's mutex and is checked: the
// calling thread holds that mutex, and no device-wide wait of another thread is in progress (one
// inside AllQueues holds every mutex). A violation is counted ([queue] line at the gate's teardown,
// the [stream] line); it aborts with a [queue] line under APS5_ASSERT_GPU_LOCK=1, and always at =2.
// The thread's held mutexes are also kept in GuestMemory (SetQueueMutexesHeld), so a GpuMutex lock
// under one aborts there under APS5_ASSERT_GPU_LOCK=1. One gate per device (VulkanDevice's State);
// Context::queueGate points to it (null in tests that build their own context: today's paths).
class QueueGate {
public:
    enum class Queue : std::uint8_t { Graphics, Present, Settle, Sparse, Count };
    static constexpr std::size_t Queues = static_cast<std::size_t>(Queue::Count);
    static const char* QueueName(Queue queue);

    // APS5_EXPERIMENT_PRESENT_QUEUE, read once: 0 unset; 1 the device gets a present queue and the
    // presenter issues vkQueuePresentKHR on it after its hold; 2 the present queue, but the present
    // stays inside the hold (the queue change without the lock change), each violation fatal. A
    // family without a free queue keeps presents on the graphics queue inside the hold.
    static int PresentQueueMode();
    // The index of the present queue in the graphics family at device creation: the next one after
    // the `used` queues (graphics, settle) when the family has `familyQueues` > `used`; 0 (none,
    // presents stay on the graphics queue) otherwise.
    static std::uint32_t PresentQueueIndex(std::uint32_t familyQueues, std::uint32_t used) { return used != 0 && familyQueues > used ? used : 0; }

    struct Modes {
        int presentQueue = 0;
    };
    static Modes SwitchModes();

    explicit QueueGate(Modes modes = SwitchModes());
    ~QueueGate();
    QueueGate(const QueueGate&) = delete;
    QueueGate& operator=(const QueueGate&) = delete;

    const Modes& Mode() const { return modes; }
    // Whether queue operations take and check their mutexes (a switch is set).
    bool Gated() const { return gated; }
    // The device's present queue (VulkanDevice sets it after device creation; null: none, presents
    // go to the graphics queue), and whether the presenter issues the present after its hold.
    void SetPresentQueue(VkQueue queue) { presentQueue = queue; }
    VkQueue PresentQueue() const { return presentQueue; }
    bool PresentAfterHold() const { return modes.presentQueue == 1 && presentQueue != VK_NULL_HANDLE; }
    // The settle and sparse mutexes (Context::settleQueueMutex, sparseQueueMutex point to them).
    std::mutex& Mutex(Queue queue) { return mutexes[static_cast<std::size_t>(queue)]; }

    // One queue's mutex around one queue operation, released at its end. `mutex` null: an empty
    // hold (an ungated graphics or present operation). `gate` null: an untracked lock (the settle
    // and sparse mutexes off the switch, or of a test's context without a gate).
    class Hold {
    public:
        Hold() = default;
        Hold(std::mutex* mutex, QueueGate* gate, Queue queue);
        ~Hold() { release(); }
        Hold(Hold&& other) noexcept : mutex(other.mutex), gate(other.gate), queue(other.queue) { other.mutex = nullptr; }
        Hold& operator=(Hold&& other) noexcept;
        Hold(const Hold&) = delete;
        Hold& operator=(const Hold&) = delete;

    private:
        void release() noexcept;
        std::mutex* mutex = nullptr;
        QueueGate* gate = nullptr;
        Queue queue = Queue::Graphics;
    };
    Hold Lock(Queue queue);

    // Every queue's mutex in the fixed order, for a device-wide wait (empty when not gated).
    class AllHold {
    public:
        AllHold() = default;
        explicit AllHold(QueueGate* gate);
        ~AllHold() { release(); }
        AllHold(AllHold&& other) noexcept : gate(other.gate) { other.gate = nullptr; }
        AllHold& operator=(AllHold&&) = delete;
        AllHold(const AllHold&) = delete;
        AllHold& operator=(const AllHold&) = delete;

    private:
        void release() noexcept;
        QueueGate* gate = nullptr;
    };
    AllHold AllQueues();

    // The check of one operation on `queue` (gated only): its mutex is held by the calling thread,
    // and no other thread's device-wide wait is in progress.
    void CheckOp(Queue queue, const char* what);

    // A gated present's result (VulkanDevice::IssuePresent, perhaps after the presenter's hold),
    // counted (NotePresent). Out of date or suboptimal only sets a flag, which the next Resize
    // takes under GpuMutex (TakePresentOutOfDate) and turns into today's zeroed extent: nothing else
    // of the device is written by a present outside the hold. False for any other failure (the
    // caller's check throws it).
    bool NotePresentResult(VkResult result);
    bool TakePresentOutOfDate() { return presentOutOfDate.exchange(false, std::memory_order_acq_rel); }

    // Cumulative over every gate of the process (the [queue] and [stream] lines). Presents by arm
    // (the APS5_EXPERIMENT_PRESENT_QUEUE value) and outcome: success, suboptimal, out of date,
    // other (a failure). Violations: an operation without its queue's mutex, one overlapping
    // another thread's device-wide wait, a queue mutex taken out of the fixed order.
    static constexpr std::size_t PresentOutcomes = 4;
    struct Counts {
        std::uint64_t ops = 0;
        std::uint64_t deviceWaits = 0;
        std::uint64_t unheld = 0;
        std::uint64_t overlapped = 0;
        std::uint64_t misordered = 0;
        std::array<std::array<std::uint64_t, PresentOutcomes>, 3> presents{};
    };
    static Counts CountsSoFar();
    static void NotePresent(VkResult result);

private:
    friend class Hold;
    friend class AllHold;
    void noteHeld(Queue queue);
    void noteReleased(Queue queue) noexcept;
    void violation(std::atomic<std::uint64_t>& counter, const char* kind, Queue queue, const char* what);
    bool fatal() const;

    Modes modes;
    bool gated = false;
    VkQueue presentQueue = VK_NULL_HANDLE;
    std::array<std::mutex, Queues> mutexes;
    // The thread token (AgcDriver::CurrentThreadToken) holding each queue's mutex, and the one in a
    // device-wide wait; null when free. Written by the holder only.
    std::array<std::atomic<const void*>, Queues> holders{};
    std::atomic<const void*> deviceWide{nullptr};
    std::atomic<bool> presentOutOfDate{false};
};

// The sites in Graphics go through their context's gate. Settle and Sparse always lock the
// context's own mutex pointer (the gate's for a device's context; a test's mutex otherwise), the
// gate only tracks; Graphics and Present lock only when the gate is on.
inline QueueGate::Hold LockQueue(const Context& context, QueueGate::Queue queue) {
    auto* gate = context.queueGate != nullptr && context.queueGate->Gated() ? context.queueGate : nullptr;
    if (queue == QueueGate::Queue::Settle || queue == QueueGate::Queue::Sparse) {
        auto* mutex = queue == QueueGate::Queue::Settle ? context.settleQueueMutex : context.sparseQueueMutex;
        return mutex != nullptr ? QueueGate::Hold(mutex, gate, queue) : QueueGate::Hold();
    }
    return gate != nullptr ? gate->Lock(queue) : QueueGate::Hold();
}

inline void CheckQueueOp(const Context& context, QueueGate::Queue queue, const char* what) {
    if (context.queueGate != nullptr && context.queueGate->Gated()) context.queueGate->CheckOp(queue, what);
}

inline QueueGate::AllHold LockAllQueues(const Context& context) {
    return context.queueGate != nullptr ? context.queueGate->AllQueues() : QueueGate::AllHold();
}

}

#endif
