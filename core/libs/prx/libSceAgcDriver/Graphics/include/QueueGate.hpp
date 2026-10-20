#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_QUEUEGATE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_QUEUEGATE_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <unordered_set>

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
// Off (the switches unset) the gate is inert and every site runs as before: graphics and present
// operations under GpuMutex alone, device-wide waits under it, the settle and sparse queues under
// their mutexes. On (=1 or =2) every queue operation takes its queue's mutex and is checked: the
// calling thread holds that mutex, and no device-wide wait of another thread is in progress (one
// inside AllQueues holds every mutex). A violation is counted ([queue] line at the gate's teardown,
// the [stream] line); it aborts with a [queue] line under APS5_ASSERT_GPU_LOCK=1, and always at =2.
// The thread's held mutexes are also kept in GuestMemory (SetQueueMutexesHeld), so a GpuMutex lock
// under one aborts there under APS5_ASSERT_GPU_LOCK=1. One gate per device (VulkanDevice's State);
// Context::queueGate points to it (null in tests that build their own context: today's paths).
//
// LOCAL EXPERIMENT (PARALLEL-COMMIT-PLAN A4, APS5_EXPERIMENT_SUBMIT_THREAD): the gate also owns the
// submitter, the thread "agcsubmit" that issues the recorder's vkQueueSubmit. Recorder::Submit still
// does everything up to vkEndCommandBuffer under GpuMutex, then seals: its batch goes into a FIFO in
// serial order (the GpuMutex order), and the submitter issues the FIFO in that order under the
// graphics queue's mutex, inside a GuestMemory::PrepareScope (it never takes GpuMutex). What keeps a
// fence or the timeline from being used before its submit: a blocking wait first awaits the batch's
// submission (AwaitSubmitted, the only wait on the submitter, never under a queue mutex); a poll
// treats a sealed batch the submitter has not issued as unsignaled (the batch's gpuSubmitted flag).
// A foreign submit on the graphics queue (CommandBatch, the presentation blit, the import probe, a
// sparse bind on it) drains the FIFO first, then takes the graphics mutex (LockForeign); its callers
// hold GpuMutex, so no seal slips in between. A device-wide wait drains the FIFO before AllQueues
// (the submitter's own device-lost wait takes every mutex without draining). A failed submit is
// stored: the FIFO behind it is dropped, nothing is sealed or issued after it, and the error is
// rethrown at the next seal or await on the caller. =2 makes the hand-off synchronous (the sealer
// waits until its batch is issued, today's latency) and checks every ordering invariant (serials
// strictly increasing, no fence polled or waited before its submit, every foreign submit finds the
// FIFO issued, one submit per fence before its reset); the first violation is fatal. The serials are the recorder's (Recorder::Submissions):
// one recorder seals through a gate.
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
    // APS5_EXPERIMENT_SUBMIT_THREAD, read once: 0 unset; 1 the submitter issues sealed batches; 2 the
    // synchronous hand-off with the ordering checks. APS5_EXPERIMENT_SUBMIT_POOLS=1 (with it): every
    // recorder batch records into a VkCommandPool of its own (reset at its release), so the submitter
    // never submits from a pool another thread is recording into (judge defect #10).
    static int SubmitThreadMode();
    static bool SubmitPools();

    struct Modes {
        int presentQueue = 0;
        int submitThread = 0;
        bool submitPools = false;
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
    // A4: whether seals go through the submitter, and whether recorder batches own their pools.
    bool Threaded() const { return modes.submitThread != 0; }
    bool OwnPools() const { return modes.submitThread != 0 && modes.submitPools; }
    // A4: the device, graphics queue and entry points the submitter uses (set once after device
    // creation, before the first seal).
    void Attach(VkDevice device, VkQueue graphics, PFN_vkQueueSubmit submit, PFN_vkDeviceWaitIdle waitIdle);

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
    // A foreign operation on the graphics queue (A4): the FIFO drained first (every earlier seal
    // issued), then the graphics queue's mutex; Lock(Graphics) without the submit thread.
    Hold LockForeign(const char* what);

    // Every queue's mutex in the fixed order, for a device-wide wait (empty when not gated); under
    // the submit thread the FIFO is drained first.
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

    // A4. A sealed batch: its gpuSubmitted flag (set once its vkQueueSubmit returned success; the
    // flag of a Recorder batch, which outlives the FIFO entry), command buffer, fence, the timeline
    // (null: none) and the serial the timeline reaches when the batch completes.
    struct Sealed {
        std::atomic<bool>* issued = nullptr;
        VkCommandBuffer commands = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkSemaphore timeline = VK_NULL_HANDLE;
        std::uint64_t serial = 0;
    };
    // Under GpuMutex: rethrows a stored failure, then hands the batch to the submitter (started at
    // the first seal, and again at a seal after StopSubmitter). =1 returns at once, =2 once the batch
    // was issued (its own failure thrown). Without a submitter thread (it could not be started) the
    // batch is issued here. The failure is checked under the FIFO's mutex, where fail() stores it
    // and clears the FIFO, so no batch is pushed behind that clear (it would be issued after the
    // batches the failure dropped, or, at =2, discarded by its sealer while still queued).
    void Seal(const Sealed& sealed);
    // The await rule: returns once the batch of `serial` was issued (at once for a serial never
    // sealed); throws the stored failure when the FIFO stopped before it.
    void AwaitSubmitted(std::uint64_t serial);
    // The await rule for a sealed batch (Recorder::finish): AwaitSubmitted, then its own flag
    // (`issued`, its Sealed::issued) must read issued, or the stored failure is thrown: the
    // submitted serial alone passes a batch only while every seal was issued in serial order, which
    // a failure ends (the batches it dropped never are). A batch found unissued without a failure is
    // an ordering violation, thrown rather than waited for (its fence was never submitted).
    void AwaitIssued(const std::atomic<bool>& issued, std::uint64_t serial);
    // Every sealed batch issued (or dropped by a failure); never throws. A no-op on the submitter.
    void Drain();
    // The poll rule for a sealed batch's flag (Recorder): false reads as unsignaled. At =2 every seal
    // is issued before Seal returns, so an unissued poll is a violation there.
    bool PollIssued(const std::atomic<bool>& issued, const char* what);
    // =2: a sealed batch's fence is about to be reset (Recorder::release): exactly one submit of it
    // must have been issued since its last reset.
    void NoteFenceReset(VkFence fence);
    // Drains the FIFO and joins the submitter (a recorder's teardown, a device replacement, the
    // gate's destructor); idempotent.
    void StopSubmitter();
    std::uint64_t SealedSerial() const { return sealedSerial.load(std::memory_order_acquire); }
    std::uint64_t SubmittedSerial() const { return submittedSerial.load(std::memory_order_acquire); }
    // Tests: hold the FIFO (the submitter issues nothing until released; a drain or await waits
    // meanwhile), make the next issue fail with `result` without calling Vulkan (a device-lost
    // injection), and the submitter threads running in the process.
    void PauseSubmitter(bool pause);
    void InjectSubmitFailure(VkResult result);
    static unsigned RunningSubmitters();

    // Cumulative over every gate of the process (the [queue] and [stream] lines). Presents by arm
    // (the APS5_EXPERIMENT_PRESENT_QUEUE value) and outcome: success, suboptimal, out of date,
    // other (a failure). Violations: an operation without its queue's mutex, one overlapping
    // another thread's device-wide wait, a queue mutex taken out of the fixed order (or a wait for
    // the submitter under one), A4's ordering checks. A4: seals, the awaits and drains that had to
    // wait for the submitter, polls that found a sealed batch unissued, failed submits.
    static constexpr std::size_t PresentOutcomes = 4;
    struct Counts {
        std::uint64_t ops = 0;
        std::uint64_t deviceWaits = 0;
        std::uint64_t unheld = 0;
        std::uint64_t overlapped = 0;
        std::uint64_t misordered = 0;
        std::uint64_t ordering = 0;
        std::uint64_t seals = 0;
        std::uint64_t waitedAwaits = 0;
        std::uint64_t waitedDrains = 0;
        std::uint64_t unissuedPolls = 0;
        std::uint64_t submitFailures = 0;
        std::array<std::array<std::uint64_t, PresentOutcomes>, 3> presents{};
    };
    static Counts CountsSoFar();
    static void NotePresent(VkResult result);

private:
    void noteHeld(Queue queue);
    void noteReleased(Queue queue) noexcept;
    void violation(std::atomic<std::uint64_t>& counter, const char* kind, Queue queue, const char* what);
    bool fatal() const;
    // A4.
    void runSubmitter();
    void issue(const Sealed& sealed);
    void fail(VkResult result, bool injected);
    bool awaitSerial(std::uint64_t target, bool drain);
    // Throws the stored failure when a submit failed (`failed` is never cleared); throwStored
    // always throws (the stored failure, else a generic one).
    void rethrowStored();
    [[noreturn]] void throwStored();
    bool onSubmitter() const;

    Modes modes;
    bool gated = false;
    VkQueue presentQueue = VK_NULL_HANDLE;
    std::array<std::mutex, Queues> mutexes;
    // The thread token (AgcDriver::CurrentThreadToken) holding each queue's mutex, and the one in a
    // device-wide wait; null when free. Written by the holder only.
    std::array<std::atomic<const void*>, Queues> holders{};
    std::atomic<const void*> deviceWide{nullptr};
    std::atomic<bool> presentOutOfDate{false};
    // A4 (Attach).
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    PFN_vkQueueSubmit queueSubmit = nullptr;
    PFN_vkDeviceWaitIdle deviceWaitIdle = nullptr;
    // A4: the FIFO and its thread, under fifoMutex: `work` wakes the submitter, `issuedWake` the
    // awaiters (`waiters` of them blocked). `sealedSerial` is the newest sealed serial,
    // `submittedSerial` the newest issued one (FIFO order is serial order); `failed` once a submit
    // failed (`failure` holds the error; the FIFO behind it was dropped).
    std::mutex fifoMutex;
    std::condition_variable work;
    std::condition_variable issuedWake;
    std::deque<Sealed> fifo;
    std::thread submitter;
    std::atomic<const void*> submitterToken{nullptr};
    std::atomic<bool> stopRequested{false};
    bool stopping = false;
    bool sleeping = false;
    bool paused = false;
    std::uint32_t waiters = 0;
    std::atomic<std::uint64_t> sealedSerial{0};
    std::atomic<std::uint64_t> submittedSerial{0};
    std::atomic<bool> failed{false};
    std::exception_ptr failure;
    std::atomic<int> injectedFailure{VK_SUCCESS};
    // A4 =2: the fences issued since their last reset.
    std::mutex fenceMutex;
    std::unordered_set<VkFence> issuedFences;
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

// A4: a foreign graphics-queue operation through the context's gate (see LockForeign).
inline QueueGate::Hold LockForeignQueue(const Context& context, const char* what) {
    return context.queueGate != nullptr && context.queueGate->Gated() ? context.queueGate->LockForeign(what) : QueueGate::Hold();
}

inline void CheckQueueOp(const Context& context, QueueGate::Queue queue, const char* what) {
    if (context.queueGate != nullptr && context.queueGate->Gated()) context.queueGate->CheckOp(queue, what);
}

inline QueueGate::AllHold LockAllQueues(const Context& context) {
    return context.queueGate != nullptr ? context.queueGate->AllQueues() : QueueGate::AllHold();
}

}

#endif
