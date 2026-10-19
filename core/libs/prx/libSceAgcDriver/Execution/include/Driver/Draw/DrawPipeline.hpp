#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

class DrawPipeline {
public:
    using Commit = std::function<void()>;
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    enum class DrainReason : std::uint8_t { Packet, Flush, Labels, Capture, Indirect, Submission, Count };
    // What a queued commit records, named by the draw and label enqueues: consecutive draws and
    // labels may share one GPU-mutex hold (APS5_EXPERIMENT_COMMIT_RUNS), anything else commits alone.
    enum class ItemKind : std::uint8_t { Other, Draw, Label };

    static DrawPipeline& Queue0();
    static std::size_t Depth();
    static std::size_t QueuedItems();
    static bool& Active();
    static std::atomic<std::uint64_t>& EpochToken();
    static void FollowEpoch(std::uint64_t token);
    static unsigned Committers();

    // LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_TICKET, room-wait draw tickets): work the producer does
    // for the item it enqueues (Enqueue's optional `prepare`) while the committer works through the
    // queue, before the push. When Enqueue finds at least `pre` items queued and the producer does
    // not hold GuestMemory::GpuMutex, it drops the pipeline mutex, runs `run` inside a
    // GuestMemory::PrepareScope (the queue may only shrink meanwhile), takes the mutex
    // again, rethrows a committer failure and re-checks closing (an inline commit then), and only
    // then waits for room and pushes as before; otherwise it calls `skipped` (under the pipeline
    // mutex, to count) and enqueues at once. The push under the mutex publishes what `run` wrote
    // into the item to the committer, which takes the front under the same mutex.
    struct PrepareBudget {
        const DrawPipeline* pipeline = nullptr;
        // Items queued when Enqueue looked, and whether that was a full pipeline.
        std::size_t depth = 0;
        bool full = false;
        std::size_t bail = 0;
        // Whether to stop between parts: the committer is close to running dry.
        bool Stop() const { return pipeline->Outstanding() <= bail; }
    };
    enum class PrepareSkip : std::uint8_t { Shallow, Locked };
    struct Prepare {
        void* self = nullptr;
        void (*run)(void* self, const PrepareBudget& budget) = nullptr;
        void (*skipped)(void* self, PrepareSkip why) = nullptr;
        std::size_t pre = 0;
        std::size_t bail = 0;
    };
    void Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress = 0, std::vector<std::byte> labelBytes = {}, ItemKind kind = ItemKind::Other, const Prepare* prepare = nullptr);
    void Drain(DrainReason reason, std::uint32_t opcode = 0x100);
    void Stop();
    static std::uint64_t DrainRequestsByOpcode(std::uint32_t opcode);
    bool Busy() const { return outstanding.load(std::memory_order_acquire) != 0; }
    std::size_t Outstanding() const { return outstanding.load(std::memory_order_acquire); }
    bool Overlaps(std::uint64_t address, std::size_t bytes);
    bool OverlapsAny(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges);
    std::optional<std::uint64_t> PendingLabel(std::uint64_t address, std::size_t bytes);
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_COMMIT_RUNS=1): the committer commits consecutive draws and
    // labels under one GPU-mutex hold (a run). A commit inside a run leaves its after-commit check to
    // the run's end with DeferToRunEnd, which returns false outside a run, where the caller runs the
    // check itself; LabelReapDue lets only the run's first label reap finished batches.
    static bool DeferToRunEnd(void (*check)());
    static bool LabelReapDue();
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_LABEL_COALESCE=1): a label queued behind a draw or a label
    // the committer has not started rides on that item instead of taking a slot of its own. It runs
    // right after the item's commit, inside the same GPU-mutex hold, through the committer the
    // driver registered (no closure per label), and until then it is one of the item's writes for
    // PendingLabel, Overlaps and OverlapsAny, the item's newest.
    struct LabelPart {
        std::uint32_t queue = 0;
        std::uint32_t opcode = 0;
        std::uint64_t address = 0;
        std::vector<std::byte> bytes;
        bool endOfPipeInterrupt = false;
    };
    using LabelCommit = void (*)(const LabelPart& label);
    static constexpr std::size_t MaxAppendedLabels = 8;
    void SetLabelCommit(LabelCommit commit) { labelCommit.store(commit, std::memory_order_release); }
    // Appends the label to the newest queued item and moves from it; false, leaving it as it was,
    // when that item may already be running (fewer than two items queued), is not a draw or a
    // label, already carries MaxAppendedLabels, when the pipeline is closing or no committer is
    // registered: the caller then enqueues the label as an item of its own. Like Enqueue, it first
    // rethrows the failure of an earlier commit.
    bool AppendLabel(LabelPart& label);
    // LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS): what queue 0's enqueues found, for the [q0prep] line:
    // the queue depth each draw was enqueued behind (index = depth, the last bucket holds deeper
    // ones), the draws enqueued at full depth (PrepareBudget::full's own test, counted apart since
    // Depth() may exceed the last bucket), and the time draws and labels waited for room.
    // TakePrepStats returns the counts since the last call and resets them.
    static constexpr std::size_t PrepDepthBuckets = 10;
    struct PrepStats {
        std::array<std::uint64_t, PrepDepthBuckets> drawDepths{};
        std::uint64_t draws = 0;
        std::uint64_t fullDraws = 0;
        std::uint64_t drawWaits = 0;
        std::uint64_t drawWaitNs = 0;
        std::uint64_t labels = 0;
        std::uint64_t labelWaits = 0;
        std::uint64_t labelWaitNs = 0;
    };
    PrepStats TakePrepStats();

private:
    struct Item {
        Commit commit;
        std::vector<Range> writes;
        std::uint64_t labelAddress = 0;
        std::vector<std::byte> labelBytes;
        ItemKind kind = ItemKind::Other;
        // Appended labels, in the order they were queued; only the producer adds them, under the
        // mutex and only while the item is not the front, so the committer reads them unlocked.
        std::vector<LabelPart> labels;
        // The queue depth the item was enqueued behind, and whether that was a full pipeline
        // (APS5_DEBUG_Q0_PREP_STATS attribution).
        std::uint32_t enqueueDepth = 0;
        bool full = false;
    };
    struct CommitRun;
    DrawPipeline() = default;
    void run();
    Item* commitFront(Item* front, CommitRun* run);
    void commitRun(Item* front);
    void retire(Item&& item);
    void reap();
    void rethrowFailure();
    void report(std::chrono::steady_clock::time_point now);

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_ROOM_WAKE): a producer waiting for room waits here instead
    // of on `idle`, so a pop wakes it only at half depth and wakes a drain only once empty.
    std::condition_variable room;
    std::deque<Item> items;
    std::exception_ptr failure;
    std::atomic<std::size_t> outstanding{0};
    std::uint32_t idleWaiters = 0;
    std::uint32_t roomWaiters = 0;
    bool committerWaiting = false;
    bool stopped = false;
    bool closing = false;
    std::thread thread;
    std::mutex retireMutex;
    std::condition_variable reapWake;
    std::deque<Item> retired;
    std::thread reaper;
    bool reaperWaiting = false;
    std::uint64_t commits = 0;
    std::uint64_t commitNs = 0;
    std::uint64_t commitErrors = 0;
    std::uint64_t enqueued = 0;
    std::uint64_t fullWaitNs = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrainReason::Count)> drains{};
    std::array<std::uint64_t, static_cast<std::size_t>(DrainReason::Count)> drainWaitNs{};
    std::array<std::uint64_t, 257> drainOpcodes{};
    std::array<std::atomic<std::uint64_t>, 257> drainRequests{};
    std::uint64_t depthSum = 0;
    std::uint64_t runs = 0;
    std::uint64_t runCommits = 0;
    std::atomic<LabelCommit> labelCommit{nullptr};
    std::uint64_t labelsAppended = 0;
    PrepStats prepStats;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

}

#endif
