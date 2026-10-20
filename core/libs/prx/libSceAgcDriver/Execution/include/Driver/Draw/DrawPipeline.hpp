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
#include <memory>
#include <mutex>
#include <optional>
#include <source_location>
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
    // Returns the item's seq (LOCAL EXPERIMENT APS5_EXPERIMENT_PREP_HELPERS, PARALLEL-COMMIT-PLAN B0:
    // assigned under the pipeline mutex at the push, from 1 up, in queue order), or 0 for a commit run
    // inline because the pipeline is closing (no item, so no helper job).
    std::uint64_t Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress = 0, std::vector<std::byte> labelBytes = {}, ItemKind kind = ItemKind::Other, const Prepare* prepare = nullptr);
    // `site` names the caller for the [stream] drain sites (APS5_DEBUG_STREAM_STATS); unused otherwise.
    void Drain(DrainReason reason, std::uint32_t opcode = 0x100, std::source_location site = std::source_location::current());
    // Stops the prep helpers first (PrepHelpers::Stop: their queued jobs' pins go to RetirePin), then
    // closes the pipeline: the committer commits what is queued and ends, later commits run inline.
    void Stop();
    static std::uint64_t DrainRequestsByOpcode(std::uint32_t opcode);
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_PREP_HELPERS, PARALLEL-COMMIT-PLAN B0): the seq of the item
    // whose commit started last (0 before the first): the committer stores it (release) right before
    // it runs an item's commit, and again at the item's pop, which also advances it past the items
    // a stopped pipeline pops without committing. A helper job whose seq is at or below it is stale:
    // its commit started, so its slot is closed or about to be. CurrentItemEnqueuedNs is the push
    // time (Stream::NowNs) of the item the calling committer is committing, 0 outside a commit and
    // without helpers: a thread slot set by the committer around every item's commit. RetirePin
    // hands a helper's pin of a closure to the reaper, so a closure's last reference never drops on
    // a helper (it drops there, on the committer or on queue 0).
    std::uint64_t CommittingSeq() const { return committingSeq.load(std::memory_order_acquire); }
    static std::uint64_t CurrentItemEnqueuedNs();
    void RetirePin(std::shared_ptr<void> pin);
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
    // LOCAL DEBUG (APS5_DEBUG_STREAM_STATS=1, PARALLEL-COMMIT-PLAN S0): the committer's idle
    // attribution. A Drain that finds items queued opens a drain episode (its reason, opcode and
    // call site), which lasts until the next Enqueue: the committer's idle inside it (from the later
    // of the episode's start and the committer running dry, to the push, and then its wake until it
    // takes the pushed item) is filed under the drain, and queue 0's time from the drain's return to
    // that push is the episode's serial phase; idle outside every episode is "producer slow". The
    // shadow dispatch classifier (Dispatch.cpp) reads DrainEpisode() before its drain and names the
    // episode its drain opened with NoteDrainClass (a Stream::DispatchClass), or an episode no
    // dispatch named yet that the gate drain queue 0 makes before a dispatch that follows a wait
    // opened (NoteGateDrain marks it; A1 skips that drain, so the dispatch's class decides), or one
    // a 0x16 packet's own drain opened (named indirect: A1 keeps that drain whatever the dispatch
    // resolves to). An episode an item-eligible class named is cut at the first later point where
    // A1 would still drain: a dispatch of a class that
    // drains (NoteDrainClass, at `drain` when that is not now), or a Drain or a packet that finds
    // nothing queued but would drain a queued dispatch item (NoteWouldDrain). The idle from the cut
    // on is filed under that point, not under the eligible class. Queue 0's GPU waits in the serial
    // phase (Stream::ThreadGpuWait, read on the draining thread) and the shadow's own work
    // (`shadowNs`) are kept beside the idle of the part they fell in. SnapshotWriters copies the
    // queued items' writes (writes and appended labels, with the item's position from the front, 0
    // first) for the reads-overlap test before the drain empties the queue, and returns how many
    // items were queued; inside the episode a gate drain opened, the first call returns the writes
    // NoteGateDrain copied before that drain (A1 skips it for queue 0). DrainEpisode and
    // StreamCuttable read atomics; SnapshotWriters, NoteDrainClass and NoteWouldDrain take the
    // pipeline mutex only when they have something to copy or an episode to name or cut. The
    // [stream] line is printed by Enqueue every 10 s.
    struct QueuedWriter {
        std::uint64_t begin;
        std::uint64_t end;
        std::uint32_t position;
    };
    // A point where A1 would drain, kept for a later NoteDrainClass: its time and the thread's
    // Stream::ThreadGpuWait then (0: the call itself is the point).
    struct StreamPoint {
        std::uint64_t ns = 0;
        std::uint64_t gpuWaitNs = 0;
    };
    std::size_t SnapshotWriters(std::vector<QueuedWriter>& writers, std::uint64_t episode);
    void NoteGateDrain();
    std::uint64_t DrainEpisode() const { return streamEpisodeSerial.load(std::memory_order_relaxed); }
    bool StreamCuttable() const { return streamCuttable.load(std::memory_order_relaxed); }
    void NoteDrainClass(std::uint64_t before, std::uint8_t dispatchClass, std::uint64_t shadowNs, StreamPoint drain);
    void NoteWouldDrain(DrainReason reason, std::uint32_t opcode);

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
        // The push time (Stream::NowNs) under APS5_DEBUG_STREAM_STATS or APS5_EXPERIMENT_PREP_HELPERS,
        // 0 otherwise, and the depth the push found (after the producer's preparation and its room
        // wait): a draw's lead and the depth it was pushed behind.
        std::uint64_t enqueuedAtNs = 0;
        std::uint32_t pushDepth = 0;
        // The item's place in the stream (Enqueue's return; CommittingSeq).
        std::uint64_t seq = 0;
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
    // LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): see SnapshotWriters. Under the pipeline mutex, except
    // reportStream, which releases it before it prints.
    void openStreamEpisode(std::uint64_t startNs, DrainReason reason, std::uint32_t opcode, const std::source_location& site);
    void closeStreamEpisode(std::uint64_t nowNs);
    void cutStreamEpisode(std::size_t slot, StreamPoint at);
    void fileStreamWake(std::uint64_t foundNs, std::uint64_t idleStartNs);
    void copyQueuedWriters(std::vector<QueuedWriter>& writers) const;
    void reportStream(std::unique_lock<std::mutex>& lock);

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
    // Under the pipeline mutex: the last seq given out. The seq of the item whose commit started
    // last (CommittingSeq).
    std::uint64_t lastSeq = 0;
    std::atomic<std::uint64_t> committingSeq{0};
    std::mutex retireMutex;
    std::condition_variable reapWake;
    std::deque<Item> retired;
    // Helper pins of closures (RetirePin), cleared by the reaper with the retired items.
    std::vector<std::shared_ptr<void>> retiredPins;
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
    // LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): the [stream] line's pipeline counts, under the mutex.
    // Busy is timed by the committer's loop around each front (a run under its front's kind),
    // idle from the loop's top to the item it then takes; `commitNs` is the [draw] pipelined
    // commit time over the same window. Drain sites are kept by call site, reason and opcode in
    // StreamSiteSlots slots (the last one takes the rest). The idle by dispatch class, with the
    // class slot LaterDrainSlot for the idle a cut at a drain or a packet took. Draw leads (push to
    // commit start) by LeadBuckets (< 5, 10, 20, 50, 100, 200, 500 us, more) and by the depth
    // pushed behind.
    static constexpr std::size_t StreamSiteSlots = 32;
    static constexpr std::size_t StreamClassSlots = 16;
    static constexpr std::size_t LaterDrainSlot = StreamClassSlots - 1;
    static constexpr std::size_t LeadBuckets = 8;
    static constexpr std::size_t LeadDepthBuckets = 10;
    struct StreamSite {
        const char* file = nullptr;
        std::uint32_t line = 0;
        std::uint32_t opcode = 0;
        DrainReason reason = DrainReason::Packet;
        std::uint64_t episodes = 0;
        std::uint64_t waitNs = 0;
        std::uint64_t idleNs = 0;
        std::uint64_t serialNs = 0;
    };
    // An open episode keeps its drain's site and wait; all of it is counted when it closes, in the
    // window of its close.
    struct StreamEpisode {
        std::uint64_t serial = 0;
        std::uint64_t startNs = 0;
        std::uint64_t returnNs = 0;
        std::uint64_t waitNs = 0;
        const char* file = nullptr;
        std::uint32_t line = 0;
        std::uint32_t opcode = 0;
        DrainReason reason = DrainReason::Packet;
        std::uint8_t dispatchClass = 0;
        std::uint32_t dispatches = 0;
        // Opened by the gate drain NoteGateDrain announced.
        bool gate = false;
        // The draining thread (queue 0) and its ThreadGpuWait at the drain's return; read again on
        // that thread at the cut and at the push.
        std::thread::id thread{};
        std::uint64_t returnGpuWaitNs = 0;
        // The cut (cutNs 0: none): the class slot its idle goes to (a dispatch class, or
        // LaterDrainSlot) and the ThreadGpuWait then; the shadow's work before it and after it.
        std::uint64_t cutNs = 0;
        std::size_t cutSlot = 0;
        std::uint64_t cutGpuWaitNs = 0;
        std::uint64_t shadowNs = 0;
        std::uint64_t cutShadowNs = 0;
    };
    // The committer's wake after the push that closed an episode in which it idled, filed when it
    // takes that item under the episode's site and the class slot its idle ended in.
    struct StreamWake {
        bool pending = false;
        std::size_t site = 0;
        std::size_t classSlot = 0;
        bool cut = false;
        std::uint64_t pushedAtNs = 0;
    };
    struct StreamCounts {
        std::array<std::uint64_t, 3> busyNs{};
        std::uint64_t idleNs = 0;
        std::uint64_t idleWaits = 0;
        std::uint64_t commits = 0;
        std::uint64_t commitNs = 0;
        std::array<StreamSite, StreamSiteSlots> sites{};
        std::uint64_t episodes = 0;
        std::uint64_t drainIdleNs = 0;
        std::uint64_t serialNs = 0;
        std::array<std::uint64_t, StreamClassSlots> classEpisodes{};
        std::array<std::uint64_t, StreamClassSlots> classIdleNs{};
        std::array<std::uint64_t, StreamClassSlots> classGpuWaitNs{};
        std::array<std::uint64_t, StreamClassSlots> classShadowNs{};
        std::uint64_t wakeNs = 0;
        // Item-eligible episodes cut: at a dispatch that drains, at a drain by reason (a packet by
        // opcode instead), and the idle filed under the cuts.
        std::uint64_t dispatchCuts = 0;
        std::array<std::uint64_t, static_cast<std::size_t>(DrainReason::Count)> drainCuts{};
        std::array<std::uint64_t, 257> packetCuts{};
        std::uint64_t cutIdleNs = 0;
        std::uint64_t leadDraws = 0;
        std::uint64_t leadAtLeast20 = 0;
        std::array<std::uint64_t, LeadBuckets> leads{};
        std::array<std::uint64_t, LeadDepthBuckets> leadDepths{};
        std::array<std::uint64_t, LeadDepthBuckets> leadDepthNs{};
    };
    StreamCounts streamCounts;
    StreamEpisode streamEpisode;
    StreamWake streamWake;
    std::uint64_t streamEpisodes = 0;
    // Written under the mutex for the unlocked checks: the open episode's serial (0: none), whether
    // it can still be cut (an item-eligible class named it and nothing cut it yet), and the episode
    // whose first SnapshotWriters takes the writers NoteGateDrain copied (0: none; a push ends it).
    std::atomic<std::uint64_t> streamEpisodeSerial{0};
    std::atomic<bool> streamCuttable{false};
    std::atomic<std::uint64_t> streamGateEpisode{0};
    std::vector<QueuedWriter> streamGateWriters;
    std::size_t streamGateQueued = 0;
    // The serial the gate drain's episode will have (0: none; a push or the next episode ends it).
    std::uint64_t streamGateSerial = 0;
    std::chrono::steady_clock::time_point streamLastReport{};
    // When the committer's loop last went looking for work (0 while it commits); read by the push
    // that closes a drain episode.
    std::atomic<std::uint64_t> streamIdleSinceNs{0};
};

}

#endif
