#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_GUESTMEMORY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_GUESTMEMORY_HPP

#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace AgcDriver::GuestMemory {

void CheckRange(const void* pointer, std::size_t bytes, std::size_t alignment, bool writable = false);
// CheckRange without throwing: whether the whole range is mapped with the requested access.
bool Accessible(const void* pointer, std::size_t bytes, bool writable = false);
// The accessible parts [begin, end) of a range, in address order. GPU heaps are often bound whole while
// the guest commits their pages on demand, so a binding can cover reserved but uncommitted pages.
std::vector<std::pair<std::uint64_t, std::uint64_t>> CommittedRanges(std::uint64_t address, std::size_t bytes, bool writable = false);
// Both facts from one walk of the page tables: the accessible parts of a range and whether they cover
// all of it (`whole`, what Accessible answers). Reserved pages are queried each time, so asking twice
// costs two VirtualQuery per run.
struct Commitment {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    bool whole = false;
};
Commitment DescribeCommitted(std::uint64_t address, std::size_t bytes, bool writable = false);
void CopyGuestBytes(void* destination, const void* source, std::size_t bytes);
void Read(std::uint64_t address, std::span<std::byte> destination, std::size_t alignment = 1);
// Variants for resources in heaps the guest commits on demand (partially resident textures, GPU
// scratch heaps): uncommitted pages read as zeros, are not compared, and are never stored.
void ReadCommitted(std::uint64_t address, std::span<std::byte> destination);
void ReadCommittedUnsynced(std::uint64_t address, std::span<std::byte> destination);
bool EqualsCommitted(std::uint64_t address, std::span<const std::byte> bytes);
// EqualsCommitted without the flush hook: the caller has decided that the recorded GPU work
// writing the range need not be waited for (the dispatch-cache validation in Driver.cpp).
bool EqualsCommittedUnsynced(std::uint64_t address, std::span<const std::byte> bytes);
// EqualsCommittedUnsynced for a caller that treats an unmapped page as a miss (the dispatch-cache
// validation's per-run compare): one walk of the page states, one memcmp per mapped run, no profile
// counter and no reason string. Unmapped: some page of the range is not readable (or the address
// space cannot be queried).
enum class Compare : std::uint8_t { Equal, Differs, Unmapped };
Compare CompareMapped(std::uint64_t address, std::span<const std::byte> bytes);
// CompareMapped's walk with a copy in place of the compare: the bytes at `address` are copied into
// `out` (Equal), or Unmapped as CompareMapped answers it (the dispatch-cache validation's masked
// re-read of a differing run).
Compare CopyMapped(std::uint64_t address, std::span<std::byte> out);
void WriteChangedCommitted(std::uint64_t address, std::span<const std::byte> current, std::span<const std::byte> original);
void Write(std::uint64_t address, std::span<const std::byte> source, std::size_t alignment = 1);
// Stores the parts of `current` that differ from `original` (the guest bytes the GPU started from), so
// guest writes made meanwhile to untouched bytes survive. Compares in 256-byte blocks.
void WriteChanged(std::uint64_t address, std::span<const std::byte> current, std::span<const std::byte> original);

// Write tracking. The guest arena is reserved with page write watching, so CollectWrites learns which
// pages of a range the CPU wrote since they were last collected, stamps their 64 KiB blocks with a new
// generation and returns it; a snapshot of the range taken after that call is current while
// UnchangedSince(range, that generation) holds. Every validation collects its own range first, so
// resources sharing pages see each other's writes. GPU writes into host-imported memory bypass the
// page tables and are reported with MarkWritten. Without write watching CollectWrites returns 0 and
// UnchangedSince is always false, so callers fall back to comparing bytes.
bool WriteWatched();
void Unwatch(std::uint64_t address, std::size_t bytes);
void Rewatch(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges);
bool ImportWatched(std::uint64_t address, std::size_t bytes, const std::function<bool()>& import, bool staysWatched = true);
void SetUnwatchHook(void (*hook)(std::uint64_t address, std::size_t bytes));
bool Watched(std::uint64_t address, std::size_t bytes);
std::uint64_t CollectWrites(std::uint64_t address, std::size_t bytes);
bool UnchangedSince(std::uint64_t address, std::size_t bytes, std::uint64_t generation);
// UnchangedSince for several ranges under one tracker lock: true only when every one holds.
struct UnchangedQuery {
    std::uint64_t address;
    std::size_t bytes;
    std::uint64_t generation;
};
bool WrittenSince(std::uint64_t address, std::size_t bytes, std::uint64_t generation);
bool UnchangedSinceAll(std::span<const UnchangedQuery> queries);
std::uint64_t CollectCommittedWrites(std::uint64_t address, std::size_t bytes, std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges);
bool UnchangedCommittedSince(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges, std::uint64_t generation);
// Returns the generation the blocks were stamped with (0 when the arena is not write-watched):
// UnchangedSince(range, it) holds until the next store over the range.
std::uint64_t MarkWritten(std::uint64_t address, std::size_t bytes);
std::uint64_t StoreOwnBytes(std::uint64_t address, std::size_t bytes, const std::function<void()>& store);
std::uint64_t StoreOwnBytes(std::uint64_t address, std::span<const std::byte> source);
bool StoredOver(std::uint64_t address, std::size_t bytes, std::uint64_t generation);
bool StoredAllSince_nid_no_patch(std::uint64_t address, std::size_t bytes, std::uint64_t generation);
// Collect epoch: within one epoch a range already collected is not walked again, CollectWrites
// returns the current generation instead. A guest write landing between two collects of the same
// epoch is seen by the next epoch, which is the ordering real hardware gives a CPU write made while a
// submission runs. The driver's own writes (Write, WriteChanged, MarkWritten) stamp their blocks, so
// they are never hidden. APS5_NO_COLLECT_MEMO=1 disables the memo. CollectWritesUncached always walks:
// for decisions about which blocks the CPU wrote while a GPU result was pending (a stale answer there
// stores old texels over reused memory).
// The epoch is per queue worker: BumpCollectEpoch advances the calling thread's epoch to a value no
// other thread ever holds (one worker's walk never satisfies another's check) at the hardware
// ordering points of its queue: the start of a submission, a WAIT_REG_MEM the CPU may have satisfied
// (at entry, polled, timed out, or by another queue's recorded label; a label this queue recorded
// itself orders no CPU write), a device drain, and before reaping completions outside a submission.
// A thread that never bumps (the presenter, game threads inside the flush hook) gets a fresh epoch
// for every collect: nothing is reused without an ordering point. APS5_PACKET_EPOCH=1 makes the
// workers bump before every packet as before. CollectEpochBumps counts the bumps ([guestmem] line).
// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_RECEIPTS, COMMITTER-SPLIT-PLAN Q3) amends "one worker's walk
// never satisfies another's check" for exactly one pair of threads: a walk that queue 0 (the owner
// of DrawPipeline's EpochToken, SetReceiptOwner) or the draw committer (its follower,
// SetReceiptFollower) completed under (EpochToken T, the walker's epoch for T, UnwatchSerial S)
// satisfies the committer's collects while it follows T in the epoch it took for T and S still
// holds. Only queue 0 advances T, in the same call as its epoch, so such a walk followed T's
// ordering point. Queue 0 never consults these receipts for its own collects; draw-ahead and
// speculative walks and threads without a role never publish one.
void BumpCollectEpoch();
void AdoptLatestCollectEpoch();
std::uint64_t CollectEpoch();
std::uint64_t CollectEpochBumps();
std::uint64_t ThreadCollectEpoch();
std::uint64_t UnwatchSerial();
std::uint64_t CollectWritesUncached(std::uint64_t address, std::size_t bytes);
// The tracker's current generation (every collect and MarkWritten bumps it; under
// APS5_EXPERIMENT_LAZY_GENERATION a collect bumps it only when it stamps): a stamp taken after
// a set of driver stores, for UnchangedSinceCollected to compare against later. Read under the
// tracker mutex, so every collect that bumped before it has finished its walk and every later
// collect stamps newer.
std::uint64_t TrackerGeneration();
std::uint64_t NewestStamp(std::uint64_t address, std::size_t bytes);
std::uint64_t UntrackedHint(std::uint64_t address, std::size_t bytes);
// Whether no CPU write touched the range since `generation`: an uncached resetting collect of the
// range's pages first (a game store not collected yet becomes a block stamp), then the compare over
// the stamps collects make (dirty pages), not the MarkWritten stamps of the driver's own GPU label
// records, which share the title's label blocks. False when the arena is not write-watched.
bool UnchangedSinceCollected(std::uint64_t address, std::size_t bytes, std::uint64_t generation);
// Per 64 KiB tracker block of [address, address + bytes) (block 0 holds `address`): `changed[k]`
// receives whether the block was stamped after generations[k] (UnchangedSince over that block
// alone), `cpu[k]`, when given, whether a collect stamped it after (a CPU store, as
// UnchangedSinceCollected tells them apart from the driver's MarkWritten stamps, but without its
// walk: the caller collected the range first). One tracker lock for the whole range. Outside the
// watched arena, or for a generation of 0, a block reads as changed.
constexpr std::uint8_t BlockUnchanged = 0;
constexpr std::uint8_t BlockWritten = 1;
bool ChangedBlocks(std::uint64_t address, std::size_t bytes, std::span<const std::uint64_t> generations, std::span<std::uint8_t> changed, std::span<std::uint8_t> cpu = {});

// Serializes device work: draws, dispatches, presentation and the deferred write-backs below. The
// mutex is recursive; it is wrapped so every acquisition (std::lock_guard at any site) measures how
// long it waited: the graphics worker's frame is mostly spent waiting for this lock behind the
// compute workers, and that wait has to stay visible. Under APS5_PROFILE_DRAW each thread reports its
// waits in a [lock] line every 10 s (APS5_NO_LOCK_PROFILE=1 turns the timing off); a queue worker tags
// its thread with TagGpuLockThread so the line names the queue.
class GpuMutexType {
public:
    void lock();
    bool try_lock();
    void unlock();
    // Whether the calling thread holds the mutex (at any depth): the APS5_ASSERT_GPU_LOCK checks
    // (AssertGpuLockHeld) and code that must know whether it runs inside the device lock.
    bool HeldByThisThread() const;
    // The calling thread's recursion depth (0 when it does not hold the mutex): 1 means its
    // current acquisition is the outermost, so code that releases the mutex around a wait
    // (Recorder::SyncThrough for the flush hook) gives up no caller's hold.
    std::uint32_t DepthOnThisThread() const;
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_COMMIT_RUNS=1): how many threads are blocked in lock() (0
    // without the switch, which counts nothing). The draw committer holds the mutex across a run of
    // commits and ends the run at the next commit when another thread waits.
    std::uint32_t Waiters() const { return waiting.load(std::memory_order_relaxed); }

private:
    void enter();
    bool tryEnter();
    void acquired();
    Mutex mutex;
    // The holder's thread token and recursion depth, written by the holder only; another thread
    // reads the owner just to see that it is not itself.
    std::atomic<const void*> owner{nullptr};
    std::uint32_t depth = 0;
    std::atomic<std::uint32_t> waiting{0};
};
GpuMutexType& GpuMutex();
// Called on the thread that just gave up its outermost hold, right after the mutex was released:
// work the hold deferred to run without it (the recorder's release of the objects completed batches
// kept). One hook, set once; it must not take GpuMutex itself.
void SetGpuUnlockHook(void (*hook)());
std::uint64_t GpuHoldSerial();
// Debug aid (APS5_ASSERT_GPU_LOCK=1): aborts with a [lock] line naming `where` when the calling
// thread does not hold GpuMutex. Recording, submits, command batches and the detiler's pools must
// only be reached under it (a resource build's stage A runs without it, see ShaderResources).
void AssertGpuLockHeld(const char* where);
// LOCAL EXPERIMENT (COMMITTER-SPLIT-PLAN, room-wait draw tickets): queue 0 prepares a draw's ticket
// inside DrawPipeline::Enqueue without GpuMutex, and must never take it there (a committer that
// waits for queue 0 while queue 0 waits for the mutex would deadlock, and nothing it prepares may
// record). The scope marks the calling thread for its lifetime; under APS5_ASSERT_GPU_LOCK=1 a
// lock() of GpuMutex inside it aborts with a [lock] line. LOCAL DEBUG (PARALLEL-COMMIT-PLAN S0): so
// does a try_lock() (GpuMutexType::tryEnter) and a HostImportFor call (AssertOutsidePrepareScope),
// which only GpuMutex holders may make (COMMITTER-MAP 3.3).
class PrepareScope {
public:
    PrepareScope();
    ~PrepareScope();
    PrepareScope(const PrepareScope&) = delete;
    PrepareScope& operator=(const PrepareScope&) = delete;

private:
    bool previous;
};
// LOCAL DEBUG (APS5_ASSERT_GPU_LOCK=1, PARALLEL-COMMIT-PLAN S0): aborts with a [lock] line naming
// `where` when the calling thread is inside a PrepareScope (a thread that must never take GpuMutex
// nor do what only its holders do). Off, one predictable branch.
void AssertOutsidePrepareScope(const char* where);
// LOCAL EXPERIMENT (PARALLEL-COMMIT-PLAN A3, Graphics::QueueGate): the Vulkan queue mutexes the
// calling thread holds, one bit per QueueGate::Queue, kept by a gate that is on. The lock order is
// GpuMutex before any queue mutex: under APS5_ASSERT_GPU_LOCK=1 a GpuMutex lock() that would
// acquire it (not a holder's nested one) while a bit is set aborts with a [lock] line.
std::uint8_t QueueMutexesHeld();
void SetQueueMutexesHeld(std::uint8_t held);
// LOCAL DEBUG (APS5_DEBUG_STREAM_STATS=1, PARALLEL-COMMIT-PLAN S0): the switch of the [stream] line
// (DrawPipeline prints it every 10 s; counters in StreamStats.hpp), read once: 0 when unset. With
// it on, threads carry a role for the per-role counters (the tracker lock, GpuMutex waits, submits):
// the draw committer (DrawPipeline::run), queue 0 and the compute queues (TagGpuLockThread), the
// presenter (MarkPresenterThread) and the prep helpers (B0). Unset roles are Other. Set only while
// the switch is on; read back as Other otherwise.
int StreamStats();
enum class StreamRole : std::uint8_t { Other, Committer, Queue0, Compute, Presenter, Helper, Count };
void SetStreamRole(StreamRole role);
StreamRole CurrentStreamRole();
// LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS=1|2|3): the level of the switch (0 when unset), read once.
// 1 counts what queue 0 finds at its draw enqueues and the committer's write-watch walks by draw
// step and enqueue depth; 2 also shadows tickets (built, compared at commit, never used); 3 also
// the byte-derived shadow. Printed every 10 s as [q0prep].
int Q0PrepStats();
// LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS): the committer's write-watch walks (completed and failed
// collects that reached the tracker, not memo hits) by the draw step that made them
// (WalkCallerScope, innermost wins) and by the queue depth the draw was enqueued behind
// (SetWalkItemDepth, set by the committer around a draw's commit: depth + 1, 0 outside one).
// Walks outside a committed draw are not counted. The time of walks in draws enqueued at full
// depth (DrawPipeline's PrepareBudget::full test, passed as `full`) is also summed apart, since
// DrawPipeline::Depth() may exceed the last depth bucket. Inputs: the collects of CopyDrawInput's
// index and vertex copies (most draw inputs under APS5_EXPERIMENT_DRAW_INPUTS_VRAM).
enum class WalkCaller : std::uint8_t { Other, Revalidate, Index, Target, Depth, Bindings, Inputs, Count };
constexpr std::size_t WalkDepthBuckets = 10;
class WalkCallerScope {
public:
    explicit WalkCallerScope(WalkCaller caller);
    ~WalkCallerScope();
    WalkCallerScope(const WalkCallerScope&) = delete;
    WalkCallerScope& operator=(const WalkCallerScope&) = delete;

private:
    std::uint32_t previous;
    bool active;
};
void SetWalkItemDepth(std::uint32_t depthPlusOne, bool full = false);
struct WalkStats {
    std::array<std::array<std::uint64_t, WalkDepthBuckets>, static_cast<std::size_t>(WalkCaller::Count)> walks{};
    std::array<std::array<std::uint64_t, WalkDepthBuckets>, static_cast<std::size_t>(WalkCaller::Count)> ns{};
    std::array<std::array<std::uint64_t, WalkDepthBuckets>, static_cast<std::size_t>(WalkCaller::Count)> bytes{};
    std::array<std::uint64_t, static_cast<std::size_t>(WalkCaller::Count)> fullNs{};
};
// The counts since the last call (reset by it).
WalkStats TakeWalkStats();
// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_RECEIPTS, COMMITTER-SPLIT-PLAN Q3): first-in-token walk
// receipts (the amended contract above BumpCollectEpoch). 0 (unset or 0) off, today's code exactly.
// 1: queue 0 walks the ranges the committer would walk first for a draw (WalkReceipts, from its
// ticket preparation) and publishes every completed chunk into a ring of 4 token slots (merged
// ranges per token and UnwatchSerial, a slot reset only by a newer token); the committer publishes
// its own completed memoized walks there too (APS5_EXPERIMENT_Q0_RECEIPT_PUBLISH, default on, 0
// off), so queue 0 skips them; and a committer collect that missed its own memos is served from
// the ring when the slot of the token it follows covers the range under the current UnwatchSerial
// and its epoch is still the one it took for that token: the range joins its epoch coverage and
// the current generation is returned with an acquire load, as a memo hit returns it, so every
// later stamp check still runs. The ring is only try_locked by the committer (a busy ring walks)
// and is never taken under the tracker lock. 2 counts only: the committer counts the hits it would
// have served (and how many of those walks then stamped a CPU write on a page of the requested
// range, the writes a receipt would have hidden) and still walks. Needs
// APS5_EXPERIMENT_Q0_TICKET, APS5_DEBUG_COVERAGE=1 (a hit joins the epoch coverage) and
// APS5_EXPERIMENT_LAZY_GENERATION=1 (queue 0's clean walks must not move the tracker generation and
// retire the committer's slice memos); without one of them it stays off (said once). Read once.
// LOCAL DEBUG (APS5_DEBUG_Q0_VERIFY_WALKS=1, Windows): at each served hit the committer peeks the
// range's write watch without resetting it and counts the dirty pages the receipt hid (writes made
// after the receipt's walk, which this token does not see; expect a small racy count).
int Q0Receipts();
// Queue 0, right after its bump made `token` (Driver::bumpEpoch): it owns that token in the epoch it
// now has. No-op while receipts are off.
void SetReceiptOwner(std::uint64_t token);
// The draw committer, right after DrawPipeline::FollowEpoch bumped its epoch for `token`.
void SetReceiptFollower(std::uint64_t token);
// The ranges queue 0 walks for one draw's receipts: page-rounded [begin, end), sorted and disjoint
// (PlanReceiptWalks), the range being walked (`next`), where in it (`cursor`, 0 at its start) and
// the bytes walked so far (the per-draw budget).
struct ReceiptWalks {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::size_t next = 0;
    std::uint64_t cursor = 0;
    std::uint64_t walked = 0;
    bool Done() const { return next >= ranges.size(); }
};
// Replaces `walks` with `planned` (address ranges [begin, end)) rounded out to whole pages, sorted
// and merged, so a page two ranges share is walked once.
void PlanReceiptWalks(std::span<const std::pair<std::uint64_t, std::uint64_t>> planned, ReceiptWalks& walks);
// Queue 0, owner of `token` in the epoch it took for it (otherwise it drops the list and walks
// nothing): walks the rest of `walks` in chunks of APS5_EXPERIMENT_Q0_RECEIPT_CHUNK_KIB (default
// 1024, chunk-aligned), each under one tracker hold of its own (collectWrites' initialize, covers
// and walkWrites), so the committer never waits for more than one chunk. Coverage is decided per
// run of a range, not per chunk: a run queue 0 already published (or found in the ring) under this
// token, serial and epoch is skipped without a lock (its own note, kept per thread and cleared
// when any of the three changes); a run its own epoch coverage holds (walked by this thread in
// this epoch) is published in one ring hold without a walk; a run the ring holds is skipped after
// one try_lock; only the rest is chunked and walked, up to the next run already held. A completed
// chunk joins its epoch coverage, as any completed collect, and the ring. `stop` is asked before
// each test or chunk and the draw stops at `budget` bytes walked; returns false when it stopped
// with ranges left. `afterPush` only attributes the time ([q0prep]).
bool WalkReceipts(std::uint64_t token, ReceiptWalks& walks, std::uint64_t budget, bool (*stop)(const void* context), const void* context, bool afterPush);
// LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS with APS5_EXPERIMENT_Q0_RECEIPTS): the [q0prep] receipt
// counts since the last call (reset by it). Queue 0: draws and ranges planned (and the planning
// time), chunks walked
// (bytes and ns: the walk time moved to it, before and after the push), runs skipped because its
// own note, the ring or its own coverage held them, its ring mutex acquisitions (blocking and
// try_lock), chunks not watched or failed, ranges published, draws it could not publish for (no
// owner role for the draw's token) or left unfinished. Committer: its own walks published, hits
// served (bytes, by draw step), would-be hits under =2 and how many of those walks stamped a page
// of the requested range (and, apart, anywhere in the walk widened to whole blocks), misses by
// reason, dirty pages a hit hid (APS5_DEBUG_Q0_VERIFY_WALKS).
enum class ReceiptMiss : std::uint8_t { Token, Epoch, Serial, Uncovered, Busy, Count };
struct ReceiptStats {
    std::uint64_t draws = 0;
    std::uint64_t ranges = 0;
    std::uint64_t rangeBytes = 0;
    std::uint64_t planNs = 0;
    std::uint64_t chunks = 0;
    std::uint64_t walkedBytes = 0;
    std::array<std::uint64_t, 2> walkNs{};
    std::array<std::uint64_t, 2> spentNs{};
    std::uint64_t ringCovered = 0;
    std::uint64_t ownCovered = 0;
    std::uint64_t notedCovered = 0;
    std::uint64_t ringLocks = 0;
    std::uint64_t unwatched = 0;
    std::uint64_t failed = 0;
    std::uint64_t published = 0;
    std::uint64_t notOwner = 0;
    std::uint64_t unfinished = 0;
    std::uint64_t committerPublished = 0;
    std::uint64_t hits = 0;
    std::uint64_t hitBytes = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(WalkCaller::Count)> hitsByCaller{};
    std::uint64_t wouldHits = 0;
    std::uint64_t wouldHitsStamped = 0;
    std::uint64_t wouldHitsWidened = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(ReceiptMiss::Count)> misses{};
    std::uint64_t hiddenPages = 0;
};
void NoteReceiptPlan(std::size_t ranges, std::uint64_t bytes, std::uint64_t ns);
ReceiptStats TakeReceiptStats();
void TagGpuLockThread(std::uint32_t queue);
// The presenter stays untagged (its tag is a queue id to the recorder) but takes its own column
// of the [lock] holder matrix, apart from the game threads.
void MarkPresenterThread();
// The queue tag of the calling thread (0xffffffff when untagged), for reports that name the thread.
std::uint32_t GpuLockThreadTag();
// Lock sites (APS5_PROFILE_DRAW): a caller names the site of its next lock() so the [lock] line
// splits acquisitions and waits per site (whose holds queue 0 waits behind is then measurable). The
// tag is consumed by that acquisition, so a nested lock inside the flush hook is attributed to the
// hook, and an acquisition nobody tagged counts as "other". Try-only acquisitions are not counted
// as acquisitions; the hold they start is timed under "try". Holds are timed per site as well (the
// outermost acquisition of a thread until its unlock; APS5_NO_HOLD_PROFILE=1 turns that off), so
// the line shows who holds the mutex, not only who waits for it.
enum class GpuLockSite : std::uint8_t { Other = 0, Dispatch, Indirect, Draw, Hook, Wait, Label, Flush, Present, Fill, Copy, End, Try, Count };
void TagGpuLockSite(GpuLockSite site);
// LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): the site tagged for the calling thread's outermost GpuMutex
// hold (Try for an untagged try_lock, Other when untagged or the switch is off), for the [stream]
// submit split; GpuLockSiteName names it.
GpuLockSite CurrentGpuHoldSite();
const char* GpuLockSiteName(GpuLockSite site);
// A GPU wait (a fence or timeline wait that found its work unfinished) made while the calling
// thread holds GpuMutex: counted on the thread's [lock] line as 'locked GPU waits' (APS5_PROFILE_DRAW).
void NoteLockedGpuWait(double ms);
// Bytes the calling thread's write-watch collects walked so far (0 without the memory profile):
// a caller reads the difference across a span of its own work.
std::uint64_t ThreadCollectedBytes();
// A code address (a return address) as an offset into its module, in the form of the [guestmem]
// callers (symbolized with nm against the driver's image). Takes the loader lock: for reports only.
unsigned long long CodeOffset(const void* address);

// Deferred GPU writes. The graphics layer keeps the results of storage images on the GPU until guest
// memory is read; it registers a hook that stores every pending result overlapping a range, which
// every read and store through this interface calls first. Code that reads guest memory directly
// calls FlushGpuWrites itself.
void SetFlushHook(void (*hook)(std::uint64_t address, std::size_t bytes));
void SetReadGuard(void (*guard)(std::uint64_t address, std::size_t bytes));
void GuardRead(std::uint64_t address, std::size_t bytes);
void FlushGpuWrites(std::uint64_t address, std::size_t bytes);
class ISpeculativeReadGate {
public:
    virtual void Enter() = 0;
    virtual void Leave() = 0;

protected:
    ~ISpeculativeReadGate() = default;
};
class SpeculativeReadRefused : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
class SpeculativeReadScope {
public:
    explicit SpeculativeReadScope(ISpeculativeReadGate& gate);
    ~SpeculativeReadScope();
    SpeculativeReadScope(const SpeculativeReadScope&) = delete;
    SpeculativeReadScope& operator=(const SpeculativeReadScope&) = delete;

private:
    ISpeculativeReadGate* previous;
};
bool SpeculativeReadsSupported();
bool SpeculativeReader();
// Moves with every ForgetPages call (the libc invalidator: memory unmapped or re-registered): odd
// while the call stores its page states, moved past it after, so a reader that loads an even
// value before consulting the page states and reading directly, and the same value after, knows
// the states it consulted were current and no page it read changed its mapping meanwhile.
std::uint64_t ForgetSerial();

// Attribution of guest memory accesses (the [hooksync] line in Recorder.cpp and the read-site
// counts of the [guestmem] line, APS5_PROFILE_DRAW): a sync the flush hook makes for an access has
// to be charged to the PM4 packet and the driver code that made the access. A queue worker names
// the packet it executes at the top of its packet loop (SetCurrentPacket; opcode 0xffff is the
// flip); a thread that runs no packets (the presenter, game threads) reads back NoPacket. The main
// read paths name themselves with a ReadSiteScope; an access made by none of them is attributed by
// its return addresses instead (CaptureCallerOffsets).
struct PacketTag {
    std::uint32_t opcode;
    std::uint32_t queue;
};
constexpr std::uint32_t NoPacket = 0xfffffffeu;
void SetCurrentPacket(std::uint32_t opcode, std::uint32_t queue);
PacketTag CurrentPacket();

// Store: a CPU write (Write/WriteChanged) rather than a read; a hook sync made for one orders a
// write after a pending GPU write, so whether the bytes change says nothing about its necessity.
// MirrorRefresh: an address-based build bringing an image mirror up to date with guest memory
// (GuestBufferMemory.cpp: the flush before a writable mirror's compare, a new mirror's fill).
// DrawCache: a draw entry's per-stage value validation (Driver.cpp drawCache), where a draw
// capture's hook waits reappear on a hit.
// Every enumerator needs its name in ReadSiteName's table (GuestMemory.cpp).
enum class ReadSite : std::uint8_t { Unknown = 0, Capture, DispatchCache, TextureCompare, TextureRead, BufferUpload, IndexBuffer, VertexBuffer, Registers, IndirectArguments, Wait, Label, Scanout, Store, MirrorRefresh, DrawCache, Count };
const char* ReadSiteName(ReadSite site);
// Sets the calling thread's read site and returns the previous one (nested scopes restore it).
ReadSite SetReadSite(ReadSite site);
ReadSite CurrentReadSite();
class ReadSiteScope {
public:
    explicit ReadSiteScope(ReadSite site) : previous(SetReadSite(site)) {}
    ~ReadSiteScope() { SetReadSite(previous); }
    ReadSiteScope(const ReadSiteScope&) = delete;
    ReadSiteScope& operator=(const ReadSiteScope&) = delete;

private:
    ReadSite previous;
};
// Return addresses above the caller as offsets into the driver's module (the same form as the
// [guestmem] callers, symbolized with nm against the driver's image; an address outside the module
// is written as 0), newest first, skipping `skip` frames above the caller; returns how many were
// written. Stack unwinding, so only for rare, profiled events.
std::size_t CaptureCallerOffsets(std::span<unsigned long long> frames, unsigned skip);

}

extern "C" void AgcDriverCheckGuestMemory_nid_postfix(const void* pointer, std::size_t bytes, std::size_t alignment, bool writable = false);
extern "C" void AgcDriverStoreGuestLabel_nid_postfix(std::uint64_t* pointer, std::uint64_t value);

#endif
