#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRAINSTATS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRAINSTATS_HPP

#include "prx/libSceAgcDriver/Execution/include/StreamStats.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

// LOCAL DEBUG (DRAIN-PLAN section 6, D0): the counters that decide the queue-0 drain levers, beside
// the [stream] line (DrawPipeline keeps C0, the window tag, and C4, the refill split, itself):
// C1 [gds] (APS5_DEBUG_GDS_STATS=1), C2 [dma0], C3 [gate-skip], C5 [recipe] dispatch and
// [dispatch-cache] indirect, C6 [dispatch-guards] and C7 [zpass] (APS5_DEBUG_STREAM_STATS=1). They
// count only; nothing here changes what a packet does. With the switches unset every entry point
// returns after a cached flag (the inline checks below) and nothing is counted. The lines of a
// [stream] window are printed by ReportWindow right after the [stream] line, tagged with its window;
// [gds] without stream stats prints on its own every 10 s, tagged window=-.
namespace AgcDriver::Drain {

using Range = std::pair<std::uint64_t, std::uint64_t>;
// Per-queue rows (C1) use this many slots, given out in first-seen order; the last one takes every
// later queue.
constexpr std::size_t QueueSlots = 8;

// APS5_DEBUG_GDS_STATS (C1), read once.
bool GdsOn();
inline bool GdsCounting() {
    static const bool on = GdsOn();
    return on;
}
// Either switch: the per-packet records of C1, C2 and C7 open.
inline bool PacketCounting() {
    static const bool on = Stream::On() || GdsCounting();
    return on;
}

// C0: what a [stream] window covers, from its presents and the draws committed per present: the
// 60 Hz menu (many light frames), the hold (heavy frames), and the hub-entry transient between them
// (and windows without presents: loading). The thresholds are MenuPresents and HoldDrawsPerPresent.
enum class Window : std::uint8_t { Menu, Entry, Hold, Count };
inline constexpr std::uint64_t MenuPresents = 300;
inline constexpr std::uint64_t HoldDrawsPerPresent = 2000;
Window ClassifyWindow(std::uint64_t presents, std::uint64_t draws);
const char* WindowName(Window window);
// Prints the window's D0 lines ([dma0], [gate-skip], [dispatch-guards], [zpass], [recipe] dispatch,
// [dispatch-cache] indirect, and [gds] when APS5_DEBUG_GDS_STATS is on) and starts the next window.
// DrawPipeline::reportStream calls it after the [stream] line, without the pipeline mutex; its
// first call (`print` false) drops what was counted before the first window.
void ReportWindow(Window window, bool print = true);

// C1/C2: a DMA_DATA by what it moves (source to destination; Prefetch has no destination) and by the
// way it went: Item (queued on the committer, APS5_PIPELINE_DMA), Nothing (no bytes, or ordered
// already without a store), GpuStore (WriteLabelOnGpu reason 0), GpuCopy (CopyBuffer paths 1/3),
// GpuFill (a DMA fill), CpuAfterWait / CpuIdle (moved on the CPU: a targeted GDS sync, a device
// drain or a host copy, then Pm4::Execute or GuestMemory::Write; after a GPU wait or not), and
// Fallback (WriteLabelOnGpu refused, its reason kept: 1 CPU store, 3/4/6 drained, 5 behind
// completions).
enum class DmaKind : std::uint8_t { ImmToMem, MemToMem, ImmToGds, MemToGds, GdsToMem, GdsToGds, Prefetch, Count };
enum class DmaPath : std::uint8_t { Item, Nothing, GpuStore, GpuCopy, GpuFill, CpuAfterWait, CpuIdle, Fallback, Count };
constexpr std::size_t DmaKinds = static_cast<std::size_t>(DmaKind::Count);
constexpr std::size_t DmaPaths = static_cast<std::size_t>(DmaPath::Count);
constexpr std::size_t FallbackReasons = 8;
DmaKind ClassifyDma(std::span<const std::uint32_t> packet);
constexpr bool TouchesGds(DmaKind kind) {
    return kind == DmaKind::ImmToGds || kind == DmaKind::MemToGds || kind == DmaKind::GdsToMem || kind == DmaKind::GdsToGds;
}
// preparePacketMemory's decision for the DMA_DATA it handles, kept for the record open on the
// thread (PacketRecord::Arm); nothing without one.
void noteDmaPath(DmaPath path, int reason);
inline void NoteDmaPath(DmaPath path, int reason = 0) {
    if (PacketCounting()) noteDmaPath(path, reason);
}

// One packet's record, on the queue worker that executes it (PacketExecution): a DMA_DATA (C2 on
// queue 0 under stream stats; C1 for one that touches the GDS under GDS stats) or a ZPASS_DONE
// sample dump (C7, queue 0 under stream stats). Start opens it, NoteDrain keeps the packet drain
// (PacketExecution's queue-0 drain before the packet: its wait and the drain episode it opened), Arm
// starts the packet's own part (after that drain: queue 0's work and GPU waits from here on; C1's
// GDS writers then), Finish files it. A record destroyed without Finish (a predicated packet that
// was skipped) counts nothing.
class PacketRecord {
public:
    PacketRecord() = default;
    // Inline: every packet of the loop has a record, which off the switches is never armed.
    ~PacketRecord() {
        if (armed) release();
    }
    PacketRecord(const PacketRecord&) = delete;
    PacketRecord& operator=(const PacketRecord&) = delete;
    void Start(std::span<const std::uint32_t> packet, std::uint32_t opcode, std::uint32_t queue);
    bool Active() const { return active; }
    void NoteDrain(std::uint64_t waitNs, std::uint64_t episodeBefore, std::uint64_t episodeAfter);
    void Arm();
    void Finish(bool wroteOnGpu);

private:
    void release();
    bool active = false;
    bool armed = false;
    bool zpass = false;
    bool stream = false;
    bool gds = false;
    bool drained = false;
    DmaKind kind = DmaKind::Prefetch;
    std::uint8_t target = 0;
    std::uint32_t queue = 0;
    std::uint64_t drainWaitNs = 0;
    std::uint64_t episode = 0;
    std::uint64_t startNs = 0;
    std::uint64_t gpuWaitNs = 0;
    std::uint64_t destination = 0;
    std::size_t bytes = 0;
};
// A DMA_DATA queued as a committer item (PacketExecution, APS5_PIPELINE_DMA).
void CountDmaItem(std::span<const std::uint32_t> packet, std::uint32_t queue);

// The tags DrawPipeline files a closed drain episode's idle under (TagStreamEpisode): the C2 row of
// the DMA_DATA whose drain opened it (TagDma0 | kind * DmaPaths + path) or C7's (TagZpass).
constexpr std::uint16_t TagDma0 = 0x100;
constexpr std::uint16_t TagZpass = 0x200;
// Under the pipeline mutex: relaxed adds only. `closed`: the episode's close (its idle and serial
// phase), else the committer's wake after the push that closed it (idle only).
void FileTaggedEpisode(std::uint16_t tag, std::uint64_t idleNs, std::uint64_t serialNs, bool closed);

// C1: the first later touch of a GDS readback's destination (a GDS-to-memory DMA_DATA; 64 per
// producer queue, the oldest aging out). Wait is a WAIT_REG_MEM on it (by the waiting queue and how
// it was satisfied); DispatchArguments readDispatchArguments; CondExec a COND_EXEC; Predication a
// predicated packet's test; DmaSource a later DMA_DATA's source; HookRead any other guest access
// through the flush hook; Overwritten a CPU store or a label store over it first.
enum class Touch : std::uint8_t { Wait, DispatchArguments, CondExec, Predication, DmaSource, HookRead, Overwritten, AgedOut, Count };
enum class WaitHow : std::uint8_t { Entry, Held, Stored, Table, Polled, Count };
void noteTouch(Touch touch, std::uint64_t address, std::size_t bytes);
inline void NoteTouch(Touch touch, std::uint64_t address, std::size_t bytes) {
    if (GdsCounting()) noteTouch(touch, address, bytes);
}
// GuestMemory's flush hook: a CPU access (a store when `store`).
void noteGuestAccess(std::uint64_t address, std::size_t bytes, bool store);
inline void NoteGuestAccess(std::uint64_t address, std::size_t bytes, bool store) {
    if (GdsCounting()) noteGuestAccess(address, bytes, store);
}
// Driver::waitMemory's record: the readback destinations it reads (taken at once, filed under Wait
// with how it ended), and for queue 0 the 0x93 blame of a wait that polled: the producer queue of
// the label that satisfied it and whether that queue spent time in a GDS DMA_DATA's GPU wait
// meanwhile. The caller sets `how`, `producer` and `polled` before it returns.
class WaitRecord {
public:
    WaitRecord(std::uint32_t queueId, std::uint64_t awaited, std::size_t bytes) : active(GdsCounting()), queue(queueId) {
        if (active) start(awaited, bytes);
    }
    ~WaitRecord() {
        if (active) finish();
    }
    WaitRecord(const WaitRecord&) = delete;
    WaitRecord& operator=(const WaitRecord&) = delete;
    bool Active() const { return active; }
    WaitHow how = WaitHow::Polled;
    std::uint32_t producer = ~0u;
    bool polled = false;

private:
    void start(std::uint64_t awaited, std::size_t bytes);
    void finish();
    bool active = false;
    std::uint32_t queue = 0;
    // The producer slots of the destinations taken (the first four), and how many were taken.
    std::array<std::uint8_t, 4> producers{};
    std::uint32_t taken = 0;
    std::uint64_t startNs = 0;
    std::array<std::uint64_t, QueueSlots> gdsWaitNs{};
    std::array<bool, QueueSlots> inGds{};
};

// C3 (APS5_DEBUG_STREAM_STATS): DRAIN-PLAN D4's shadow. The queue-0 gate drain before a 0x15 that
// follows a gate (PacketExecution) with items queued: whether D4 would skip it or the first refusal,
// in the plan's order; then the window from that drain's return to the first exit, on queue 0's
// thread: the fill queued as an item, the dispatch returning before any other exit, the MSAA
// single-clear or memory-value-fill note (D1's guards), a read of shader code or fill inputs that
// overlaps the writes queued at the gate or comes after a hidden-write draw (Capture: D4 would drain
// there), or the non-fill dispatch's own drain (Dispatch.cpp). W is the gate drain's wait; G the
// window up to a FillItem or NonFill exit; D4's overlap estimate is the sum of min(W, G).
enum class GateRefusal : std::uint8_t { None, DeferredLabels, LabelBatchSubmit, A1, GroupCapture, UnorderedFills, HiddenWrites, Count };
enum class GateExit : std::uint8_t { FillItem, Return, MsaaGuard, ValueFillGuard, Capture, NonFill, Count };
// The thread's window writes, cleared, for the caller to fill before the drain.
std::vector<Range>& GateWrites();
void GateOpen(GateRefusal refusal, std::uint64_t waitNs, bool hidden);
// A gate-first 0x15 with items queued that opens no window (A1 skips the drain).
void GateCounted(GateRefusal refusal);
bool GateWindowOpen();
void GateCapture(std::uint64_t address, std::size_t bytes);
void GateClose(GateExit exit);

// LOCAL EXPERIMENT (DRAIN-PLAN D1): the queue-0 ordering guards. APS5_EXPERIMENT_DISPATCH_NOTE_GUARDS
// drains before TakeMultisampledSingleClear and NoteMemoryValueFill when items are queued (G-msaa,
// G-vfill: both notes read guest memory and change state queued draws read at their commit);
// APS5_EXPERIMENT_HIDDEN_WRITE_DRAIN drains before an unregistered code read (ReadRawShader) or a
// fill's input reads (drainPendingInputWrites) when a draw that may write guest memory it does not
// declare is queued (HiddenWritesQueued). Each =1 drains, =2 only counts (C6); unset (or 0) is
// today. Read once.
int noteGuardMode();
int hiddenWriteMode();
inline int NoteGuards() {
    static const int mode = noteGuardMode();
    return mode;
}
inline int HiddenWriteDrain() {
    static const int mode = hiddenWriteMode();
    return mode;
}

// C6: D1's guard conditions met with items queued (a hit), and the drains D1 made for them
// (APS5_EXPERIMENT_DISPATCH_NOTE_GUARDS=1, APS5_EXPERIMENT_HIDDEN_WRITE_DRAIN=1). Counted under
// stream stats or the guard's switch. Msaa: TakeMultisampledSingleClear's kernel; ValueFill: NoteMemoryValueFill's; the hidden
// ones: an unregistered code read (ReadRawShader) or fill inputs (drainPendingInputWrites) with a
// hidden-write draw queued and no overlap drain. Cumulative totals for the tests; the [dispatch-guards]
// line prints each window's share.
enum class Guard : std::uint8_t { Msaa, ValueFill, HiddenShader, HiddenFill, Count };
constexpr std::size_t Guards = static_cast<std::size_t>(Guard::Count);
void NoteGuard(Guard guard, bool drained);
struct GuardCounts {
    std::array<std::uint64_t, Guards> hits{};
    std::array<std::uint64_t, Guards> drains{};
};
GuardCounts GuardTotals();

// C5 (APS5_DEBUG_STREAM_STATS, or APS5_DEBUG_RECIPE_REFUSALS extended to dispatches): queue 0's
// direct dispatches that hit the dispatch cache, by how the device call went: a recipe (RecipeHit,
// Rebuild after its proof failed), PrecheckRefused (a recipe the precheck refused), StampValidate
// (no recipe under stamp validation), NoAttach (nowhere to attach one), or the device call's recipe
// build: Built, or the first failing term of its condition (MovedOverlay: a template hit with moved
// buffers, served through an overlay; NotReusable by ShaderResources::UnreusableReason). Each split
// by whether the resource cache served a template (template) or the object was built (built).
enum class RecipeOutcome : std::uint8_t { RecipeHit, Rebuild, PrecheckRefused, StampValidate, NoAttach, Built, RecipesOff, Uncacheable, MovedOverlay, NotReusable, NeedsCompletion, NoObjects, NoContentKey, Count };
constexpr std::size_t RecipeOutcomes = static_cast<std::size_t>(RecipeOutcome::Count);
bool recipeRefusalStats();
inline bool RecipeCounting() {
    static const bool on = Stream::On() || recipeRefusalStats();
    return on;
}
// Arms the thread's note for the device call (Dispatch.cpp), which VulkanDevice::dispatch fills.
class RecipeNote {
public:
    explicit RecipeNote(bool armedNote) : armed(armedNote) {
        if (armed) arm(true);
    }
    ~RecipeNote() {
        if (armed) arm(false);
    }
    RecipeNote(const RecipeNote&) = delete;
    RecipeNote& operator=(const RecipeNote&) = delete;
    // Whether the device call noted its build decision, and what.
    bool Taken(RecipeOutcome& outcome, std::uint8_t& unreusable, bool& fromCache) const;

private:
    static void arm(bool on);
    bool armed;
};
bool RecipeNoteArmed();
void NoteRecipeBuild(RecipeOutcome outcome, std::uint8_t unreusable, bool fromCache);
void CountDispatchRecipe(RecipeOutcome outcome, std::uint8_t unreusable, bool fromCache);
// The key parts of a queue-0 dispatch from a 0x16 packet, against that shader's previous one; a miss
// is filed by the parts that changed.
void NoteIndirectKey(std::uint64_t address, std::span<const std::uint32_t> userData, const std::array<std::uint32_t, 3>& partialThreads, const std::array<std::uint32_t, 5>& registers, const void* shader, std::uint32_t wave, bool missed);

// C7: CountSamples' segment syncs are Stream::counters (Recorder::SamplesTotal); the dumps are
// PacketRecords.

}

#endif
