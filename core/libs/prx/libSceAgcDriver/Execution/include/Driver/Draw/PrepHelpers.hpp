#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_PREPHELPERS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_PREPHELPERS_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawTicketBuild.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

// LOCAL EXPERIMENT (APS5_EXPERIMENT_PREP_HELPERS=N, PARALLEL-COMMIT-PLAN B0): prep helper threads
// agcprep0..N-1 build the Q1/Q2/Q2b ticket parts (BuildTicket) of pipelined draws that are already
// queued, in place of queue 0, which then passes Enqueue no Prepare and walks no receipts. The
// committer never waits for a helper:
// - Queue 0 publishes a PrepJob after the draw's Enqueue returned its seq. The job owns a copy of
//   the draw's TicketInputs (whose views point into the closure) and a pin of the closure (a copy
//   of the item's shared closure object), so the closure outlives the job.
// - A helper takes the oldest job whose commit has not started (seq > CommittingSeq), dropping the
//   older ones, builds the ticket, stopping between parts once the commit starts, and offers the
//   complete ticket to the closure's slot with one compare-exchange from null (release). It hands
//   its pin to DrawPipeline::RetirePin in every case, so the closure is never destroyed on a helper.
// - The commit takes the slot with one exchange to ClosedSentinel (acquire): a ticket is used; an
//   empty slot is closed, and a helper that finishes later fails its compare-exchange and destroys
//   its ticket, whose ~DrawTicket hands every strong reference to the ticket release list (drained
//   by the committer under GuestMemory::GpuMutex on every draw while helpers are on).
// - The PrepQueue (mutex, deque, condition variable) holds at most DrawPipeline::Depth() + 8 jobs;
//   when full, Publish drops the oldest (its pin to the reaper), so queue 0 never blocks on helpers.
// Helpers run inside a GuestMemory::PrepareScope for their whole life (APS5_ASSERT_GPU_LOCK aborts a
// GpuMutex lock or try_lock and a HostImportFor there), never record and never take GpuMutex.
// Unset, nothing here runs: queue 0 prepares its tickets inside Enqueue as before.
namespace AgcDriver::DriverDetail {

// The value a commit leaves in a closed slot: never a ticket, never dereferenced.
inline Graphics::DrawTicket* ClosedSentinel() {
    return reinterpret_cast<Graphics::DrawTicket*>(std::uintptr_t{1});
}

namespace PrepHelpers {
// The switch: the helper count, read once. 0 when unset, 0 or not a number; 3 when set without a
// value; at most 16. It requires APS5_EXPERIMENT_Q0_TICKET (said once when that is unset) and uses
// APS5_EXPERIMENT_Q0_TEMPLATE and APS5_EXPERIMENT_Q0_WORD_BUFFERS as queue 0's tickets do.
unsigned Threads();
inline bool On() {
    static const bool on = Threads() != 0;
    return on;
}
}

// A pipelined draw's helper slots, owned by its closure (Draw.cpp's PipelinedDraw; the tests' own
// closures). `cheap` holds the Q1/Q2/Q2b ticket (null: none yet; ClosedSentinel: the commit took
// it); B1/B6's heavy parts get a slot of their own here. Helpers off, queue 0's preparation stores
// its ticket in `cheap` before the push (whose pipeline mutex publishes it) and nothing else
// touches the slot.
struct PrepSlots {
    std::atomic<Graphics::DrawTicket*> cheap{nullptr};
    // When the helper's ticket was complete (Stream::NowNs, timed only with [q0prep] or [stream]
    // on): stored before the compare-exchange that publishes it.
    std::atomic<std::uint64_t> readyAtNs{0};
    // The ticket the commit took; it lives as long as the closure, as before.
    std::unique_ptr<Graphics::DrawTicket> taken;
    // Set by the producer before the push when it will publish a job for these slots (the push
    // publishes it to the committer): only such slots are claimed and counted.
    bool offered = false;
    PrepSlots() = default;
    PrepSlots(const PrepSlots&) = delete;
    PrepSlots& operator=(const PrepSlots&) = delete;
    // Destroys a ticket left in `cheap` (a stopped pipeline popped the item without committing it),
    // on the thread that drops the closure: the reaper, the committer or queue 0, never a helper
    // (counted there, GuestMemory::HelperDestroyed::Closure).
    ~PrepSlots();
    // The commit's claim, once, on the committer: the ticket, or null (the commit runs today's code
    // for every part). Helpers off (or no job offered): whatever queue 0 stored before the push.
    Graphics::DrawTicket* Take() {
        if (!offered || !PrepHelpers::On()) {
            if (auto* stored = cheap.load(std::memory_order_relaxed); stored != nullptr) {
                cheap.store(nullptr, std::memory_order_relaxed);
                taken.reset(stored);
            }
            return taken.get();
        }
        return takeClaimed();
    }

private:
    Graphics::DrawTicket* takeClaimed();
};

// One draw's job for the helpers (see the top). `build` is BuildTicket; the tests substitute their
// own builders.
struct PrepJob {
    std::uint64_t seq = 0;
    std::uint64_t token = 0;
    // The publish time (Stream::NowNs, timed only), and the depth queued when the item was pushed.
    std::uint64_t enqueuedAtNs = 0;
    std::uint32_t depth = 0;
    std::shared_ptr<void> pin;
    TicketInputs inputs;
    PrepSlots* slots = nullptr;
    std::unique_ptr<Graphics::DrawTicket> (*build)(TicketInputs& inputs, const HelperBudget& budget) = &BuildTicket;
};

namespace PrepHelpers {
// APS5_EXPERIMENT_PREP_HELPERS_STRESS=1 (with the switch): helpers add random delays (spins up to
// about 200 us, and 1 ms sleeps) before a job, between parts and before the compare-exchange, so
// the tests and the verify runs see tickets land early, late and not at all. Read once.
bool Stress();
// Whether the leads are timed: APS5_DEBUG_Q0_PREP_STATS or APS5_DEBUG_STREAM_STATS (the reports
// that print them). Read once.
bool Timed();
// Whether Stop began (helpers stop between parts and drop what they take).
bool Stopping();
// Starts the helpers (once; never after Stop). Publish starts them at the first job.
void Start();
// Queue 0, after the draw's Enqueue returned a nonzero seq: queues the job (see the top). A job
// whose commit already started is dropped at once (its pin on the caller), and so is every job
// after Stop.
void Publish(PrepJob job);
// Stops the helpers for good: wakes and joins them (each finishes or cuts its build, offers what
// it built and retires its pin) and hands the queued jobs' pins to DrawPipeline::RetirePin.
// DrawPipeline::Stop calls it first. Never on a helper.
void Stop();
// The helper threads running now.
unsigned Running();
// Test hook: frozen helpers take no job (they wait; Stop still ends them); the jobs stay queued.
void Freeze(bool frozen);

// Counters (cumulative relaxed atomics, never reset; each report takes its own window):
// - queue 0 (Publish): jobs published; the oldest dropped from a full queue; jobs whose commit had
//   started at the publish; jobs published after the stop;
// - helpers: jobs dropped at the pickup because their commit had started (stale) and at the stop;
//   builds started, builds that made no ticket, builds cut between parts (their commit started);
//   tickets made ready in their slot, and wasted (a commit closed the slot first) with their
//   parts; timed: publish to pickup, pickup to ready, publish to ready;
// - committer (PrepSlots::Take): slots closed with a ticket and empty; timed, for a used ticket,
//   ready to commit start by bucket (< 5, < 20, < 100, >= 100 us) and in total, and for both kinds
//   push to commit start (with their counts).
enum class Counter : std::uint8_t {
    Published, Overflowed, StaleAtPublish, AfterStop,
    Stale, AtStop, Builds, Empty, Cut, Ready, Wasted, WastedParts, PickupNs, BuildNs, PushToReadyNs,
    ClosedReady, ClosedEmpty, ReadyLeadUnder5, ReadyLeadUnder20, ReadyLeadUnder100, ReadyLeadOver100, ReadyLeadNs, TimedReady, ReadyItemLeadNs, TimedEmpty, EmptyItemLeadNs,
    Count
};
constexpr std::size_t Counters = static_cast<std::size_t>(Counter::Count);
using Counts = std::array<std::uint64_t, Counters>;
inline std::uint64_t Get(const Counts& counts, Counter which) {
    return counts[static_cast<std::size_t>(which)];
}
Counts Totals();
// The counts since the last call for `report` ([q0prep] on queue 0, [stream] in Enqueue); the first
// call starts the window.
enum class Report : std::uint8_t { Q0Prep, Stream, Count };
Counts TakeWindow(Report report);
}

}

#endif
