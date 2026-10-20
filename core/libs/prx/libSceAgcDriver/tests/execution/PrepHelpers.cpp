#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/PrepHelpers.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawTicket.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

// LOCAL EXPERIMENT (APS5_EXPERIMENT_PREP_HELPERS, PARALLEL-COMMIT-PLAN B0): the prep helpers' hand-off
// without a device. Plain closures with PrepSlots stand in for pipelined draws, and builders that
// make bare tickets stand in for BuildTicket (the matrix rows prep_helpers and prep_helpers_stress
// run the real tickets through the draw tests, verified at the commit). Checked: (a) a ticket ready
// before its commit is used; (b) a helper that finishes after the commit started wastes its ticket,
// whose references go to the release list, never dropped on a helper; (c) a stopped pipeline with
// jobs in flight (--stopped); (d) the stop: helpers first, queued jobs retired, then inline commits
// that publish nothing (--closing); (e) a ticket built for a device replaced since the push is
// refused at the commit; (f) closures are never destroyed on a helper; (g) frozen helpers never hold
// a commit back; (h) jobs at or behind the front are dropped unbuilt. Unset, the slots work as queue
// 0's tickets did and no helper starts.

namespace {

namespace Driver = AgcDriver::DriverDetail;
namespace GuestMemory = AgcDriver::GuestMemory;
namespace Graphics = AgcDriver::Graphics;
namespace PrepHelpers = AgcDriver::DriverDetail::PrepHelpers;
using Driver::DrawPipeline;
using Counter = PrepHelpers::Counter;
using Builder = std::unique_ptr<Graphics::DrawTicket> (*)(Driver::TicketInputs& inputs, const Driver::HelperBudget& budget);

constexpr auto Draw = DrawPipeline::ItemKind::Draw;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// A wait that never ends is a lost hand-off: it is reported and the process ends, as the threads
// that still wait cannot be joined.
template<typename TDone>
void WaitUntil(TDone&& done, const char* what) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::fprintf(stderr, "prep helpers: timed out waiting for %s\n", what);
            std::fflush(stderr);
            std::_Exit(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

std::uint64_t Since(const PrepHelpers::Counts& before, Counter which) {
    return PrepHelpers::Get(PrepHelpers::Totals(), which) - PrepHelpers::Get(before, which);
}

// Every published job ended (dropped or built) and every build ended (ready, wasted or empty).
bool HelpersSettled() {
    const auto counts = PrepHelpers::Totals();
    const auto get = [&](Counter which) { return PrepHelpers::Get(counts, which); };
    return get(Counter::Published) == get(Counter::Overflowed) + get(Counter::Stale) + get(Counter::AtStop) + get(Counter::Builds) && get(Counter::Builds) == get(Counter::Ready) + get(Counter::Wasted) + get(Counter::Empty);
}

// A pipelined draw's closure: the slots and the commit's body, which gets the claimed ticket. Its
// destruction is counted, and counted apart on a helper.
std::atomic<int> closuresCreated{0};
std::atomic<int> closuresDestroyed{0};
std::atomic<int> closuresOnHelpers{0};

struct TestDraw {
    Driver::PrepSlots slots;
    std::function<void(Graphics::DrawTicket*)> body;
    TestDraw() { ++closuresCreated; }
    ~TestDraw() {
        if (GuestMemory::OnPrepHelperThread()) ++closuresOnHelpers;
        ++closuresDestroyed;
    }
};

// Every closure made so far was destroyed (the reaper drops items and pins every 2 ms or so).
bool ClosuresSettled() {
    return closuresDestroyed == closuresCreated;
}

struct Queued {
    std::shared_ptr<TestDraw> held;
    std::uint64_t seq = 0;
};

Queued EnqueueDraw(std::function<void(Graphics::DrawTicket*)> body) {
    auto held = std::make_shared<TestDraw>();
    held->body = std::move(body);
    held->slots.offered = true;
    const auto seq = DrawPipeline::Queue0().Enqueue([held] { held->body(held->slots.Take()); }, {}, 0, {}, Draw);
    return {std::move(held), seq};
}

void Publish(const Queued& queued, Builder build, std::uint64_t token) {
    Driver::PrepJob job;
    job.seq = queued.seq;
    job.token = token;
    job.pin = queued.held;
    job.inputs.token = token;
    job.slots = &queued.held->slots;
    job.build = build;
    PrepHelpers::Publish(std::move(job));
}

bool SlotFilled(const Queued& queued) {
    auto* ticket = queued.held->slots.cheap.load(std::memory_order_acquire);
    return ticket != nullptr && ticket != Driver::ClosedSentinel();
}

// A commit that holds the committer until opened, so the test queues items behind it.
struct Gate {
    std::promise<void> opened;
    std::shared_future<void> released = opened.get_future().share();
    std::atomic<bool> entered{false};
    std::uint64_t seq = 0;

    void Enqueue() {
        seq = DrawPipeline::Queue0().Enqueue([this, released = released] {
            entered = true;
            released.wait();
        }, {});
        WaitUntil([this] { return entered.load(); }, "the committer to reach the gate");
    }

    void Open() { opened.set_value(); }
};

// The builders, each a bare ticket marked with the job's token.
std::atomic<int> builds{0};

std::unique_ptr<Graphics::DrawTicket> MarkedTicket(Driver::TicketInputs& inputs, const Driver::HelperBudget&) {
    ++builds;
    auto ticket = std::make_unique<Graphics::DrawTicket>();
    ticket->token = inputs.token;
    ticket->parts = Graphics::DrawTicket::AbiChecked | Graphics::DrawTicket::WriterRanges;
    return ticket;
}

// Up to about 50 us of work first, so tickets land before, during and after their commits.
std::unique_ptr<Graphics::DrawTicket> SlowTicket(Driver::TicketInputs& inputs, const Driver::HelperBudget& budget) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(inputs.token % 50);
    while (std::chrono::steady_clock::now() < until) {
        if (budget.Stop()) return nullptr;
    }
    return MarkedTicket(inputs, budget);
}

// (e) The device the job was published for rides in its token, as a helper's ticket takes the
// device of the job's inputs (the closure's device at the push).
std::unique_ptr<Graphics::DrawTicket> DeviceTicket(Driver::TicketInputs& inputs, const Driver::HelperBudget& budget) {
    auto ticket = MarkedTicket(inputs, budget);
    ticket->device = reinterpret_cast<VkDevice>(static_cast<std::uintptr_t>(inputs.token));
    return ticket;
}

// (b) Blocks until released, then makes a ticket holding a strong reference (a stand-in pipeline
// that is never dereferenced; its deleter records where the reference was dropped).
const VkDevice LateDevice = reinterpret_cast<VkDevice>(std::uintptr_t{0x7000});
std::atomic<bool> lateEntered{false};
std::atomic<bool> lateReleased{false};
std::atomic<int> referenceDrops{0};
std::atomic<int> referenceDropsOnHelpers{0};
alignas(64) unsigned char standInPipeline[64];

std::unique_ptr<Graphics::DrawTicket> LateTicket(Driver::TicketInputs& inputs, const Driver::HelperBudget& budget) {
    lateEntered = true;
    WaitUntil([] { return lateReleased.load(); }, "the test to release the late builder");
    auto ticket = MarkedTicket(inputs, budget);
    ticket->device = LateDevice;
    ticket->pipeline = std::shared_ptr<Graphics::Pipeline>(reinterpret_cast<Graphics::Pipeline*>(standInPipeline), [](Graphics::Pipeline*) {
        ++referenceDrops;
        if (GuestMemory::OnPrepHelperThread()) ++referenceDropsOnHelpers;
    });
    return ticket;
}

// (a) A ticket in its slot before the commit starts is the one the commit gets.
void ReadyTicketIsUsed() {
    auto& pipeline = DrawPipeline::Queue0();
    WaitUntil(HelpersSettled, "the helpers to settle");
    const auto before = PrepHelpers::Totals();
    std::atomic<std::uint64_t> seen{0};
    Gate gate;
    gate.Enqueue();
    auto queued = EnqueueDraw([&seen](Graphics::DrawTicket* ticket) { seen = ticket != nullptr ? ticket->token : ~std::uint64_t{0}; });
    Require(queued.seq > gate.seq, "the draw's seq does not follow the gate's");
    Publish(queued, &MarkedTicket, 0xa11ce);
    WaitUntil([&] { return SlotFilled(queued); }, "the helper's ticket in its slot");
    gate.Open();
    pipeline.Drain(DrawPipeline::DrainReason::Flush);
    WaitUntil(HelpersSettled, "the helpers to settle");
    Require(seen == 0xa11ce, "the commit did not get the ticket that was ready before it");
    Require(Since(before, Counter::Ready) == 1 && Since(before, Counter::ClosedReady) == 1 && Since(before, Counter::ClosedEmpty) == 0, "a used ticket was not counted ready and closed ready once");
    Require(pipeline.CommittingSeq() >= queued.seq, "the committing seq did not reach the committed draw");
}

// (b) The commit starts while the helper builds: the slot is closed empty, the helper's
// compare-exchange fails, and its ticket's references reach the release list. They are dropped
// where the list is drained (here, standing in for the committer), never on the helper.
void LateTicketIsWasted() {
    auto& pipeline = DrawPipeline::Queue0();
    WaitUntil(HelpersSettled, "the helpers to settle");
    const auto before = PrepHelpers::Totals();
    lateEntered = false;
    lateReleased = false;
    std::atomic<int> nulls{0};
    Gate gate;
    gate.Enqueue();
    auto queued = EnqueueDraw([&nulls](Graphics::DrawTicket* ticket) {
        if (ticket == nullptr) ++nulls;
    });
    Publish(queued, &LateTicket, 0xb);
    WaitUntil([] { return lateEntered.load(); }, "the helper to start the late build");
    gate.Open();
    pipeline.Drain(DrawPipeline::DrainReason::Flush);
    Require(nulls == 1, "the commit waited for the helper or got its ticket");
    lateReleased = true;
    WaitUntil([&] { return Since(before, Counter::Wasted) == 1; }, "the late ticket to be wasted");
    WaitUntil(HelpersSettled, "the helpers to settle");
    Require(Since(before, Counter::ClosedEmpty) == 1 && Since(before, Counter::Ready) == 0 && Since(before, Counter::WastedParts) == 2, "the late ticket was not counted wasted with its two parts against one empty slot");
    Require(referenceDrops == 0, "the wasted ticket's reference was dropped before the release list was drained");
    Graphics::Context context{};
    context.device = LateDevice;
    WaitUntil([&] {
        Graphics::DrainTicketRelease(context);
        return referenceDrops != 0;
    }, "the wasted ticket's reference on the release list");
    Require(referenceDrops == 1 && referenceDropsOnHelpers == 0, "the wasted ticket's reference was not dropped exactly once, off the helpers (" + std::to_string(referenceDrops.load()) + " drops, " + std::to_string(referenceDropsOnHelpers.load()) + " on helpers)");
    Require(GuestMemory::HelperDestructions(GuestMemory::HelperDestroyed::ShaderResources) == 0 && GuestMemory::HelperDestructions(GuestMemory::HelperDestroyed::Pipeline) == 0, "a ShaderResources or a Pipeline was destroyed on a helper");
}

// (e) The device is replaced between the push and the helper's build: the ticket names the device
// of the push, and the commit's TicketFor (the current device's context) refuses it.
void ReplacedDeviceRefusesTicket() {
    auto& pipeline = DrawPipeline::Queue0();
    WaitUntil(HelpersSettled, "the helpers to settle");
    const auto pushed = reinterpret_cast<VkDevice>(std::uintptr_t{0x1000});
    const auto replacement = reinterpret_cast<VkDevice>(std::uintptr_t{0x2000});
    std::atomic<VkDevice> current{pushed};
    std::atomic<bool> got{false}, refused{false}, acceptedForPushed{false};
    PrepHelpers::Freeze(true);
    Gate gate;
    gate.Enqueue();
    auto queued = EnqueueDraw([&](Graphics::DrawTicket* ticket) {
        got = ticket != nullptr;
        Graphics::Context context{};
        context.device = current.load();
        refused = ticket != nullptr && Graphics::TicketFor(ticket, context) == nullptr;
        context.device = pushed;
        acceptedForPushed = ticket != nullptr && Graphics::TicketFor(ticket, context) == ticket;
    });
    Publish(queued, &DeviceTicket, reinterpret_cast<std::uintptr_t>(pushed));
    current = replacement;
    PrepHelpers::Freeze(false);
    WaitUntil([&] { return SlotFilled(queued); }, "the helper's ticket for the replaced device");
    gate.Open();
    pipeline.Drain(DrawPipeline::DrainReason::Flush);
    Require(got && refused && acceptedForPushed, "the ticket built for the device of the push was not refused for the replacement device (or not accepted for its own)");
}

// (h) A job at the front (its commit is running) is dropped without a build; a job behind it is built.
void JobsAtTheFrontAreDropped() {
    auto& pipeline = DrawPipeline::Queue0();
    WaitUntil(HelpersSettled, "the helpers to settle");
    const auto before = PrepHelpers::Totals();
    const int buildsBefore = builds;
    std::promise<void> opened;
    auto released = opened.get_future().share();
    std::atomic<bool> entered{false};
    std::atomic<std::uint64_t> frontSeen{0}, behindSeen{0};
    auto front = EnqueueDraw([&, released](Graphics::DrawTicket* ticket) {
        frontSeen = ticket != nullptr ? ticket->token : 0;
        entered = true;
        released.wait();
    });
    WaitUntil([&] { return entered.load(); }, "the front draw's commit to start");
    Require(pipeline.CommittingSeq() == front.seq, "the committing seq is not the running draw's");
    Publish(front, &MarkedTicket, 0xf);
    auto behind = EnqueueDraw([&](Graphics::DrawTicket* ticket) { behindSeen = ticket != nullptr ? ticket->token : 0; });
    Publish(behind, &MarkedTicket, 0xbe);
    WaitUntil([&] { return SlotFilled(behind); }, "the ticket of the draw behind the front");
    opened.set_value();
    pipeline.Drain(DrawPipeline::DrainReason::Flush);
    WaitUntil(HelpersSettled, "the helpers to settle");
    Require(frontSeen == 0 && behindSeen == 0xbe, "the running draw got a ticket, or the draw behind it did not");
    Require(Since(before, Counter::StaleAtPublish) + Since(before, Counter::Stale) == 1 && builds == buildsBefore + 1, "the job at the front was built, or the one behind it was not");
}

// (g) Frozen helpers take no job: every commit runs at once, closes its slot empty and runs
// today's code. (h) Once thawed, the helpers drop the jobs whose commits are done, unbuilt.
void FrozenHelpersNeverBlockCommits() {
    auto& pipeline = DrawPipeline::Queue0();
    WaitUntil(HelpersSettled, "the helpers to settle");
    const auto before = PrepHelpers::Totals();
    const int buildsBefore = builds;
    constexpr int Draws = 5;
    std::atomic<int> nulls{0};
    PrepHelpers::Freeze(true);
    Gate gate;
    gate.Enqueue();
    for (int i = 0; i < Draws; ++i) {
        auto queued = EnqueueDraw([&nulls](Graphics::DrawTicket* ticket) {
            if (ticket == nullptr) ++nulls;
        });
        Publish(queued, &MarkedTicket, static_cast<std::uint64_t>(i + 1));
    }
    const auto start = std::chrono::steady_clock::now();
    gate.Open();
    pipeline.Drain(DrawPipeline::DrainReason::Flush);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    Require(nulls == Draws, "a commit got a ticket from frozen helpers");
    Require(elapsed < std::chrono::seconds(5), "the commits waited for frozen helpers");
    Require(Since(before, Counter::ClosedEmpty) == Draws && Since(before, Counter::Published) == Draws, "the frozen jobs were not published, or their slots were not closed empty");
    Require(builds == buildsBefore, "a frozen helper built a ticket");
    PrepHelpers::Freeze(false);
    WaitUntil([&] { return Since(before, Counter::Stale) == Draws; }, "the thawed helpers to drop the committed draws' jobs");
    Require(builds == buildsBefore, "a thawed helper built a ticket for a committed draw");
}

// (f) Many draws with helpers racing their commits: every closure is destroyed off the helpers,
// every job and build ends exactly one way, and every ticket made ready is the one its commit used.
void ClosuresNeverDieOnHelpers() {
    auto& pipeline = DrawPipeline::Queue0();
    WaitUntil(HelpersSettled, "the helpers to settle");
    WaitUntil(ClosuresSettled, "the reaper to destroy the earlier closures");
    const auto before = PrepHelpers::Totals();
    constexpr int Draws = 400;
    std::atomic<int> used{0}, empty{0};
    for (int i = 0; i < Draws; ++i) {
        auto queued = EnqueueDraw([&used, &empty, i](Graphics::DrawTicket* ticket) {
            if (ticket != nullptr) {
                Require(ticket->token == static_cast<std::uint64_t>(i), "a commit got another draw's ticket");
                ++used;
            } else {
                ++empty;
            }
            if (i % 5 == 0) std::this_thread::sleep_for(std::chrono::microseconds(30));
        });
        Publish(queued, &SlowTicket, static_cast<std::uint64_t>(i));
    }
    pipeline.Drain(DrawPipeline::DrainReason::Flush);
    WaitUntil(HelpersSettled, "the helpers to settle");
    WaitUntil(ClosuresSettled, "the reaper to destroy every closure");
    Require(closuresOnHelpers == 0 && GuestMemory::HelperDestructions(GuestMemory::HelperDestroyed::Closure) == 0, "a closure was destroyed on a helper");
    Require(used + empty == Draws, "a commit did not run");
    Require(Since(before, Counter::ClosedReady) == static_cast<std::uint64_t>(used.load()) && Since(before, Counter::ClosedEmpty) == static_cast<std::uint64_t>(empty.load()), "the closed slots do not match what the commits got");
    Require(Since(before, Counter::Ready) == static_cast<std::uint64_t>(used.load()), "a ticket made ready was not used, or one was used without being made ready");
    Require(Since(before, Counter::Published) + Since(before, Counter::StaleAtPublish) == Draws, "a job was neither published nor dropped at the publish");
}

// (c) A commit fails with jobs in flight: the stopped pipeline pops the items behind it without
// committing them, the frozen jobs are dropped as stale once thawed, and the tickets already in
// slots go with their closures, off the helpers.
void StoppedPipelineDropsJobs() {
    auto& pipeline = DrawPipeline::Queue0();
    const auto before = PrepHelpers::Totals();
    const int buildsBefore = builds;
    std::atomic<int> ran{0};
    std::vector<Queued> queued;
    {
        Gate gate;
        gate.Enqueue();
        pipeline.Enqueue([] { throw std::runtime_error("injected commit failure"); }, {}, 0, {}, Draw);
        for (int i = 0; i < 2; ++i) {
            queued.push_back(EnqueueDraw([&ran](Graphics::DrawTicket*) { ++ran; }));
            Publish(queued.back(), &MarkedTicket, static_cast<std::uint64_t>(i + 1));
        }
        WaitUntil([&] { return SlotFilled(queued[0]) && SlotFilled(queued[1]); }, "the first two tickets in their slots");
        PrepHelpers::Freeze(true);
        for (int i = 2; i < 5; ++i) {
            queued.push_back(EnqueueDraw([&ran](Graphics::DrawTicket*) { ++ran; }));
            Publish(queued.back(), &MarkedTicket, static_cast<std::uint64_t>(i + 1));
        }
        queued.clear();
        gate.Open();
        WaitUntil([&] { return pipeline.Outstanding() == 0; }, "the stopped committer to drop the queued draws");
    }
    Require(ran == 0, "a draw behind the failed commit ran");
    Require(pipeline.CommittingSeq() != 0, "the stopped pipeline's pops did not advance the committing seq");
    PrepHelpers::Freeze(false);
    WaitUntil([&] { return Since(before, Counter::Stale) == 3; }, "the thawed helpers to drop the stopped pipeline's jobs");
    WaitUntil(ClosuresSettled, "the reaper to destroy the dropped draws' closures");
    Require(closuresOnHelpers == 0 && GuestMemory::HelperDestructions(GuestMemory::HelperDestroyed::Closure) == 0, "a dropped draw's closure was destroyed on a helper");
    Require(builds == buildsBefore + 2 && Since(before, Counter::ClosedReady) + Since(before, Counter::ClosedEmpty) == 0, "the stopped pipeline built or closed more than it should");
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception&) {
        // The driver reports the failure again at its shutdown.
    }
}

// (d) The stop ends the helpers first, while the committer still runs (held at a gate): the queued
// jobs are retired, the commits close their slots empty, and afterwards a commit runs inline and
// publishes nothing.
void StopEndsHelpersFirst() {
    auto& pipeline = DrawPipeline::Queue0();
    const auto before = PrepHelpers::Totals();
    std::atomic<std::uint64_t> firstSeen{0};
    std::atomic<int> nulls{0};
    Gate gate;
    gate.Enqueue();
    {
        auto first = EnqueueDraw([&firstSeen](Graphics::DrawTicket* ticket) { firstSeen = ticket != nullptr ? ticket->token : 0; });
        Publish(first, &MarkedTicket, 0x1);
        WaitUntil([&] { return SlotFilled(first); }, "the first ticket in its slot");
        Require(PrepHelpers::Running() == PrepHelpers::Threads(), "the helpers did not all start");
        PrepHelpers::Freeze(true);
        for (int i = 0; i < 3; ++i) {
            auto queued = EnqueueDraw([&nulls](Graphics::DrawTicket* ticket) {
                if (ticket == nullptr) ++nulls;
            });
            Publish(queued, &MarkedTicket, static_cast<std::uint64_t>(i + 2));
        }
    }
    std::atomic<bool> stopped{false};
    std::thread stopper([&] {
        pipeline.Stop();
        stopped = true;
    });
    WaitUntil([] { return PrepHelpers::Running() == 0; }, "the helpers to stop");
    Require(!stopped, "the stop returned while the committer was held at the gate");
    Require(Since(before, Counter::AtStop) == 3, "the stop did not retire the three queued jobs");
    gate.Open();
    stopper.join();
    PrepHelpers::Freeze(false);
    Require(firstSeen == 0x1 && nulls == 3, "the committer did not use the ready ticket and close the others empty after the helpers stopped");
    const auto caller = std::this_thread::get_id();
    bool ranHere = false;
    const auto seq = pipeline.Enqueue([&] { ranHere = std::this_thread::get_id() == caller; }, {});
    Require(seq == 0 && ranHere, "a commit after the stop did not run inline without a seq");
    auto late = std::make_shared<TestDraw>();
    Driver::PrepJob job;
    job.seq = ~std::uint64_t{0};
    job.pin = late;
    job.slots = &late->slots;
    job.build = &MarkedTicket;
    PrepHelpers::Publish(std::move(job));
    late.reset();
    Require(Since(before, Counter::AfterStop) == 1 && PrepHelpers::Running() == 0, "a job published after the stop was queued or restarted the helpers");
    WaitUntil(ClosuresSettled, "the reaper to destroy every closure");
    Require(closuresOnHelpers == 0 && GuestMemory::HelperDestructions(GuestMemory::HelperDestroyed::Closure) == 0, "a closure was destroyed on a helper");
    AgcDriverShutdown_nid_postfix();
}

// Unset: queue 0's preparation fills the slot before the push and the commit takes that ticket;
// nothing is counted and no helper starts; every item still gets a seq.
void UnsetKeepsQueueZeroSlots() {
    auto& pipeline = DrawPipeline::Queue0();
    Require(!PrepHelpers::On() && PrepHelpers::Running() == 0, "helpers run without APS5_EXPERIMENT_PREP_HELPERS");
    std::atomic<std::uint64_t> seen{0};
    auto held = std::make_shared<TestDraw>();
    held->body = [&seen](Graphics::DrawTicket* ticket) { seen = ticket != nullptr ? ticket->token : 0; };
    auto ticket = std::make_unique<Graphics::DrawTicket>();
    ticket->token = 0x51;
    held->slots.cheap.store(ticket.release(), std::memory_order_relaxed);
    const auto first = pipeline.Enqueue([held] { held->body(held->slots.Take()); }, {}, 0, {}, Draw);
    const auto second = pipeline.Enqueue([] {}, {});
    pipeline.Drain(DrawPipeline::DrainReason::Flush);
    Require(seen == 0x51, "the commit did not get the ticket stored before the push");
    Require(first != 0 && second > first && pipeline.CommittingSeq() == second, "the items' seqs do not follow the queue");
    const auto counts = PrepHelpers::Totals();
    Require(PrepHelpers::Get(counts, Counter::ClosedReady) + PrepHelpers::Get(counts, Counter::ClosedEmpty) + PrepHelpers::Get(counts, Counter::Published) == 0 && PrepHelpers::Running() == 0, "the unset switch counted slots or started helpers");
    AgcDriverShutdown_nid_postfix();
}

}

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc > 1 ? argv[1] : "";
        if (!PrepHelpers::On()) {
            if (!mode.empty()) {
                std::cout << "prep helper tests skipped: APS5_EXPERIMENT_PREP_HELPERS (with APS5_EXPERIMENT_Q0_TICKET) is not set\n";
                return 77;
            }
            UnsetKeepsQueueZeroSlots();
        } else if (mode == "--stopped") {
            StoppedPipelineDropsJobs();
        } else if (mode == "--closing") {
            StopEndsHelpersFirst();
        } else {
            ReadyTicketIsUsed();
            LateTicketIsWasted();
            ReplacedDeviceRefusesTicket();
            JobsAtTheFrontAreDropped();
            FrozenHelpersNeverBlockCommits();
            ClosuresNeverDieOnHelpers();
            AgcDriverShutdown_nid_postfix();
        }
        Require(GuestMemory::HelperDestructions(GuestMemory::HelperDestroyed::ShaderResources) == 0 && GuestMemory::HelperDestructions(GuestMemory::HelperDestroyed::Pipeline) == 0 && GuestMemory::HelperDestructions(GuestMemory::HelperDestroyed::Closure) == 0, "an object was destroyed on a helper");
        std::cout << "prep helper tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
