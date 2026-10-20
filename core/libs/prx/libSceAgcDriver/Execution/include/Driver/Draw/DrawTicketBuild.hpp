#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWTICKETBUILD_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWTICKETBUILD_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawTicket.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>

namespace AgcDriver {
class VulkanDevice;
}

namespace AgcDriver::Graphics {
struct PreparedDraw;
}

namespace AgcDriver::DriverDetail {

// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_TICKET, COMMITTER-SPLIT-PLAN): what one pipelined draw's ticket
// is built from. Every view points into objects the draw's commit closure owns (moved into it before
// the Enqueue: a moved vector keeps its elements where they were), so the ticket is derived from
// exactly what the committer later reads. Queue 0 hands it to DrawPipeline::Enqueue as its Prepare,
// which stores the ticket in `slot` (the closure's PrepSlots::cheap) before the push. LOCAL
// EXPERIMENT (APS5_EXPERIMENT_PREP_HELPERS, PARALLEL-COMMIT-PLAN B0 fix N3): with prep helpers each
// PrepJob owns its own copy (its views stay valid because the job's pin keeps the closure alive);
// BuildTicket returns the ticket, `slot` stays unused and the helper offers the ticket to the job's
// PrepSlots.
struct TicketInputs {
    std::atomic<Graphics::DrawTicket*>* slot = nullptr;
    VulkanDevice* device = nullptr;
    const Graphics::State* state = nullptr;
    Pm4::DrawParameters parameters{};
    std::span<const Graphics::CompiledShader> stages;
    const Graphics::PreparedDraw* prepared = nullptr;
    std::uint64_t token = 0;
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_RECEIPTS): set when queue 0 planned this draw's receipt
    // walks (into its own thread's list, copied from the closure's objects while it built the
    // ticket, so the walks after the push read nothing the closure owns).
    bool receipts = false;
};
// What BuildTicket may spend: it stops between parts once Stop() holds. Queue 0's own preparation
// inside Enqueue (helpers off, today's path) passes its PrepareBudget as `queue0`, whose Stop decides
// as before; only that path builds the byte shadow and plans receipts. LOCAL EXPERIMENT
// (APS5_EXPERIMENT_PREP_HELPERS, PARALLEL-COMMIT-PLAN B0): a prep helper's budget is its item's seq:
// Stop once the committer started that item's commit (CommittingSeq() >= seq on `pipeline`) or the
// helpers stop (under APS5_EXPERIMENT_PREP_HELPERS_STRESS after a random delay). Helpers leave the
// byte shadow and the receipt walks to B6 / B5.
struct HelperBudget {
    const DrawPipeline* pipeline = nullptr;
    std::uint64_t seq = 0;
    // The depth queued when the item was pushed (the ticket's enqueueDepth).
    std::uint32_t depth = 0;
    const DrawPipeline::PrepareBudget* queue0 = nullptr;
    // Set when a helper's Stop held (the build was cut between parts).
    mutable bool cut = false;
    bool Stop() const { return queue0 != nullptr ? queue0->Stop() : helperStop(); }

private:
    bool helperStop() const;
};
// The ticket of `inputs`, or null when a stage has no compiled program or the build failed; a part
// whose checks throw is only left out (the committer runs them). Nothing here takes GpuMutex,
// records or reads guest bytes (queue 0's byte shadow only collects).
std::unique_ptr<Graphics::DrawTicket> BuildTicket(TicketInputs& inputs, const HelperBudget& budget);
// The Prepare that builds the ticket of `inputs` (which must outlive the Enqueue call).
DrawPipeline::Prepare DrawTicketPrepare(TicketInputs& inputs);
// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_RECEIPTS, COMMITTER-SPLIT-PLAN Q3): on queue 0 right after the
// draw's Enqueue returned: walks the receipt ranges its preparation left while more than the
// post-push depth (APS5_EXPERIMENT_Q0_PREP_DEPTH's third field) of items are queued, never while
// queue 0 holds GuestMemory::GpuMutex, inside a GuestMemory::PrepareScope.
void WalkTicketReceipts(const TicketInputs& inputs);

}

#endif
