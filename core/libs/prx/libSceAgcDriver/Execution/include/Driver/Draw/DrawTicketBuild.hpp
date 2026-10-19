#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWTICKETBUILD_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWTICKETBUILD_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawTicket.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
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

// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_TICKET, COMMITTER-SPLIT-PLAN): what queue 0 builds one
// pipelined draw's ticket from, handed to DrawPipeline::Enqueue as its Prepare. Every view points
// into objects the draw's commit closure owns (moved into it before the Enqueue: a moved vector
// keeps its elements where they were), so the ticket is derived from exactly what the committer
// later reads; `slot` is the closure's ticket, filled before the push.
struct TicketInputs {
    std::unique_ptr<Graphics::DrawTicket>* slot = nullptr;
    VulkanDevice* device = nullptr;
    const Graphics::State* state = nullptr;
    Pm4::DrawParameters parameters{};
    std::span<const Graphics::CompiledShader> stages;
    const Graphics::PreparedDraw* prepared = nullptr;
    std::uint64_t token = 0;
};
// The Prepare that builds the ticket of `inputs` (which must outlive the Enqueue call).
DrawPipeline::Prepare DrawTicketPrepare(TicketInputs& inputs);

}

#endif
