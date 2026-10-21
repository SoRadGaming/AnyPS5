#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWWRITERANGES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWWRITERANGES_HPP

#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

std::vector<std::pair<std::uint64_t, std::uint64_t>> DrawWriteRanges(const Graphics::State& graphics, std::span<const Graphics::CompiledShader> stages, bool exact);
// `bounded` is false when the shader may write outside the returned ranges. LOCAL DEBUG
// (APS5_DEBUG_STREAM_STATS, the [stream] dispatch classes): `unboundedRoles`, when given, receives
// the DispatchUnbounded bits of every reason it found. Bda (a nonzero bdaAbiVersion) comes exactly
// with Fault (a FaultBuffer binding), and Pagetable always with both (ShaderAddressResources).
namespace DispatchUnbounded {
constexpr std::uint32_t Gds = 1u;
constexpr std::uint32_t Fault = 2u;
constexpr std::uint32_t Pagetable = 4u;
constexpr std::uint32_t Bda = 8u;
constexpr std::uint32_t Image = 16u;
}
std::vector<std::pair<std::uint64_t, std::uint64_t>> DispatchWriteRanges(const ShaderRecompiler::RecompileResult& compiled, bool& bounded, std::uint32_t* unboundedRoles = nullptr);
bool ExactDrawWrites();
// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_TICKET): the ranges Driver::noteDrawWriters notes for the
// stages, in its order (a ticket's WriterRanges part, noted at the commit by noteDrawWriterRanges).
// False, leaving `ranges` as it was, when a stage has no compiled program.
bool AppendDrawWriterRanges(std::span<const Graphics::CompiledShader> stages, std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges);
std::vector<std::pair<std::uint64_t, std::uint64_t>> QueuedDrawReadRanges(const Pm4::DrawParameters& draw, std::span<const Graphics::CompiledShader> stages);

}

#endif
