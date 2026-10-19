#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWWRITERANGES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWWRITERANGES_HPP

#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

std::vector<std::pair<std::uint64_t, std::uint64_t>> DrawWriteRanges(const Graphics::State& graphics, std::span<const Graphics::CompiledShader> stages, bool exact);
std::vector<std::pair<std::uint64_t, std::uint64_t>> DispatchWriteRanges(const ShaderRecompiler::RecompileResult& compiled, bool& bounded);
bool ExactDrawWrites();
// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_TICKET): the ranges Driver::noteDrawWriters notes for the
// stages, in its order (a ticket's WriterRanges part, noted at the commit by noteDrawWriterRanges).
// False, leaving `ranges` as it was, when a stage has no compiled program.
bool AppendDrawWriterRanges(std::span<const Graphics::CompiledShader> stages, std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges);

}

#endif
