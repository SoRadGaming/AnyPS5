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
// A written image table element (DRAIN-PLAN D1, O5) makes it unbounded too, with Image.
std::vector<std::pair<std::uint64_t, std::uint64_t>> DispatchWriteRanges(const ShaderRecompiler::RecompileResult& compiled, bool& bounded, std::uint32_t* unboundedRoles = nullptr);
bool ExactDrawWrites();
// DRAIN-PLAN D1 (O5): whether a binding is an image table (ImageTable or ImageStorageTable) with a
// written element: the shader stores to a storage image over a guest surface that neither write
// list declares.
bool WritesImageTable(const ShaderRecompiler::DescriptorBinding& binding);
// DRAIN-PLAN D1 (O5): whether a draw may write guest memory that DrawWriteRanges leaves out: a stage
// with a Gds or BdaPagetable binding (GDS stores; stores through guest addresses) or a written image
// table, or a depth target with HTILE. A fault buffer alone (rect-list validation, with its BDA ABI
// version) is an internal buffer and is not. A queued draw flagged so is counted by
// HiddenWritesQueued.
bool DrawHiddenWrites(const Graphics::State& graphics, std::span<const Graphics::CompiledShader> stages);
// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_TICKET): the ranges Driver::noteDrawWriters notes for the
// stages, in its order (a ticket's WriterRanges part, noted at the commit by noteDrawWriterRanges).
// False, leaving `ranges` as it was, when a stage has no compiled program.
bool AppendDrawWriterRanges(std::span<const Graphics::CompiledShader> stages, std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges);
std::vector<std::pair<std::uint64_t, std::uint64_t>> QueuedDrawReadRanges(const Pm4::DrawParameters& draw, std::span<const Graphics::CompiledShader> stages);

}

#endif
