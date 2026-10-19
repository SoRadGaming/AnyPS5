#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWINPUTMEMORY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWINPUTMEMORY_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstddef>
#include <cstdint>

// Where the GPU reads a draw's index and vertex inputs and its per-draw snapshots from
// (GPU-MEMORY-PLAN C0-C2): two local experiments that move them into ReBAR VRAM and a local debug
// report of the bytes. Defined in Draw.cpp.
namespace AgcDriver::Graphics {

class Buffer;

// LOCAL EXPERIMENT (APS5_EXPERIMENT_DRAW_INPUTS_VRAM=0|1|2, GPU-MEMORY-PLAN C2): 0 (or unset) reads
// index and vertex inputs as before (in place from the host import where InPlaceDrawInput allows it,
// else a copy in a default host buffer). 1: InPlaceDrawInput refuses every range except a vertex
// range with a pending GPU write (which stays in place, ordered behind its producer), so the draw
// binds copies, and CopyDrawInput / CopyZeroPaddedDrawInput make them in DEVICE_LOCAL|HOST_VISIBLE|
// HOST_COHERENT memory (ReBAR VRAM). 2: the same copies in default host buffers (the control arm).
// The CPU reads of a device-local index copy (the highest-index scan, the mesh restart table) read
// its guest range instead; the clamped-index buffer stays in host memory either way.
int DrawInputsVram();
// The memory properties of CopyDrawInput's and CopyZeroPaddedDrawInput's buffers under the switch.
VkMemoryPropertyFlags DrawInputCopyProperties();

// LOCAL EXPERIMENT (APS5_EXPERIMENT_DRAW_SNAPSHOTS_VRAM=1, GPU-MEMORY-PLAN C1): the small per-draw
// snapshot chunks the CPU only writes (the SnapshotArena chunk, the DrawUpload chunk and the moved
// word buffers of PrepareDrawBindings) are made in DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT memory.
// The large reusable snapshots stay in host memory: refreshDrawSnapshot compares their bytes on the
// CPU. Only APS5_VERIFY_DRAW_SNAPSHOTS and APS5_CAPTURE_INPUTS read the chunks back. Off: as before.
bool DrawSnapshotsVram();
// The memory properties of those chunks under the switch.
VkMemoryPropertyFlags DrawSnapshotChunkProperties();

// LOCAL DEBUG (APS5_DEBUG_DRAW_INPUT_BYTES=1, GPU-MEMORY-PLAN C0): every 10 s a "[draw-inputs]"
// line (ProfilePrint, independent of APS5_PROFILE_DRAW) with the bytes the draws bound in place,
// from copies and from snapshots, the copy reuse hits and the draw counts; each distinct memory type
// that host imports and draw copies or snapshots were given is printed once, and host buffers that
// VK_EXT_memory_budget moved to another heap are counted. Relaxed atomics, touched only when on.
bool DebugDrawInputBytes();
enum class DrawInputBytes : std::uint8_t {
    IndexInPlace16,
    IndexInPlace32,
    IndexCopied16,
    IndexCopied32,
    IndexReused16,
    IndexReused32,
    IndexClamped16,
    IndexClamped32,
    VertexInPlace,
    VertexCopied,
    VertexReused,
    Solo,
    SliceCopied,
    SliceEpoch,
    SliceKept,
    Upload,
    SnapshotCopied,
    SnapshotEpoch,
    SnapshotReused,
    MovedCopied,
    MovedReused,
    Count
};
// Where a draw-input or snapshot buffer was made, for the memory type lines.
enum class DrawInputSite : std::uint8_t { IndexCopy, VertexCopy, SoloCopy, ClampedIndices, SnapshotArena, DrawUpload, ReusableSnapshot, MovedWords, Count };
// One more use of `kind` with `bytes`; callers check DebugDrawInputBytes first.
void CountDrawInputBytes(DrawInputBytes kind, std::uint64_t bytes);
// One more draw whose inputs were prepared; prints the line every 10 s.
void CountDrawInputDraw(const Context& context, bool indexed);
// A buffer of `bytes` made at `site`: its properties and memory type (printed once per distinct
// pair) and whether its heap is device-local.
void NoteDrawInputBuffer(const Context& context, DrawInputSite site, const Buffer& buffer, std::size_t bytes);
// The memory type a host import took (bindImport), printed once per distinct triple.
void NoteHostImportMemory(const Context& context, std::uint32_t importTypes, std::uint32_t bufferTypes, std::uint32_t chosen);
// A Buffer whose properties VK_EXT_memory_budget moved to another heap (Context::BufferMemoryType);
// counted and the first few printed when the switch is on.
void NoteBufferHeapSpill(const Context& context, VkMemoryPropertyFlags properties, std::uint32_t type, VkDeviceSize bytes);

}

#endif
