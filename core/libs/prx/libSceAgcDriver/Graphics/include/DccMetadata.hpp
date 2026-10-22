#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DCCMETADATA_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DCCMETADATA_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace AgcDriver::Graphics {

struct Context;

// Delta color compression keeps one key byte per 256 bytes of a color surface. Surfaces are written
// uncompressed here, so the keys that matter are the fast-clear codes a title writes into the metadata
// (the surface then reads as a constant whatever its texels hold) and "uncompressed", which the driver
// stores after it writes a surface so later reads see the texels.
enum class DccKeys { Uncompressed, Clear0000, Clear0001, Clear1110, Clear1111, ClearRegister, ClearSingle, Mixed, Unreadable };

const char* DccKeysName(DccKeys keys);
std::size_t DccKeyBytes(std::uint64_t surfaceBytes);
std::size_t DccKeyCount(TextureTileMode tileMode, std::uint32_t elementBytes, std::uint32_t width, std::uint32_t height, std::uint64_t surfaceBytes);
std::size_t DccKeyCount(const GuestTextureResource& surface, std::uint64_t surfaceBytes);

template<typename ReadFollowed, typename ReadNamed>
bool KeysServeSurface(std::uint64_t followedDcc, DccKeys uploaded, DccKeys filled, std::uint64_t namedDcc, ReadFollowed&& readFollowed, ReadNamed&& readNamed) {
    if (namedDcc == 0 || namedDcc == followedDcc) return true;
    if (uploaded != DccKeys::Uncompressed || filled != DccKeys::Uncompressed) return false;
    if (followedDcc != 0 && readFollowed() != DccKeys::Uncompressed) return false;
    return readNamed() == DccKeys::Uncompressed;
}
// The keys covering a surface of `surfaceBytes`, when they all agree.
DccKeys ReadDccKeys(std::uint64_t metaAddress, std::uint64_t surfaceBytes);
bool IsDccClear(DccKeys keys);
std::optional<std::uint8_t> LandedUniformMetadataByte(std::uint64_t address, std::size_t bytes);
DccKeys CurrentDccKeys(std::uint64_t metaAddress, std::uint64_t surfaceBytes);
DccKeys CurrentDccKeys(std::uint64_t metaAddress, std::uint64_t surfaceBytes, std::size_t keyCount);
std::optional<DccKeys> KnownPendingDccKeys(std::uint64_t metaAddress, std::uint64_t surfaceBytes);
std::optional<std::uint8_t> CurrentUniformMetadataByte(std::uint64_t address, std::size_t bytes);
// Stores "uncompressed" keys over the surface's metadata on the CPU (a guest memory write: it waits
// for recorded GPU work that writes the keys first).
void MarkDccUncompressed(std::uint64_t metaAddress, std::uint64_t surfaceBytes);
void MarkDccUncompressed(std::uint64_t metaAddress, std::uint64_t surfaceBytes, std::size_t keyCount);
// The same after a write-back recorded under GuestMemory::GpuMutex: when the keys are in host-imported
// memory and a recorder is active, the store is a fill recorded into the open batch (ordered behind
// the title's key-writing kernels like the write-back itself, nothing waits on the CPU) and the range
// reads as uncompressed from ReadDccKeys while that batch is pending; otherwise the CPU store above.
// APS5_CPU_DCC_KEYS=1 always stores on the CPU.
void MarkDccUncompressed(const Context& context, std::uint64_t metaAddress, std::uint64_t surfaceBytes);
void MarkDccUncompressed(const Context& context, std::uint64_t metaAddress, std::uint64_t surfaceBytes, std::size_t keyCount);
// Fills `bytes` with the texel a 0000/0001/1110/1111 clear code stands for ("1" is 1.0 or the integer
// maximum; the alpha channel is the last one in memory when alphaOnMsb, else the first, and 3-channel
// formats have none). False when the format has no encoding here.
bool FillDccClear(VkFormat format, DccKeys keys, bool alphaOnMsb, std::span<std::byte> bytes);

// Where a color target's metadata puts alpha: the last channel in memory unless the component swap is
// reversed (single-channel formats: only with the alternate reversed swap).
bool DccAlphaOnMsb(VkFormat format, std::uint32_t componentSwap);

// The keys of a texture's DCC surface when they fast-clear it to a value encodable here, else
// Uncompressed (the texels are read as stored; other keys are reported once).
DccKeys TextureClearKeys(const GuestTextureResource& resource, std::uint64_t guestBytes);
// The last stable key scan of one surface's metadata (ProvedClearKeys): the keys read and the write
// generation the key range was collected at before the scan. Held by the image or texture of the
// surface, read and written under GuestMemory::GpuMutex only.
struct DccKeyProof {
    DccKeys keys = DccKeys::Uncompressed;
    std::uint64_t generation = 0;
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_KEY_PIN, GPU-OWNED-PLAN 9.3, K1): nonzero for a pinned proof,
    // whose keys are Uncompressed by construction: the note of the driver's own pending key store
    // that today's key read answers from (the pending-store memo, not the bytes). It answers only
    // while every use finds that still so (DccKeyPinRefusal). Never set while the switch is unset.
    std::uint64_t pinNote = 0;
};
// TextureClearKeys with the scan skipped while the key range is unchanged since `proof` was taken
// (CollectWrites over the keys, then UnchangedSince its generation), so a surface's keys are scanned
// once per change instead of on every use. A scan becomes the proof only when it read the bytes
// (not the pending-store memo) and no recorded work still writes them: a recorded key store stamps
// the range at its note, before it lands, so a proof taken meanwhile would outlive the change. Every
// other key writer stamps the range too (CPU writes through the page write watch, GPU writes at
// their record), which makes the proof exactly as sound as UnchangedSince over the texels.
// APS5_NO_KEY_FAST_PATH=1 scans on every call and stores nothing.
DccKeys ProvedClearKeys(const GuestTextureResource& resource, std::uint64_t guestBytes, DccKeyProof& proof);
bool KeyFastPath();
// LOCAL EXPERIMENT (APS5_EXPERIMENT_KEY_PIN, GPU-OWNED-PLAN 9.3, K1): pinned key proofs. While the
// driver's own "uncompressed" key store over a surface's keys is pending, today's key read answers
// Uncompressed from the pending-store memo without reading the bytes, so its scan is never kept
// (and a refresh proof is refused: "key proof empty"). A pin keeps that answer as the proof, taken
// by ProvedClearKeys after such a scan, and it answers only while today's read would still take the
// memo's note branch (C1-C5): the GPU mutex held with a recorder active (C1); a write of that
// recorder pending over the keys (C2); the memo's recorder that one, and its first entry with a note
// covering the keys holding the pinned note and Uncompressed (C3); no unit shadow over the keys
// (C4); the pinned note the newest writer of the keys (C5). Every use also needs today's
// UnchangedSince over the keys first (C6), and a refresh answered from a pinned proof C7-C10
// (StorageTexture::refreshProved). Unset or 0: today's code (no pin is taken; [dcc]'s pinnable
// count under APS5_PROFILE_DRAW only). 1: on. 2: verify: pins are taken and kept, but every answer
// comes from today's path, which must agree; a disagreement aborts with a [key-pin] line.
int KeyPins();
// Why a pin does not answer (Holds: it does; Unlocked .. NotNewest: C1, C2, C3, C4, C5; DepthPlane ..
// HandOver: a refresh's C7, C8, C9, C10), and the counted events: pins taken, memoized scans whose
// memo entry holds fill keys other than Uncompressed (not pinned), key reads and refreshes answered,
// =2's verified key reads and refreshes, and verifies overtaken by a change after the prediction.
enum class KeyPin : std::uint8_t { Holds, Unlocked, NotPending, MemoMoved, Shadowed, NotNewest, DepthPlane, FilledKeys, Scoped, HandOver, Taken, FillNotPinned, KeyAnswer, RefreshHit, KeysVerified, RefreshVerified, Raced, Count };
// C1-C5 for the pin `pinNote` over the `count` keys at `dcc`, cheapest and most selective first;
// Holds or the first that fails. No side effect (the memo is read, never cleared).
KeyPin DccKeyPinRefusal(std::uint64_t dcc, std::size_t count, std::uint64_t pinNote);
// The note a pin over the keys would hold now: that of the memo's first entry with a note covering
// them, when its keys are Uncompressed and C1-C5 hold for it; else 0.
std::uint64_t DccKeyPinNote(std::uint64_t dcc, std::size_t count);
// Cumulative counts per caller (a committer target lookup, InTargetLookup, or another), kept while
// the switch is set (relaxed; the [proofs] line prints 10 s deltas).
void CountKeyPin(KeyPin kind, bool targetLookup);
struct KeyPinCounts {
    std::array<std::uint64_t, static_cast<std::size_t>(KeyPin::Count)> target{};
    std::array<std::uint64_t, static_cast<std::size_t>(KeyPin::Count)> other{};
    std::uint64_t operator[](KeyPin kind) const { return target[static_cast<std::size_t>(kind)] + other[static_cast<std::size_t>(kind)]; }
};
KeyPinCounts KeyPinTotals();
const char* KeyPinName(KeyPin kind);
// LOCAL DEBUG (APS5_PROFILE_DRAW, GPU-OWNED-PLAN L2-0): whether this thread's refreshes are a
// committer target lookup's (StorageTexture::TargetLookupScope, set only while profiling; defined
// in Texture.cpp).
bool InTargetLookup();
// Cumulative outcomes of ProvedClearKeys: calls answered by a proof, scans made, and scans not kept.
struct DccKeyProofCounts {
    std::uint64_t proved;
    std::uint64_t scanned;
    std::uint64_t unstable;
    std::uint64_t rangeProved;
    std::uint64_t rangeScanned;
};
DccKeyProofCounts KeyProofCounts();
struct DccRangeProof {
    std::uint64_t address = 0;
    std::uint64_t count = 0;
    DccKeys keys = DccKeys::Uncompressed;
    std::uint64_t generation = 0;
};
DccKeys ProvedCurrentDccKeys(std::uint64_t metaAddress, std::uint64_t surfaceBytes, DccRangeProof& proof);
// A surface's texels as a read sees them: the guest bytes, or the clear value of fast-cleared keys.
void ReadTextureSurface(const GuestTextureResource& resource, DccKeys keys, std::span<std::byte> bytes);
void NoteKeysFillOnGpu(std::uint64_t begin, std::size_t count, DccKeys keys);
std::optional<DccKeys> WaitForKeyWriters(const GuestTextureResource& resource, std::uint64_t guestBytes);
bool DccKeyWritersPending(std::uint64_t metaAddress, std::size_t count);
bool DccKeyWriterNeedsSync(std::uint64_t metaAddress, std::size_t count);
struct DccKeyWaitCounts {
    std::uint64_t calls;
    std::uint64_t noPending;
    std::uint64_t known;
    std::uint64_t fillsUncompressed;
    std::uint64_t synced;
};
DccKeyWaitCounts KeyWaitCounts();

}

#endif
