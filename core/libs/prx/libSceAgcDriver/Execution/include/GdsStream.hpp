#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_GDSSTREAM_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_GDSSTREAM_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace AgcDriver::Graphics {
class Recorder;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_GDS_IN_STREAM, DRAIN-PLAN section 5, D2): DMA_DATA transfers that
// touch the GDS recorded in the stream, so the queue that issues them stops waiting on the CPU for the
// GDS's unfinished users (queue 0x40's GDS waits, which set the length of queue 0's 0x93 phases, and
// the GDS share of queue 0's own 0x50 phases). Driver::preparePacketMemory (PacketMemory.cpp) makes
// the decisions; this file keeps the switch, the whole-GDS note, the checks and the [gds-stream] line.
//   R, copies (the readback GDS -> memory; also GDS -> GDS and memory -> GDS): Pm4::DecodeGdsCopy,
//     recorded by copyBehindPendingSource as VulkanDevice::CopyBuffer when the source has unfinished
//     writers. With nothing pending the store path below reads the source on the CPU without waiting,
//     as today. A copy the GPU path refuses (a completion over a range, no host import, a decode
//     refusal) takes today's path whole.
//   S, stores into the GDS (immediate, memory, or an idle GDS source): Pm4::ResolveStore's GDS case,
//     recorded by WriteLabelOnGpu, reason 0 only. Reasons 1 (CPU store), 2, 3, 4 and 5 (the completion
//     memcpy at a reap: refused before anything is recorded, `behindCompletions` false), a store
//     ResolveStore does not take (not a multiple of 4 bytes or over 64 KiB), and a memory source with
//     pending writes whose copy was refused take today's targeted path
//     (APS5_EXPERIMENT_GDS_TARGETED_SYNC: a SyncThrough of the GDS, then the store on the CPU).
// Judge fix 1 (write-after-read): CopyBuffer notes its source only as a read (nothing tracks that
// without read tracking) and SyncThrough waits for writes only, so a later fallback CPU store into the
// GDS could land before a recorded readback ran and the readback would copy the new value.
// NoteRecorded notes the whole GDS as a pending write of the open batch after every recorded operation,
// under the hold that recorded it (every GPU user of the GDS notes all of it already: O6).
// Switch: =1 on; =2 on with a value check of every recorded operation (VerifyRecorded). It requires
// APS5_EXPERIMENT_GDS_TARGETED_SYNC (its path is the fallback; the same test as that path's) and refuses
// APS5_PIPELINE_DMA (its queue-0 DMA items can store into the GDS on the CPU ahead of queued draws,
// DRAIN-PLAN section 9 item 7): either way the mode is 0 and a line says why. Unset (or 0) is today:
// nothing here runs. Read once, at the first DMA_DATA that touches the GDS.
namespace AgcDriver::GdsStream {

int mode();
inline int Mode() {
    static const int value = mode();
    return value;
}
inline bool On() {
    return Mode() != 0;
}
inline bool Verify() {
    return Mode() == 2;
}

// What was recorded: a readback (GDS -> memory copy), a GDS -> GDS copy, a memory -> GDS copy, and a
// store into the GDS of an immediate, of a memory source's bytes, or of an idle GDS range's bytes.
enum class Op : std::uint8_t { Readback, GdsCopy, CopyIn, Store, StoreFromMemory, StoreFromGds, Count };
// Why a transfer took today's path: WriteLabelOnGpu's reasons 1-5 for a store, a store ResolveStore
// did not take (StoreShape); for a copy, a completion that still stores over a range
// (VulkanDevice::CopyWaitsOnCompletions), no host import (the rest of GpuCopyPossible's refusals), an
// inaccessible range, and Pm4::DecodeGdsCopy's refusals.
enum class Fallback : std::uint8_t { StoreCpu, StoreDrain, StoreNoImport, StoreUnsuitable, StoreBehindCompletions, StoreShape, CopyCompletions, CopyNoImport, CopyInaccessible, CopyHighWord, CopyRange, CopySize, CopyShape, Count };
constexpr std::size_t Ops = static_cast<std::size_t>(Op::Count);
constexpr std::size_t Fallbacks = static_cast<std::size_t>(Fallback::Count);
const char* OpName(Op op);
const char* FallbackName(Fallback reason);
// WriteLabelOnGpu's refusal (1-5; anything else counts as 4) as a store fallback.
Fallback StoreFallback(int reason);

// Under GpuMutex, right after `op` was recorded into the open batch (`copyPath`: CopyBuffer's path for
// a copy, 0 for a store): notes the whole GDS as a pending write of that batch (judge fix 1) and keeps
// the batch for the invariant below. The first call arms the guest-access check.
void NoteRecorded(Graphics::Recorder& recorder, Op op, int copyPath = 0);
void CountFallback(Fallback reason);
// A GDS copy whose source had no unfinished writer: the store path reads it on the CPU without waiting.
void CountIdleCopy();
// A DMA_DATA into the GDS that reached the targeted path (each fallback whose destination is the GDS).
void CountTargetedSync();

// =1's invariant (judge fix 2): cpu-gds-access-unnoted counts a CPU access to the GDS, through the flush
// hook (CheckGuestAccess) or a targeted sync (CheckTargetedSync), while a recorded GDS operation sits in
// an unfinished batch newer than the newest batch that notes a write of the GDS. Fix 1 makes it 0 by
// construction: the note is made under the hold that recorded the operation, and the check runs under
// GpuMutex. Checked in both modes once an operation was recorded (Armed).
bool Armed();
void CheckTargetedSync(const Graphics::Recorder& recorder);
// From the guest-access counter (Drain::GuestAccess, called before the flush hook on any thread): takes
// GpuMutex, as the hook does, when the access overlaps the GDS and the newest recorded operation may
// be unfinished.
void CheckGuestAccess(std::uint64_t address, std::size_t bytes);

// =2, kept only as a value check (judge fix 2: it syncs around every operation, so it cannot see an
// ordering defect). Expected holds the bytes the operation must leave at its destination, taken under
// the caller's hold before the record: a copy's source read through the flush hook (a locked
// SyncThrough of it, pending results stored), or a store's own bytes, with the tracker generation taken
// before either. VerifyRecorded then runs a locked SyncThrough of the destination and compares. A
// disagreement is explained only by a CPU stamp newer than that generation on a page of the source or
// the destination (GuestMemory::CpuWrittenSince): an H host write, or a CPU store a collect stamped.
// Never by a driver stamp: the operation's own MarkWritten, the write-backs the reaps of that
// SyncThrough run, or another queue's CPU store into the GDS (the game orders those; the GDS gets no
// CPU stamps, so a disagreement over it always aborts). Otherwise it prints
// "[gds-stream] =2 disagreement queue 0x%x kind %s dst 0x%llx bytes %zu", flushes and aborts (the
// TicketDisagrees convention).
struct Expected {
    std::vector<std::byte> bytes;
    std::uint64_t source = 0;
    std::uint64_t generation = 0;
};
Expected ExpectCopy(std::uint64_t source, std::size_t bytes);
Expected ExpectStore(std::span<const std::byte> bytes);
void VerifyRecorded(Graphics::Recorder& recorder, std::uint32_t queue, Op op, std::uint64_t destination, const Expected& expected);

// Cumulative counts (the tests read them; the [gds-stream] line prints each window's share).
struct Totals {
    std::array<std::uint64_t, Ops> recorded{};
    // By CopyBuffer's path (1: a transfer between the host imports, 3: a device copy between storage
    // images).
    std::array<std::uint64_t, 4> copyPaths{};
    std::uint64_t notes = 0;
    std::uint64_t idleCopies = 0;
    std::array<std::uint64_t, Fallbacks> fallbacks{};
    std::uint64_t targetedSyncs = 0;
    std::uint64_t accessChecks = 0;
    std::uint64_t unnoted = 0;
    std::uint64_t verified = 0;
    std::uint64_t explained = 0;
};
Totals CurrentTotals();
// The [gds-stream] line with what was counted since the last one: from Drain::ReportWindow under the
// stream stats (tagged with its window; `print` false drops what was counted before the first window),
// else every 10 s from a thread that counts (window=-). Nothing while the switch is off.
void ReportWindow(const char* window, bool print);

}

#endif
