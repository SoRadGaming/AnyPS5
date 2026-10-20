// LOCAL EXPERIMENT (APS5_EXPERIMENT_NARROW_FLUSH_SYNC, PARALLEL-COMMIT-PLAN section 5, F1): the
// storage hand-over guard of StorageTexture::FlushPending under each mode of the switch. A storage
// image's store leaves its results pending; a newer writer over part of its memory is recorded into
// the open batch; the image is flushed. The guard waits (Recorder::Sync of the open batch) unless F1
// skips it, which only =1 does and only for a writer stamped when it was recorded:
// - a fill (record-stamped) over a 2 MiB image (no byte baseline): no wait at =1, a wait otherwise;
//   the fill's 64 KiB block is kept and a texel in another block is stored either way;
// - an address-based store (a lease: its pages are marked at completion): a wait in every mode;
//   the stored dword is kept;
// - a completion label in the open batch over the image: a wait in every mode, so the label lands
//   before the write-back and its block is left to the CPU (the image's texel there is not stored);
// - the fill over a 128 KiB image, which holds a generation baseline (the write-back compares the
//   block's bytes with it): a wait in every mode, without which the block would still read as the
//   baseline and the image's texels would be stored over the fill.
// Each case uses its own 4 MiB of a write-watched guest block. Without write watching, or when the
// block cannot be host-imported (no recorded fill), the cases are skipped.
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/FlushGuard.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::size_t CaseBytes = std::size_t{4} << 20u;
constexpr std::size_t BlockBytes = 4 * CaseBytes;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t Initial = 0x3f800000u;
constexpr std::uint32_t Stored = 0x11111111u;
constexpr std::uint32_t Overwritten = 0x22222222u;
constexpr std::uint32_t Filled = 0x33333333u;
constexpr std::size_t OverwrittenDword = 0x100u / 4u;
constexpr std::size_t FillBytes = 65536;

// A linear 32-bit image of `width` x `height` texels: 512 x 1024 is 2 MiB (above the 1 MiB up to
// which an image keeps a generation baseline), 256 x 128 is 128 KiB (below it).
struct Surface {
    std::uint32_t width;
    std::uint32_t height;
    std::size_t Bytes() const { return std::size_t{width} * height * 4u; }
    // Texel (0, 100), which StoreImageCode stores: in the image's second 64 KiB block or later.
    std::size_t FarDword() const { return std::size_t{100} * width; }
};

constexpr Surface Large{512, 1024};
constexpr Surface Small{256, 128};

// Stores `Stored` at texels (0, 0) and (0, 100) of the image in user data 4-7.
alignas(256) constexpr std::array<std::uint32_t, 11> StoreImageCode{0x7e000280u, 0x7e020280u, 0x7e0402ffu, Stored, 0xf0200108u, 0x00010200u, 0x7e0202ffu, 0x00000064u, 0xf0200108u, 0x00010200u, 0xbf810000u};
// Stores `Overwritten` at the global address in user data 8-9 plus 0x100 (an address-based store).
alignas(256) constexpr std::array<std::uint32_t, 7> StoreGlobalCode{0x7e000208u, 0x7e020209u, 0x7e0402ffu, Overwritten, 0xdc708100u, 0x007d0200u, 0xbf810000u};

class GuestBlock {
public:
    explicit GuestBlock(bool watched) : watched(watched) {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, PAGE_READWRITE, BlockBytes);
#else
        constexpr std::uintptr_t alignment = 65536;
        void* mapped = mmap(nullptr, BlockBytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + BlockBytes != begin + BlockBytes + alignment) munmap(reinterpret_cast<void*>(aligned + BlockBytes), begin + alignment - aligned);
            block = reinterpret_cast<std::uint8_t*>(aligned);
            if (watched) GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, BlockBytes);
        }
#endif
        Require(block != nullptr, "narrow flush sync: cannot allocate the guest block");
        Require(!watched || AgcDriver::GuestMemory::Watched(Address(), BlockBytes), "narrow flush sync: the guest block is not write-watched");
        std::fill(Words(), Words() + BlockBytes / 4u, Initial);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
        AgcDriver::GuestMemory::CollectWritesUncached(Address(), BlockBytes);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(block, BlockBytes);
        GuestArena::GuestArenaRelease_nid_postfix(block, BlockBytes);
#else
        munmap(block, BlockBytes);
        if (watched) GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, BlockBytes);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint32_t* Words() const { return reinterpret_cast<std::uint32_t*>(block); }
    std::uint64_t Address() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block)); }

private:
    bool watched;
    std::uint8_t* block = nullptr;
};

ShaderRecompiler::RecompileResult Compile(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const ShaderRecompiler::SpirvTarget& target) {
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{64u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64u, 0, userData, compute, std::nullopt, std::nullopt, memory},
        target,
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

std::vector<std::uint32_t> ImageUserData(std::uint64_t base, const Surface& surface) {
    std::vector<std::uint32_t> userData(12, 0u);
    userData[4] = static_cast<std::uint32_t>(base >> 8u);
    userData[5] = static_cast<std::uint32_t>((base >> 40u) & 0xffu) | (Format32UInt << 20u) | (((surface.width - 1u) & 3u) << 30u);
    userData[6] = ((surface.width - 1u) >> 2u) | ((surface.height - 1u) << 14u);
    userData[7] = 0xfacu | (Type2D << 28u);
    return userData;
}

std::vector<std::uint32_t> GlobalUserData(std::uint64_t base) {
    std::vector<std::uint32_t> userData(10, 0u);
    userData[8] = static_cast<std::uint32_t>(base);
    userData[9] = static_cast<std::uint32_t>(base >> 32u);
    return userData;
}

void StoreImage(AgcDriver::VulkanDevice& device, std::uint64_t image, const Surface& surface) {
    const auto shader = Compile(StoreImageCode, ImageUserData(image, surface), device.Target());
    device.Dispatch(shader, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreImageCode.data()));
}

// A recorded fill of the image's first 64 KiB block (stamped when it is recorded); false when the
// memory is not host-imported (nothing recorded).
bool FillFirstBlock(AgcDriver::VulkanDevice& device, std::uint64_t image) {
    const std::array<std::uint32_t, 4> pattern{Filled, Filled, Filled, Filled};
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    return device.FillBuffer(image, FillBytes, pattern);
}

// Flushes the image and returns whether the guard synced the open batch (the recorder submitted).
bool FlushSynced(std::uint64_t image, const Surface& surface, const std::string& what) {
    std::uint64_t before = 0;
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        const auto* recorder = AgcDriver::Graphics::Recorder::Active();
        Require(recorder != nullptr && recorder->Recording(), what + "the writer's batch is not open at the flush");
        before = recorder->Submissions();
    }
    Require(AgcDriver::Graphics::StorageTexture::FlushPending(image, surface.Bytes(), nullptr, "test"), what + "the image store left no results pending");
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    return AgcDriver::Graphics::Recorder::Active()->Submissions() != before;
}

std::string Name(const char* name) {
    return std::string("narrow flush sync (APS5_EXPERIMENT_NARROW_FLUSH_SYNC=") + std::to_string(AgcDriver::Graphics::NarrowFlushSync()) + "), " + name + ": ";
}

// (i) A record-stamped writer in the open batch: =1 records the write-back behind it without a wait.
bool RunStampedWriter(AgcDriver::VulkanDevice& device, const GuestBlock& guest, std::size_t offset) {
    const auto what = Name("record-stamped writer");
    const auto image = guest.Address() + offset;
    const auto* words = guest.Words() + offset / 4u;
    StoreImage(device, image, Large);
    if (!FillFirstBlock(device, image)) {
        std::puts("the guest block is not host-imported: the record-stamped writer case is skipped");
        device.WaitIdle();
        return false;
    }
    const bool synced = FlushSynced(image, Large, what);
    const bool skips = AgcDriver::Graphics::NarrowFlushSync() == 1;
    Require(synced != skips, what + (skips ? "the guard waited for a writer stamped when it was recorded" : "the guard did not wait for the open batch"));
    device.WaitIdle();
    Require(words[0] == Filled && words[OverwrittenDword] == Filled, what + "the image's store put its older texels over the newer fill");
    Require(words[Large.FarDword()] == Stored, what + "the pending image's texel away from the fill was not stored");
    return true;
}

// (ii) An address-based store (a lease) in the open batch: every mode waits.
void RunLeaseWriter(AgcDriver::VulkanDevice& device, const GuestBlock& guest, std::size_t offset) {
    const auto what = Name("lease writer");
    const auto image = guest.Address() + offset;
    const auto* words = guest.Words() + offset / 4u;
    StoreImage(device, image, Large);
    const auto storeGlobal = Compile(StoreGlobalCode, GlobalUserData(image), device.Target());
    device.Dispatch(storeGlobal, 1u, 1u, 1u, {}, reinterpret_cast<std::uintptr_t>(StoreGlobalCode.data()));
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        const auto* recorder = AgcDriver::Graphics::Recorder::Active();
        // Past the image, as StorageFlushOrdering looks: inside it the image's own upload read is found.
        const auto read = recorder != nullptr ? recorder->DescribePendingRead(image + Large.Bytes(), 4u) : std::nullopt;
        Require(read.has_value() && read->kind == AgcDriver::Graphics::Recorder::ReadKind::AddressBased && read->open, what + "the address-based store was not pending in the open batch at the flush");
    }
    Require(FlushSynced(image, Large, what), what + "the guard did not wait for an address-based lease");
    device.WaitIdle();
    Require(words[OverwrittenDword] == Overwritten, what + "the pending image's store put its older texels over a newer address-based store");
    Require(words[Large.FarDword()] == Stored, what + "the pending image's texel away from the newer write was not stored");
}

// (iii) A completion label of the open batch over the image: every mode waits, so the label is
// stored (and stamped) before the write-back, which leaves its block to the CPU.
void RunCompletionLabel(AgcDriver::VulkanDevice& device, const GuestBlock& guest, std::size_t offset) {
    const auto what = Name("completion label");
    const auto image = guest.Address() + offset;
    const auto* words = guest.Words() + offset / 4u;
    StoreImage(device, image, Large);
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        auto* recorder = AgcDriver::Graphics::Recorder::Active();
        Require(recorder != nullptr && recorder->Recording(), what + "the image store's batch is not open");
        const std::uint32_t label = Overwritten;
        recorder->AfterCompletions(image + OverwrittenDword * 4u, std::as_bytes(std::span(&label, 1)), 1, 0, false);
    }
    Require(FlushSynced(image, Large, what), what + "the guard did not wait for a completion label over the image");
    device.WaitIdle();
    Require(words[OverwrittenDword] == Overwritten, what + "the completion label is not in memory");
    Require(words[0] == Initial, what + "the write-back stored the image's texel into the block the completion label wrote");
    Require(words[Large.FarDword()] == Stored, what + "the pending image's texel away from the label was not stored");
}

// (iv) The fill over an image holding a generation baseline: every mode waits.
bool RunBaseline(AgcDriver::VulkanDevice& device, const GuestBlock& guest, std::size_t offset) {
    const auto what = Name("baseline held");
    const auto image = guest.Address() + offset;
    const auto* words = guest.Words() + offset / 4u;
    StoreImage(device, image, Small);
    if (!FillFirstBlock(device, image)) {
        std::puts("the guest block is not host-imported: the baseline case is skipped");
        device.WaitIdle();
        return false;
    }
    Require(FlushSynced(image, Small, what), what + "the guard did not wait for a writer over an image whose write-back compares bytes with its baseline");
    device.WaitIdle();
    Require(words[0] == Filled, what + "the image's store put its older texels over the newer fill");
    Require(words[Small.FarDword()] == Stored, what + "the pending image's texel away from the fill was not stored");
    return true;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (!AgcDriver::GuestMemory::WriteWatched()) {
            std::puts("guest memory has no write watch: the narrow flush sync cases are skipped");
            return VulkanTestSkipped;
        }
        GuestBlock guest(true);
        const bool stamped = RunStampedWriter(*device, guest, 0);
        RunLeaseWriter(*device, guest, CaseBytes);
        RunCompletionLabel(*device, guest, 2 * CaseBytes);
        const bool baseline = RunBaseline(*device, guest, 3 * CaseBytes);
        if (!stamped && !baseline) std::puts("narrow flush sync: only the lease and completion label cases ran");
        std::puts("narrow flush sync tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
