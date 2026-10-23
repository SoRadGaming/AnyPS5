#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthLease.hpp"
#include "prx/libSceAgcDriver/Graphics/include/FieldCount.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include "RdnaDecoder/include/RdnaDecoder/RdnaDescriptorFormat.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <shared_mutex>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace AgcDriver::Graphics {
namespace {

std::uint64_t depthWrites = 0;
std::uint64_t mirrorEpoch = 0;

std::pair<std::uint32_t, std::uint32_t> DepthBlockExtent(std::uint32_t bytesPerTexel) {
    return {bytesPerTexel == 4 ? 128u : 256u, bytesPerTexel == 1 ? 256u : 128u};
}

Recorder* RecorderFor(const Context& context) {
    auto* recorder = Recorder::Active();
    return recorder != nullptr && recorder->Device() == context.device ? recorder : nullptr;
}

bool storageFits(const DepthTarget& target, const GuestTextureResource& descriptor) {
    const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
    const auto format = ResolveTextureFormat(descriptor.format);
    const bool sized = d16 ? (format == VK_FORMAT_R16_UINT || format == VK_FORMAT_R16_UNORM || format == VK_FORMAT_R16_SINT || format == VK_FORMAT_R16_SNORM || format == VK_FORMAT_R16_SFLOAT) : (format == VK_FORMAT_R32_SFLOAT || format == VK_FORMAT_R32_UINT || format == VK_FORMAT_R32_SINT);
    return sized && descriptor.width == target.extent.width && descriptor.height == target.extent.height && descriptor.mipCount == 1 && (descriptor.dimension == TextureDimension::k2D || descriptor.dimension == TextureDimension::k2DArray);
}

struct DepthGuard {
    VkDevice device;
    std::uint64_t address;
    std::size_t bytes;
    std::shared_ptr<std::atomic<bool>> active;
    std::vector<GuestMemory::UnchangedQuery> live;
};

std::shared_mutex& depthGuardMutex() {
    static std::shared_mutex mutex;
    return mutex;
}

std::vector<DepthGuard>& depthGuards() {
    static std::vector<DepthGuard> ranges;
    return ranges;
}

std::atomic<bool>& anyDepthGuard() {
    static std::atomic<bool> any{false};
    return any;
}

struct DepthSamplingState {
    std::array<GuestMemory::UnchangedQuery, 2> initial{};
    std::string refusal;

    void Refuse(const char* reason) {
        if (refusal.empty()) refusal = reason;
    }
};

bool htileProvesClear(std::uint32_t word, bool depthOnly, float clearDepth) {
    if ((word & 15u) == 15u) return false;
    const auto depth = depthOnly ? word : word & 0xfffff00fu;
    if (depth == 0u) return clearDepth == 0.0f;
    if (depth == (depthOnly ? 0xfffffff0u : 0xfffc0000u)) return clearDepth == 1.0f;
    return false;
}

DepthSamplingState prepareDepthSampling(const Context& context, const DepthTarget& target) {
    DepthSamplingState state;
    const auto refuse = [&](const char* reason) {
        state.refusal = reason;
        return state;
    };
    if (target.samples != 2u && target.samples != 4u && target.samples != 8u) return refuse("unsupported native depth sample count");
    if (target.format != VK_FORMAT_D16_UNORM && target.format != VK_FORMAT_D32_SFLOAT) return refuse("native multisample stencil or depth format import is not implemented");
    if (target.depthSwizzle != 24u || target.firstSlice != 0u || target.lastSlice != 0u) return refuse("native multisample depth arrays or non-Z_X layout are not implemented");
    if ((target.pipeBankXor & ~0xcf00u) != 0u) return refuse("unsupported native multisample depth XOR");
    if (target.address == 0u || (target.address & 65535u) != 0u) return refuse("native multisample depth base is not aligned");
    if (target.stencilAddress != 0u || target.htileStencil) return refuse("native multisample stencil import is not implemented");
    if (target.htileAddress == 0u || (target.htileAddress & 255u) != 0u || !target.htilePipeAligned || target.htileVrsEncoding != 0u) return refuse("native multisample depth initial import requires pipe-aligned HTILE without VRS");
    if (target.clearDepth != 0.0f && target.clearDepth != 1.0f) return refuse("native multisample depth initial clear value is not implemented");
    state.initial = {{{target.address, static_cast<std::size_t>(MultisampledDepthBytes_nid_no_patch(target.extent, target.format == VK_FORMAT_D16_UNORM ? 2u : 4u, target.samples)), 0u}, {target.htileAddress, static_cast<std::size_t>(HtileMetadataBytes(target.extent)), 0u}}};
    for (const auto& range : state.initial) {
        if (range.address > std::numeric_limits<std::uint64_t>::max() - range.bytes) return refuse("native depth import range overflows");
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(range.address), range.bytes) || !GuestMemory::Watched(range.address, range.bytes)) return refuse("native multisample depth import requires watched accessible guest memory");
    }
    const auto& depth = state.initial[0];
    const auto& htile = state.initial[1];
    if (depth.address + depth.bytes > htile.address && htile.address + htile.bytes > depth.address) return refuse("native depth and HTILE initial ranges overlap");
    for (auto& range : state.initial) {
        if (auto* recorder = RecorderFor(context)) recorder->SyncThrough(range.address, range.bytes);
        StorageTexture::FlushPending(range.address, range.bytes);
        GuestMemory::FlushGpuWrites(range.address, range.bytes);
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(range.address), range.bytes) || !GuestMemory::Watched(range.address, range.bytes)) return refuse("native multisample depth import requires watched accessible guest memory");
        range.generation = GuestMemory::CollectWritesUncached(range.address, range.bytes);
        if (range.generation == 0u) return refuse("native multisample depth import requires tracked generations");
    }
    std::vector<std::uint32_t> metadata(htile.bytes / sizeof(std::uint32_t));
    GuestMemory::Read(target.htileAddress, std::as_writable_bytes(std::span(metadata)), 256u);
    for (const auto word : metadata) {
        if (!htileProvesClear(word, target.htileDepthOnly, target.clearDepth)) return refuse("native multisample depth initial HTILE does not prove its clear value");
    }
    for (const auto& range : state.initial) GuestMemory::CollectWritesUncached(range.address, range.bytes);
    if (!GuestMemory::UnchangedSinceAll(state.initial)) return refuse("guest writes raced native multisample depth import");
    return state;
}

std::atomic<bool> depthPlaneClearEnabled{true};

enum DepthClearDecline : std::size_t { DeclineLayers, DeclineFormat, DeclineValue, DeclineColor, DeclinePending, DeclineCoverage, DeclineSize, DeclineCount };
static_assert(DeclineCount <= 8, "DepthClearStatistics::declined holds eight reasons");

struct DepthClearCounters {
    std::atomic<std::uint64_t> matched{0};
    std::atomic<std::uint64_t> taken{0};
    std::atomic<std::uint64_t> copiesSkipped{0};
    std::atomic<std::uint64_t> clears{0};
    std::atomic<std::uint64_t> bytesSkipped{0};
    std::array<std::atomic<std::uint64_t>, 8> declined{};
};

DepthClearCounters& depthClearCounters() {
    static DepthClearCounters counters;
    return counters;
}

struct DepthPlaneStoreCounters {
    std::atomic<std::uint64_t> stores{0};
    std::atomic<std::uint64_t> units{0};
    std::atomic<std::uint64_t> bytes{0};
};

DepthPlaneStoreCounters& depthPlaneStoreCounters() {
    static DepthPlaneStoreCounters counters;
    return counters;
}

bool anyUnit(const std::vector<bool>& units) {
    return std::find(units.begin(), units.end(), true) != units.end();
}

thread_local DepthPlaneOverwrite* currentOverwrite = nullptr;

std::optional<DepthPlaneOverwrite> decodeOverwrite(const StorageImageClear& clear) {
    auto& counters = depthClearCounters();
    ++counters.matched;
    const auto decline = [&](DepthClearDecline why) -> std::optional<DepthPlaneOverwrite> {
        ++counters.declined[why];
        return std::nullopt;
    };
    DepthPlaneOverwrite overwrite;
    overwrite.image = clear.image;
    try {
        overwrite.view = DecodeTextureResource(clear.image);
    } catch (...) {
        return decline(DeclineFormat);
    }
    const auto& view = overwrite.view;
    const bool oneLayer = view.dimension == TextureDimension::k2D || (view.dimension == TextureDimension::k2DArray && view.baseArray == 0 && view.depthOrLastArray == 0);
    if (!oneLayer || view.baseLevel != 0) return decline(DeclineLayers);
    VkFormat format = VK_FORMAT_UNDEFINED;
    try {
        format = ResolveTextureFormat(view.format);
    } catch (...) {
        return decline(DeclineFormat);
    }
    enum class Kind { Unorm16, Uint16, Float32, Uint32 };
    Kind kind;
    switch (format) {
        case VK_FORMAT_R16_UNORM: kind = Kind::Unorm16; break;
        case VK_FORMAT_R16_UINT: kind = Kind::Uint16; break;
        case VK_FORMAT_R32_SFLOAT: kind = Kind::Float32; break;
        case VK_FORMAT_R32_UINT:
        case VK_FORMAT_R32_SINT: kind = Kind::Uint32; break;
        default: return decline(DeclineFormat);
    }
    const std::uint64_t base = clear.color[0] | (static_cast<std::uint64_t>(clear.color[1] & 0xffffu) << 32u);
    const std::uint64_t stride = (clear.color[1] >> 16u) & 0x3fffu;
    const std::uint64_t extent = stride == 0 ? clear.color[2] : static_cast<std::uint64_t>(clear.color[2]) * stride;
    if (base == 0 || (base & 3u) != 0u || extent < 16u || !GuestMemory::Accessible(reinterpret_cast<const void*>(base), 16)) return decline(DeclineColor);
    if (const auto* recorder = Recorder::Active(); recorder != nullptr && recorder->PendingWriteOverlaps(base, 16)) return decline(DeclinePending);
    const std::pair<std::uint64_t, std::uint64_t> colorRange{base, base + 16u};
    if (StorageTexture::AnyPendingOverlaps(std::span(&colorRange, 1))) return decline(DeclinePending);
    std::uint32_t c0 = 0;
    try {
        GuestMemory::Read(base, std::as_writable_bytes(std::span(&c0, 1)), 4);
    } catch (...) {
        return decline(DeclineColor);
    }
    switch (kind) {
        case Kind::Unorm16:
            if (c0 == 0u || c0 == 0x80000000u) overwrite.raw = 0u;
            else if (c0 == 0x3f800000u) overwrite.raw = 0xffffu;
            else return decline(DeclineValue);
            overwrite.bits = 16u;
            overwrite.depth = static_cast<float>(overwrite.raw) / 65535.0f;
            break;
        case Kind::Uint16:
            if (c0 != 0u && c0 != 0xffffu) return decline(DeclineValue);
            overwrite.raw = c0;
            overwrite.bits = 16u;
            overwrite.depth = static_cast<float>(overwrite.raw) / 65535.0f;
            break;
        case Kind::Float32:
        case Kind::Uint32: {
            const auto depth = std::bit_cast<float>(c0);
            if (c0 != 0u && !(std::isnormal(depth) && depth > 0.0f && depth <= 1.0f)) return decline(DeclineValue);
            overwrite.raw = c0;
            overwrite.bits = 32u;
            overwrite.depth = depth;
            break;
        }
    }
    overwrite.colorAddress = base;
    overwrite.colorBits = c0;
    overwrite.width = clear.width;
    overwrite.height = clear.height;
    return overwrite;
}

class DepthSurface {
public:
    DepthSurface(const Context& context, const DepthTarget& target, DepthSamplingState sampling = {}) : context(context), target(target), hostFormat(HostDepthFormat(context, target.format)), sampling(std::move(sampling)) {
        this->context.bufferPool.reset();
        VkFormatProperties properties{};
        context.formatProperties(context.physical, hostFormat, &properties);
        Require((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0, "depth/stencil format " + std::to_string(hostFormat) + " cannot be an attachment on this device");
        Require(target.samples != 1u || (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0, "depth/stencil format " + std::to_string(hostFormat) + " cannot be sampled on this device");
        Require(target.extent.width <= context.limits.maxFramebufferWidth && target.extent.height <= context.limits.maxFramebufferHeight, "depth target exceeds framebuffer limits");
        const VkImageAspectFlags aspects = VK_IMAGE_ASPECT_DEPTH_BIT | (target.stencilAddress != 0 ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
        try {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            if (SampleLocationsCompatibleDepth(context, hostFormat, target.samples)) info.flags = VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = hostFormat;
            info.extent = {target.extent.width, target.extent.height, 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = static_cast<VkSampleCountFlagBits>(target.samples);
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            sampleable = target.samples == 1;
            if (target.samples != 1) {
                Require(context.imageFormatProperties != nullptr, "native multisample depth requires image format queries");
                VkImageFormatProperties shadowProperties{};
                VkImageFormatProperties directProperties{};
                const auto sampledUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                const auto shadowResult = context.imageFormatProperties(context.physical, hostFormat, info.imageType, info.tiling, sampledUsage, 0, &shadowProperties);
                if (shadowResult != VK_ERROR_FORMAT_NOT_SUPPORTED) Check(shadowResult, "vkGetPhysicalDeviceImageFormatProperties sampled depth shadow");
                const auto directResult = context.imageFormatProperties(context.physical, hostFormat, info.imageType, info.tiling, info.usage | VK_IMAGE_USAGE_SAMPLED_BIT, info.flags, &directProperties);
                if (directResult != VK_ERROR_FORMAT_NOT_SUPPORTED) Check(directResult, "vkGetPhysicalDeviceImageFormatProperties sampled depth attachment");
                const auto supports = [&](VkResult result, const VkImageFormatProperties& supported) {
                    return (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0 && result == VK_SUCCESS && (supported.sampleCounts & info.samples) != 0 && supported.maxMipLevels >= 1u && supported.maxArrayLayers >= 1u && supported.maxExtent.depth >= 1u && info.extent.width <= supported.maxExtent.width && info.extent.height <= supported.maxExtent.height;
                };
                useSampleShadow = supports(shadowResult, shadowProperties);
                const bool directSampleable = supports(directResult, directProperties);
                sampleable = useSampleShadow || directSampleable;
                if (!useSampleShadow && directSampleable) info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
            }
            if (target.samples == 1 && sampleable) info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &image), "vkCreateImage depth");
            VkMemoryRequirements requirements{};
            context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            memory = AllocateImageMemory(context, allocation, requirements.memoryTypeBits, "vkAllocateMemory depth target");
            Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory depth");
            if (target.samples != 1u) owned = std::make_shared<OwnedImage>(this->context, image, memory);
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = hostFormat;
            viewInfo.subresourceRange = {aspects, 0, 1, 0, 1};
            Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &view), "vkCreateImageView depth");
            if (target.samples != 1u && sampleable && this->sampling.refusal.empty()) CheckCurrent();
            auto* recorder = RecorderFor(context);
            if (recorder != nullptr && owned != nullptr) recorder->Keep(owned);
            std::unique_ptr<CommandBatch> batch;
            if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
            const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
            VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            toGeneral.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.image = image;
            toGeneral.subresourceRange = viewInfo.subresourceRange;
            const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
            barrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
            const VkClearDepthStencilValue clear{target.clearDepth, target.clearStencil};
            context.Function<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage")(commands, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &toGeneral.subresourceRange);
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
            if (batch) batch->SubmitAndWait();
            else Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
        } catch (...) {
            release();
            throw;
        }
    }

    ~DepthSurface() { release(); }
    DepthSurface(const DepthSurface&) = delete;
    DepthSurface& operator=(const DepthSurface&) = delete;

    void CheckCurrent() {
        if (target.samples == 1u) return;
        const auto clearLease = clearGeneration == 0u ? GuestAllocations::Lease{} : GuestAllocations::GuestAllocationsAcquireAll_nid_postfix();
        Require(active->load(std::memory_order_acquire), "the native multisample depth owner was released");
        Require(sampling.refusal.empty(), sampling.refusal);
        for (const auto& range : sampling.initial) {
            Require(GuestMemory::Accessible(reinterpret_cast<const void*>(range.address), range.bytes) && GuestMemory::Watched(range.address, range.bytes), "native multisample depth requires watched accessible guest memory");
            if (const auto* recorder = RecorderFor(context)) Require(!recorder->PendingWriteOverlaps(range.address, range.bytes), "external GPU writes over native multisample depth are not implemented");
            GuestMemory::CollectWritesUncached(range.address, range.bytes);
        }
        if (clearGeneration != 0u) {
            const auto& depth = sampling.initial[0];
            const auto& metadata = sampling.initial[1];
            Require(GuestMemory::UnchangedSince(depth.address, depth.bytes, depth.generation), "guest writes over native multisample depth are not implemented");
            Require(GuestMemory::UnchangedSinceCollected(metadata.address, metadata.bytes, clearGeneration), "CPU writes raced native multisample depth clear");
            Require(GuestMemory::StoredAllSince_nid_no_patch(metadata.address, metadata.bytes, clearGeneration), "native multisample depth clear has no complete tracked HTILE store");
            Require(clearDepth == 0.0f || clearDepth == 1.0f, "native multisample depth clear value is not implemented");
            auto current = sampling.initial;
            current[1].generation = GuestMemory::CollectWritesUncached(metadata.address, metadata.bytes);
            std::vector<std::uint32_t> words(metadata.bytes / sizeof(std::uint32_t));
            std::memcpy(words.data(), reinterpret_cast<const void*>(metadata.address), metadata.bytes);
            for (const auto word : words) Require(htileProvesClear(word, target.htileDepthOnly, clearDepth), "native multisample depth completed HTILE does not prove its clear value");
            for (const auto& range : current) GuestMemory::CollectWritesUncached(range.address, range.bytes);
            Require(GuestMemory::UnchangedSinceAll(current), "guest writes raced native multisample depth clear");
            Require(GuestMemory::UnchangedSinceCollected(metadata.address, metadata.bytes, clearGeneration), "CPU writes raced native multisample depth clear");
            RecordClear(VK_IMAGE_ASPECT_DEPTH_BIT);
            sampling.initial = current;
            clearGeneration = 0u;
            pendingClear = 0u;
        }
        Require(GuestMemory::UnchangedSinceAll(sampling.initial), "guest writes over native multisample depth are not implemented");
    }

    void CheckResource(const GuestTextureResource& resource) const {
        if (target.samples == 1u) return;
        Require(resource.baseAddress == target.address && resource.width == target.extent.width && resource.height == target.extent.height && resource.samples == target.samples && resource.format == (target.format == VK_FORMAT_D16_UNORM ? 7u : 22u), "native multisample depth sampled identity or format mismatch");
        Require(resource.dimension == TextureDimension::k2DMsaa && resource.tileMode == TextureTileMode::kZ64KBX && resource.depthOrLastArray == 0u && resource.baseArray == 0u && resource.baseLevel == 0u && resource.lastLevel == 0u && resource.mipCount == 1u && resource.allocatedMipCount == 1u && resource.minLod == 0u, "native multisample depth sampled array, mip or layout is not implemented");
        Require(resource.pipeBankXor == target.pipeBankXor && resource.dccAddress == target.htileAddress && resource.iterate256 == target.iterate256 && resource.dccPipeAligned == target.htilePipeAligned && !resource.dccAlphaOnMsb, "native multisample depth sampled metadata or XOR mismatch");
    }

    void ApplyFastClear() {
        if (samplingActive) {
            CheckCurrent();
            Require(pendingClear == 0u, "native multisample depth fast-clear transition is not implemented");
        }
        const auto aspects = pendingClear & (VK_IMAGE_ASPECT_DEPTH_BIT | (target.stencilAddress != 0 ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u));
        pendingClear = 0;
        if (aspects == 0) return;
        RecordClear(aspects);
    }

    void RecordClear(VkImageAspectFlags aspects) {
        const float depth = depthFill.value_or(clearDepth);
        const std::uint8_t stencil = stencilFill.value_or(clearStencil);
        depthFill.reset();
        stencilFill.reset();
        RecordClearValue(aspects, depth, stencil);
    }

    void RecordClearValue(VkImageAspectFlags aspects, float depth, std::uint8_t stencil) {
        written = ++depthWrites;
        auto* recorder = RecorderFor(context);
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        const VkImageSubresourceRange range{aspects, 0, 1, 0, 1};
        constexpr VkAccessFlags access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, access, VK_ACCESS_TRANSFER_WRITE_BIT);
        const VkClearDepthStencilValue clear{depth, stencil};
        context.Function<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage")(commands, image, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, access);
        if (batch) batch->SubmitAndWait();
        else {
            Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
            recorder->Keep(owned);
        }
    }

    bool UnormStencilView(const GuestTextureResource& resource) const {
        return target.samples == 1 && target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress && ResolveTextureFormat(resource.format) == VK_FORMAT_R8_UNORM;
    }

    bool SampledAccepts(std::span<const std::uint32_t> words, const GuestTextureResource& resource) const {
        const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
        const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
        const bool oneSlice = target.samples != 1 ? resource.dimension == TextureDimension::k2DMsaa : resource.dimension == TextureDimension::k2D || (resource.dimension == TextureDimension::k2DArray && resource.depthOrLastArray == 0);
        return (ResolveTextureFormat(resource.format) == expected || depthBits || UnormStencilView(resource)) && oneSlice && resource.width == target.extent.width && resource.height == target.extent.height && resource.baseLevel == 0 && resource.lastLevel == 0 && resource.baseArray == 0;
    }

    std::shared_ptr<Texture> Sampled(std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components, bool drawn = false, VkImageViewType multisampleViewType = VK_IMAGE_VIEW_TYPE_2D) {
        CheckCurrent();
        CheckResource(resource);
        std::array<std::uint32_t, 13> key{};
        std::copy_n(words.begin(), std::min<std::size_t>(words.size(), 8), key.begin());
        key[8] = components.r;
        key[9] = components.g;
        key[10] = components.b;
        key[11] = components.a;
        key[12] = static_cast<std::uint32_t>(multisampleViewType);
        const bool multisampled = target.samples != 1;
        Require(!multisampled || sampleable, "the device cannot sample the native depth target's format and sample count");
        const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
        const auto format = ResolveTextureFormat(resource.format);
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
        const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
        if (multisampled) {
            Require(!stencil && !depthBits && format == expected, "native multisampled depth sampling needs a numeric depth-plane format");
            Require(resource.samples == target.samples, "the sampled depth descriptor's sample count differs from its native target");
            Require(hostFormat == target.format, "sampling a native multisampled depth plane that the device stores in another depth format is not implemented");
        }
        if (stencil && UnormStencilView(resource) && SampledAccepts(words, resource)) {
            return unormStencil(key, components, resource.dimension == TextureDimension::k2DArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, drawn);
        }
        if (!SampledAccepts(words, resource)) {
            char text[448];
            std::snprintf(text, sizeof(text), "AGC graphics: sampling the %s plane of depth surface 0x%llx (%ux%u, vk format %d) as a %ux%u texture of guest format %u (vk %d), tile mode %u, dimension %d, levels %u-%u, slice %u is not implemented (T# %08x %08x %08x %08x %08x %08x %08x %08x)",
                          stencil ? "stencil" : "depth", static_cast<unsigned long long>(target.address), target.extent.width, target.extent.height, static_cast<int>(target.format), resource.width, resource.height, resource.format, static_cast<int>(format),
                          static_cast<unsigned>(resource.tileMode), static_cast<int>(resource.dimension), resource.baseLevel, resource.lastLevel, resource.baseArray, key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7]);
            throw std::runtime_error(text);
        }
        if (multisampled && useSampleShadow) {
            if (sampleShadow == nullptr) createSampleShadow();
            refreshSampleShadow(drawn);
        }
        if (const auto found = textures.find(key); found != textures.end()) return found->second;
        const auto viewType = multisampled ? multisampleViewType : resource.dimension == TextureDimension::k2DArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
        const auto sampledImage = useSampleShadow ? sampleShadow : owned;
        auto texture = std::make_shared<Texture>(context, useSampleShadow ? sampleShadow->image : image, hostFormat, stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, components, viewType, 0, sampledImage);
        if (multisampled) {
            Require(samplingActive, "native multisample depth sampling requires its creation-time owner");
            texture->MarkRefreshedPerUse();
        }
        textures.emplace(key, texture);
        return texture;
    }

    void ForgetClear(VkImageAspectFlags aspects) {
        pendingClear &= ~aspects;
        if ((aspects & VK_IMAGE_ASPECT_DEPTH_BIT) != 0) depthFill.reset();
        if ((aspects & VK_IMAGE_ASPECT_STENCIL_BIT) != 0) stencilFill.reset();
    }

    void NoteHtileClear(VkImageAspectFlags aspects) {
        ForgetClear(aspects);
        pendingClear |= aspects;
    }

    void NotePlaneFill(VkImageAspectFlagBits aspect, float depth, std::uint8_t stencil) {
        pendingClear |= aspect;
        if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) depthFill = depth;
        else stencilFill = stencil;
    }

    void CopyAspectFrom(const DepthSurface& source, VkImageAspectFlagBits aspect) {
        ForgetClear(aspect);
        auto* recorder = RecorderFor(context);
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        constexpr VkAccessFlags all = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, all, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        VkImageCopy region{};
        region.srcSubresource = {static_cast<VkImageAspectFlags>(aspect), 0, 0, 1};
        region.dstSubresource = region.srcSubresource;
        region.extent = {target.extent.width, target.extent.height, 1};
        context.Function<PFN_vkCmdCopyImage>("vkCmdCopyImage")(commands, source.image, VK_IMAGE_LAYOUT_GENERAL, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, all);
        if (batch) batch->SubmitAndWait();
        else Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
    }

    void CopySample(std::uint32_t sample, StorageTexture& destination, VkImageView destinationView) {
        auto* recorder = Recorder::Active();
        Require(recorder != nullptr, "depth sample copies need an active recorder");
        Require(sampleable, "DB depth copy from a depth surface the device cannot sample at its sample count");
        ApplyFastClear();
        if (useSampleShadow) {
            if (sampleShadow == nullptr) createSampleShadow();
            refreshSampleShadow(false);
            recorder->Keep(sampleShadow);
        }
        if (depthView == VK_NULL_HANDLE) {
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = useSampleShadow ? sampleShadow->image : image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = hostFormat;
            viewInfo.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &depthView), "vkCreateImageView depth sample copy");
        }
        Require(context.detiler != nullptr, "depth sample copies need the texture detiler");
        context.detiler->DispatchDepthSample(recorder->Commands(), depthView, target.samples > 1, destinationView, target.extent.width, target.extent.height, sample);
        Recorder::CountBarriers(Recorder::CommandClass::Copy, 2);
        destination.MarkDirty();
    }

    bool Holds(const Texture* texture) const {
        const auto held = [&](const auto& entry) { return entry.second.get() == texture; };
        return std::any_of(textures.begin(), textures.end(), held) || std::any_of(stencilCopy.textures.begin(), stencilCopy.textures.end(), held);
    }

    bool SampledElsewhere() const {
        return std::any_of(textures.begin(), textures.end(), [](const auto& entry) { return entry.second.use_count() > 1; });
    }

    std::uint64_t written = ++depthWrites;
    bool retired = false;

    bool Overlaps(std::uint64_t address, std::uint64_t bytes) const {
        const auto end = address + bytes;
        if (address < target.address + depthBytes() && target.address < end) return true;
        return target.stencilAddress != 0 && address < target.stencilAddress + stencilBytes() && target.stencilAddress < end;
    }

    bool CoveredBy(std::span<const std::pair<std::uint64_t, std::uint64_t>> written, std::span<const WrittenStorageExtent> extents) const {
        const auto reaches = [&](std::uint64_t address, std::uint32_t bytesPerTexel) {
            const auto [blockWidth, blockHeight] = DepthBlockExtent(bytesPerTexel);
            const std::array<std::uint32_t, 3> block{65536u, blockWidth, blockHeight};
            const auto blocksPerRow = (target.extent.width + blockWidth - 1u) / blockWidth;
            return std::any_of(extents.begin(), extents.end(), [&](const auto& extent) {
                return extent.address == address && extent.bytesPerElement == bytesPerTexel && extent.block == block && extent.blocksPerRow == blocksPerRow && extent.width >= target.extent.width && extent.height >= target.extent.height;
            });
        };
        const auto covers = [&](std::uint64_t begin, std::uint64_t end) {
            for (const auto& [writtenBegin, writtenEnd] : written) {
                if (writtenEnd <= begin) continue;
                if (writtenBegin > begin) return false;
                begin = writtenEnd;
                if (begin >= end) return true;
            }
            return false;
        };
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        if (target.samples == 1u && (reaches(target.address, d16 ? 2u : 4u) || (target.stencilAddress != 0 && reaches(target.stencilAddress, 1u)))) return true;
        return covers(target.address, target.address + depthBytes()) || (target.stencilAddress != 0 && covers(target.stencilAddress, target.stencilAddress + stencilBytes()));
    }

    void NoteMemory() {
        depthMemory = GuestMemory::CollectWrites(target.address, static_cast<std::size_t>(depthBytes()));
        stencilMemory = target.stencilAddress != 0 ? GuestMemory::CollectWrites(target.stencilAddress, static_cast<std::size_t>(stencilBytes())) : 0;
    }

    // LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.6): NoteMemory where both collects
    // are memo hits answering `generation` (a DepthSurfaceView lease's hit). Mirror tripwire (11.13):
    // NoteMemory.
    void NoteMemoryAt(std::uint64_t generation) {
        depthMemory = generation;
        stencilMemory = target.stencilAddress != 0 ? generation : 0;
    }

    // LOCAL DEBUG (APS5_PROFILE_DRAW, GPU-OWNED-PLAN 9.5 D5-0): the depth and stencil ranges with the
    // generations NoteMemory last took (the stencil range empty without a stencil plane).
    std::array<GuestMemory::UnchangedQuery, 2> NotedMemory() const {
        return {{{target.address, static_cast<std::size_t>(depthBytes()), depthMemory}, {target.stencilAddress, target.stencilAddress != 0 ? static_cast<std::size_t>(stencilBytes()) : std::size_t{0}, stencilMemory}}};
    }

    bool OverwrittenInMemory() const {
        GuestMemory::CollectWrites(target.address, static_cast<std::size_t>(depthBytes()));
        if (GuestMemory::WrittenSince(target.address, static_cast<std::size_t>(depthBytes()), depthMemory)) return true;
        if (target.stencilAddress == 0) return false;
        GuestMemory::CollectWrites(target.stencilAddress, static_cast<std::size_t>(stencilBytes()));
        return GuestMemory::WrittenSince(target.stencilAddress, static_cast<std::size_t>(stencilBytes()), stencilMemory);
    }

    VkImageAspectFlags pendingClear = 0;
    float clearDepth = 0.0f;
    std::uint8_t clearStencil = 0;
    std::optional<float> depthFill;
    std::optional<std::uint8_t> stencilFill;

    bool StorageViewMatches(std::span<const std::uint32_t> words, const GuestTextureResource& resource) const {
        if (resource.width != target.extent.width || resource.height != target.extent.height) return false;
        const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
        const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
        return ResolveTextureFormat(resource.format) == expected || depthBits;
    }

    void RequireStorable(const GuestTextureResource& resource) const {
        const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
        Require(target.samples == 1, "storage image access to a multisampled depth surface is not implemented");
        if (!stencil && hostFormat != target.format) {
            char text[224];
            std::snprintf(text, sizeof(text), "AGC graphics: storage image access to the depth plane of depth surface 0x%llx (vk format %d) stored as vk format %d is not implemented", static_cast<unsigned long long>(target.address), static_cast<int>(target.format), static_cast<int>(hostFormat));
            throw std::runtime_error(text);
        }
    }

    std::shared_ptr<StorageTexture> Storage(std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint32_t mip) {
        if (!StorageViewMatches(words, resource)) return nullptr;
        const bool stencil = target.stencilAddress != 0 && resource.baseAddress == target.stencilAddress;
        const bool oneLayer = resource.dimension == TextureDimension::k2DArray && resource.baseArray == 0 && resource.depthOrLastArray == 0;
        if (resource.dimension != TextureDimension::k2D && !oneLayer) {
            char text[256];
            std::snprintf(text, sizeof(text), "AGC graphics: storage image access to depth/stencil surface 0x%llx is not implemented (storage=%ux%u guest format=%u mip=%u dimension=%d tile=%d)", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, mip, static_cast<int>(resource.dimension), static_cast<int>(resource.tileMode));
            throw std::runtime_error(text);
        }
        RequireStorable(resource);
        auto* overwrite = stencil ? nullptr : PlaneOverwritten(words);
        ApplyFastClear();
        auto& plane = stencil ? stencilPlane : depthPlane;
        if (plane.storage == nullptr || plane.storage->Descriptor().format != resource.format || plane.storage->Descriptor().tileMode != resource.tileMode || plane.storage->Descriptor().baseAddress != resource.baseAddress || plane.storage->Descriptor().dimension != resource.dimension) {
            std::erase_if(plane.replaced, [](const auto& old) { return old.expired(); });
            if (plane.storage != nullptr) plane.replaced.push_back(plane.storage);
            plane.storage = StorageTexture::Create(context, *context.detiler, resource, 0);
        }
        if (overwrite != nullptr) {
            ++depthClearCounters().copiesSkipped;
        } else {
            copyPlane(plane, plane.storage, stencil, true);
        }
        return plane.storage;
    }

    bool OverwritesWholePlane(const DepthPlaneOverwrite& overwrite) const {
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        return overwrite.bits == (d16 ? 16u : 32u) && overwrite.width >= target.extent.width && overwrite.height >= target.extent.height;
    }

    DepthPlaneOverwrite* PlaneOverwritten(std::span<const std::uint32_t> words) const {
        auto* overwrite = currentOverwrite;
        if (overwrite == nullptr || words.size() < overwrite->image.size() || !std::equal(overwrite->image.begin(), overwrite->image.end(), words.begin())) return nullptr;
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        auto& declined = depthClearCounters().declined;
        if (overwrite->bits != (d16 ? 16u : 32u)) {
            ++declined[DeclineSize];
            return nullptr;
        }
        if (overwrite->width < target.extent.width || overwrite->height < target.extent.height) {
            ++declined[DeclineCoverage];
            return nullptr;
        }
        return overwrite;
    }

    bool StoreOverwrite(const StorageTexture& mirror) {
        auto* overwrite = currentOverwrite;
        if (overwrite == nullptr || !OverwritesWholePlane(*overwrite)) return false;
        const auto& described = mirror.Descriptor();
        const auto& view = overwrite->view;
        if (described.baseAddress != view.baseAddress || described.format != view.format || described.tileMode != view.tileMode || described.dimension != view.dimension) return false;
        std::uint32_t current = 0;
        try {
            GuestMemory::Read(overwrite->colorAddress, std::as_writable_bytes(std::span(&current, 1)), 4);
        } catch (...) {
            return false;
        }
        if (current != overwrite->colorBits) return false;
        auto* recorder = RecorderFor(context);
        Require(recorder != nullptr, "depth surface storage clears need the recorder of the surface's device");
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto bytes = static_cast<std::size_t>(target.extent.width) * target.extent.height * (d16 ? 2u : 4u);
        recorder->Commands();
        const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::Copy);
        RecordClearValue(VK_IMAGE_ASPECT_DEPTH_BIT, overwrite->depth, clearStencil);
        recorder->EndGpuTiming(timing, bytes);
        auto& counters = depthClearCounters();
        ++counters.clears;
        counters.bytesSkipped += 2u * bytes;
        return true;
    }

    bool StoreBack(const StorageTexture& candidate) {
        for (const bool stencil : {false, true}) {
            auto& plane = stencil ? stencilPlane : depthPlane;
            if (plane.storage.get() == &candidate) {
                if (stencil || !StoreOverwrite(candidate)) copyPlane(plane, plane.storage, stencil, false);
                written = ++depthWrites;
                return true;
            }
            if (std::none_of(plane.replaced.begin(), plane.replaced.end(), [&](const auto& old) { return old.lock().get() == &candidate; })) continue;
            char text[288];
            std::snprintf(text, sizeof(text), "AGC graphics: storage image writes to the %s plane of depth surface 0x%llx through a view that another view of the plane (guest format %u, tile mode %d) replaced are not implemented", stencil ? "stencil" : "depth", static_cast<unsigned long long>(target.address), plane.storage->Descriptor().format, static_cast<int>(plane.storage->Descriptor().tileMode));
            throw std::runtime_error(text);
        }
        return false;
    }

    bool StoreBackCovering(const StorageTexture& candidate) {
        for (const bool stencil : {false, true}) {
            if (!CoversPlane(candidate, stencil)) continue;
            auto& plane = stencil ? stencilPlane : depthPlane;
            const auto source = std::const_pointer_cast<StorageTexture>(candidate.shared_from_this());
            copyPlane(plane, source, stencil, false);
            written = ++depthWrites;
            return true;
        }
        return false;
    }

    void TakeStorage() {
        if (target.samples != 1 || hostFormat != target.format) return;
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const auto bytes = DepthSliceBytes(target.extent, d16 ? 2u : 4u);
        auto storage = StorageTexture::FindPending(target.address, bytes);
        if (storage == nullptr) storage = StorageTexture::FindLive(target.address, bytes);
        if (storage == nullptr || !storageFits(target, storage->Descriptor())) return;
        storage->Refresh();
        if (depthPlane.staging == nullptr) depthPlane.staging = std::make_unique<DeviceBuffer>(context, static_cast<std::size_t>(target.extent.width) * target.extent.height * (d16 ? 2u : 4u), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        auto* recorder = RecorderFor(context);
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        if (recorder != nullptr) recorder->Keep(storage);
        VkBufferImageCopy depthRegion{0, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {target.extent.width, target.extent.height, 1}};
        VkBufferImageCopy colorRegion = depthRegion;
        colorRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, storage->Image(), VK_IMAGE_LAYOUT_GENERAL, depthPlane.staging->Handle(), 1, &colorRegion);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, depthPlane.staging->Handle(), image, VK_IMAGE_LAYOUT_GENERAL, 1, &depthRegion);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        if (batch) batch->SubmitAndWait();
        else Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
    }

    std::shared_ptr<StorageTexture> Mirror(const GuestTextureResource& resource) {
        if (mirrorPlane.storage == nullptr) {
            mirrorPlane.storage = StorageTexture::Create(context, *context.detiler, resource, 0);
            ++mirrorEpoch;
        }
        return mirrorPlane.storage;
    }

    bool StoresUnits(const StorageTexture& mirror) const {
        return mirror.BlockUnits() && mirror.TrackedUnits() == staleUnits.size();
    }

    std::shared_ptr<StorageTexture> StoreDepthPlane(const GuestTextureResource& resource, std::vector<bool>& units) {
        ApplyFastClear();
        auto mirror = Mirror(resource);
        auto& counters = depthPlaneStoreCounters();
        ++counters.stores;
        const auto planeBytes = mirror->GuestBytes();
        if (!StoresUnits(*mirror)) {
            units.assign(units.size(), true);
            copyPlane(mirrorPlane, mirror, false, true, true);
            mirror->MarkWholeWritten();
            counters.units += units.size();
            counters.bytes += planeBytes;
            return mirror;
        }
        const auto rects = mirror->UnitRects(units);
        copyPlane(mirrorPlane, mirror, false, true, true, &rects);
        mirror->MarkUnitsWritten(units);
        for (std::size_t unit = 0; unit < units.size(); ++unit) {
            if (!units[unit]) continue;
            ++counters.units;
            counters.bytes += std::min<std::uint64_t>(unitBytes, planeBytes - unit * unitBytes);
        }
        return mirror;
    }

    void SizeUnits(std::uint64_t planeBytes) {
        static const bool whole = std::getenv("APS5_DEPTH_PLANE_WHOLE_STORE") != nullptr;
        const std::uint64_t bytes = !whole && target.address % 65536u == 0 ? std::uint64_t{65536} : std::max<std::uint64_t>(planeBytes, 1u);
        const auto count = static_cast<std::size_t>(std::max<std::uint64_t>((planeBytes + bytes - 1u) / bytes, 1u));
        if (unitBytes == bytes && staleUnits.size() == count) return;
        unitBytes = bytes;
        staleUnits.assign(count, false);
        staleCount = 0;
        unitGeneration.assign(count, 0);
    }

    bool AnyStale() const {
        return staleCount != 0;
    }

    void MarkAllStale(std::uint64_t planeBytes) {
        SizeUnits(planeBytes);
        staleUnits.assign(staleUnits.size(), true);
        staleCount = staleUnits.size();
    }

    void ClearStale() {
        staleUnits.assign(staleUnits.size(), false);
        staleCount = 0;
    }

    void ClearStale(const std::vector<bool>& units) {
        for (std::size_t unit = 0; unit < units.size() && unit < staleUnits.size(); ++unit) {
            if (!units[unit] || !staleUnits[unit]) continue;
            staleUnits[unit] = false;
            --staleCount;
        }
    }

    std::vector<bool> StaleUnitsIn(std::uint64_t planeBytes, std::uint64_t begin, std::uint64_t end) const {
        std::vector<bool> units(staleUnits.size(), false);
        const auto plane = target.address;
        if (units.empty() || unitBytes == 0 || begin >= end || begin >= plane + planeBytes || plane >= end) return units;
        const auto from = std::max(begin, plane) - plane;
        const auto to = std::min(end, plane + planeBytes) - plane;
        const auto last = std::min<std::uint64_t>((to - 1u) / unitBytes, units.size() - 1u);
        for (auto unit = from / unitBytes; unit <= last; ++unit) units[static_cast<std::size_t>(unit)] = staleUnits[static_cast<std::size_t>(unit)];
        return units;
    }

    std::pair<std::uint64_t, std::uint64_t> UnitHull(const std::vector<bool>& units, std::uint64_t planeBytes) const {
        std::uint64_t first = planeBytes;
        std::uint64_t last = 0;
        for (std::size_t unit = 0; unit < units.size(); ++unit) {
            if (!units[unit]) continue;
            first = std::min<std::uint64_t>(first, unit * unitBytes);
            last = std::max<std::uint64_t>(last, std::min<std::uint64_t>(planeBytes, (unit + 1u) * unitBytes));
        }
        return {target.address + first, target.address + std::max(first, last)};
    }

    void NoteStored(const std::vector<bool>& units, std::uint64_t generation) {
        for (std::size_t unit = 0; unit < units.size() && unit < unitGeneration.size(); ++unit) {
            if (units[unit]) unitGeneration[unit] = generation;
        }
    }

    std::vector<bool> WrittenUnits(const GuestTextureResource& resource, std::uint64_t& generation) {
        const auto bytes = static_cast<std::size_t>(DescribeSurface(resource).guestBytes);
        SizeUnits(bytes);
        GuestMemory::CollectWrites(resource.baseAddress, bytes);
        generation = GuestMemory::TrackerGeneration();
        std::vector<bool> changed(unitGeneration.size(), false);
        if (unitGeneration.size() == 1u) {
            changed[0] = GuestMemory::WrittenSince(resource.baseAddress, bytes, unitGeneration[0]);
            return changed;
        }
        const auto oldest = *std::min_element(unitGeneration.begin(), unitGeneration.end());
        if (oldest != 0u && !GuestMemory::WrittenSince(resource.baseAddress, bytes, oldest)) return changed;
        std::vector<std::uint8_t> stamped(unitGeneration.size(), GuestMemory::BlockUnchanged);
        if (!GuestMemory::ChangedBlocks(resource.baseAddress, bytes, unitGeneration, stamped)) return changed;
        for (std::size_t unit = 0; unit < changed.size(); ++unit) changed[unit] = unitGeneration[unit] != 0u && stamped[unit] == GuestMemory::BlockWritten;
        return changed;
    }

    void SyncFromMemory(const GuestTextureResource& resource) {
        std::uint64_t generation = 0;
        const auto changed = WrittenUnits(resource, generation);
        if ((pendingClear & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 || !anyUnit(changed)) {
            unitGeneration.assign(unitGeneration.size(), generation);
            return;
        }
        ClearStale(changed);
        {
            struct Syncing {
                bool& flag;
                ~Syncing() { flag = false; }
            } guard{syncing};
            syncing = true;
            if (mirrorPlane.storage == nullptr) {
                mirrorPlane.storage = StorageTexture::Create(context, *context.detiler, resource, 0);
                ++mirrorEpoch;
            } else {
                static_cast<void>(mirrorPlane.storage->Refresh());
            }
        }
        unitGeneration.assign(unitGeneration.size(), generation);
        const auto mirror = mirrorPlane.storage;
        if (StoresUnits(*mirror)) {
            const auto rects = mirror->UnitRects(changed);
            copyPlane(mirrorPlane, mirror, false, false, true, &rects);
        } else {
            ClearStale();
            copyPlane(mirrorPlane, mirror, false, false, true);
        }
        written = ++depthWrites;
    }

    // LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.6): for a DepthSurfaceView lease,
    // under the surfaces mutex. UnitsSized: SizeUnits(planeBytes) would return early (the same unit size
    // and count; the switch is read once, as SizeUnits reads it). OldestUnitGeneration: the generation
    // WrittenUnits tests the plane from (the only unit's, or the oldest of several; 0 without units).
    // SyncUnchanged: SyncFromMemory's exit with nothing changed (SizeUnits, then every unit assigned
    // `generation`, TrackerGeneration's answer). Mirror tripwire (11.13): SizeUnits, WrittenUnits,
    // SyncFromMemory. HasMirror: a test hook (SyncFromMemory made the mirror plane).
    bool UnitsSized(std::uint64_t planeBytes) const {
        static const bool whole = std::getenv("APS5_DEPTH_PLANE_WHOLE_STORE") != nullptr;
        const std::uint64_t bytes = !whole && target.address % 65536u == 0 ? std::uint64_t{65536} : std::max<std::uint64_t>(planeBytes, 1u);
        const auto count = static_cast<std::size_t>(std::max<std::uint64_t>((planeBytes + bytes - 1u) / bytes, 1u));
        return unitBytes == bytes && staleUnits.size() == count && unitGeneration.size() == count;
    }

    std::uint64_t OldestUnitGeneration() const {
        return unitGeneration.empty() ? 0 : *std::min_element(unitGeneration.begin(), unitGeneration.end());
    }

    void SyncUnchanged(std::uint64_t planeBytes, std::uint64_t generation) {
        SizeUnits(planeBytes);
        unitGeneration.assign(unitGeneration.size(), generation);
    }

    bool HasMirror() const {
        return mirrorPlane.storage != nullptr;
    }

    void DropMirror() {
        mirrorPlane = {};
    }

    std::vector<bool> staleUnits;
    std::size_t staleCount = 0;
    std::vector<std::uint64_t> unitGeneration;
    std::uint64_t unitBytes = 0;
    bool syncing = false;
    std::uint64_t scannedEpoch = 0;

    const Context context;
    const DepthTarget target;
    const VkFormat hostFormat;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageView depthView = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::shared_ptr<std::atomic<bool>> active = std::make_shared<std::atomic<bool>>(true);
    bool sampleable = true;
    bool samplingActive = false;
    DepthSamplingState sampling;
    std::uint64_t clearGeneration = 0u;
    std::shared_ptr<OwnedImage> owned;
    std::shared_ptr<OwnedImage> sampleShadow;
    bool useSampleShadow = false;
    std::uint64_t sampleShadowWritten = 0u;

private:
    struct PlaneStorage {
        std::shared_ptr<StorageTexture> storage;
        std::vector<std::weak_ptr<StorageTexture>> replaced;
        std::unique_ptr<DeviceBuffer> staging;
    };
    struct StencilCopy {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        std::unique_ptr<DeviceBuffer> staging;
        std::map<std::array<std::uint32_t, 13>, std::shared_ptr<Texture>> textures;
        std::uint64_t written = 0;
        bool filled = false;
    };
    std::map<std::array<std::uint32_t, 13>, std::shared_ptr<Texture>> textures;
    PlaneStorage depthPlane;
    PlaneStorage stencilPlane;
    std::uint64_t depthMemory = 0;
    std::uint64_t stencilMemory = 0;
    StencilCopy stencilCopy;

    void createSampleShadow() {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        try {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = hostFormat;
            info.extent = {target.extent.width, target.extent.height, 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = static_cast<VkSampleCountFlagBits>(target.samples);
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &image), "vkCreateImage sampled depth shadow");
            VkMemoryRequirements requirements{};
            context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory), "vkAllocateMemory sampled depth shadow");
            Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory sampled depth shadow");
            sampleShadow = std::make_shared<OwnedImage>(context, image, memory);
            image = VK_NULL_HANDLE;
            memory = VK_NULL_HANDLE;
            auto* recorder = RecorderFor(context);
            if (recorder != nullptr) recorder->Keep(sampleShadow);
            std::unique_ptr<CommandBatch> batch;
            if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
            const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = sampleShadow->image;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            if (batch) batch->SubmitAndWait();
            else Recorder::CountBarriers(Recorder::CommandClass::Draw, 1);
        } catch (...) {
            if (image != VK_NULL_HANDLE) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
            if (memory != VK_NULL_HANDLE) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
            throw;
        }
    }

    void refreshSampleShadow(bool drawn) {
        Require(sampleShadow != nullptr, "native multisample depth shadow was not created");
        if (!drawn && sampleShadowWritten == written) return;
        auto* recorder = RecorderFor(context);
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        if (recorder != nullptr) recorder->Keep(sampleShadow);
        constexpr VkAccessFlags accesses = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, accesses, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        VkImageCopy region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        region.dstSubresource = region.srcSubresource;
        region.extent = {target.extent.width, target.extent.height, 1};
        context.Function<PFN_vkCmdCopyImage>("vkCmdCopyImage")(commands, image, VK_IMAGE_LAYOUT_GENERAL, sampleShadow->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, accesses);
        if (batch) batch->SubmitAndWait();
        else Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
        sampleShadowWritten = written;
    }

    std::shared_ptr<Texture> unormStencil(const std::array<std::uint32_t, 13>& key, VkComponentMapping components, VkImageViewType viewType, bool drawn) {
        if (stencilCopy.image == VK_NULL_HANDLE) {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = VK_FORMAT_R8_UNORM;
            info.extent = {target.extent.width, target.extent.height, 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            try {
                Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &stencilCopy.image), "vkCreateImage stencil copy");
                VkMemoryRequirements requirements{};
                context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, stencilCopy.image, &requirements);
                VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocation.allocationSize = requirements.size;
                allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                stencilCopy.memory = AllocateImageMemory(context, allocation, requirements.memoryTypeBits, "vkAllocateMemory stencil copy");
                Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, stencilCopy.image, stencilCopy.memory, 0), "vkBindImageMemory stencil copy");
                stencilCopy.staging = std::make_unique<DeviceBuffer>(context, static_cast<std::size_t>(target.extent.width) * target.extent.height, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            } catch (...) {
                releaseStencilCopy();
                throw;
            }
        }
        if (drawn || !stencilCopy.filled || stencilCopy.written != written) {
            copyStencil();
            stencilCopy.written = drawn ? 0 : written;
        }
        auto& texture = stencilCopy.textures[key];
        if (texture == nullptr) {
            texture = std::make_shared<Texture>(context, stencilCopy.image, VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, components, viewType);
            texture->MarkRefreshedPerUse();
        }
        return texture;
    }

    void copyStencil() {
        auto* recorder = RecorderFor(context);
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{0, 0, 0, {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1}, {0, 0, 0}, {target.extent.width, target.extent.height, 1}};
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image, VK_IMAGE_LAYOUT_GENERAL, stencilCopy.staging->Handle(), 1, &region);
        VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toTransfer.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toTransfer.oldLayout = stencilCopy.filled ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = stencilCopy.image;
        toTransfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const VkBufferMemoryBarrier staged{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, stencilCopy.staging->Handle(), 0, VK_WHOLE_SIZE};
        barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &staged, 1, &toTransfer);
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, stencilCopy.staging->Handle(), stencilCopy.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        VkImageMemoryBarrier toGeneral = toTransfer;
        toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
        stencilCopy.filled = true;
        if (batch) batch->SubmitAndWait();
        else Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
    }

    void releaseStencilCopy() noexcept {
        stencilCopy.textures.clear();
        stencilCopy.staging.reset();
        if (stencilCopy.image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, stencilCopy.image, nullptr);
        if (stencilCopy.memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, stencilCopy.memory, nullptr);
        stencilCopy = {};
    }

    std::uint64_t depthBytes() const {
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        return DepthSliceBytes(target.extent, d16 ? 2u : 4u);
    }

    std::uint64_t stencilBytes() const {
        return DepthSliceBytes(target.extent, 1u);
    }
    PlaneStorage mirrorPlane;

    bool CoversPlane(const StorageTexture& storage, bool stencil) const {
        const auto& resource = storage.Descriptor();
        const auto address = stencil ? target.stencilAddress : target.address;
        if (address == 0 || resource.baseAddress != address || resource.width != target.extent.width || resource.height != target.extent.height || resource.baseLevel != 0 || resource.mipCount != 1) return false;
        if (resource.dimension != TextureDimension::k2D && (resource.dimension != TextureDimension::k2DArray || resource.baseArray != 0 || resource.depthOrLastArray != 0)) return false;
        if (target.samples != 1 || (!stencil && hostFormat != target.format)) return false;
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        return ResolveTextureFormat(resource.format) == (stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT);
    }

    void copyPlane(PlaneStorage& plane, const std::shared_ptr<StorageTexture>& storage, bool stencil, bool toStorage, bool ownBatch = false, const std::vector<VkRect2D>* rects = nullptr) {
        if (rects != nullptr && rects->empty()) return;
        auto* recorder = Recorder::Active();
        Require(recorder != nullptr || ownBatch, "depth surface storage copies need an active recorder");
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        const std::uint32_t texelBytes = stencil ? 1u : d16 ? 2u : 4u;
        const auto bytes = static_cast<std::size_t>(target.extent.width) * target.extent.height * texelBytes;
        if (plane.staging == nullptr) plane.staging = std::make_unique<DeviceBuffer>(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        if (recorder != nullptr) recorder->Keep(storage);
        const std::uint32_t timing = recorder != nullptr ? recorder->BeginGpuTiming(Recorder::CommandClass::Copy) : 0u;
        const VkBufferImageCopy planeRegion{0, 0, 0, {stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {target.extent.width, target.extent.height, 1}};
        std::vector<VkBufferImageCopy> planeRegions;
        std::size_t moved = bytes;
        if (rects == nullptr) {
            planeRegions.push_back(planeRegion);
        } else {
            moved = 0;
            for (const auto& rect : *rects) {
                auto region = planeRegion;
                region.bufferOffset = (static_cast<VkDeviceSize>(rect.offset.y) * target.extent.width + static_cast<VkDeviceSize>(rect.offset.x)) * texelBytes;
                region.bufferRowLength = target.extent.width;
                region.imageOffset = {rect.offset.x, rect.offset.y, 0};
                region.imageExtent = {rect.extent.width, rect.extent.height, 1};
                planeRegions.push_back(region);
                moved += static_cast<std::size_t>(rect.extent.width) * rect.extent.height * texelBytes;
            }
        }
        auto colorRegions = planeRegions;
        for (auto& region : colorRegions) region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        const auto count = static_cast<std::uint32_t>(planeRegions.size());
        const auto sourceImage = toStorage ? image : storage->Image();
        const auto destinationImage = toStorage ? storage->Image() : image;
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, sourceImage, VK_IMAGE_LAYOUT_GENERAL, plane.staging->Handle(), count, toStorage ? planeRegions.data() : colorRegions.data());
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, plane.staging->Handle(), destinationImage, VK_IMAGE_LAYOUT_GENERAL, count, toStorage ? colorRegions.data() : planeRegions.data());
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        if (batch) {
            batch->SubmitAndWait();
            return;
        }
        Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
        recorder->EndGpuTiming(timing, moved);
    }

    void release() noexcept {
        depthPlane = {};
        stencilPlane = {};
        releaseStencilCopy();
        mirrorPlane = {};
        textures.clear();
        if (depthView) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, depthView, nullptr);
        if (view) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, view, nullptr);
        if (owned != nullptr) owned.reset();
        else {
            if (image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
            if (memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
        }
        view = VK_NULL_HANDLE;
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
    }
};

struct DepthPlaneSlice {
    VkImage image = VK_NULL_HANDLE;
    std::uint64_t written = 0;
};

class DepthPlaneCopy {
public:
    DepthPlaneCopy(const Context& context, VkExtent2D extent, VkFormat format, std::uint32_t layers) : context(context), extent(extent), format(format), layers(layers), guestLayers(layers), surfaceLayers(layers) {
        this->context.bufferPool.reset();
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.flags = layers % 6u == 0 && extent.width == extent.height ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {extent.width, extent.height, 1};
        info.mipLevels = 1;
        info.arrayLayers = layers;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        try {
            Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &image), "vkCreateImage depth plane copy");
            VkMemoryRequirements requirements{};
            context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            memory = AllocateImageMemory(context, allocation, requirements.memoryTypeBits, "vkAllocateMemory depth plane copy");
            Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory depth plane copy");
            stagingSize = static_cast<std::size_t>(sliceBytes()) * layers;
            staging = std::make_unique<DeviceBuffer>(context, stagingSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        } catch (...) {
            release();
            throw;
        }
    }
    ~DepthPlaneCopy() { release(); }
    DepthPlaneCopy(const DepthPlaneCopy&) = delete;
    DepthPlaneCopy& operator=(const DepthPlaneCopy&) = delete;

    std::shared_ptr<Texture> Refresh(std::span<const DepthPlaneSlice> slices, const GuestTextureResource& resource, VkComponentMapping components, VkImageViewType viewType, std::uint32_t baseLayer) {
        Require(slices.size() == layers, "depth plane copy slices do not match its layers");
        const auto geometry = DescribeSurface(resource);
        const bool guestFill = std::any_of(slices.begin(), slices.end(), [](const auto& slice) { return slice.image == VK_NULL_HANDLE; });
        if (guestFill && (geometry.layers != layers || geometry.sliceLinearBytes < sliceBytes() || geometry.mips.empty())) {
            char text[320];
            std::snprintf(text, sizeof(text), "AGC graphics: depth plane copy geometry does not match its layers (copy %ux%u, %u layers, %llu bytes a slice; texture %ux%u guest format %u, dimension %d, tile mode %u, %u layers, %llu bytes a slice, %zu mips)", extent.width, extent.height, layers, static_cast<unsigned long long>(sliceBytes()), resource.width, resource.height, resource.format, static_cast<int>(resource.dimension), resource.tileMode, geometry.layers, static_cast<unsigned long long>(geometry.sliceLinearBytes), geometry.mips.size());
            throw std::runtime_error(text);
        }
        const auto stagingBytes = static_cast<std::size_t>(geometry.sliceLinearBytes) * layers;
        if (stagingBytes > stagingSize) {
            retiredStaging.push_back(std::move(staging));
            staging = std::make_unique<DeviceBuffer>(context, stagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            stagingSize = stagingBytes;
        }
        std::vector<std::shared_ptr<Buffer>> uploads;
        std::vector<std::uint32_t> uploadLayers;
        std::vector<std::uint32_t> copiedLayers;
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            auto& held = guestLayers[layer];
            auto& surface = surfaceLayers[layer];
            if (slices[layer].image != VK_NULL_HANDLE) {
                held = {};
                if (slices[layer].written != 0 && surface.image == slices[layer].image && surface.written == slices[layer].written) continue;
                surface = slices[layer];
                copiedLayers.push_back(layer);
                continue;
            }
            surface = {};
            const GuestLayer current{resource.baseAddress + geometry.GuestLayerOffset(layer), geometry.layerBytes, resource.tileMode, geometry.thick, GuestMemory::CollectWrites(resource.baseAddress + geometry.GuestLayerOffset(layer), static_cast<std::size_t>(geometry.layerBytes))};
            if (held.generation != 0 && held.address == current.address && held.bytes == current.bytes && held.tileMode == current.tileMode && held.thick == current.thick && !Recorder::SnapshotWriteOverlaps(current.address, static_cast<std::size_t>(current.bytes)) && GuestMemory::UnchangedSince(current.address, static_cast<std::size_t>(current.bytes), held.generation)) {
                held.generation = current.generation;
                continue;
            }
            auto upload = std::make_shared<Buffer>(context, static_cast<std::size_t>(geometry.layerBytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            GuestMemory::ReadCommitted(current.address, upload->Bytes().first(static_cast<std::size_t>(geometry.layerBytes)));
            uploads.push_back(std::move(upload));
            uploadLayers.push_back(layer);
            held = current;
        }
        if (!copiedLayers.empty() || !uploads.empty()) record(slices, resource, geometry, copiedLayers, uploads, uploadLayers);
        const std::array<std::uint32_t, 6> key{static_cast<std::uint32_t>(components.r), static_cast<std::uint32_t>(components.g), static_cast<std::uint32_t>(components.b), static_cast<std::uint32_t>(components.a), static_cast<std::uint32_t>(viewType), baseLayer};
        auto& texture = textures[key];
        if (texture == nullptr) {
            texture = std::make_shared<Texture>(context, image, format, aspect(), components, viewType, baseLayer);
            texture->MarkRefreshedPerUse();
        }
        return texture;
    }

    const Context context;
    const VkExtent2D extent;
    const VkFormat format;
    const std::uint32_t layers;

private:
    struct GuestLayer {
        std::uint64_t address = 0;
        std::uint64_t bytes = 0;
        TextureTileMode tileMode = TextureTileMode::kLinear;
        bool thick = false;
        std::uint64_t generation = 0;
    };

    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::unique_ptr<DeviceBuffer> staging;
    std::size_t stagingSize = 0;
    std::vector<std::unique_ptr<DeviceBuffer>> retiredStaging;
    std::map<std::array<std::uint32_t, 6>, std::shared_ptr<Texture>> textures;
    std::vector<GuestLayer> guestLayers;
    std::vector<DepthPlaneSlice> surfaceLayers;
    bool filled = false;

    void record(std::span<const DepthPlaneSlice> slices, const GuestTextureResource& resource, const SurfaceGeometry& geometry, std::span<const std::uint32_t> copiedLayers, std::span<const std::shared_ptr<Buffer>> uploads, std::span<const std::uint32_t> uploadLayers) {
        if (!uploads.empty()) {
            Require(context.detiler != nullptr, "depth plane copy requires a texture detiler");
            context.detiler->BeginBatch();
        }
        auto* recorder = RecorderFor(context);
        std::unique_ptr<CommandBatch> batch;
        if (recorder == nullptr) batch = std::make_unique<CommandBatch>(context);
        const auto commands = recorder != nullptr ? recorder->Commands() : batch->Handle();
        const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        std::vector<VkBufferImageCopy> regions;
        const auto& mip = geometry.mips.front();
        const auto region = [&](std::uint32_t layer, VkImageAspectFlags aspectMask, std::uint32_t imageLayer) {
            VkBufferImageCopy copy{};
            copy.bufferOffset = geometry.LinearLayerOffset(layer) + mip.linearOffset;
            copy.bufferRowLength = mip.pitchBytes / (format == VK_FORMAT_D32_SFLOAT ? 4u : 2u);
            copy.imageSubresource = {aspectMask, 0, imageLayer, 1};
            copy.imageExtent = {extent.width, extent.height, 1};
            return copy;
        };
        for (const auto layer : copiedLayers) {
            const auto copy = region(layer, VK_IMAGE_ASPECT_DEPTH_BIT, 0);
            context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, slices[layer].image, VK_IMAGE_LAYOUT_GENERAL, staging->Handle(), 1, &copy);
            regions.push_back(region(layer, aspect(), layer));
        }
        if (!uploads.empty()) {
            RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            for (std::size_t upload = 0; upload < uploads.size(); ++upload) {
                const auto layer = uploadLayers[upload];
                context.detiler->Dispatch(commands, resource.tileMode, format == VK_FORMAT_D32_SFLOAT ? 4u : 2u, uploads[upload]->Handle(), 0, staging->Handle(), geometry.LinearLayerOffset(layer), geometry.mips.front(), false, layer, geometry.thick);
                regions.push_back(region(layer, aspect(), layer));
            }
            if (recorder != nullptr) {
                for (const auto& upload : uploads) recorder->Keep(upload, upload->Bytes().size());
            }
        }
        VkImageMemoryBarrier toTransfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toTransfer.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toTransfer.oldLayout = filled ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
        toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toTransfer.image = image;
        toTransfer.subresourceRange = {aspect(), 0, 1, 0, layers};
        const VkBufferMemoryBarrier staged{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, staging->Handle(), 0, VK_WHOLE_SIZE};
        barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &staged, 1, &toTransfer);
        context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, staging->Handle(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<std::uint32_t>(regions.size()), regions.data());
        VkImageMemoryBarrier toGeneral = toTransfer;
        toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &toGeneral);
        filled = true;
        if (batch) batch->SubmitAndWait();
        else Recorder::CountBarriers(Recorder::CommandClass::Draw, 3);
    }

    VkDeviceSize sliceBytes() const {
        return static_cast<VkDeviceSize>(extent.width) * extent.height * (format == VK_FORMAT_D32_SFLOAT ? 4u : 2u);
    }

    VkImageAspectFlags aspect() const {
        return format == VK_FORMAT_R16_UINT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    }

    void release() noexcept {
        textures.clear();
        staging.reset();
        if (image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
        if (memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
    }
};

std::map<std::tuple<VkDevice, std::uint64_t, std::uint32_t, std::uint32_t, VkFormat, std::uint32_t>, std::unique_ptr<DepthPlaneCopy>>& planeCopies() {
    static auto* copies = new std::map<std::tuple<VkDevice, std::uint64_t, std::uint32_t, std::uint32_t, VkFormat, std::uint32_t>, std::unique_ptr<DepthPlaneCopy>>();
    return *copies;
}

struct LayeredSlice {
    DepthSurface* surface;
    std::uint32_t layer;
};

class LayeredDepthStorage {
public:
    LayeredDepthStorage(const Context& context, const GuestTextureResource& resource, const DepthTarget& base, std::uint32_t bytesPerTexel, std::uint32_t firstLayer, std::uint32_t layers) : context(context), base(base), extent(base.extent), bytesPerTexel(bytesPerTexel), firstLayer(firstLayer), layers(layers) {
        const auto bytes = DescribeSurface(resource).guestBytes;
        StorageTexture::FlushPending(resource.baseAddress, static_cast<std::size_t>(bytes), nullptr, "storage image creation", PublishScope::None);
        storage = StorageTexture::Create(context, *context.detiler, resource, 0);
        Require(storage->ImageLayers() == layers, "a layered depth storage view's image does not hold its layers");
        staging = std::make_unique<DeviceBuffer>(context, static_cast<std::size_t>(sliceBytes()) * layers, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    void Copy(std::span<const LayeredSlice> slices, bool toStorage) {
        if (slices.empty()) return;
        auto* recorder = Recorder::Active();
        Require(recorder != nullptr, "depth surface storage copies need an active recorder");
        const auto commands = recorder->Commands();
        recorder->Keep(storage);
        const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::Copy);
        std::vector<VkBufferImageCopy> storageRegions;
        const auto region = [&](std::uint32_t layer, VkImageAspectFlags aspect, std::uint32_t imageLayer) {
            VkBufferImageCopy copy{};
            copy.bufferOffset = sliceBytes() * layer;
            copy.imageSubresource = {aspect, 0, imageLayer, 1};
            copy.imageExtent = {extent.width, extent.height, 1};
            return copy;
        };
        for (const auto& slice : slices) storageRegions.push_back(region(slice.layer, VK_IMAGE_ASPECT_COLOR_BIT, slice.layer));
        const auto copyToBuffer = context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer");
        const auto copyToImage = context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage");
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        if (toStorage) {
            for (const auto& slice : slices) {
                const auto copy = region(slice.layer, VK_IMAGE_ASPECT_DEPTH_BIT, 0);
                copyToBuffer(commands, slice.surface->image, VK_IMAGE_LAYOUT_GENERAL, staging->Handle(), 1, &copy);
            }
        } else {
            copyToBuffer(commands, storage->Image(), VK_IMAGE_LAYOUT_GENERAL, staging->Handle(), static_cast<std::uint32_t>(storageRegions.size()), storageRegions.data());
        }
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        if (toStorage) {
            copyToImage(commands, staging->Handle(), storage->Image(), VK_IMAGE_LAYOUT_GENERAL, static_cast<std::uint32_t>(storageRegions.size()), storageRegions.data());
        } else {
            for (const auto& slice : slices) {
                const auto copy = region(slice.layer, VK_IMAGE_ASPECT_DEPTH_BIT, 0);
                copyToImage(commands, staging->Handle(), slice.surface->image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
            }
        }
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
        recorder->EndGpuTiming(timing, sliceBytes() * slices.size());
    }

    const Context context;
    const DepthTarget base;
    const VkExtent2D extent;
    const std::uint32_t bytesPerTexel;
    const std::uint32_t firstLayer;
    const std::uint32_t layers;
    std::shared_ptr<StorageTexture> storage;

private:
    std::unique_ptr<DeviceBuffer> staging;

    VkDeviceSize sliceBytes() const {
        return static_cast<VkDeviceSize>(extent.width) * extent.height * bytesPerTexel;
    }
};

std::map<std::tuple<VkDevice, std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, int, int, std::uint32_t>, std::unique_ptr<LayeredDepthStorage>>& layeredStorages() {
    static auto* storages = new std::map<std::tuple<VkDevice, std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, int, int, std::uint32_t>, std::unique_ptr<LayeredDepthStorage>>();
    return *storages;
}

bool sameSurface(const DepthTarget& a, const DepthTarget& b) {
    return a.address == b.address && a.stencilAddress == b.stencilAddress && a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.format == b.format && a.samples == b.samples && (a.samples == 1u || (a.htileAddress == b.htileAddress && a.htileStencil == b.htileStencil && a.depthSwizzle == b.depthSwizzle && a.pipeBankXor == b.pipeBankXor && a.firstSlice == b.firstSlice && a.lastSlice == b.lastSlice && a.iterate256 == b.iterate256 && a.htilePipeAligned == b.htilePipeAligned && a.htileVrsEncoding == b.htileVrsEncoding && a.htileDepthOnly == b.htileDepthOnly));
}

// Recursive since main 84e3b73f1 (a store reached from a flush under DepthSurfaceTexture re-enters
// it), so it stays std::recursive_mutex rather than the non-recursive AgcDriver::Mutex of PR #6273.
std::recursive_mutex& surfacesMutex() {
    static std::recursive_mutex mutex;
    return mutex;
}

std::vector<std::unique_ptr<DepthSurface>>& surfaces() {
    static auto* list = new std::vector<std::unique_ptr<DepthSurface>>();
    return *list;
}

struct PlaneOwner {
    DepthSurface* surface = nullptr;
    std::uint64_t serial = 0;
};

using PlaneOwners = std::map<std::pair<VkDevice, std::uint64_t>, PlaneOwner>;

std::uint64_t planeBinds = 0;

PlaneOwners& depthOwners() {
    static auto* owners = new PlaneOwners();
    return *owners;
}

PlaneOwners& stencilOwners() {
    static auto* owners = new PlaneOwners();
    return *owners;
}

void takePlane(DepthSurface& surface, PlaneOwners& owners, std::uint64_t address, VkImageAspectFlagBits aspect, bool created, VkImageAspectFlags clearAspects) {
    auto& entry = owners[{surface.context.device, address}];
    auto* owner = entry.surface;
    if (owner != nullptr && owner->active->load(std::memory_order_acquire) && owner != &surface && owner->target.format == surface.target.format && owner->target.samples == surface.target.samples && owner->target.extent.width == surface.target.extent.width && owner->target.extent.height == surface.target.extent.height && (!created || (clearAspects & aspect) == 0)) {
        if ((owner->pendingClear & aspect) != 0) {
            if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) surface.depthFill = owner->depthFill;
            else surface.stencilFill = owner->stencilFill;
            if (!created || (aspect == VK_IMAGE_ASPECT_DEPTH_BIT ? surface.depthFill.has_value() : surface.stencilFill.has_value())) surface.pendingClear |= aspect;
        } else {
            surface.CopyAspectFrom(*owner, aspect);
        }
    }
    entry = {&surface, ++planeBinds};
}

void takePlanes(DepthSurface& surface, bool created, VkImageAspectFlags clearAspects) {
    takePlane(surface, depthOwners(), surface.target.address, VK_IMAGE_ASPECT_DEPTH_BIT, created, clearAspects);
    if (surface.target.stencilAddress != 0) takePlane(surface, stencilOwners(), surface.target.stencilAddress, VK_IMAGE_ASPECT_STENCIL_BIT, created, clearAspects);
}

DepthSurface* depthSlice(VkDevice device, std::uint64_t address, const DepthTarget& base) {
    if (const auto found = depthOwners().find({device, address}); found != depthOwners().end()) {
        const auto& target = found->second.surface->target;
        if (found->second.surface->active->load(std::memory_order_acquire) && target.extent.width == base.extent.width && target.extent.height == base.extent.height && target.format == base.format) return found->second.surface;
    }
    const auto& list = surfaces();
    const auto slice = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) { return surface->active->load(std::memory_order_acquire) && surface->context.device == device && !surface->retired && surface->target.address == address && surface->target.extent.width == base.extent.width && surface->target.extent.height == base.extent.height && surface->target.format == base.format; });
    return slice == list.rend() ? nullptr : slice->get();
}

DepthSurface* planeSurface(VkDevice device, std::uint64_t address) {
    const auto depth = depthOwners().find({device, address});
    const auto stencil = stencilOwners().find({device, address});
    const bool hasDepth = depth != depthOwners().end() && depth->second.surface->active->load(std::memory_order_acquire);
    const bool hasStencil = stencil != stencilOwners().end() && stencil->second.surface->active->load(std::memory_order_acquire);
    if (hasDepth && (!hasStencil || depth->second.serial > stencil->second.serial)) return depth->second.surface;
    if (hasStencil) return stencil->second.surface;
    const auto& list = surfaces();
    const auto found = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) {
        return surface->active->load(std::memory_order_acquire) && surface->context.device == device && !surface->retired && (surface->target.address == address || (surface->target.stencilAddress != 0 && surface->target.stencilAddress == address));
    });
    return found == list.rend() ? nullptr : found->get();
}

std::map<std::uint64_t, std::size_t>& clearedHtiles() {
    static auto* cleared = new std::map<std::uint64_t, std::size_t>();
    return *cleared;
}

bool takeClearedHtile(const DepthTarget& target) {
    if (target.htileAddress == 0) return false;
    auto& cleared = clearedHtiles();
    const auto found = std::find_if(cleared.begin(), cleared.end(), [&](const auto& entry) { return HtileFillCovers(target.htileAddress, target.extent, entry.first, entry.second); });
    if (found == cleared.end()) return false;
    cleared.erase(found);
    return true;
}

DepthSurface* drawnDepth = nullptr;

std::atomic<std::uint64_t> surfacesSerial{0};
std::atomic<std::uint64_t> surfacesReleased{0};

void releaseRetired(const Context& context) {
    auto* recorder = Recorder::Active();
    if (recorder != nullptr && recorder->Device() != context.device) return;
    auto& list = surfaces();
    for (auto it = list.begin(); it != list.end();) {
        const auto& surface = **it;
        if (!surface.retired || surface.context.device != context.device || surface.target.samples != 1u || surface.SampledElsewhere()) {
            ++it;
            continue;
        }
        if (drawnDepth == &surface) drawnDepth = nullptr;
        std::shared_ptr<DepthSurface> released = std::move(*it);
        it = list.erase(it);
        surfacesReleased.fetch_add(1, std::memory_order_release);
        if (recorder != nullptr) recorder->Keep(std::move(released));
    }
}

std::uint32_t expandedHtileWord(std::uint32_t word, bool stencilInHtile, VkImageAspectFlags cleared) {
    if (!stencilInHtile) return (cleared & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 ? 0xfffc000fu : word;
    if ((cleared & VK_IMAGE_ASPECT_DEPTH_BIT) != 0) word = (word & 0x3f0u) | 0xfffff00fu;
    if ((cleared & VK_IMAGE_ASPECT_STENCIL_BIT) != 0) word |= 0x3f0u;
    return word;
}

std::mutex& htileReadsMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<std::uint64_t, std::uint64_t>& htileReads() {
    static auto* reads = new std::map<std::uint64_t, std::uint64_t>();
    return *reads;
}

void forgetHtileReads(std::uint64_t address, std::size_t bytes) {
    std::lock_guard lock(htileReadsMutex());
    auto& reads = htileReads();
    reads.erase(reads.lower_bound(address), reads.lower_bound(address + bytes));
}

bool RoutesToNativeDepthOwner(const DepthTarget& target) {
    if (target.samples != 1u) return true;
    if (target.htileAddress == 0 || !anyDepthGuard().load(std::memory_order_acquire)) return false;
    const auto bytes = static_cast<std::uint64_t>(HtileMetadataBytes(target.extent));
    std::shared_lock lock(depthGuardMutex());
    return std::ranges::any_of(depthGuards(), [&](const auto& range) { return range.active->load(std::memory_order_acquire) && target.htileAddress < range.address + range.bytes && range.address < target.htileAddress + bytes; });
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.8): `covered`, when given, receives
// whether the pending fills covered the whole range (so no guest word was read) once a word landed.
std::optional<std::uint32_t> htileAfterPendingFills(const Recorder& recorder, std::uint64_t address, std::size_t bytes, bool* covered = nullptr) {
    auto writes = recorder.PendingWritesOver(address, bytes);
    std::sort(writes.begin(), writes.end(), [](const Recorder::PendingWrite& a, const Recorder::PendingWrite& b) { return a.note > b.note; });
    std::vector<std::pair<std::uint64_t, std::uint64_t>> unwritten{{address, address + bytes}};
    std::optional<std::uint32_t> landed;
    for (const auto& write : writes) {
        if (unwritten.empty()) break;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> rest;
        bool covered = false;
        for (const auto& [from, to] : unwritten) {
            if (write.end <= from || to <= write.begin) {
                rest.emplace_back(from, to);
                continue;
            }
            covered = true;
            if (from < write.begin) rest.emplace_back(from, write.begin);
            if (write.end < to) rest.emplace_back(write.end, to);
        }
        unwritten = std::move(rest);
        if (!covered) continue;
        if (write.word < 0 || (landed.has_value() && *landed != static_cast<std::uint32_t>(write.word))) return std::nullopt;
        landed = static_cast<std::uint32_t>(write.word);
    }
    if (!landed.has_value()) return std::nullopt;
    if (covered != nullptr) *covered = unwritten.empty();
    for (const auto& [from, to] : unwritten) {
        if (from % 4u != 0 || to % 4u != 0) return std::nullopt;
        const auto* words = reinterpret_cast<const volatile std::uint32_t*>(from);
        for (std::uint64_t index = 0; index < (to - from) / 4u; ++index) {
            if (words[index] != *landed) return std::nullopt;
        }
    }
    return landed;
}

bool storeHtileOnGpu(const Context& context, Recorder& recorder, std::uint64_t address, std::size_t bytes, std::uint32_t word) {
    if (address % 4u != 0 || bytes % 4u != 0) return false;
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr) return false;
    recorder.FlushKeyStoresOverlapping(address, bytes);
    recorder.FlushStoresOverlapping(address, bytes);
    const auto commands = recorder.Commands();
    const auto timing = recorder.BeginGpuTiming(Recorder::CommandClass::Fill);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    Recorder::CountBarriers(Recorder::CommandClass::Fill);
    context.Resolved(&DeviceFunctions::cmdFillBuffer, "vkCmdFillBuffer")(commands, import->buffer, address - import->base, bytes, word);
    constexpr VkAccessFlags filledAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, filledAccess);
    Recorder::CountBarriers(Recorder::CommandClass::Fill);
    recorder.EndGpuTiming(timing, bytes);
    recorder.MarkCovered(filledAccess);
    recorder.NotePendingWordFill(address, bytes, word);
    GuestMemory::MarkWritten(address, bytes);
    return true;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.10): storeBackClearedHtile's switch,
// read once, hoisted so a DepthSurfaceView lease's hit reads the same answer.
bool pendingHtileFills() {
    static const bool pendingFills = std::getenv("APS5_NO_PENDING_HTILE_FILLS") == nullptr;
    return pendingFills;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.8): what storeBackClearedHtile
// decided, for a DepthSurfaceView lease's creation and =2's check, each value read at its decision
// before any store of its own: where it returned (DepthLease.hpp), the Accessible answer (when read),
// the collect's generation and the recorder's finished serial and newest write note (read together,
// right before the pending test: GPU-OWNED-PLAN 11.2 row 22), and the not-pending read-proof lookup.
struct HtileVerdict {
    HtileVerdictKind kind = HtileVerdictKind::NoHtile;
    bool accessibleRead = false;
    bool accessible = false;
    bool collectedRead = false;
    std::uint64_t collected = 0;
    std::uint64_t finished = 0;
    std::uint64_t notes = 0;
    bool lookedUp = false;
    bool found = false;
    std::uint64_t foundValue = 0;
};

// LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES): `verdict` (null: today exactly) receives HtileVerdict.
// Mirror tripwire (GPU-OWNED-PLAN 9.6, 11.13): a DepthSurfaceView lease's hit (depthViewLeaseHit) stands
// in for this function's routes; any change here re-runs the =2 rows and one in-game =2 run.
VkImageAspectFlags storeBackClearedHtile(const Context& context, const DepthTarget& target, HtileVerdict* verdict = nullptr) {
    if (target.htileAddress == 0) return 0;
    const auto bytes = static_cast<std::size_t>(HtileMetadataBytes(target.extent));
    const bool accessible = GuestMemory::Accessible(reinterpret_cast<const void*>(target.htileAddress), bytes, true);
    if (verdict != nullptr) *verdict = {HtileVerdictKind::NotAccessible, true, accessible};
    if (!accessible) return 0;
    const auto collected = GuestMemory::CollectWrites(target.htileAddress, bytes);
    auto* recorder = RecorderFor(context);
    if (verdict != nullptr) {
        verdict->collectedRead = true;
        verdict->collected = collected;
        verdict->finished = recorder != nullptr ? recorder->FinishedSerial() : 0;
        verdict->notes = recorder != nullptr ? recorder->WriteNotes() : 0;
    }
    const bool pending = recorder != nullptr && recorder->PendingWriteOverlaps(target.htileAddress, bytes);
    if (!pending) {
        std::lock_guard lock(htileReadsMutex());
        const auto found = htileReads().find(target.htileAddress);
        if (verdict != nullptr) {
            verdict->kind = HtileVerdictKind::NotPendingProved;
            verdict->lookedUp = true;
            verdict->found = found != htileReads().end();
            verdict->foundValue = verdict->found ? found->second : 0;
        }
        if (found != htileReads().end() && found->second != 0 && GuestMemory::UnchangedSince(target.htileAddress, bytes, found->second)) return 0;
    }
    if (pending && pendingHtileFills()) {
        std::lock_guard gpu(GuestMemory::GpuMutex());
        bool covered = false;
        if (const auto landed = htileAfterPendingFills(*recorder, target.htileAddress, bytes, verdict != nullptr ? &covered : nullptr)) {
            const VkImageAspectFlags cleared = (VK_IMAGE_ASPECT_DEPTH_BIT | (target.htileStencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u)) & HtileFillClears(*landed, target.htileStencil);
            if (cleared == 0) {
                std::lock_guard lock(htileReadsMutex());
                htileReads()[target.htileAddress] = collected;
                if (verdict != nullptr) verdict->kind = covered ? HtileVerdictKind::PendingCovered : HtileVerdictKind::PendingNotCovering;
                return 0;
            }
            if (storeHtileOnGpu(context, *recorder, target.htileAddress, bytes, expandedHtileWord(*landed, target.htileStencil, cleared))) {
                const auto generation = GuestMemory::CollectWrites(target.htileAddress, bytes);
                std::lock_guard lock(htileReadsMutex());
                htileReads()[target.htileAddress] = generation;
                if (verdict != nullptr) verdict->kind = HtileVerdictKind::StoredBack;
                return cleared;
            }
        }
    }
    std::vector<std::uint32_t> words(bytes / 4u);
    GuestMemory::Read(target.htileAddress, std::as_writable_bytes(std::span(words)), 4);
    VkImageAspectFlags cleared = VK_IMAGE_ASPECT_DEPTH_BIT | (target.htileStencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
    for (const auto word : words) cleared &= HtileFillClears(word, target.htileStencil);
    auto generation = collected;
    if (cleared != 0) {
        for (auto& word : words) word = expandedHtileWord(word, target.htileStencil, cleared);
        GuestMemory::Write(target.htileAddress, std::as_bytes(std::span(words)), 4);
        generation = GuestMemory::CollectWrites(target.htileAddress, bytes);
    }
    std::lock_guard lock(htileReadsMutex());
    htileReads()[target.htileAddress] = generation;
    if (verdict != nullptr) verdict->kind = cleared != 0 ? HtileVerdictKind::StoredBack : !pending ? HtileVerdictKind::NotPendingRead : pendingHtileFills() ? HtileVerdictKind::PendingNotCovering : HtileVerdictKind::PendingFillsOff;
    return cleared;
}

struct StalePlanes {
    std::mutex mutex;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    std::atomic<std::uint64_t> begin{std::numeric_limits<std::uint64_t>::max()};
    std::atomic<std::uint64_t> end{0};
};

StalePlanes& stalePlanes() {
    static auto* planes = new StalePlanes();
    return *planes;
}

void publishStaleHull(StalePlanes& planes) {
    auto begin = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t end = 0;
    for (const auto& range : planes.ranges) {
        begin = std::min(begin, range.first);
        end = std::max(end, range.second);
    }
    planes.begin.store(begin, std::memory_order_release);
    planes.end.store(end, std::memory_order_release);
}

void noteStalePlane(std::uint64_t begin, std::uint64_t end) {
    auto& planes = stalePlanes();
    std::lock_guard lock(planes.mutex);
    if (std::find(planes.ranges.begin(), planes.ranges.end(), std::pair{begin, end}) != planes.ranges.end()) return;
    planes.ranges.emplace_back(begin, end);
    publishStaleHull(planes);
}

bool overlapsStalePlane(std::uint64_t begin, std::uint64_t end) {
    auto& planes = stalePlanes();
    if (begin >= planes.end.load(std::memory_order_acquire) || planes.begin.load(std::memory_order_acquire) >= end) return false;
    std::lock_guard lock(planes.mutex);
    return std::any_of(planes.ranges.begin(), planes.ranges.end(), [&](const auto& range) { return begin < range.second && range.first < end; });
}

bool mirrorable(const DepthTarget& target) {
    return (target.format == VK_FORMAT_D32_SFLOAT || target.format == VK_FORMAT_D32_SFLOAT_S8_UINT) && target.samples == 1 && target.depthSwizzle == XorSwizzleMode(TextureTileMode::kZ64KBX);
}

GuestTextureResource mirrorResource(const DepthTarget& target) {
    GuestTextureResource resource{};
    resource.baseAddress = target.address;
    resource.width = target.extent.width;
    resource.height = target.extent.height;
    resource.mipCount = 1;
    resource.allocatedMipCount = 1;
    resource.tileMode = TextureTileMode::kZ64KBX;
    resource.dimension = TextureDimension::k2D;
    resource.format = 22;
    resource.dstSelX = 4;
    resource.dstSelY = 5;
    resource.dstSelZ = 6;
    resource.dstSelW = 7;
    return resource;
}

std::pair<std::uint64_t, std::uint64_t> stalePlaneRange(const DepthSurface& surface) {
    return {surface.target.address, surface.target.address + DescribeSurface(mirrorResource(surface.target)).guestBytes};
}

void retire(DepthSurface& surface) {
    surface.retired = true;
    for (auto* owners : {&depthOwners(), &stencilOwners()}) {
        std::erase_if(*owners, [&](const auto& entry) { return entry.second.surface == &surface; });
    }
    if (!surface.AnyStale()) return;
    surface.ClearStale();
    const auto plane = stalePlaneRange(surface);
    if (std::any_of(surfaces().begin(), surfaces().end(), [&](const auto& other) { return other->AnyStale() && stalePlaneRange(*other) == plane; })) return;
    auto& planes = stalePlanes();
    std::lock_guard lock(planes.mutex);
    std::erase(planes.ranges, plane);
    publishStaleHull(planes);
}

void retireStoringStale(DepthSurface& surface) {
    if (!surface.AnyStale() || !mirrorable(surface.target)) {
        retire(surface);
        return;
    }
    const auto resource = mirrorResource(surface.target);
    const auto planeBytes = DescribeSurface(resource).guestBytes;
    std::uint64_t generation = 0;
    const auto written = surface.WrittenUnits(resource, generation);
    auto units = surface.staleUnits;
    for (std::size_t unit = 0; unit < units.size() && unit < written.size(); ++unit) {
        if (written[unit]) units[unit] = false;
    }
    std::shared_ptr<StorageTexture> mirror;
    if (anyUnit(units) && (!anyUnit(written) || surface.StoresUnits(*surface.Mirror(resource)))) mirror = surface.StoreDepthPlane(resource, units);
    retire(surface);
    if (mirror == nullptr) return;
    const auto [begin, end] = surface.UnitHull(units, planeBytes);
    if (StorageTexture::FlushPending(begin, static_cast<std::size_t>(end - begin), nullptr, "depth plane store")) surface.NoteStored(units, GuestMemory::TrackerGeneration());
}

// LOCAL DEBUG (APS5_PROFILE_DRAW, GPU-OWNED-PLAN 9.5 D5-0, DepthLease.hpp): the DepthSurfaceView lease
// counter. Read once. Its two serials move only while profiling, under the surfaces mutex: a
// clearedHtiles entry set (NoteHtileDepthClear), and a surface DepthSurfaceView found retired.
bool depthLeaseProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

std::uint64_t clearedHtileSets = 0;
std::uint64_t depthUnretires = 0;

// Relaxed, counted only while profiling; DepthLeaseTotals reads them, the [proofs] line prints deltas.
struct DepthLeaseCounters {
    std::atomic<std::uint64_t> calls{0};
    std::array<std::atomic<std::uint64_t>, DepthLeaseIneligibleCount> ineligible{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DepthLease::Count)> outcomes{};
    std::atomic<std::uint64_t> hitsWriteGeneration{0};
    std::atomic<std::uint64_t> missesWriteGeneration{0};
    std::array<std::atomic<std::uint64_t>, 3> hitWork{};
    std::atomic<std::uint64_t> evictions{0};
    std::atomic<std::uint64_t> counterNs{0};
    std::atomic<std::uint64_t> recipeCounterNs{0};
};
constinit DepthLeaseCounters depthLeaseCounters;

void countDepthLease(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

// Set by DepthLeaseRecipeScope around DrawWithRecipe's call (trivially destructible): the counter's
// time there goes to recipeCounterNs, since that timer books the call to readTarget, not prepare.
thread_local bool depthLeaseInRecipe = false;

void countDepthLeaseNs(std::uint64_t ns) {
    countDepthLease(depthLeaseInRecipe ? depthLeaseCounters.recipeCounterNs : depthLeaseCounters.counterNs, ns);
}

// The key holds every DepthTarget field (clearDepth by its bits). A member added by a merge, wherever
// it lands, fails the field count until it is compared here and the count is raised (FieldCount.hpp).
static_assert(AggregateFieldCount<DepthTarget, 17>, "a DepthTarget field was added: compare it in sameDepthKey (the D5-0 lease key holds every field)");
bool sameDepthKey(const DepthTarget& a, const DepthTarget& b) {
    return a.address == b.address && a.stencilAddress == b.stencilAddress && a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.format == b.format && std::bit_cast<std::uint32_t>(a.clearDepth) == std::bit_cast<std::uint32_t>(b.clearDepth) && a.clearStencil == b.clearStencil && a.htileAddress == b.htileAddress && a.htileStencil == b.htileStencil && a.samples == b.samples && a.depthSwizzle == b.depthSwizzle && a.pipeBankXor == b.pipeBankXor && a.firstSlice == b.firstSlice && a.lastSlice == b.lastSlice && a.iterate256 == b.iterate256 && a.htilePipeAligned == b.htilePipeAligned && a.htileVrsEncoding == b.htileVrsEncoding && a.htileDepthOnly == b.htileDepthOnly;
}

// 16 sets x 4 ways, the set from the target's 64 KiB block, as the D4-0 target lease table. Per thread
// (ThreadScratch).
constexpr std::size_t DepthLeaseSets = 16;
constexpr std::size_t DepthLeaseWays = 4;

struct DepthLeaseEntry {
    bool valid = false;
    const Recorder* recorder = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    DepthTarget target{};
    DepthLeaseMark mark;
    // What the next call's stamp test reads: the depth and stencil planes with the generations
    // NoteMemory took, the HTILE with the generation its read proof (htileReads) held at the end.
    std::array<GuestMemory::UnchangedQuery, 3> ranges{};
    std::uint64_t used = 0;
};

struct DepthLeaseTable {
    std::array<std::array<DepthLeaseEntry, DepthLeaseWays>, DepthLeaseSets> sets{};
    // Per set, fingerprints of the last keys a set conflict evicted (0 empty).
    std::array<std::array<std::uint64_t, DepthLeaseWays>, DepthLeaseSets> evicted{};
    std::array<std::uint8_t, DepthLeaseSets> nextEvicted{};
    std::uint64_t tick = 0;
};
struct DepthLeaseTableTag {};

std::size_t depthLeaseSet(std::uint64_t address) {
    return static_cast<std::size_t>(((address >> 16u) * 0x9E3779B97F4A7C15ull) >> 60u);
}

std::uint64_t depthLeaseFingerprint(const Recorder* recorder, VkDevice device, const DepthTarget& target) {
    std::uint64_t hash = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(recorder)) ^ (static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(device)) << 1u);
    for (const std::uint64_t word : {target.address, target.stencilAddress, target.htileAddress, (std::uint64_t{target.extent.width} << 32u) | target.extent.height, static_cast<std::uint64_t>(static_cast<std::uint32_t>(target.format)) | (std::uint64_t{target.samples} << 32u)}) {
        hash = (hash ^ word) * 0x9E3779B97F4A7C15ull;
    }
    return hash | 1u;
}

DepthLeaseEntry* findDepthLease(DepthLeaseTable& table, std::size_t set, const Recorder* recorder, VkDevice device, const DepthTarget& target) {
    for (auto& way : table.sets[set]) {
        if (way.valid && way.recorder == recorder && way.device == device && sameDepthKey(way.target, target)) return &way;
    }
    return nullptr;
}

// One DepthSurfaceView call while profiling: depthLeaseStart fills it before any of the call's own
// work, depthLeaseLocked under the surfaces mutex, the call its un-retire and mirror sync, and
// depthLeaseEnd classifies, counts and leaves the target's entry.
struct DepthLeaseUse {
    std::uint8_t ineligible = DepthLeaseEligible;
    Recorder* recorder = nullptr;
    bool found = false;
    bool evicted = false;
    DepthLeaseMark previous;
    std::array<GuestMemory::UnchangedQuery, 3> ranges{};
    DepthLeaseMark now;
    DepthLease stamped = DepthLease::WouldHit;
    bool htileWritten = false;
    std::uint64_t depthWrites = 0;
    bool synced = false;
    std::uint64_t ns = 0;
};

std::uint64_t depthLeaseNs(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

// Reads only: the memo key, the recorder under the GPU lock (eligibility requires it held) and, while
// the memo key and the finished serial held, one tracker lock for the stamps, taken without the drain
// (none when a range has no generation; not counted on [guestmem]'s tracker line), whose per-range
// answers name a failed range (unproven before stamped, depth before stencil before HTILE), and with
// every range unstamped, a host write that ended unstamped over a range's pages (a miss of its own:
// the drain would stamp that range) and else the HTILE's newer writes. The host writes queued
// meanwhile stay queued for the call's own first collect, so profiling moves neither the tracker's
// drain order nor its counted locks.
void depthLeaseStart(DepthLeaseUse& use, const Context& context, const DepthTarget& target, bool native) {
    const auto start = std::chrono::steady_clock::now();
    use.recorder = RecorderFor(context);
    if (target.samples != 1u || native) use.ineligible = DepthLeaseMultisampled;
    else if (use.recorder != nullptr && !GuestMemory::GpuMutex().HeldByThisThread()) use.ineligible = DepthLeaseUnlocked;
    else if (use.now.memo = GuestMemory::CurrentCollectMemoKey(); !use.now.memo.Valid()) use.ineligible = DepthLeaseNoMemoKey;
    if (use.ineligible == DepthLeaseEligible) {
        use.now.finished = use.recorder != nullptr ? use.recorder->FinishedSerial() : 0;
        use.now.writeGeneration = Recorder::WriteGeneration();
        auto& table = ThreadScratch<DepthLeaseTable, DepthLeaseTableTag>();
        const auto set = depthLeaseSet(target.address);
        if (const auto* entry = findDepthLease(table, set, use.recorder, context.device, target)) {
            use.found = true;
            use.previous = entry->mark;
            use.ranges = entry->ranges;
        } else {
            const auto fingerprint = depthLeaseFingerprint(use.recorder, context.device, target);
            for (auto& evicted : table.evicted[set]) {
                if (evicted != fingerprint) continue;
                evicted = 0;
                use.evicted = true;
                break;
            }
        }
    }
    if (use.found && use.now.memo == use.previous.memo && use.now.finished == use.previous.finished) {
        std::array<GuestMemory::UnchangedQuery, 3> queries{};
        // The plane of each query (0 depth, 1 stencil, 2 HTILE), in that order.
        std::array<std::size_t, 3> planes{};
        std::size_t count = 0;
        bool unproven = false;
        for (std::size_t plane = 0; plane < use.ranges.size(); ++plane) {
            const auto& range = use.ranges[plane];
            if (range.bytes == 0) continue;
            planes[count] = plane;
            queries[count++] = range;
            if (range.generation == 0) unproven = true;
        }
        std::array<GuestMemory::StampRow, 3> rows{};
        GuestMemory::StampsHoldReport report{std::span(rows.data(), count)};
        // A range with no proof (generation 0, or no longer covered by the tracker, as after an import
        // unwatched it) fails the test though no stamp moved it: it misses as unproven, named before a
        // stamp (and a failure with no range named: the tracker watches nothing).
        if (unproven) {
            use.stamped = DepthLease::Unproven;
        } else if (!GuestMemory::StampsHold(std::span(queries.data(), count), {}, false, nullptr, &report)) {
            use.stamped = DepthLease::Unproven;
            if (std::ranges::none_of(std::span(rows.data(), count), [](GuestMemory::StampRow row) { return row == GuestMemory::StampRow::Unproven; })) {
                for (std::size_t i = 0; i < count; ++i) {
                    if (rows[i] != GuestMemory::StampRow::Stamped) continue;
                    use.stamped = planes[i] == 0 ? DepthLease::StampedDepth : planes[i] == 1 ? DepthLease::StampedStencil : DepthLease::StampedHtile;
                    break;
                }
            }
        } else if (report.hostWritesOver) {
            use.stamped = DepthLease::HostWrites;
        } else if (use.recorder != nullptr && use.ranges[2].bytes != 0) {
            use.htileWritten = use.recorder->WrittenAfterNote(use.ranges[2].address, use.ranges[2].bytes, use.previous.writeNotes);
        }
    }
    use.ns += depthLeaseNs(start);
}

// The serials today's lookup reads, at the point it takes the surfaces mutex (the HTILE store-back
// before it moves none of them).
void depthLeaseLocked(DepthLeaseUse& use) {
    if (!use.found) return;
    use.now.surfaces = surfacesSerial.load(std::memory_order_acquire);
    use.now.planeBinds = planeBinds;
    use.now.clearedHtiles = clearedHtileSets;
    use.now.mirrorEpoch = mirrorEpoch;
}

// Where the call un-retires its surface: the serial moves for a surface found retired (counted in
// every call while profiling, eligible or not), and an eligible call compares it after its own move.
void depthLeaseUnretire(DepthLeaseUse& use, bool retired) {
    if (retired) ++depthUnretires;
    use.now.unretired = depthUnretires;
}

// Under the surfaces mutex, at the call's end: classifies against the target's entry, counts, and
// leaves the entry the state at this end.
void depthLeaseEnd(DepthLeaseUse& use, const Context& context, const DepthTarget& target, const DepthSurface& surface, bool created, bool storedBack, bool clearedTaken) {
    const auto start = std::chrono::steady_clock::now();
    auto& counters = depthLeaseCounters;
    countDepthLease(counters.calls);
    countDepthLease(counters.ineligible[use.ineligible]);
    if (use.ineligible != DepthLeaseEligible) {
        countDepthLeaseNs(use.ns + depthLeaseNs(start));
        return;
    }
    if (!use.found) {
        CountDepthLease({use.evicted ? DepthLease::Evicted : DepthLease::NoEntry, false}, 0);
    } else {
        use.now.surface = created ? nullptr : &surface;
        const auto work = static_cast<std::uint8_t>((storedBack ? DepthLeaseHtileStoreBack : 0) | (clearedTaken ? DepthLeaseClearedHtileTaken : 0) | (use.synced ? DepthLeaseMirrorSync : 0));
        CountDepthLease(ClassifyDepthLease(use.previous, use.now, use.stamped, use.htileWritten), work);
    }
    DepthLeaseMark mark;
    mark.surface = &surface;
    mark.memo = GuestMemory::CurrentCollectMemoKey();
    mark.surfaces = surfacesSerial.load(std::memory_order_acquire);
    mark.unretired = depthUnretires;
    mark.planeBinds = planeBinds;
    mark.clearedHtiles = clearedHtileSets;
    mark.mirrorEpoch = mirrorEpoch;
    mark.finished = use.recorder != nullptr ? use.recorder->FinishedSerial() : 0;
    mark.writeNotes = use.recorder != nullptr ? use.recorder->WriteNotes() : 0;
    mark.writeGeneration = Recorder::WriteGeneration();
    // takeClearedHtile took at most one covering entry; another left (inserted before this end, so the
    // insert serial will not move for it) is taken by the next call. The map is normally empty.
    if (target.htileAddress != 0) {
        const auto& cleared = clearedHtiles();
        mark.clearedHtileLeft = std::any_of(cleared.begin(), cleared.end(), [&](const auto& entry) { return HtileFillCovers(target.htileAddress, target.extent, entry.first, entry.second); });
    }
    const auto noted = surface.NotedMemory();
    std::array<GuestMemory::UnchangedQuery, 3> ranges{noted[0], noted[1], GuestMemory::UnchangedQuery{0, 0, 0}};
    if (target.htileAddress != 0) {
        std::uint64_t generation = 0;
        {
            std::lock_guard lock(htileReadsMutex());
            if (const auto found = htileReads().find(target.htileAddress); found != htileReads().end()) generation = found->second;
        }
        ranges[2] = {target.htileAddress, static_cast<std::size_t>(HtileMetadataBytes(target.extent)), generation};
    }
    auto& table = ThreadScratch<DepthLeaseTable, DepthLeaseTableTag>();
    const auto set = depthLeaseSet(target.address);
    auto* entry = findDepthLease(table, set, use.recorder, context.device, target);
    if (entry == nullptr) {
        entry = &table.sets[set][0];
        for (auto& way : table.sets[set]) {
            if (!way.valid) {
                entry = &way;
                break;
            }
            if (way.used < entry->used) entry = &way;
        }
        if (entry->valid) {
            countDepthLease(counters.evictions);
            table.evicted[set][table.nextEvicted[set]++ % DepthLeaseWays] = depthLeaseFingerprint(entry->recorder, entry->device, entry->target);
        }
        entry->valid = true;
        entry->recorder = use.recorder;
        entry->device = context.device;
        entry->target = target;
    }
    entry->mark = mark;
    entry->ranges = ranges;
    entry->used = ++table.tick;
    countDepthLeaseNs(use.ns + depthLeaseNs(start));
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11 D5, DepthLease.hpp): DepthSurfaceView
// leases. depthViewLeaseHit runs right after RoutesToNativeDepthOwner, before any of the call's own
// work; depthViewPoint reads =2's race evidence at the points of today's path; depthViewLeaseEnd, under
// the surfaces mutex at the call's end, renews a hit's lease, checks =2's prediction against today's
// path and makes (or refuses) the lease after a full path. Why a hit is exact is GPU-OWNED-PLAN 11.7;
// in short, a hit re-reads live every input of the steps it skips:
// - the HTILE store-back: on the not-pending route, no write was open or in flight over the HTILE at
//   the decision that made the lease, writes enter batches only by an append with a fresh global note
//   and leave only from the in-flight front, so a pending one now carries a note newer than the
//   lease's (WrittenAfterNote); the read proof is found live under its mutex and tested with
//   UnchangedSince's own predicate in the hit's tracker lock. On the pending route, no batch finished
//   (FinishedSerial) and no write over the HTILE was noted since, so PendingWritesOver returns the
//   writes the decision saw, the same fill word lands and clears nothing, and today writes the read
//   proof as a collect answers it: the hit writes memoCurrent. Accessible runs live (page protection
//   is not in the memo key);
// - takeClearedHtile: no covering entry, found without the erase under the surfaces mutex every
//   clearedHtiles writer takes;
// - the surface scan: the surfaces and released serials (each list change moves one) prove the same
//   members in the same order; `active` only falls, and only for native guard ranges (samples != 1);
//   so the scan returns the lease's surface, nothing is created and the owner is never read;
// - SyncFromMemory: SizeUnits would return early (UnitsSized), and the plane's not-written row from the
//   live oldest unit generation (every unitGeneration writer holds the surfaces mutex) is WrittenUnits'
//   test, so nothing changed and every unit gets TrackerGeneration's answer (current);
// - the skipped collects (store-back's, WrittenUnits', NoteMemory's): the lease's creation proved them
//   covered under its memo key, which still holds (coverage only grows under a key), so each would be
//   a memo hit answering hitGeneration: memoCurrent, read under the hit's draining tracker lock.
// When another thread moves the tracker between the hit's lock and today's last read, the hit's values
// are today's under the schedule in which that update landed after them: only ever lower, which only
// makes a later check fail (a miss, a sync, a retire through OverwrittenInMemory).
// Mirror tripwire (GPU-OWNED-PLAN 9.6, 11.13): this block copies the behaviour of, or reads the state
// written by, in this file DepthSurfaceView, storeBackClearedHtile, htileAfterPendingFills,
// takeClearedHtile, takePlane, releaseRetired, forgetHtileReads, NoteHtileDepthClear,
// NoteDepthMetadataFill, SizeUnits, WrittenUnits, SyncFromMemory, NoteStored, NoteMemory, Mirror,
// retire, retireStoringStale, FlushDepthPlanes, ClearDepthSurfaces, RetireDepthSurfaces,
// RetireUnmappedDepth_nid_no_patch and RoutesToNativeDepthOwner; in E/GuestMemory.cpp WrittenSince,
// UnchangedSince, rangeUnchanged, unchanged, stamp, hitGeneration, TrackerGeneration, lockTracker,
// CollectCovered, CurrentCollectMemoKey and verify/Accessible; in G/Recorder.cpp PendingWriteOverlaps,
// PendingWritesOver, WrittenAfterNote, FinishedSerial, WriteNotes and publishPendingWrites. An upstream
// change to any of them re-runs the =2 ctest rows and one in-game =2 run before the merged build is
// adopted. The writers the channels rest on, at 367000dc5 (GPU-OWNED-PLAN 11.18 risk 1): unitGeneration
// (SizeUnits; NoteStored from retireStoringStale and FlushDepthPlanes; SyncFromMemory); htileReads
// (storeBackClearedHtile; forgetHtileReads from NoteHtileDepthClear and NoteDepthMetadataFill);
// clearedHtiles (NoteHtileDepthClear's insert, takeClearedHtile's erase); the surface list
// (DepthSurfaceView's creation, releaseRetired, ClearDepthSurfaces); retired (retire, DepthSurfaceView);
// active (RetireUnmappedDepth_nid_no_patch only); recorder batches (an append, and pop_front under the
// GPU lock). A writer a merge adds breaks a channel silently until a =2 run.

struct DepthViewCounters {
    std::atomic<std::uint64_t> calls{0};
    std::array<std::atomic<std::uint64_t>, DepthViewIndex(DepthViewIneligible::Count)> ineligible{};
    std::atomic<std::uint64_t> hits{0};
    std::array<std::atomic<std::uint64_t>, DepthViewIndex(DepthViewRoute::Count)> hitRoutes{};
    std::array<std::atomic<std::uint64_t>, DepthViewIndex(HtileVerdictKind::Count)> verdicts{};
    std::atomic<std::uint64_t> created{0};
    std::atomic<std::uint64_t> renewed{0};
    std::array<std::atomic<std::uint64_t>, DepthViewIndex(DepthViewRefusal::Count)> refusals{};
    std::array<std::atomic<std::uint64_t>, DepthViewIndex(DepthViewMiss::Count)> misses{};
    std::atomic<std::uint64_t> evictions{0};
    std::atomic<std::uint64_t> predicted{0};
    std::atomic<std::uint64_t> verified{0};
    std::array<std::atomic<std::uint64_t>, 4> raced{};
};
// Relaxed and cumulative, counted only while the switch is set (DepthViewLeaseTotals).
constinit DepthViewCounters depthViewCounters;

void countView(std::atomic<std::uint64_t>& counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
}

// A lease (GPU-OWNED-PLAN 11.5): no generation of any kind. The key is D5-0's (sameDepthKey, under its
// field-count tripwire), with the recorder and the device; the set is D5-0's (depthLeaseSet).
struct DepthViewLeaseEntry {
    bool valid = false;
    bool leased = false;
    const Recorder* recorder = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    DepthTarget target{};
    std::uint64_t used = 0;
    // The memo key the creation proved every skipped collect's range covered under (renewals keep it).
    GuestMemory::CollectMemoKey memo;
    // The surface, and the serials it was found under (read before it is dereferenced).
    DepthSurface* surface = nullptr;
    std::uint64_t surfaces = 0;
    std::uint64_t released = 0;
    DepthViewRoute route = DepthViewRoute::NoHtile;
    // Recorder::FinishedSerial and WriteNotes, read at the decision (the creation's store-back, a
    // renewing hit's check), never at a call's end (GPU-OWNED-PLAN 11.2 row 22).
    std::uint64_t finished = 0;
    std::uint64_t notes = 0;
    // DescribeSurface(mirrorResource(target)).guestBytes (pure) and HtileMetadataBytes(extent).
    std::uint64_t planeBytes = 0;
    std::size_t htileBytes = 0;
};

struct DepthViewLeaseTable {
    std::array<std::array<DepthViewLeaseEntry, DepthLeaseWays>, DepthLeaseSets> sets{};
    // Per set, fingerprints of the last keys a set conflict evicted (0 empty), as D5-0's.
    std::array<std::array<std::uint64_t, DepthLeaseWays>, DepthLeaseSets> evicted{};
    std::array<std::uint8_t, DepthLeaseSets> nextEvicted{};
    std::uint64_t tick = 0;
};
struct DepthViewLeaseTableTag {};

DepthViewLeaseEntry* findDepthViewLease(DepthViewLeaseTable& table, std::size_t set, const Recorder* recorder, VkDevice device, const DepthTarget& target) {
    for (auto& way : table.sets[set]) {
        if (way.valid && way.recorder == recorder && way.device == device && sameDepthKey(way.target, target)) return &way;
    }
    return nullptr;
}

// A new entry: a free way, else the set's least recently used one (counted, its fingerprint kept for
// the "evicted" miss); the new key's own fingerprint leaves the evicted list (it has an entry again).
DepthViewLeaseEntry& insertDepthViewLease(DepthViewLeaseTable& table, std::size_t set, const Recorder* recorder, VkDevice device, const DepthTarget& target) {
    const auto fingerprint = depthLeaseFingerprint(recorder, device, target);
    for (auto& evicted : table.evicted[set]) {
        if (evicted == fingerprint) evicted = 0;
    }
    auto* entry = &table.sets[set][0];
    for (auto& way : table.sets[set]) {
        if (!way.valid) {
            entry = &way;
            break;
        }
        if (way.used < entry->used) entry = &way;
    }
    if (entry->valid) {
        countView(depthViewCounters.evictions);
        table.evicted[set][table.nextEvicted[set]++ % DepthLeaseWays] = depthLeaseFingerprint(entry->recorder, entry->device, entry->target);
    }
    *entry = DepthViewLeaseEntry{};
    entry->valid = true;
    entry->recorder = recorder;
    entry->device = device;
    entry->target = target;
    return *entry;
}

// takeClearedHtile's find, without its erase (under the surfaces mutex; the map is normally empty).
bool clearedHtileCovers(const DepthTarget& target) {
    if (target.htileAddress == 0) return false;
    const auto& cleared = clearedHtiles();
    return std::any_of(cleared.begin(), cleared.end(), [&](const auto& entry) { return HtileFillCovers(target.htileAddress, target.extent, entry.first, entry.second); });
}

enum class DepthViewPoint : std::uint8_t { StoreBack, Identity, Sync };

// =2: what the prediction read (GPU-OWNED-PLAN 11.9), in the call's stack frame.
struct DepthViewPrediction {
    DepthViewRoute route = DepthViewRoute::NoHtile;
    DepthSurface* surface = nullptr;
    std::uint64_t oldest = 0;
    std::uint64_t h = 0;
    std::uint64_t current = 0;
    std::uint64_t memoCurrent = 0;
    std::uint64_t surfaces = 0;
    std::uint64_t released = 0;
    GuestMemory::CollectMemoKey memo;
    bool accessible = false;
    std::uint64_t planeBytes = 0;
    std::size_t htileBytes = 0;
};

// One DepthSurfaceView call while the switch is set (in the call's stack frame, so no thread_local).
struct DepthViewLease {
    int mode = 0;
    // Eligible: depthViewLeaseEnd runs at the call's end.
    bool tracking = false;
    // =1: the lease answered (the call holds the surfaces mutex from the check on).
    bool hit = false;
    // =2: the prediction held; today's path runs with its points and is checked at the end.
    bool predicting = false;
    Recorder* recorder = nullptr;
    // The memo key, read before today's store-back.
    GuestMemory::CollectMemoKey memo;
    // Today's store-back (a full path).
    HtileVerdict verdict;
    // A hit: its route, the recorder's serials its check read, the plane's size and the generations
    // its tracker lock read.
    DepthViewRoute route = DepthViewRoute::NoHtile;
    std::uint64_t finished = 0;
    std::uint64_t notes = 0;
    std::uint64_t planeBytes = 0;
    std::uint64_t current = 0;
    std::uint64_t memoCurrent = 0;
    // Today's path: the surface found retired, and whether SyncFromMemory synced (depthWrites moved).
    bool retiredBefore = false;
    std::uint64_t writes = 0;
    bool synced = false;
    DepthViewPrediction prediction;
    DepthViewStoreBackEvidence storeBack;
    // At the identity point, then the sync point.
    std::array<DepthViewEvidence, 2> evidence{};

    HtileVerdict* VerdictOut() {
        return tracking && !hit ? &verdict : nullptr;
    }
};

// The check (GPU-OWNED-PLAN 11.6 L1-L13). Any failure is a miss counted by its reason that releases what
// the check took and changed nothing; today's path then runs whole. =1: a hit writes the pending
// route's read proof (today's store-back write), moves the surfaces mutex into `held` and returns the
// surface. =2: the same check without the drain (a queued host write makes it a miss) and without any
// write, which keeps what it read and returns null. Lock order: the GPU lock (held), the surfaces
// mutex, htileReads', the tracker's: every edge exists in today's path (NoteHtileDepthClear takes the
// first two, storeBackClearedHtile's not-pending test the last two, SyncFromMemory under the surfaces
// mutex the tracker's); DepthSurfaceView's own GPU lock after a hit re-enters it, adding no edge.
DepthSurface* depthViewLeaseHit(DepthViewLease& view, const Context& context, const DepthTarget& target, bool native, std::unique_lock<std::recursive_mutex>& held) {
    auto& counters = depthViewCounters;
    view.mode = DepthViewLeases();
    countView(counters.calls);
    const auto ineligible = [&](DepthViewIneligible reason) -> DepthSurface* {
        countView(counters.ineligible[DepthViewIndex(reason)]);
        return nullptr;
    };
    // Eligibility (11.4). Native excludes samples != 1 too.
    if (native) return ineligible(DepthViewIneligible::Native);
    if (!mirrorable(target)) return ineligible(DepthViewIneligible::NotMirrorable);
    auto* recorder = RecorderFor(context);
    if (recorder == nullptr || !GuestMemory::GpuMutex().HeldByThisThread()) return ineligible(DepthViewIneligible::Unlocked);
    const auto memo = GuestMemory::CurrentCollectMemoKey();
    if (!memo.Valid()) return ineligible(DepthViewIneligible::NoMemoKey);
    countView(counters.ineligible[DepthViewIndex(DepthViewIneligible::Eligible)]);
    view.tracking = true;
    view.recorder = recorder;
    view.memo = memo;
    const auto missed = [&](DepthViewMiss miss) -> DepthSurface* {
        countView(counters.misses[DepthViewIndex(miss)]);
        return nullptr;
    };
    // L1. The table (only read here; a full path's end writes it).
    auto& table = ThreadScratch<DepthViewLeaseTable, DepthViewLeaseTableTag>();
    const auto set = depthLeaseSet(target.address);
    const auto* entry = findDepthViewLease(table, set, recorder, context.device, target);
    if (entry == nullptr) {
        const auto fingerprint = depthLeaseFingerprint(recorder, context.device, target);
        const bool evicted = std::find(table.evicted[set].begin(), table.evicted[set].end(), fingerprint) != table.evicted[set].end();
        return missed(evicted ? DepthViewMiss::Evicted : DepthViewMiss::NoLease);
    }
    if (!entry->leased) return missed(DepthViewMiss::NoLease);
    // L2. The memo key, part by part.
    if (entry->memo.epoch != memo.epoch) return missed(DepthViewMiss::Epoch);
    if (entry->memo.mapping != memo.mapping) return missed(DepthViewMiss::Allocations);
    if (entry->memo.unwatched != memo.unwatched) return missed(DepthViewMiss::Unwatch);
    // L3. The recorder, which no other thread moves while this one holds the GPU lock. Pending: no batch
    // finished and no write over the HTILE noted since the decision (the pending fills switch is read
    // once and a pending lease is made only with it on). Not pending: no such write.
    const auto route = entry->route;
    const bool htile = route == DepthViewRoute::NotPending || route == DepthViewRoute::Pending;
    if (route == DepthViewRoute::Pending && (!pendingHtileFills() || recorder->FinishedSerial() != entry->finished)) return missed(DepthViewMiss::Reaped);
    if (htile && recorder->WrittenAfterNote(target.htileAddress, entry->htileBytes, entry->notes)) return missed(DepthViewMiss::HtileNote);
    view.finished = recorder->FinishedSerial();
    view.notes = recorder->WriteNotes();
    // L4. The HTILE's page protection, live (not in the memo key).
    bool accessible = false;
    if (route != DepthViewRoute::NoHtile) {
        accessible = GuestMemory::Accessible(reinterpret_cast<const void*>(target.htileAddress), entry->htileBytes, true);
        if (accessible != (route != DepthViewRoute::NotAccessible)) return missed(DepthViewMiss::Accessibility);
    }
    // L5-L6. Under the surfaces mutex (today takes it after the GPU lock, as here): the serials, before
    // the surface is dereferenced.
    std::unique_lock<std::recursive_mutex> lock(surfacesMutex());
    if (DepthSurfacesSerial() != entry->surfaces || DepthSurfacesReleased() != entry->released) return missed(DepthViewMiss::Surfaces);
    auto* surface = entry->surface;
    // L7-L8.
    if (!surface->active->load(std::memory_order_acquire) || surface->retired) return missed(DepthViewMiss::Retired);
    if (clearedHtileCovers(target)) return missed(DepthViewMiss::ClearedHtile);
    // L9. The plane's units: sized, and the oldest generation live (the refutation of a stored one:
    // GPU-OWNED-PLAN 11.2 row 1); 0 proves nothing.
    if (!surface->UnitsSized(entry->planeBytes)) return missed(DepthViewMiss::UnitsResized);
    const auto oldest = surface->OldestUnitGeneration();
    if (oldest == 0) return missed(DepthViewMiss::NoPlaneProof);
    // L10. Not pending: the HTILE's read proof, found as today finds it (no insert), held through L11.
    std::unique_lock<std::mutex> readsLock;
    std::uint64_t h = 0;
    if (route == DepthViewRoute::NotPending) {
        readsLock = std::unique_lock<std::mutex>(htileReadsMutex());
        const auto found = htileReads().find(target.htileAddress);
        if (found == htileReads().end() || found->second == 0) return missed(DepthViewMiss::HtileReadProofMoved);
        h = found->second;
    }
    // L11. One tracker lock (draining at =1, where today's first collect would drain): the HTILE's
    // unchanged row (not pending: UnchangedSince's predicate), the plane's not-written row (WrittenUnits'
    // predicate), and the generations TrackerGeneration and a memo-hit collect answer under it.
    const bool apply = view.mode == 1;
    const std::size_t htileRows = route == DepthViewRoute::NotPending ? 1 : 0;
    const GuestMemory::UnchangedQuery htileRow{target.htileAddress, entry->htileBytes, h};
    const GuestMemory::UnchangedQuery planeRow{target.address, static_cast<std::size_t>(entry->planeBytes), oldest};
    std::array<GuestMemory::StampRow, 2> rows{};
    GuestMemory::StampsHoldReport report{std::span(rows.data(), htileRows + 1)};
    std::uint64_t current = 0;
    std::uint64_t memoCurrent = 0;
    if (!GuestMemory::StampsHold(std::span(&htileRow, htileRows), {}, std::span(&planeRow, 1), apply, &current, &memoCurrent, &report)) {
        const auto plane = rows[htileRows];
        if (plane != GuestMemory::StampRow::Held) return missed(plane == GuestMemory::StampRow::Stamped ? DepthViewMiss::PlaneStamped : DepthViewMiss::PlaneUnproven);
        return missed(rows[0] == GuestMemory::StampRow::Stamped ? DepthViewMiss::HtileStamped : DepthViewMiss::HtileUnproven);
    }
    // =2 took the lock without the drain: any host write queued then would move the generation at
    // today's first tracker lock.
    if (!apply && report.hostWritesPending) return missed(DepthViewMiss::HostWritesPending);
    view.route = route;
    view.planeBytes = entry->planeBytes;
    view.current = current;
    view.memoCurrent = memoCurrent;
    if (!apply) {
        // L12. The prediction; every lock is released on return, and today's path runs.
        auto& prediction = view.prediction;
        prediction.route = route;
        prediction.surface = surface;
        prediction.oldest = oldest;
        prediction.h = h;
        prediction.current = current;
        prediction.memoCurrent = memoCurrent;
        prediction.surfaces = entry->surfaces;
        prediction.released = entry->released;
        prediction.memo = memo;
        prediction.accessible = accessible;
        prediction.planeBytes = entry->planeBytes;
        prediction.htileBytes = entry->htileBytes;
        view.predicting = true;
        countView(counters.predicted);
        return nullptr;
    }
    // L13. The pending route's read proof, as today's store-back writes it (DS's htileReads write in
    // storeBackClearedHtile's covered branch), with the memo-hit generation its collect would answer.
    if (route == DepthViewRoute::Pending) {
        std::lock_guard reads(htileReadsMutex());
        htileReads()[target.htileAddress] = memoCurrent;
    }
    view.hit = true;
    countView(counters.hits);
    countView(counters.hitRoutes[DepthViewIndex(route)]);
    held = std::move(lock);
    return surface;
}

// =2: the evidence at one point of today's path (DepthLease.hpp). The store-back's point, before it,
// reads the memo key only (its other evidence is its verdict's); the identity and sync points hold
// the surfaces mutex.
void depthViewPoint(DepthViewLease& view, DepthViewPoint point, const DepthTarget& target) {
    const auto& prediction = view.prediction;
    const bool memoMoved = GuestMemory::CurrentCollectMemoKey() != prediction.memo;
    if (point == DepthViewPoint::StoreBack) {
        view.storeBack.memoMoved = memoMoved;
        return;
    }
    auto& evidence = view.evidence[point == DepthViewPoint::Identity ? 0 : 1];
    evidence.memoMoved = memoMoved;
    evidence.surfacesMoved = DepthSurfacesSerial() != prediction.surfaces || DepthSurfacesReleased() != prediction.released;
    if (!evidence.surfacesMoved) {
        const auto& surface = *prediction.surface;
        evidence.retired = !surface.active->load(std::memory_order_acquire) || surface.retired;
        evidence.oldestMoved = !surface.UnitsSized(prediction.planeBytes) || surface.OldestUnitGeneration() != prediction.oldest;
    }
    evidence.clearedHtile = clearedHtileCovers(target);
    evidence.clockMoved = GuestMemory::TrackerGenerationNoDrain() > prediction.current;
}

// =2, at the call's end under the surfaces mutex: today's path against the prediction
// (CheckDepthViewPrediction). Raced calls are counted (the first 8 printed); a disagreement no evidence
// explains prints [depth-lease] and aborts.
void verifyDepthViewPrediction(DepthViewLease& view, const DepthTarget& target, const DepthSurface& surface, bool created, VkImageAspectFlags htileCleared, bool cleared) {
    const auto& prediction = view.prediction;
    const auto& verdict = view.verdict;
    DepthViewFacts facts;
    facts.route = prediction.route;
    facts.stencil = target.stencilAddress != 0;
    facts.h = prediction.h;
    facts.current0 = prediction.current;
    facts.memoCurrent0 = prediction.memoCurrent;
    facts.verdict = verdict.kind;
    facts.htileCleared = static_cast<std::uint32_t>(htileCleared);
    facts.storeBack = view.storeBack;
    facts.storeBack.accessibleDiffers = verdict.accessibleRead && verdict.accessible != prediction.accessible;
    facts.storeBack.foundDiffers = prediction.route == DepthViewRoute::NotPending && verdict.lookedUp && (!verdict.found || verdict.foundValue != prediction.h);
    facts.storeBack.collectedAbove = verdict.collectedRead && verdict.collected > prediction.memoCurrent;
    facts.cleared = cleared;
    facts.created = created;
    facts.sameSurface = &surface == prediction.surface;
    facts.retiredBefore = view.retiredBefore;
    facts.identity = view.evidence[0];
    facts.synced = view.synced;
    facts.sync = view.evidence[1];
    // A sync (already a disagreement): whether the plane itself was stamped since the prediction, the
    // only update besides the sync point's memo key, serials and units that makes WrittenUnits change a
    // unit (the r73 review: a clock move elsewhere excuses nothing). The prediction's plane row, from
    // its oldest, in one tracker lock without the drain (surfaces -> tracker, SyncFromMemory's order;
    // a report's plain lock, not counted on [guestmem]); a stamp before today's WrittenSince is still
    // there, as stamps only rise. A row with no proof left counts as stamped.
    if (facts.synced) {
        const GuestMemory::UnchangedQuery planeRow{target.address, static_cast<std::size_t>(prediction.planeBytes), prediction.oldest};
        auto row = GuestMemory::StampRow::Held;
        GuestMemory::StampsHoldReport report{std::span(&row, 1)};
        static_cast<void>(GuestMemory::StampsHold({}, {}, std::span(&planeRow, 1), false, nullptr, nullptr, &report));
        facts.planeStamped = row != GuestMemory::StampRow::Held;
    }
    if (prediction.route == DepthViewRoute::NotPending || prediction.route == DepthViewRoute::Pending) {
        std::lock_guard reads(htileReadsMutex());
        if (const auto found = htileReads().find(target.htileAddress); found != htileReads().end()) {
            facts.htileReadFound = true;
            facts.htileRead = found->second;
        }
    }
    if (!surface.unitGeneration.empty()) {
        const auto [low, high] = std::minmax_element(surface.unitGeneration.begin(), surface.unitGeneration.end());
        facts.unitsLow = *low;
        facts.unitsHigh = *high;
    }
    const auto noted = surface.NotedMemory();
    facts.depthMemory = noted[0].generation;
    facts.stencilMemory = noted[1].generation;
    const auto check = CheckDepthViewPrediction(facts);
    auto& counters = depthViewCounters;
    if (check.part == DepthViewPart::None) {
        countView(counters.verified);
        return;
    }
    const auto evidence = [](const DepthViewEvidence& at) {
        char text[160];
        std::snprintf(text, sizeof(text), "memo key %d, surfaces %d, retired %d, units %d, cleared HTILE %d, clock %d", at.memoMoved ? 1 : 0, at.surfacesMoved ? 1 : 0, at.retired ? 1 : 0, at.oldestMoved ? 1 : 0, at.clearedHtile ? 1 : 0, at.clockMoved ? 1 : 0);
        return std::string(text);
    };
    char text[1536];
    std::snprintf(text, sizeof(text), "target 0x%llx (stencil 0x%llx, HTILE 0x%llx): the lease predicted a hit (route %s, oldest %llu, h %llu, current %llu, memo-current %llu) but today's %s part disagrees: %s (today: store-back %s returned 0x%x, cleared HTILE taken %d, created %d, the predicted surface %d, retired %d, synced %d; HTILE read proof %s%llu, units %llu..%llu, depth memory %llu, stencil memory %llu; evidence: store-back memo key %d, Accessible %d, read proof %d, collect above %d; identity: %s; sync: %s, plane stamped %d)",
                  static_cast<unsigned long long>(target.address), static_cast<unsigned long long>(target.stencilAddress), static_cast<unsigned long long>(target.htileAddress), DepthViewRouteName(prediction.route), static_cast<unsigned long long>(prediction.oldest), static_cast<unsigned long long>(prediction.h), static_cast<unsigned long long>(prediction.current), static_cast<unsigned long long>(prediction.memoCurrent),
                  DepthViewPartName(check.part), check.reason, HtileVerdictName(verdict.kind), static_cast<unsigned>(htileCleared), cleared ? 1 : 0, created ? 1 : 0, facts.sameSurface ? 1 : 0, view.retiredBefore ? 1 : 0, view.synced ? 1 : 0, facts.htileReadFound ? "" : "absent, ", static_cast<unsigned long long>(facts.htileRead),
                  static_cast<unsigned long long>(facts.unitsLow), static_cast<unsigned long long>(facts.unitsHigh), static_cast<unsigned long long>(facts.depthMemory), static_cast<unsigned long long>(facts.stencilMemory), facts.storeBack.memoMoved ? 1 : 0, facts.storeBack.accessibleDiffers ? 1 : 0, facts.storeBack.foundDiffers ? 1 : 0, facts.storeBack.collectedAbove ? 1 : 0,
                  evidence(facts.identity).c_str(), evidence(facts.sync).c_str(), facts.planeStamped ? 1 : 0);
    if (check.raced) {
        countView(counters.raced[DepthViewIndex(check.part) - 1]);
        static std::atomic<int> reports{0};
        if (reports.fetch_add(1, std::memory_order_relaxed) < 8) std::fprintf(stderr, "[depth-lease] %s, after an update that landed after the prediction (not a disagreement)\n", text);
        return;
    }
    std::fprintf(stderr, "[depth-lease] APS5_EXPERIMENT_DEPTH_LEASES=2: %s\n", text);
    std::fflush(stderr);
    std::abort();
}

// A full path's lease (GPU-OWNED-PLAN 11.8), or its refusal by the first reason that held (an entry
// of the key then stops being a lease). The entry is found again by its key: no pointer is kept
// across the call. =2 makes leases here too, so its predictions read leases full paths made.
void createDepthViewLease(DepthViewLease& view, const Context& context, const DepthTarget& target, DepthSurface& surface, bool created, VkImageAspectFlags htileCleared, bool cleared) {
    auto& table = ThreadScratch<DepthViewLeaseTable, DepthViewLeaseTableTag>();
    const auto set = depthLeaseSet(target.address);
    auto* entry = findDepthViewLease(table, set, view.recorder, context.device, target);
    const auto refuse = [&](DepthViewRefusal refusal) {
        countView(depthViewCounters.refusals[DepthViewIndex(refusal)]);
        if (entry != nullptr) entry->leased = false;
    };
    if (created) return refuse(DepthViewRefusal::Created);
    if (htileCleared != 0 || cleared) return refuse(DepthViewRefusal::StoredBackOrClearedTaken);
    auto route = DepthViewRoute::NoHtile;
    switch (view.verdict.kind) {
        case HtileVerdictKind::NoHtile: route = DepthViewRoute::NoHtile; break;
        case HtileVerdictKind::NotAccessible: route = DepthViewRoute::NotAccessible; break;
        // A CPU read that cleared nothing wrote the read proof the next call's test finds.
        case HtileVerdictKind::NotPendingProved:
        case HtileVerdictKind::NotPendingRead: route = DepthViewRoute::NotPending; break;
        case HtileVerdictKind::PendingCovered: route = DepthViewRoute::Pending; break;
        case HtileVerdictKind::PendingNotCovering: return refuse(DepthViewRefusal::PendingNotCovering);
        case HtileVerdictKind::PendingFillsOff: return refuse(DepthViewRefusal::PendingFillsOff);
        default: return refuse(DepthViewRefusal::StoredBackOrClearedTaken);
    }
    // The memo key read before the store-back still holds, and the recorder is the key's.
    const auto memo = GuestMemory::CurrentCollectMemoKey();
    if (!memo.Valid() || memo != view.memo) return refuse(DepthViewRefusal::MemoKeyMoved);
    if (RecorderFor(context) != view.recorder) return refuse(DepthViewRefusal::RecorderChanged);
    // Every collect a hit skips is covered under that key: the HTILE's (store-back), the plane's
    // (WrittenUnits) and the depth and stencil slices' (NoteMemory).
    const auto planeBytes = static_cast<std::uint64_t>(DescribeSurface(mirrorResource(target)).guestBytes);
    const auto htileBytes = static_cast<std::size_t>(HtileMetadataBytes(target.extent));
    const auto noted = surface.NotedMemory();
    const bool htile = route == DepthViewRoute::NotPending || route == DepthViewRoute::Pending;
    if ((htile && !GuestMemory::CollectCovered(target.htileAddress, htileBytes)) || !GuestMemory::CollectCovered(target.address, static_cast<std::size_t>(planeBytes)) || !GuestMemory::CollectCovered(noted[0].address, noted[0].bytes) || (noted[1].bytes != 0 && !GuestMemory::CollectCovered(noted[1].address, noted[1].bytes))) return refuse(DepthViewRefusal::NotCovered);
    if (entry == nullptr) entry = &insertDepthViewLease(table, set, view.recorder, context.device, target);
    entry->leased = true;
    entry->memo = memo;
    entry->surface = &surface;
    entry->surfaces = DepthSurfacesSerial();
    entry->released = DepthSurfacesReleased();
    entry->route = route;
    entry->finished = view.verdict.finished;
    entry->notes = view.verdict.notes;
    entry->planeBytes = planeBytes;
    entry->htileBytes = htileBytes;
    entry->used = ++table.tick;
    countView(depthViewCounters.created);
}

// At the end of every eligible call, under the surfaces mutex. A hit renews its lease with the
// recorder's serials its check read (memo key, route, surface and serials are unchanged: the surfaces
// mutex has been held since its check). A full path counts its store-back verdict, is checked
// against =2's prediction, then makes or refuses the lease.
void depthViewLeaseEnd(DepthViewLease& view, const Context& context, const DepthTarget& target, DepthSurface& surface, bool created, VkImageAspectFlags htileCleared, bool cleared) {
    auto& counters = depthViewCounters;
    if (view.hit) {
        auto& table = ThreadScratch<DepthViewLeaseTable, DepthViewLeaseTableTag>();
        if (auto* entry = findDepthViewLease(table, depthLeaseSet(target.address), view.recorder, context.device, target); entry != nullptr && entry->leased) {
            entry->finished = view.finished;
            entry->notes = view.notes;
            entry->used = ++table.tick;
            countView(counters.renewed);
        }
        return;
    }
    countView(counters.verdicts[DepthViewIndex(view.verdict.kind)]);
    if (view.predicting) verifyDepthViewPrediction(view, target, surface, created, htileCleared, cleared);
    createDepthViewLease(view, context, target, surface, created, htileCleared, cleared);
}
}

bool DepthLeaseSetInRecipe(bool inside) {
    const bool previous = depthLeaseInRecipe;
    depthLeaseInRecipe = inside;
    return previous;
}

DepthLeaseVerdict ClassifyDepthLease(const DepthLeaseMark& previous, const DepthLeaseMark& now, DepthLease stamped, bool htileWritten) {
    DepthLeaseVerdict verdict;
    verdict.writeGenerationMoved = now.writeGeneration != previous.writeGeneration;
    auto& outcome = verdict.outcome;
    if (now.surface == nullptr || now.surface != previous.surface) outcome = DepthLease::Identity;
    else if (now.memo.epoch != previous.memo.epoch) outcome = DepthLease::Epoch;
    else if (now.memo.mapping != previous.memo.mapping) outcome = DepthLease::Allocations;
    else if (now.memo.unwatched != previous.memo.unwatched) outcome = DepthLease::Unwatch;
    else if (now.surfaces != previous.surfaces) outcome = DepthLease::Surfaces;
    else if (now.unretired != previous.unretired) outcome = DepthLease::Unretired;
    else if (now.planeBinds != previous.planeBinds) outcome = DepthLease::PlaneBinds;
    else if (now.clearedHtiles != previous.clearedHtiles || previous.clearedHtileLeft) outcome = DepthLease::ClearedHtile;
    else if (now.mirrorEpoch != previous.mirrorEpoch) outcome = DepthLease::MirrorEpoch;
    else if (now.finished != previous.finished) outcome = DepthLease::Reaped;
    else if (stamped != DepthLease::WouldHit) outcome = stamped;
    else if (htileWritten) outcome = DepthLease::HtileNote;
    return verdict;
}

void CountDepthLease(const DepthLeaseVerdict& verdict, std::uint8_t work) {
    auto& counters = depthLeaseCounters;
    countDepthLease(counters.outcomes[static_cast<std::size_t>(verdict.outcome)]);
    if (verdict.outcome != DepthLease::WouldHit) {
        if (verdict.writeGenerationMoved) countDepthLease(counters.missesWriteGeneration);
        return;
    }
    if (verdict.writeGenerationMoved) countDepthLease(counters.hitsWriteGeneration);
    for (std::size_t bit = 0; bit < counters.hitWork.size(); ++bit) {
        if ((work & (1u << bit)) != 0) countDepthLease(counters.hitWork[bit]);
    }
}

DepthLeaseCounts DepthLeaseTotals() {
    const auto& counters = depthLeaseCounters;
    const auto load = [](const std::atomic<std::uint64_t>& counter) { return counter.load(std::memory_order_relaxed); };
    DepthLeaseCounts totals;
    totals.calls = load(counters.calls);
    for (std::size_t i = 0; i < DepthLeaseIneligibleCount; ++i) totals.ineligible[i] = load(counters.ineligible[i]);
    for (std::size_t i = 0; i < static_cast<std::size_t>(DepthLease::Count); ++i) totals.outcomes[i] = load(counters.outcomes[i]);
    totals.hitsWriteGeneration = load(counters.hitsWriteGeneration);
    totals.missesWriteGeneration = load(counters.missesWriteGeneration);
    for (std::size_t i = 0; i < 3; ++i) totals.hitWork[i] = load(counters.hitWork[i]);
    totals.evictions = load(counters.evictions);
    totals.counterNs = load(counters.counterNs);
    totals.recipeCounterNs = load(counters.recipeCounterNs);
    return totals;
}

const char* DepthLeaseName(DepthLease lease) {
    constexpr std::array<const char*, static_cast<std::size_t>(DepthLease::Count)> names{"would-hit", "no entry", "evicted by a set conflict", "identity", "memo key moved: epoch", "allocations", "unwatch", "surfaces serial", "un-retired", "plane binds", "cleared HTILE set or left", "mirror epoch", "reaped", "no proof (unwatched or generation 0)", "stamped: depth", "stencil", "HTILE", "host writes pending", "HTILE note"};
    const auto index = static_cast<std::size_t>(lease);
    return index < names.size() ? names[index] : "?";
}

std::string DepthLeaseReport() {
    static DepthLeaseCounts reported;
    const auto now = DepthLeaseTotals();
    const auto delta = [](std::uint64_t value, std::uint64_t before) { return static_cast<unsigned long long>(value - before); };
    std::string line;
    char text[512];
    std::snprintf(text, sizeof(text), "; depth view leases (D5-0): eligible %llu of %llu calls (ineligible: multisampled or native owner %llu, unlocked %llu, no memo key %llu), would-hit %llu (writeGeneration moved in %llu; today's call still did work: HTILE stored back %llu, cleared HTILE taken %llu, mirror synced %llu), misses by reason:",
                  delta(now.ineligible[DepthLeaseEligible], reported.ineligible[DepthLeaseEligible]), delta(now.calls, reported.calls), delta(now.ineligible[DepthLeaseMultisampled], reported.ineligible[DepthLeaseMultisampled]), delta(now.ineligible[DepthLeaseUnlocked], reported.ineligible[DepthLeaseUnlocked]),
                  delta(now.ineligible[DepthLeaseNoMemoKey], reported.ineligible[DepthLeaseNoMemoKey]), delta(now.outcomes[0], reported.outcomes[0]), delta(now.hitsWriteGeneration, reported.hitsWriteGeneration), delta(now.hitWork[0], reported.hitWork[0]), delta(now.hitWork[1], reported.hitWork[1]), delta(now.hitWork[2], reported.hitWork[2]));
    line += text;
    for (std::size_t i = 1; i < static_cast<std::size_t>(DepthLease::Count); ++i) {
        std::snprintf(text, sizeof(text), "%s %s %llu", i == 1 ? "" : ",", DepthLeaseName(static_cast<DepthLease>(i)), delta(now.outcomes[i], reported.outcomes[i]));
        line += text;
    }
    std::snprintf(text, sizeof(text), "; writeGeneration moved in %llu of the classified misses; table evictions %llu; counter time %.1f ms in the [draws] prepare phase (Draw) and %.1f ms in readTarget (DrawWithRecipe)", delta(now.missesWriteGeneration, reported.missesWriteGeneration), delta(now.evictions, reported.evictions), static_cast<double>(now.counterNs - reported.counterNs) / 1e6,
                  static_cast<double>(now.recipeCounterNs - reported.recipeCounterNs) / 1e6);
    line += text;
    reported = now;
    return line;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.3): read once, as TargetLeases.
int DepthViewLeases() {
    static const int mode = [] {
        const char* text = std::getenv("APS5_EXPERIMENT_DEPTH_LEASES");
        if (text == nullptr) return 0;
        const auto value = std::strtol(text, nullptr, 10);
        return value == 0 ? 0 : value == 2 ? 2 : 1;
    }();
    return mode;
}

DepthViewLeaseCounts DepthViewLeaseTotals() {
    const auto& counters = depthViewCounters;
    const auto load = [](const std::atomic<std::uint64_t>& counter) { return counter.load(std::memory_order_relaxed); };
    DepthViewLeaseCounts totals;
    totals.calls = load(counters.calls);
    for (std::size_t i = 0; i < counters.ineligible.size(); ++i) totals.ineligible[i] = load(counters.ineligible[i]);
    totals.hits = load(counters.hits);
    for (std::size_t i = 0; i < counters.hitRoutes.size(); ++i) totals.hitRoutes[i] = load(counters.hitRoutes[i]);
    for (std::size_t i = 0; i < counters.verdicts.size(); ++i) totals.verdicts[i] = load(counters.verdicts[i]);
    totals.created = load(counters.created);
    totals.renewed = load(counters.renewed);
    for (std::size_t i = 0; i < counters.refusals.size(); ++i) totals.refusals[i] = load(counters.refusals[i]);
    for (std::size_t i = 0; i < counters.misses.size(); ++i) totals.misses[i] = load(counters.misses[i]);
    totals.evictions = load(counters.evictions);
    totals.predicted = load(counters.predicted);
    totals.verified = load(counters.verified);
    for (std::size_t i = 0; i < counters.raced.size(); ++i) totals.raced[i] = load(counters.raced[i]);
    return totals;
}

const char* DepthViewRouteName(DepthViewRoute route) {
    constexpr std::array<const char*, DepthViewIndex(DepthViewRoute::Count)> names{"no HTILE", "not accessible", "not pending", "pending"};
    const auto index = DepthViewIndex(route);
    return index < names.size() ? names[index] : "?";
}

const char* HtileVerdictName(HtileVerdictKind kind) {
    constexpr std::array<const char*, DepthViewIndex(HtileVerdictKind::Count)> names{"no HTILE", "not accessible", "not pending proved", "not pending CPU read", "pending covered", "pending not covering", "pending fills off", "stored back"};
    const auto index = DepthViewIndex(kind);
    return index < names.size() ? names[index] : "?";
}

const char* DepthViewMissName(DepthViewMiss miss) {
    constexpr std::array<const char*, DepthViewIndex(DepthViewMiss::Count)> names{"no lease", "evicted", "memo key: epoch", "allocations", "unwatch", "reaped", "HTILE note", "accessibility", "surfaces", "retired", "cleared HTILE", "units resized", "no plane proof", "HTILE read proof moved", "plane: unproven", "stamped", "HTILE: unproven", "stamped", "host writes pending"};
    const auto index = DepthViewIndex(miss);
    return index < names.size() ? names[index] : "?";
}

const char* DepthViewRefusalName(DepthViewRefusal refusal) {
    constexpr std::array<const char*, DepthViewIndex(DepthViewRefusal::Count)> names{"created", "HTILE stored back or cleared taken", "pending not covering", "pending fills off", "memo key moved", "recorder changed", "not covered"};
    const auto index = DepthViewIndex(refusal);
    return index < names.size() ? names[index] : "?";
}

const char* DepthViewPartName(DepthViewPart part) {
    constexpr std::array<const char*, DepthViewIndex(DepthViewPart::Count)> names{"none", "store-back", "identity", "sync", "refresh"};
    const auto index = DepthViewIndex(part);
    return index < names.size() ? names[index] : "?";
}

std::string DepthViewLeaseReport() {
    static DepthViewLeaseCounts reported;
    const auto now = DepthViewLeaseTotals();
    const auto delta = [](std::uint64_t value, std::uint64_t before) { return static_cast<unsigned long long>(value - before); };
    const auto route = [&](DepthViewRoute at) { return delta(now.hitRoutes[DepthViewIndex(at)], reported.hitRoutes[DepthViewIndex(at)]); };
    const auto ineligible = [&](DepthViewIneligible at) { return delta(now.ineligible[DepthViewIndex(at)], reported.ineligible[DepthViewIndex(at)]); };
    std::string line;
    char text[512];
    std::snprintf(text, sizeof(text), "; D5 (APS5_EXPERIMENT_DEPTH_LEASES=%d): eligible %llu of %llu (ineligible: native or multisampled %llu, not mirrorable %llu, no recorder or unlocked %llu, no memo key %llu); hits %llu (pending %llu, not pending %llu, no HTILE %llu, not accessible %llu); full-path verdicts:", DepthViewLeases(),
                  ineligible(DepthViewIneligible::Eligible), delta(now.calls, reported.calls), ineligible(DepthViewIneligible::Native), ineligible(DepthViewIneligible::NotMirrorable), ineligible(DepthViewIneligible::Unlocked), ineligible(DepthViewIneligible::NoMemoKey),
                  delta(now.hits, reported.hits), route(DepthViewRoute::Pending), route(DepthViewRoute::NotPending), route(DepthViewRoute::NoHtile), route(DepthViewRoute::NotAccessible));
    line += text;
    for (std::size_t i = 0; i < DepthViewIndex(HtileVerdictKind::Count); ++i) {
        std::snprintf(text, sizeof(text), "%s %s %llu", i == 0 ? "" : ",", HtileVerdictName(static_cast<HtileVerdictKind>(i)), delta(now.verdicts[i], reported.verdicts[i]));
        line += text;
    }
    std::snprintf(text, sizeof(text), "; created %llu, renewed %llu, refused:", delta(now.created, reported.created), delta(now.renewed, reported.renewed));
    line += text;
    for (std::size_t i = 0; i < DepthViewIndex(DepthViewRefusal::Count); ++i) {
        std::snprintf(text, sizeof(text), "%s %s %llu", i == 0 ? "" : ",", DepthViewRefusalName(static_cast<DepthViewRefusal>(i)), delta(now.refusals[i], reported.refusals[i]));
        line += text;
    }
    line += "; misses:";
    for (std::size_t i = 0; i < DepthViewIndex(DepthViewMiss::Count); ++i) {
        std::snprintf(text, sizeof(text), "%s %s %llu", i == 0 ? "" : ",", DepthViewMissName(static_cast<DepthViewMiss>(i)), delta(now.misses[i], reported.misses[i]));
        line += text;
    }
    std::uint64_t raced = 0;
    std::uint64_t racedBefore = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        raced += now.raced[i];
        racedBefore += reported.raced[i];
    }
    std::snprintf(text, sizeof(text), "; table evictions %llu; =2: predicted hits %llu, verified %llu, raced %llu (store-back %llu, identity %llu, sync %llu, refresh %llu)", delta(now.evictions, reported.evictions), delta(now.predicted, reported.predicted), delta(now.verified, reported.verified), delta(raced, racedBefore),
                  delta(now.raced[0], reported.raced[0]), delta(now.raced[1], reported.raced[1]), delta(now.raced[2], reported.raced[2]), delta(now.raced[3], reported.raced[3]));
    line += text;
    reported = now;
    return line;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.9): see DepthLease.hpp.
DepthViewCheck CheckDepthViewPrediction(const DepthViewFacts& facts) {
    const auto expected = facts.route == DepthViewRoute::NoHtile ? HtileVerdictKind::NoHtile : facts.route == DepthViewRoute::NotAccessible ? HtileVerdictKind::NotAccessible : facts.route == DepthViewRoute::NotPending ? HtileVerdictKind::NotPendingProved : HtileVerdictKind::PendingCovered;
    // Each part is excused only by the evidence that can change it (the r73 review; DepthLease.hpp).
    if (facts.verdict != expected || facts.htileCleared != 0) return {DepthViewPart::StoreBack, facts.storeBack.Explains(facts.route), facts.htileCleared != 0 ? "today's store-back cleared the HTILE" : "today's store-back took another route"};
    if (facts.cleared || facts.created || !facts.sameSurface || facts.retiredBefore) {
        const char* reason = facts.cleared ? "today took a cleared HTILE" : facts.created ? "today created the surface" : !facts.sameSurface ? "today found another surface" : "today found the surface retired";
        return {DepthViewPart::Identity, facts.identity.ExplainsIdentity(), reason};
    }
    // A sync: the memo key (a walk, or an unwatch), the surfaces serials (then the units were not read)
    // or the units moved; or a clock move that a stamp on the plane itself explains. Retired and a
    // cleared HTILE entry change no unit.
    if (facts.synced) {
        const auto& at = facts.sync;
        const bool clock = at.clockMoved || facts.unitsLow > facts.current0;
        return {DepthViewPart::Sync, at.memoMoved || at.surfacesMoved || at.oldestMoved || (clock && facts.planeStamped), "today synced the depth plane"};
    }
    bool lower = false;
    bool higher = false;
    const auto compare = [&](std::uint64_t value, std::uint64_t predicted) {
        if (value < predicted) lower = true;
        else if (value > predicted) higher = true;
    };
    if (facts.route == DepthViewRoute::Pending || facts.route == DepthViewRoute::NotPending) {
        if (!facts.htileReadFound) lower = true;
        else compare(facts.htileRead, facts.route == DepthViewRoute::Pending ? facts.memoCurrent0 : facts.h);
    }
    if (facts.unitsLow != facts.unitsHigh) lower = true;
    compare(facts.unitsLow, facts.current0);
    compare(facts.unitsHigh, facts.current0);
    compare(facts.depthMemory, facts.memoCurrent0);
    if (facts.stencil) compare(facts.stencilMemory, facts.memoCurrent0);
    else if (facts.stencilMemory != 0) lower = true;
    if (lower) return {DepthViewPart::Refresh, false, "a refresh value is below its prediction, missing, or unequal across the units"};
    if (higher) return {DepthViewPart::Refresh, true, "a refresh value is above its prediction"};
    return {};
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.12): test hooks (DepthLease.hpp).
DepthViewLeaseState DepthViewLeaseProbe(const Context& context, const DepthTarget& target) {
    DepthViewLeaseState state;
    std::lock_guard lock(surfacesMutex());
    DepthSurface* surface = nullptr;
    for (const auto& candidate : surfaces()) {
        if (candidate->active->load(std::memory_order_acquire) && candidate->context.device == context.device && sameSurface(candidate->target, target)) {
            surface = candidate.get();
            break;
        }
    }
    if (surface == nullptr) return state;
    state.found = true;
    state.units = surface->unitGeneration;
    const auto noted = surface->NotedMemory();
    state.depthMemory = noted[0].generation;
    state.stencilMemory = noted[1].generation;
    state.staleCount = surface->staleCount;
    state.staleUnits = surface->staleUnits.size();
    state.pendingClear = surface->pendingClear;
    state.written = surface->written;
    state.mirrored = surface->HasMirror();
    state.retired = surface->retired;
    if (target.htileAddress != 0) {
        std::lock_guard reads(htileReadsMutex());
        if (const auto found = htileReads().find(target.htileAddress); found != htileReads().end()) {
            state.htileRead = true;
            state.htileReadValue = found->second;
        }
    }
    if (mirrorable(target)) {
        const std::pair<std::uint64_t, std::uint64_t> plane{target.address, target.address + DescribeSurface(mirrorResource(target)).guestBytes};
        auto& planes = stalePlanes();
        std::lock_guard staleLock(planes.mutex);
        state.stalePlaneListed = std::find(planes.ranges.begin(), planes.ranges.end(), plane) != planes.ranges.end();
    }
    return state;
}

bool DepthSurfaceNoteStoredForTest(const Context& context, const DepthTarget& target, std::uint64_t generation, std::size_t units) {
    std::lock_guard lock(surfacesMutex());
    for (const auto& candidate : surfaces()) {
        if (!candidate->active->load(std::memory_order_acquire) || candidate->context.device != context.device || !sameSurface(candidate->target, target)) continue;
        std::vector<bool> stored(candidate->unitGeneration.size(), false);
        for (std::size_t unit = 0; unit < stored.size() && unit < units; ++unit) stored[unit] = true;
        candidate->NoteStored(stored, generation);
        return true;
    }
    return false;
}

std::uint64_t DepthSurfacesSerial() {
    return surfacesSerial.load(std::memory_order_acquire);
}

std::uint64_t DepthSurfacesReleased() {
    return surfacesReleased.load(std::memory_order_acquire);
}

std::uint64_t DepthSliceBytes(VkExtent2D extent, std::uint32_t bytesPerTexel) {
    const auto [blockWidth, blockHeight] = DepthBlockExtent(bytesPerTexel);
    const auto width = static_cast<std::uint64_t>((extent.width + blockWidth - 1) / blockWidth * blockWidth);
    const auto height = static_cast<std::uint64_t>((extent.height + blockHeight - 1) / blockHeight * blockHeight);
    return width * height * bytesPerTexel;
}

std::uint64_t MultisampledDepthBytes_nid_no_patch(VkExtent2D extent, std::uint32_t bytesPerTexel, std::uint32_t samples) {
    Require(extent.width != 0u && extent.width <= 16384u && extent.height != 0u && extent.height <= 16384u, "invalid multisample depth extent");
    Require((bytesPerTexel == 1u || bytesPerTexel == 2u || bytesPerTexel == 4u) && (samples == 2u || samples == 4u || samples == 8u), "unsupported multisample depth footprint");
    const auto sampleBits = std::countr_zero(samples);
    const auto pixelBits = 16u - std::countr_zero(bytesPerTexel) - sampleBits;
    const auto widthBits = (pixelBits + ((sampleBits & 1u) == 0u ? 1u : 0u)) / 2u;
    const auto blockWidth = 1u << widthBits;
    const auto blockHeight = 1u << (pixelBits - widthBits);
    const auto width = static_cast<std::uint64_t>((extent.width + blockWidth - 1u) / blockWidth * blockWidth);
    const auto height = static_cast<std::uint64_t>((extent.height + blockHeight - 1u) / blockHeight * blockHeight);
    return width * height * bytesPerTexel * samples;
}

void RequireNoResidentDepthRead_nid_no_patch(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0u || !anyDepthGuard().load(std::memory_order_acquire)) return;
    Require(address <= std::numeric_limits<std::uint64_t>::max() - bytes, "native depth read range overflows");
    std::shared_lock lock(depthGuardMutex());
    for (const auto& range : depthGuards()) Require(address >= range.address + range.bytes || range.address >= address + bytes, "sample-preserving guest publication of native multisample depth is not implemented");
}

bool HasResidentDepthGuards_nid_no_patch() {
    return anyDepthGuard().load(std::memory_order_acquire);
}

void RetireUnmappedDepth_nid_no_patch(std::uintptr_t address, std::size_t bytes) {
    if (bytes == 0u || !HasResidentDepthGuards_nid_no_patch()) return;
    Require(address <= std::numeric_limits<std::uint64_t>::max() - bytes, "native depth release range overflows");
    const auto end = address + bytes;
    std::unique_lock lock(depthGuardMutex());
    for (auto& range : depthGuards()) {
        std::vector<GuestMemory::UnchangedQuery> remaining;
        remaining.reserve(range.live.size() + 1u);
        for (const auto& part : range.live) {
            const auto finish = part.address + part.bytes;
            const auto first = std::max<std::uint64_t>(address, part.address);
            const auto last = std::min<std::uint64_t>(end, finish);
            if (first >= last) remaining.push_back(part);
            else {
                if (part.address < first) remaining.push_back({part.address, static_cast<std::size_t>(first - part.address), 0u});
                if (last < finish) remaining.push_back({last, static_cast<std::size_t>(finish - last), 0u});
            }
        }
        range.live = std::move(remaining);
    }
    for (const auto& range : depthGuards()) {
        if (std::ranges::all_of(depthGuards(), [&](const auto& part) { return part.active != range.active || part.live.empty(); })) range.active->store(false, std::memory_order_release);
    }
    std::erase_if(depthGuards(), [](const auto& range) { return !range.active->load(std::memory_order_acquire); });
    anyDepthGuard().store(!depthGuards().empty(), std::memory_order_release);
}

bool SampleLocationsCompatibleDepth(const Context& context, VkFormat format, std::uint32_t samples) {
    if (samples == 0 || (context.sampleLocationSampleCounts & samples) == 0) return false;
    VkImageFormatProperties properties{};
    const auto result = context.imageFormatProperties(context.physical, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT, &properties);
    if (result == VK_ERROR_FORMAT_NOT_SUPPORTED) return false;
    Check(result, "vkGetPhysicalDeviceImageFormatProperties sample locations depth");
    return (properties.sampleCounts & samples) != 0;
}

VkFormat HostDepthFormat(const Context& context, VkFormat format) {
    if (format != VK_FORMAT_D16_UNORM_S8_UINT) return format;
    static AgcDriver::Mutex mutex;
    static std::map<VkPhysicalDevice, VkFormat> chosen;
    std::lock_guard lock(mutex);
    if (const auto found = chosen.find(context.physical); found != chosen.end()) return found->second;
    constexpr VkFormatFeatureFlags required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    auto host = format;
    for (const auto candidate : {VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT}) {
        VkFormatProperties properties{};
        context.formatProperties(context.physical, candidate, &properties);
        if ((properties.optimalTilingFeatures & required) != required) continue;
        host = candidate;
        break;
    }
    chosen.emplace(context.physical, host);
    return host;
}

float HostDepthBiasConstant(const Context& context, VkFormat format, float constant) {
    const auto host = HostDepthFormat(context, format);
    if (constant == 0.0f || host == format) return constant;
    Require(host == VK_FORMAT_D24_UNORM_S8_UINT, "a constant depth bias on a 16-bit depth target with stencil that the device stores as 32-bit float depth is not implemented");
    return constant * 256.0f;
}

VkImageView DepthSurfaceView(const Context& context, const DepthTarget& target, bool depthWritten) {
    const bool native = RoutesToNativeDepthOwner(target);
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.6, DepthLease.hpp): the lease's
    // check, before any of the call's own work. A hit (=1) returns the surface and holds the surfaces
    // mutex in `held`; the steps it proved to be no-ops are skipped below, every other step runs. =2's
    // prediction holds nothing, and today's path runs with its points. Unset: nothing beyond the test.
    std::optional<DepthViewLease> view;
    std::unique_lock<std::recursive_mutex> held;
    DepthSurface* const leased = DepthViewLeases() != 0 ? depthViewLeaseHit(view.emplace(), context, target, native, held) : nullptr;
    // LOCAL DEBUG (APS5_PROFILE_DRAW, GPU-OWNED-PLAN 9.5 D5-0, DepthLease.hpp): the lease counter, while
    // profiling only (and not while the D5 leases' switch is set: its table and time would sit in the
    // phase they are measured by); its start reads the state the call's own work begins from.
    std::optional<DepthLeaseUse> lease;
    if (depthLeaseProfiled() && DepthViewLeases() == 0) depthLeaseStart(lease.emplace(), context, target, native);
    if (view && view->predicting) depthViewPoint(*view, DepthViewPoint::StoreBack, target);
    const auto htileCleared = native || leased != nullptr ? VkImageAspectFlags{0} : storeBackClearedHtile(context, target, view ? view->VerdictOut() : nullptr);
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::unique_lock<std::recursive_mutex> lock = leased != nullptr ? std::move(held) : std::unique_lock<std::recursive_mutex>(surfacesMutex());
    if (lease) depthLeaseLocked(*lease);
    if (view && view->predicting) depthViewPoint(*view, DepthViewPoint::Identity, target);
    const bool cleared = leased == nullptr && takeClearedHtile(target);
    DepthSurface* surface = leased;
    if (surface == nullptr) {
        for (const auto& candidate : surfaces()) {
            if (candidate->active->load(std::memory_order_acquire) && candidate->context.device == context.device && sameSurface(candidate->target, target)) {
                surface = candidate.get();
                break;
            }
        }
    }
    const bool created = surface == nullptr;
    GuestAllocations::Lease creationLease;
    if (created) {
        for (const auto& surface : surfaces()) {
            if (!surface->active->load(std::memory_order_acquire) || !surface->samplingActive || surface->context.device != context.device) continue;
            const auto bytes = target.samples == 1u ? DepthSliceBytes(target.extent, target.format == VK_FORMAT_D16_UNORM ? 2u : 4u) : MultisampledDepthBytes_nid_no_patch(target.extent, target.format == VK_FORMAT_D16_UNORM ? 2u : 4u, target.samples);
            Require(target.address <= std::numeric_limits<std::uint64_t>::max() - bytes && target.htileAddress <= std::numeric_limits<std::uint64_t>::max() - HtileMetadataBytes(target.extent), "native depth attachment range overflows");
            for (const auto& range : surface->sampling.initial) {
                Require(target.address >= range.address + range.bytes || range.address >= target.address + bytes, "native multisample depth attachment layout or metadata changed");
                if (target.stencilAddress != 0u) {
                    const auto stencilBytes = target.samples == 1u ? DepthSliceBytes(target.extent, 1u) : MultisampledDepthBytes_nid_no_patch(target.extent, 1u, target.samples);
                    Require(target.stencilAddress <= std::numeric_limits<std::uint64_t>::max() - stencilBytes && (target.stencilAddress >= range.address + range.bytes || range.address >= target.stencilAddress + stencilBytes), "native multisample stencil attachment aliases a resident depth surface");
                }
                if (target.htileAddress != 0u) Require(target.htileAddress >= range.address + range.bytes || range.address >= target.htileAddress + HtileMetadataBytes(target.extent), "native multisample depth attachment layout or metadata changed");
            }
        }
        lock.unlock();
        auto sampling = target.samples == 1u ? DepthSamplingState{} : prepareDepthSampling(context, target);
        creationLease = target.samples == 1u || !sampling.refusal.empty() ? GuestAllocations::Lease{} : GuestAllocations::GuestAllocationsAcquireAll_nid_postfix();
        lock.lock();
        releaseRetired(context);
        surfacesSerial.fetch_add(1, std::memory_order_release);
        surfaces().push_back(std::make_unique<DepthSurface>(context, target, std::move(sampling)));
        ++mirrorEpoch;
        auto& fresh = *surfaces().back();
        if (target.samples != 1u && fresh.sampleable && fresh.sampling.refusal.empty()) {
            std::unique_lock guardLock(depthGuardMutex());
            for (const auto& range : fresh.sampling.initial) depthGuards().push_back({context.device, range.address, range.bytes, fresh.active, {{range.address, range.bytes, 0u}}});
            anyDepthGuard().store(true, std::memory_order_release);
            InstallResidentImageReadGuard_nid_no_patch();
            fresh.samplingActive = true;
        }
        surface = surfaces().back().get();
    }
    if (lease) depthLeaseUnretire(*lease, surface->retired);
    if (view) view->retiredBefore = surface->retired;
    surface->retired = false;
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.6): `inherits` is read only for a
    // created surface or a cleared HTILE taken, which a hit has neither of, so a hit skips the lookup.
    bool inherits = false;
    if (leased == nullptr) {
        const auto owner = depthOwners().find({context.device, target.address});
        inherits = owner != depthOwners().end() && owner->second.surface->active->load(std::memory_order_acquire) && !owner->second.surface->retired;
    }
    surface->clearDepth = target.clearDepth;
    surface->clearStencil = target.clearStencil;
    takePlanes(*surface, created, htileCleared | (cleared ? VK_IMAGE_ASPECT_DEPTH_BIT : 0u));
    if (created && !inherits && target.htileAddress == 0) surface->TakeStorage();
    if (cleared && (!created || inherits)) {
        if (target.samples != 1u) surface->sampling.Refuse("native multisample depth fast-clear transition is not implemented");
        if (!created) surface->NoteHtileClear(VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    if (!created) surface->NoteHtileClear(htileCleared);
    if (lease) lease->depthWrites = depthWrites;
    if (mirrorable(target)) {
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.6): a hit's sync is
        // SyncFromMemory's unchanged exit with the generation TrackerGeneration answered under its lock;
        // a full path's is today's, watched for =2 (its point, and whether it synced) and the lease.
        if (leased != nullptr) {
            surface->SyncUnchanged(view->planeBytes, view->current);
        } else {
            if (view && view->predicting) depthViewPoint(*view, DepthViewPoint::Sync, target);
            if (view) view->writes = depthWrites;
            surface->SyncFromMemory(mirrorResource(target));
            if (view) view->synced = depthWrites != view->writes;
        }
    }
    if (lease) lease->synced = depthWrites != lease->depthWrites;
    surface->ApplyFastClear();
    if (drawnDepth != nullptr && drawnDepth != surface) drawnDepth->written = ++depthWrites;
    surface->written = ++depthWrites;
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES): a hit's NoteMemory, whose collects are memo hits.
    if (leased != nullptr) surface->NoteMemoryAt(view->memoCurrent);
    else surface->NoteMemory();
    if (mirrorable(target) && surface->scannedEpoch != mirrorEpoch) {
        surface->scannedEpoch = mirrorEpoch;
        const auto plane = stalePlaneRange(*surface);
        for (const auto& other : surfaces()) {
            if (other.get() == surface || other->context.device != context.device || other->written >= surface->written || !mirrorable(other->target)) continue;
            const auto otherPlane = stalePlaneRange(*other);
            if (otherPlane == plane || otherPlane.first >= plane.second || plane.first >= otherPlane.second) continue;
            other->ClearStale();
            other->DropMirror();
        }
    }
    if ((depthWritten || cleared) && mirrorable(target)) {
        const auto plane = target.address;
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES): a hit has the plane's size (pure) in its lease.
        const std::uint64_t planeBytes = leased != nullptr ? view->planeBytes : DescribeSurface(mirrorResource(target)).guestBytes;
        surface->MarkAllStale(planeBytes);
        noteStalePlane(plane, plane + planeBytes);
    }
    drawnDepth = surface;
    if (lease) depthLeaseEnd(*lease, context, target, *surface, created, htileCleared != 0, cleared);
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_DEPTH_LEASES, GPU-OWNED-PLAN 11.8): renewal, =2's check and the
    // lease's creation, under the surfaces mutex.
    if (view && view->tracking) depthViewLeaseEnd(*view, context, target, *surface, created, htileCleared, cleared);
    return surface->view;
}

SampleOrderTarget DepthSurfaceOrderTarget(const Context& context, const DepthTarget& target) {
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::unique_lock lock(surfacesMutex());
    for (const auto& candidate : surfaces()) {
        if (!candidate->active->load(std::memory_order_acquire) || candidate->context.device != context.device || !sameSurface(candidate->target, target)) continue;
        Require(candidate->sampleable, "the multisampled depth surface whose samples are put in the guest's order cannot be sampled");
        return {candidate->image, candidate->view, candidate->hostFormat, target.extent, target.samples, DepthSampleOrderMethod(context, target), candidate->owned};
    }
    throw std::runtime_error("AGC graphics: the multisampled depth surface whose samples are put in the guest's order is not resident");
}

DepthSampleLocations DepthSurfaceSampleLocations(const Context& context, const DepthTarget& target) {
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::unique_lock lock(surfacesMutex());
    for (const auto& candidate : surfaces()) {
        if (!candidate->active->load(std::memory_order_acquire) || candidate->context.device != context.device || !sameSurface(candidate->target, target)) continue;
        DepthSampleLocations depth;
        depth.image = candidate->image;
        depth.format = candidate->hostFormat;
        depth.samples = target.samples;
        depth.keep = candidate->owned;
        return depth;
    }
    throw std::runtime_error("AGC graphics: the depth surface drawn at custom sample locations is not resident");
}

bool DepthPlanesStale() {
    auto& planes = stalePlanes();
    return planes.begin.load(std::memory_order_acquire) < planes.end.load(std::memory_order_acquire);
}

bool DepthPlanesStaleOver(std::uint64_t address, std::size_t bytes) {
    const auto end = bytes > std::numeric_limits<std::uint64_t>::max() - address ? std::numeric_limits<std::uint64_t>::max() : address + bytes;
    return overlapsStalePlane(address, end);
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> FlushDepthPlanes(std::uint64_t address, std::size_t bytes, DepthPlaneScope scope) {
    struct FlushingTag {};
    using Flushing = HostThreadSlot<bool, FlushingTag>;
    const auto end = bytes > std::numeric_limits<std::uint64_t>::max() - address ? std::numeric_limits<std::uint64_t>::max() : address + bytes;
    if (Flushing::Get() || !overlapsStalePlane(address, end)) return {};
    if (scope == DepthPlaneScope::Overlapped) {
        std::lock_guard lock(surfacesMutex());
        const auto staleInRange = [&](const auto& surface) { return surface->AnyStale() && !surface->syncing && ((surface->pendingClear & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 || anyUnit(surface->StaleUnitsIn(DescribeSurface(mirrorResource(surface->target)).guestBytes, address, end))); };
        if (std::none_of(surfaces().begin(), surfaces().end(), staleInRange)) return {};
    }
    Flushing::Set(true);
    struct Reset {
        ~Reset() { Flushing::Set(false); }
    } reset;
    struct Stored {
        DepthSurface* surface;
        std::shared_ptr<StorageTexture> mirror;
        std::vector<bool> units;
        std::uint64_t begin;
        std::uint64_t end;
        bool flushed;
    };
    std::vector<Stored> stored;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    {
        std::lock_guard gpu(GuestMemory::GpuMutex());
        std::lock_guard lock(surfacesMutex());
        std::vector<std::pair<std::uint64_t, std::uint64_t>> remaining;
        for (const auto& surface : surfaces()) {
            if (!surface->AnyStale()) continue;
            const auto resource = mirrorResource(surface->target);
            const auto plane = surface->target.address;
            const auto planeBytes = DescribeSurface(resource).guestBytes;
            const bool overlapped = !surface->syncing && plane < end && address < plane + planeBytes;
            if (overlapped && (surface->pendingClear & VK_IMAGE_ASPECT_DEPTH_BIT) != 0) surface->MarkAllStale(planeBytes);
            if (overlapped && (scope == DepthPlaneScope::Whole || anyUnit(surface->StaleUnitsIn(planeBytes, address, end)))) {
                surface->SyncFromMemory(resource);
                auto units = scope == DepthPlaneScope::Whole ? surface->staleUnits : surface->StaleUnitsIn(planeBytes, address, end);
                if (anyUnit(units)) {
                    auto mirror = surface->StoreDepthPlane(resource, units);
                    surface->ClearStale(units);
                    const auto hull = surface->UnitHull(units, planeBytes);
                    ranges.emplace_back(hull.first, hull.second - hull.first);
                    stored.push_back({surface.get(), std::move(mirror), std::move(units), hull.first, hull.second, false});
                }
            }
            if (surface->AnyStale()) remaining.emplace_back(plane, plane + planeBytes);
        }
        auto& planes = stalePlanes();
        std::lock_guard staleLock(planes.mutex);
        planes.ranges = std::move(remaining);
        publishStaleHull(planes);
    }
    for (auto& entry : stored) entry.flushed = StorageTexture::FlushPending(entry.begin, static_cast<std::size_t>(entry.end - entry.begin), nullptr, "depth plane store");
    const auto generation = GuestMemory::TrackerGeneration();
    std::lock_guard lock(surfacesMutex());
    for (const auto& entry : stored) {
        if (!entry.flushed || std::none_of(surfaces().begin(), surfaces().end(), [&](const auto& candidate) { return candidate.get() == entry.surface; })) continue;
        entry.surface->NoteStored(entry.units, generation);
        entry.surface->NoteMemory();
    }
    return ranges;
}

DepthPlaneStoreStatistics DepthPlaneStoreCounts() {
    const auto& counters = depthPlaneStoreCounters();
    DepthPlaneStoreStatistics statistics;
    statistics.stores = counters.stores.load();
    statistics.units = counters.units.load();
    statistics.bytes = counters.bytes.load();
    return statistics;
}

std::uint64_t HtileDepthClearAddress(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const std::array<std::uint32_t, 3>& numThreads) {
    static constexpr std::array<std::uint32_t, 17> htileMaskKernel{0xd7460000u, 0x04010c06u, 0x34000084u, 0xdc388000u, 0x04020000u, 0xbf8c3f70u, 0xd7710008u, 0x00120805u, 0xd7710009u, 0x00120a05u, 0xd771000au, 0x00120c05u, 0xd771000bu, 0x00120e05u, 0xdc788000u, 0x00020800u, 0xbf810000u};
    if (userData.size() < 6 || numThreads[0] != 64 || numThreads[1] != 1 || numThreads[2] != 1) return 0;
    if (code.size() < htileMaskKernel.size() || !std::equal(htileMaskKernel.begin(), htileMaskKernel.end(), code.begin())) return 0;
    if ((userData[5] & 0xfu) != 0 || (userData[4] & 0xfu) != 0) return 0;
    return userData[2] | (static_cast<std::uint64_t>(userData[3]) << 32u);
}

std::optional<MemoryValueFill> MatchMemoryValueFill(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const std::array<std::uint32_t, 3>& numThreads, const std::array<std::uint32_t, 3>& groups) {
    static constexpr std::array<std::uint32_t, 9> kernel{0xd7460000u, 0x04010c08u, 0xf4201a82u, 0xfa000000u, 0xbf8cc07fu, 0x7e02026au, 0xe0102000u, 0x80000100u, 0xbf810000u};
    if (userData.size() < 8 || numThreads != std::array<std::uint32_t, 3>{64u, 1u, 1u} || groups[1] != 1u || groups[2] != 1u) return std::nullopt;
    if (code.size() < kernel.size() || !std::equal(kernel.begin(), kernel.end(), code.begin())) return std::nullopt;
    const auto stride = (userData[1] >> 16u) & 0x3fffu;
    const bool swizzled = ((userData[1] >> 31u) & 1u) != 0;
    const bool addTid = ((userData[3] >> 23u) & 1u) != 0;
    const auto type = userData[3] >> 30u;
    const auto format = (userData[3] >> 12u) & 0x7fu;
    const auto valueStride = (userData[5] >> 16u) & 0x3fffu;
    const auto valueBytes = valueStride == 0 ? static_cast<std::uint64_t>(userData[6]) : static_cast<std::uint64_t>(userData[6]) * valueStride;
    if (type != 0 || stride != 4 || swizzled || addTid || format < 20u || format > 22u || valueBytes < 4u) return std::nullopt;
    const auto dwords = std::min<std::uint64_t>(static_cast<std::uint64_t>(groups[0]) * 64u, userData[2]);
    const auto address = userData[0] | (static_cast<std::uint64_t>(userData[1] & 0xffffu) << 32u);
    const auto valueAddress = userData[4] | (static_cast<std::uint64_t>(userData[5] & 0xffffu) << 32u);
    if (dwords == 0 || valueAddress == 0 || (valueAddress & 3u) != 0) return std::nullopt;
    return MemoryValueFill{address, static_cast<std::size_t>(dwords * 4u), valueAddress};
}

void NoteMemoryValueFill(const MemoryValueFill& fill) {
    std::uint32_t value = 0;
    GuestMemory::Read(fill.valueAddress, std::as_writable_bytes(std::span(&value, 1)), 4);
    NoteDepthMetadataFill(fill.address, fill.bytes, value);
}

std::optional<StorageImageClear> MatchStorageImageClear(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderComputeStageInfo& compute, const std::array<std::uint32_t, 3>& groups) {
    static constexpr std::array<std::uint32_t, 15> kernel{0xd7460004u, 0x0401060cu, 0xf4280204u, 0xfa000000u, 0xd7460005u, 0x0405060du, 0x7e0c020eu, 0xbf8cc07fu, 0x7e000208u, 0x7e020209u, 0x7e04020au, 0x7e06020bu, 0xf0200f28u, 0x00000004u, 0xbf810000u};
    if (code.size() < kernel.size() || code[0] != kernel[0] || !std::equal(kernel.begin(), kernel.end(), code.begin())) return std::nullopt;
    if (userData.size() != 12u) return std::nullopt;
    if (compute.numThreads != std::array<std::uint32_t, 3>{8u, 8u, 1u}) return std::nullopt;
    if (!compute.groupIdEnable[0] || !compute.groupIdEnable[1] || !compute.groupIdEnable[2]) return std::nullopt;
    if (compute.threadIdComponentCount < 2u) return std::nullopt;
    std::array<std::uint64_t, 3> covered{};
    for (std::size_t axis = 0; axis < 3u; ++axis) covered[axis] = compute.partialThreads[axis] != 0u ? compute.partialThreads[axis] : static_cast<std::uint64_t>(groups[axis]) * compute.numThreads[axis];
    if (covered[0] == 0u || covered[1] == 0u || covered[2] != 1u) return std::nullopt;
    StorageImageClear clear;
    std::copy_n(userData.begin(), clear.image.size(), clear.image.begin());
    std::copy_n(userData.begin() + static_cast<std::ptrdiff_t>(clear.image.size()), clear.color.size(), clear.color.begin());
    clear.width = static_cast<std::uint32_t>(std::min<std::uint64_t>(covered[0], std::numeric_limits<std::uint32_t>::max()));
    clear.height = static_cast<std::uint32_t>(std::min<std::uint64_t>(covered[1], std::numeric_limits<std::uint32_t>::max()));
    return clear;
}

std::uint64_t StorageImageClearAddress(const StorageImageClear& clear) {
    try {
        return DecodeTextureResource(clear.image).baseAddress;
    } catch (...) {
        return 0u;
    }
}

DepthPlaneClear::DepthPlaneClear(const StorageImageClear& clear) {
    if (!depthPlaneClearEnabled.load(std::memory_order_relaxed)) return;
    const auto decoded = decodeOverwrite(clear);
    if (decoded) {
        value = *decoded;
        previous = currentOverwrite;
        currentOverwrite = &value;
        installed = true;
        ++depthClearCounters().taken;
    }
}

DepthPlaneClear::~DepthPlaneClear() {
    if (installed) currentOverwrite = previous;
}

DepthClearStatistics DepthClearCounts() {
    const auto& counters = depthClearCounters();
    DepthClearStatistics statistics;
    statistics.matched = counters.matched.load();
    statistics.taken = counters.taken.load();
    statistics.copiesSkipped = counters.copiesSkipped.load();
    statistics.clears = counters.clears.load();
    statistics.bytesSkipped = counters.bytesSkipped.load();
    for (std::size_t reason = 0; reason < statistics.declined.size(); ++reason) statistics.declined[reason] = counters.declined[reason].load();
    return statistics;
}

void SetDepthPlaneClearForTest(bool enabled) {
    depthPlaneClearEnabled.store(enabled, std::memory_order_relaxed);
}

std::size_t HtileDepthClearBytes_nid_no_patch(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderComputeStageInfo& compute, std::span<const std::uint32_t> packet, std::uint64_t indirectArguments) {
    if (userData.size() != 6u || packet.size() < 5u || indirectArguments != 0u || compute.PartialGroups() || !compute.groupIdEnable[0] || compute.threadIdComponentCount == 0u || compute.threadIdComponentCount > 3u) return 0u;
    if (packet[1] == 0u || packet[1] > 0x400000u || packet[2] != 1u || packet[3] != 1u || (packet[4] & 0x23u) != 1u) return 0u;
    const auto address = HtileDepthClearAddress(code, userData, compute.numThreads);
    if (address == 0u || (address & 15u) != 0u) return 0u;
    const auto bytes = static_cast<std::uint64_t>(packet[1]) * 1024u;
    return address <= std::numeric_limits<std::uint64_t>::max() - bytes ? static_cast<std::size_t>(bytes) : 0u;
}

void NoteHtileDepthClear(std::uint64_t htileAddress, std::size_t bytes, std::size_t nativeBytes) {
    std::lock_guard lock(surfacesMutex());
    forgetHtileReads(htileAddress, bytes);
    bool existing = false;
    for (const auto& surface : surfaces()) {
        if (!surface->active->load(std::memory_order_acquire)) continue;
        if (surface->samplingActive && surface->target.htileAddress == htileAddress) {
            const auto& metadata = surface->sampling.initial[1];
            if (nativeBytes != bytes || nativeBytes < metadata.bytes || !GuestMemory::Accessible(reinterpret_cast<const void*>(metadata.address), metadata.bytes) || !GuestMemory::Watched(metadata.address, metadata.bytes)) surface->sampling.Refuse("native multisample depth incomplete or unproven fast-clear transition is not implemented");
            else if (surface->clearGeneration != 0u) surface->sampling.Refuse("overlapping native multisample depth clear transitions are not implemented");
            else surface->clearGeneration = GuestMemory::CollectWritesUncached(metadata.address, metadata.bytes);
            existing = true;
            continue;
        }
        if (!HtileFillCovers(surface->target.htileAddress, surface->target.extent, htileAddress, bytes)) continue;
        if (surface->target.samples != 1u) surface->sampling.Refuse("native multisample depth fast-clear transition is not implemented");
        surface->NoteHtileClear(VK_IMAGE_ASPECT_DEPTH_BIT);
        existing = true;
    }
    if (!existing) {
        clearedHtiles()[htileAddress] = bytes;
        // LOCAL DEBUG (APS5_PROFILE_DRAW, GPU-OWNED-PLAN 9.5 D5-0): the clearedHtiles insert serial.
        if (depthLeaseProfiled()) ++clearedHtileSets;
    }
}

void ClearDepthSurfaces(VkDevice device) {
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::lock_guard lock(surfacesMutex());
    drawnDepth = nullptr;
    surfacesSerial.fetch_add(1, std::memory_order_release);
    std::erase_if(depthOwners(), [&](const auto& entry) { return entry.first.first == device; });
    std::erase_if(stencilOwners(), [&](const auto& entry) { return entry.first.first == device; });
    std::erase_if(surfaces(), [&](const auto& surface) { return surface->context.device == device; });
    std::erase_if(planeCopies(), [&](const auto& entry) { return std::get<0>(entry.first) == device; });
    std::erase_if(layeredStorages(), [&](const auto& entry) { return std::get<0>(entry.first) == device; });
    std::unique_lock guardLock(depthGuardMutex());
    std::erase_if(depthGuards(), [&](const auto& range) { return range.device == device; });
    anyDepthGuard().store(!depthGuards().empty(), std::memory_order_release);
    UpdateResidentImageUnmapObserver_nid_no_patch();
}

std::shared_ptr<Texture> DepthSurfaceTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components, VkImageViewType multisampleViewType) {
    const auto held = [&](const auto& surface) {
        return surface->active->load(std::memory_order_acquire) && surface->context.device == context.device && !surface->retired && (surface->target.address == resource.baseAddress || (surface->target.stencilAddress != 0 && surface->target.stencilAddress == resource.baseAddress));
    };
    {
        std::lock_guard lock(surfacesMutex());
        if (std::none_of(surfaces().begin(), surfaces().end(), held)) return nullptr;
    }
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::lock_guard lock(surfacesMutex());
    auto* found = planeSurface(context.device, resource.baseAddress);
    if (found == nullptr) return nullptr;
    const auto& base = found->target;
    const bool stencil = base.stencilAddress != 0 && resource.baseAddress == base.stencilAddress;
    if (stencil && BytesPerElement(resource.format) != 1u) return nullptr;
    if (base.samples != 1u || resource.dimension == TextureDimension::k2DMsaa) {
        Require(base.samples > 1u, "multisample reader overlaps a single-sample depth attachment");
        auto* recorder = RecorderFor(context);
        Require(recorder != nullptr, "a native multisampled depth target is sampled only by recorded draws and dispatches");
        found->CheckCurrent();
        found->CheckResource(resource);
        Require(found->pendingClear == 0u, "native multisample depth fast-clear transition is not implemented");
        RecordMemoryBarrier(context, recorder->Commands(), VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        Recorder::CountBarriers(Recorder::CommandClass::Draw, 1);
        return found->Sampled(words, resource, components, found == drawnDepth, multisampleViewType);
    }
    if (resource.width != base.extent.width || resource.height != base.extent.height || resource.baseLevel != 0) return nullptr;
    const bool d16 = base.format == VK_FORMAT_D16_UNORM || base.format == VK_FORMAT_D16_UNORM_S8_UINT;
    const auto format = ResolveTextureFormat(resource.format);
    const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
    const bool integer = d16 && format == VK_FORMAT_R16_UINT && !depthBits;
    if (!stencil && format != (d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT) && !depthBits && !integer) return nullptr;
    const bool cube = resource.dimension == TextureDimension::kCube;
    const bool layered = cube || (resource.dimension == TextureDimension::k2DArray && resource.depthOrLastArray != 0);
    const bool sampled = stencil || (!layered && !integer);
    const auto copyFormat = integer ? VK_FORMAT_R16_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
    const bool planes = (format == copyFormat || depthBits) && resource.width == base.extent.width && resource.height == base.extent.height && resource.baseLevel == 0 && resource.lastLevel == 0 && !(resource.baseArray != 0 && (cube || !layered || resource.baseArray > resource.depthOrLastArray)) && !(cube && resource.depthOrLastArray != 0 && resource.depthOrLastArray != 5) && (resource.dimension == TextureDimension::k2D || resource.dimension == TextureDimension::k2DArray || cube);
    if (!(sampled ? found->SampledAccepts(words, resource) : planes) && found->OverwrittenInMemory()) {
        retireStoringStale(*found);
        return nullptr;
    }
    found->ApplyFastClear();
    if (sampled) return found->Sampled(words, resource, components, found == drawnDepth);
    Require(found->hostFormat == base.format, "sampling several slices or an integer view of a depth plane that the device stores in another depth format is not implemented");
    const auto layers = cube ? 6u : layered ? resource.depthOrLastArray + 1u : 1u;
    if (!planes) {
        char text[320];
        std::snprintf(text, sizeof(text), "AGC graphics: sampling the depth planes of depth surface 0x%llx (%ux%u, vk format %d) as a %ux%u texture of guest format %u (vk %d), dimension %d, %u slices, levels %u-%u, base slice %u is not implemented", static_cast<unsigned long long>(base.address), base.extent.width, base.extent.height, static_cast<int>(base.format), resource.width, resource.height, resource.format, static_cast<int>(format), static_cast<int>(resource.dimension), layers, resource.baseLevel, resource.lastLevel, resource.baseArray);
        throw std::runtime_error(text);
    }
    std::vector<DepthPlaneSlice> slices;
    const auto geometry = DescribeSurface(resource);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const auto address = base.address + geometry.GuestLayerOffset(layer);
        auto* slice = depthSlice(context.device, address, base);
        if (slice != nullptr) slice->ApplyFastClear();
        if (slice == nullptr) slices.push_back({});
        else slices.push_back({slice->image, slice == drawnDepth ? 0u : slice->written});
    }
    const auto imageFormat = integer ? VK_FORMAT_R16_UINT : d16 ? VK_FORMAT_D16_UNORM : VK_FORMAT_D32_SFLOAT;
    auto& copy = planeCopies()[{context.device, base.address, base.extent.width, base.extent.height, imageFormat, layers}];
    if (copy == nullptr) {
        copy = std::make_unique<DepthPlaneCopy>(context, base.extent, imageFormat, layers);
    }
    return copy->Refresh(slices, resource, components, cube || resource.dimension == TextureDimension::k2DArray ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, layered && !cube ? resource.baseArray : 0u);
}

bool DepthSurfaceHolds(const Context& context, const GuestTextureResource& resource, const Texture* texture) {
    if ((resource.dimension != TextureDimension::k2D && resource.dimension != TextureDimension::k2DMsaa) || resource.baseArray != 0) return false;
    std::lock_guard gpu(GuestMemory::GpuMutex());
    const auto address = resource.baseAddress;
    std::lock_guard lock(surfacesMutex());
    auto* found = planeSurface(context.device, address);
    if (found == nullptr || !found->Holds(texture)) return false;
    found->CheckCurrent();
    found->CheckResource(resource);
    return true;
}

void RequireNoNativeDepthFeedback_nid_no_patch(const Context& context, const DepthTarget& target, std::span<const std::shared_ptr<Texture>> sampled) {
    if (target.samples == 1u || sampled.empty()) return;
    std::lock_guard gpu(GuestMemory::GpuMutex());
    std::lock_guard lock(surfacesMutex());
    for (const auto& surface : surfaces()) {
        if (!surface->active->load(std::memory_order_acquire)) continue;
        if (surface->context.device != context.device || !sameSurface(surface->target, target)) continue;
        Require(std::none_of(sampled.begin(), sampled.end(), [&](const auto& texture) { return texture != nullptr && surface->Holds(texture.get()); }), "sampling a native multisampled depth target the draw attaches is not implemented");
    }
}

VkImageAspectFlags HtileFillClears(std::uint32_t pattern, bool stencilInHtile) {
    VkImageAspectFlags cleared = (pattern & 0xfu) == 0 ? VK_IMAGE_ASPECT_DEPTH_BIT : 0u;
    if (stencilInHtile && ((pattern >> 8u) & 0x3u) == 0) cleared |= VK_IMAGE_ASPECT_STENCIL_BIT;
    return cleared;
}

std::uint64_t HtileMetadataBytes(VkExtent2D extent) {
    const auto blocks = static_cast<std::uint64_t>((extent.width + 1023u) / 1024u) * ((extent.height + 511u) / 512u);
    return blocks * 32768u;
}

bool HtileFillCovers(std::uint64_t htile, VkExtent2D extent, std::uint64_t address, std::size_t bytes) {
    if (htile == 0 || htile < address || htile >= address + bytes) return false;
    return address + bytes - htile >= HtileMetadataBytes(extent);
}

bool PlaneFillCovers(std::uint64_t plane, std::uint64_t planeBytes, std::uint64_t address, std::size_t bytes) {
    return plane != 0 && plane >= address && plane + planeBytes <= address + bytes;
}

std::optional<std::uint8_t> StencilFillValue(std::uint32_t pattern) {
    const auto byte = static_cast<std::uint8_t>(pattern & 0xffu);
    if (pattern != byte * 0x01010101u) return std::nullopt;
    return byte;
}

std::optional<float> DepthFillValue(std::uint32_t pattern, bool d16) {
    if (d16) {
        if ((pattern >> 16u) != (pattern & 0xffffu)) return std::nullopt;
        return static_cast<float>(pattern & 0xffffu) / 65535.0f;
    }
    const auto value = std::bit_cast<float>(pattern);
    if (!(value >= 0.0f && value <= 1.0f)) return std::nullopt;
    return value;
}

void NoteDepthMetadataFill(std::uint64_t address, std::size_t bytes, std::uint32_t pattern) {
    std::lock_guard lock(surfacesMutex());
    forgetHtileReads(address, bytes);
    for (const auto& surface : surfaces()) {
        if (!surface->active->load(std::memory_order_acquire)) continue;
        const auto& target = surface->target;
        if (target.samples != 1u && target.htileAddress != 0u && address < target.htileAddress + HtileMetadataBytes(target.extent) && target.htileAddress < address + bytes) surface->sampling.Refuse("native multisample depth metadata fill transition is not implemented");
        if (HtileFillCovers(target.htileAddress, target.extent, address, bytes)) {
            const VkImageAspectFlags written = VK_IMAGE_ASPECT_DEPTH_BIT | (target.htileStencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0u);
            surface->ForgetClear(written);
            surface->NoteHtileClear(HtileFillClears(pattern, target.htileStencil));
            continue;
        }
        if (target.samples != 1) continue;
        const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
        if (PlaneFillCovers(target.address, DepthSliceBytes(target.extent, d16 ? 2u : 4u), address, bytes)) {
            if (const auto depth = DepthFillValue(pattern, d16)) surface->NotePlaneFill(VK_IMAGE_ASPECT_DEPTH_BIT, *depth, 0);
        }
        if (PlaneFillCovers(target.stencilAddress, DepthSliceBytes(target.extent, 1u), address, bytes)) {
            if (const auto stencil = StencilFillValue(pattern)) surface->NotePlaneFill(VK_IMAGE_ASPECT_STENCIL_BIT, 0.0f, *stencil);
        }
    }
}

bool DepthSurfaceAt(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource) {
    std::lock_guard lock(surfacesMutex());
    const auto* found = planeSurface(context.device, resource.baseAddress);
    if (found == nullptr) return false;
    const auto& target = found->target;
    if (resource.width != target.extent.width || resource.height != target.extent.height) return false;
    const bool stencil = target.stencilAddress != 0 && target.stencilAddress == resource.baseAddress;
    const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
    const auto expected = stencil ? VK_FORMAT_R8_UINT : d16 ? VK_FORMAT_R16_UNORM : VK_FORMAT_R32_SFLOAT;
    const bool depthBits = !stencil && words.size() >= 4 && ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) == (d16 ? 16u : 32u);
    return ResolveTextureFormat(resource.format) == expected || depthBits;
}

namespace {

std::vector<LayeredSlice> layeredSlices(const Context& context, const LayeredDepthStorage& storage) {
    const auto& base = storage.base;
    std::vector<LayeredSlice> slices;
    const auto& list = surfaces();
    for (std::uint32_t layer = storage.firstLayer; layer < storage.layers; ++layer) {
        const auto address = base.address + static_cast<std::uint64_t>(layer) * DepthSliceBytes(base.extent, storage.bytesPerTexel);
        const auto slice = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) { return surface->context.device == context.device && !surface->retired && surface->target.address == address && surface->target.extent.width == base.extent.width && surface->target.extent.height == base.extent.height && surface->target.format == base.format && surface->target.samples == 1; });
        if (slice != list.rend()) slices.push_back({slice->get(), layer});
    }
    return slices;
}

std::shared_ptr<StorageTexture> layeredDepthStorage(const Context& context, DepthSurface& surface, const GuestTextureResource& resource, std::uint32_t mip) {
    const auto& target = surface.target;
    const auto layers = resource.depthOrLastArray + 1u;
    const bool d16 = target.format == VK_FORMAT_D16_UNORM || target.format == VK_FORMAT_D16_UNORM_S8_UINT;
    const auto bytesPerTexel = d16 ? 2u : 4u;
    surface.RequireStorable(resource);
    const auto geometry = DescribeSurface(resource);
    if (layers > 1 && geometry.GuestLayerOffset(1) != DepthSliceBytes(target.extent, bytesPerTexel)) {
        char text[256];
        std::snprintf(text, sizeof(text), "AGC graphics: storage image access to the depth planes of depth surface 0x%llx with a layer pitch of 0x%llx bytes, not the depth slice size 0x%llx, is not implemented", static_cast<unsigned long long>(target.address), static_cast<unsigned long long>(geometry.GuestLayerOffset(1)), static_cast<unsigned long long>(DepthSliceBytes(target.extent, bytesPerTexel)));
        throw std::runtime_error(text);
    }
    auto& entry = layeredStorages()[{context.device, resource.baseAddress, resource.width, resource.height, layers, resource.format, static_cast<int>(resource.tileMode), static_cast<int>(resource.dimension), resource.baseArray}];
    if (entry == nullptr) entry = std::make_unique<LayeredDepthStorage>(context, resource, target, bytesPerTexel, resource.baseArray, layers);
    else static_cast<void>(entry->storage->Refresh());
    const auto slices = layeredSlices(context, *entry);
    for (const auto& slice : slices) slice.surface->ApplyFastClear();
    entry->Copy(slices, true);
    return entry->storage;
}

}

std::shared_ptr<StorageTexture> DepthSurfaceStorage(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint32_t mip) {
    if (mip != 0 || resource.baseLevel != 0) return nullptr;
    std::lock_guard lock(surfacesMutex());
    const auto& list = surfaces();
    const auto found = std::find_if(list.rbegin(), list.rend(), [&](const auto& surface) {
        return surface->active->load(std::memory_order_acquire) && surface->context.device == context.device && !surface->retired && (surface->target.address == resource.baseAddress || (surface->target.stencilAddress != 0 && surface->target.stencilAddress == resource.baseAddress));
    });
    if (found == list.rend()) return nullptr;
    auto& surface = **found;
    const bool stencil = surface.target.stencilAddress != 0 && surface.target.stencilAddress == resource.baseAddress;
    if (stencil || resource.dimension != TextureDimension::k2DArray || (resource.baseArray == 0 && resource.depthOrLastArray == 0)) return surface.Storage(words, resource, mip);
    if (!surface.StorageViewMatches(words, resource)) return nullptr;
    return layeredDepthStorage(context, surface, resource, mip);
}

void CopyDepthSample(const Context& context, std::uint64_t depthAddress, std::uint32_t samples, std::uint32_t sample, StorageTexture& destination, VkImageView destinationView, VkExtent2D extent) {
    std::lock_guard lock(surfacesMutex());
    auto* found = planeSurface(context.device, depthAddress);
    Require(found != nullptr && found->target.address == depthAddress, "DB depth copy from a depth plane no draw rendered");
    Require(found->target.format == VK_FORMAT_D32_SFLOAT || found->target.format == VK_FORMAT_D32_SFLOAT_S8_UINT, "DB depth copy from a depth surface that is not 32-bit float");
    Require(found->target.samples == samples, "DB depth copy from a depth surface of another sample count");
    Require(found->target.extent.width == extent.width && found->target.extent.height == extent.height, "DB depth copy from a depth surface of another size");
    Require(sample < found->target.samples, "DB depth copy of a sample the depth surface does not have");
    found->CopySample(sample, destination, destinationView);
}

bool DepthSurfaceStoreBack(const Context& context, const StorageTexture& storage) {
    std::lock_guard lock(surfacesMutex());
    for (auto& [key, entry] : layeredStorages()) {
        if (std::get<0>(key) != context.device || entry->storage.get() != &storage) continue;
        const auto slices = layeredSlices(context, *entry);
        entry->Copy(slices, false);
        for (const auto& slice : slices) slice.surface->written = ++depthWrites;
        return true;
    }
    for (const auto& surface : surfaces()) {
        if (surface->active->load(std::memory_order_acquire) && surface->context.device == context.device && surface->StoreBack(storage)) return true;
    }
    for (auto surface = surfaces().rbegin(); surface != surfaces().rend(); ++surface) {
        if ((*surface)->context.device == context.device && (*surface)->StoreBackCovering(storage)) return true;
    }
    return false;
}

bool DepthSurfaceAt(std::uint64_t address) {
    std::lock_guard lock(surfacesMutex());
    return std::any_of(surfaces().begin(), surfaces().end(), [&](const auto& surface) { return surface->active->load(std::memory_order_acquire) && !surface->retired && (surface->target.address == address || surface->target.stencilAddress == address); });
}

bool DepthSurfaceOverlaps(VkDevice device, std::uint64_t address, std::uint64_t bytes) {
    std::lock_guard lock(surfacesMutex());
    return std::any_of(surfaces().begin(), surfaces().end(), [&](const auto& surface) { return surface->active->load(std::memory_order_acquire) && !surface->retired && surface->context.device == device && surface->Overlaps(address, bytes); });
}

void RetireDepthSurfaces(VkDevice device, std::uint64_t address, std::uint64_t bytes) {
    std::lock_guard lock(surfacesMutex());
    for (const auto& surface : surfaces()) {
        if (surface->context.device == device && !surface->retired && surface->Overlaps(address, bytes)) retire(*surface);
    }
}

void RetireCoveredDepthSurfaces(VkDevice device, std::span<const std::pair<std::uint64_t, std::uint64_t>> written, std::span<const WrittenStorageExtent> extents) {
    std::lock_guard lock(surfacesMutex());
    for (const auto& surface : surfaces()) {
        if (surface->context.device == device && !surface->retired && surface->CoveredBy(written, extents)) retire(*surface);
    }
}

}
