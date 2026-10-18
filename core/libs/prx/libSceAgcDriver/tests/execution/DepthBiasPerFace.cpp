#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 16;
constexpr std::uint32_t WaveSize = 64;
constexpr std::byte Kept{0x40};
constexpr float SurfaceDepth = 0.5f;
constexpr float Bias = 1048576.0f;
alignas(256) std::array<std::byte, Width * Height * 4> Pixels{};
alignas(256) std::array<float, Width * Height> DepthMemory{};

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 4> PixelCode{
    0x7e0e02f2, 0xf800180f, 0x07070707, 0xbf810000,
};

constexpr std::array<std::array<float, 2>, 6> Corners{{
    {1.0f, 1.0f}, {1.0f, 15.0f}, {15.0f, 1.0f},
    {17.0f, 1.0f}, {31.0f, 1.0f}, {17.0f, 15.0f},
}};

alignas(256) std::array<std::array<float, 4>, Corners.size()> Vertices{};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t stride, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), count, 0x01016facu};
}

void Draw(AgcDriver::VulkanDevice& device, float frontBias, float backBias) {
    Pixels.fill(Kept);
    for (std::size_t index = 0; index < Corners.size(); ++index) {
        Vertices[index] = {Corners[index][0] * 2.0f / Width - 1.0f, 1.0f - Corners[index][1] * 2.0f / Height, SurfaceDepth, 1.0f};
    }
    const auto target = device.Target();

    std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexBuffer = BufferDescriptor(Vertices.data(), 16u, static_cast<std::uint32_t>(Vertices.size()));
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {WaveSize, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    const auto vertexResult = ShaderRecompiler::Recompile(vertex);
    const auto vertexPush = static_cast<std::uint32_t>(vertexResult.pushConstants.size());

    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX) | ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionY);
    pixel.posX = true;
    pixel.posY = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    const std::vector<std::uint32_t> pixelUserData(8, 0u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(PixelCode.data()), std::as_bytes(std::span(PixelCode))}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(PixelCode.data()), PixelCode, 0, {}},
        {WaveSize, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory},
        target,
        PixelPushLayout(vertexPush, target)
    };
    fragment.useCache = false;
    const auto pixelResult = ShaderRecompiler::Recompile(fragment);
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &vertexResult, 0},
        {ShaderStage::Fragment, &pixelResult, PixelPushOffset(vertexPush, target)}
    }};

    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, WaveSize, WaveSize, std::nullopt, std::nullopt};
    state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, Pixels.size(), 0xe4u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.depth = AgcDriver::Graphics::DepthTarget{reinterpret_cast<std::uintptr_t>(DepthMemory.data()), 0, {Width, Height}, VK_FORMAT_D32_SFLOAT, SurfaceDepth, 0};
    state.depthTest = true;
    state.depthWrite = false;
    state.depthCompare = VK_COMPARE_OP_LESS;
    state.depthBias = true;
    state.depthBiasConstant = frontBias;
    state.depthBiasPerFace = frontBias != backBias;
    state.backDepthBiasConstant = backBias;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(Vertices.size()), 0, 1, 0, false};
    device.Draw(state, draw, shaders);
    device.WaitIdle();
}

void Check(bool frontDrawn, bool backDrawn, std::string_view what) {
    for (const auto& [x, y, front] : std::array<std::tuple<std::uint32_t, std::uint32_t, bool>, 2>{{{4, 4, true}, {20, 4, false}}}) {
        const bool drawn = front ? frontDrawn : backDrawn;
        const auto expected = drawn ? std::byte{255} : Kept;
        const auto actual = Pixels[(y * Width + x) * 4u];
        Require(actual == expected, std::string(what) + ": the " + (front ? "front" : "back") + "-facing triangle at (" + std::to_string(x) + ", " + std::to_string(y) + ") is " + std::to_string(static_cast<unsigned>(actual)) + ", expected " + (drawn ? "drawn" : "depth-rejected"));
    }
}

class GuestBlock {
public:
    static constexpr std::size_t BlockBytes = 65536;

    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "cannot allocate a guest block");
        std::memset(block, 0, BlockBytes);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }

    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;

    std::uint8_t* Data() const { return block; }
    std::uint64_t Address() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block)); }

private:
    std::uint8_t* block = nullptr;
};

void DepthDrawRecipe(AgcDriver::VulkanDevice& device) {
    constexpr std::uint32_t TargetWidth = 128;
    constexpr std::uint32_t TargetHeight = 128;
    const AgcDriver::Graphics::ColorTargetLayout colorLayout(TargetWidth, TargetHeight, AgcDriver::Graphics::ColorTileMode::RenderTarget, 4);

    const GuestBlock guestVertices;
    const GuestBlock guestTarget;
    alignas(256) static std::array<float, TargetWidth * TargetHeight> RecipeDepthMemory{};
    alignas(256) static std::array<float, TargetWidth * TargetHeight> RecipeAlternateDepthMemory{};

    std::memset(guestTarget.Data(), static_cast<int>(Kept), GuestBlock::BlockBytes);
    RecipeDepthMemory.fill(1.0f);
    RecipeAlternateDepthMemory.fill(1.0f);

    auto* targetVertices = reinterpret_cast<std::array<float, 4>*>(guestVertices.Data());
    for (std::size_t index = 0; index < Corners.size(); ++index) {
        targetVertices[index] = {Corners[index][0] * 2.0f / TargetWidth - 1.0f, 1.0f - Corners[index][1] * 2.0f / TargetHeight, SurfaceDepth, 1.0f};
    }
    const auto target = device.Target();

    std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexBuffer = BufferDescriptor(targetVertices, 16u, static_cast<std::uint32_t>(Corners.size()));
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {WaveSize, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    const auto vertexResult = ShaderRecompiler::Recompile(vertex);
    const auto vertexPush = static_cast<std::uint32_t>(vertexResult.pushConstants.size());

    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX) | ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionY);
    pixel.posX = true;
    pixel.posY = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    const std::vector<std::uint32_t> pixelUserData(8, 0u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(PixelCode.data()), std::as_bytes(std::span(PixelCode))}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(PixelCode.data()), PixelCode, 0, {}},
        {WaveSize, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory},
        target,
        PixelPushLayout(vertexPush, target)
    };
    fragment.useCache = false;
    const auto pixelResult = ShaderRecompiler::Recompile(fragment);
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &vertexResult, 0},
        {ShaderStage::Fragment, &pixelResult, PixelPushOffset(vertexPush, target)}
    }};

    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, WaveSize, WaveSize, std::nullopt, std::nullopt};
    state.color = {guestTarget.Address(), {TargetWidth, TargetHeight}, VK_FORMAT_R8G8B8A8_UNORM, colorLayout.Bytes(), 0xe4u, AgcDriver::Graphics::ColorTileMode::RenderTarget};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.depth = AgcDriver::Graphics::DepthTarget{reinterpret_cast<std::uintptr_t>(RecipeDepthMemory.data()), 0, {TargetWidth, TargetHeight}, VK_FORMAT_D32_SFLOAT, SurfaceDepth, 0};
    state.depthTest = true;
    state.depthWrite = false;
    state.depthCompare = VK_COMPARE_OP_LESS;
    state.depthBias = true;
    state.depthBiasConstant = -Bias;
    state.depthBiasPerFace = false;
    state.backDepthBiasConstant = -Bias;
    state.renderExtent = {TargetWidth, TargetHeight};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(TargetHeight), static_cast<float>(TargetWidth), -static_cast<float>(TargetHeight), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {TargetWidth, TargetHeight}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(Corners.size()), 0, 1, 0, false};

    std::shared_ptr<const AgcDriver::DrawRecipe> recipe;
    device.Draw(state, draw, shaders, {}, &recipe);
    device.WaitIdle();
    Require(recipe != nullptr, "a direct draw with depth target must produce a recipe");
    Require(recipe->hasDepth, "the recipe must record depth target presence");
    Require(recipe->targetViews.size() == 2, "the recipe must record color and depth views");

    const auto outcome = device.DrawFromRecipe(state, draw, shaders, {}, recipe);
    device.WaitIdle();
    Require(outcome == AgcDriver::RecipeOutcome::Recorded, "drawing with depth from recipe must record successfully");

    auto mutatedAddressState = state;
    mutatedAddressState.depth = AgcDriver::Graphics::DepthTarget{reinterpret_cast<std::uintptr_t>(RecipeAlternateDepthMemory.data()), 0, {TargetWidth, TargetHeight}, VK_FORMAT_D32_SFLOAT, SurfaceDepth, 0};
    const auto rejectedAddressOutcome = device.DrawFromRecipe(mutatedAddressState, draw, shaders, {}, recipe);
    Require(rejectedAddressOutcome == AgcDriver::RecipeOutcome::Rebuild, "a recipe with depth must be rejected when depth target address changes");

    auto removedDepthState = state;
    removedDepthState.depth.reset();
    const auto rejectedDepthOutcome = device.DrawFromRecipe(removedDepthState, draw, shaders, {}, recipe);
    Require(rejectedDepthOutcome == AgcDriver::RecipeOutcome::Rebuild, "a recipe with depth must be rejected when depth target is removed");
}

// buffer_load_dword v0, off, s[0:3], 0; s_waitcnt vmcnt(0); exp mrt0 v0, v0, v0, v0 done vm: every
// pixel gets the first dword of the buffer the V# in the pixel shader's user data names.
alignas(256) constexpr std::array<std::uint32_t, 8> ConstantPixelCode{
    0xe0300000, 0x80000000, 0xbf8c3f70, 0xf800180f, 0x00000000, 0xbf810000, 0xbf810000, 0xbf810000,
};

std::array<std::uint32_t, 4> ConstantDescriptor(std::uint64_t address, std::uint32_t bytes) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_MOVED_RECIPES): the recipe of a draw whose pixel shader reads
// its color through a read-only V#. The first draw builds the template and a recipe; a draw whose
// V# moved to other bytes is a template hit with a moved buffer (the content key leaves its
// address out), which builds a recipe only under the switch. Under it a hit of either recipe
// renders the bytes its own V# names, misses while they are behind a pending GPU write, a queued
// label store or pending storage-image results (and records once they settled), and retires a
// depth surface made over its color target since the recipe was built. Without the switch only
// the refusal of the second recipe is checked.
void MovedBufferDrawRecipe(AgcDriver::VulkanDevice& device) {
    using AgcDriver::RecipeOutcome;
    const bool movedRecipes = std::getenv("APS5_EXPERIMENT_MOVED_RECIPES") != nullptr;
    constexpr std::uint32_t TargetWidth = 128;
    constexpr std::uint32_t TargetHeight = 128;
    constexpr std::uint32_t ConstantBytes = 16;
    constexpr std::size_t SlotBytes = 256;
    const AgcDriver::Graphics::ColorTargetLayout colorLayout(TargetWidth, TargetHeight, AgcDriver::Graphics::ColorTileMode::RenderTarget, 4);
    Require(colorLayout.Bytes() == GuestBlock::BlockBytes, "the moved-buffer target does not fill its block");

    const GuestBlock guestVertices;
    const GuestBlock guestTarget;
    const GuestBlock guestOther;
    const GuestBlock guestConstants;
    const auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    if (AgcDriver::Graphics::HostImportFor(context, guestConstants.Address(), GuestBlock::BlockBytes) == nullptr) {
        std::puts("moved-buffer draw recipe test skipped: the constants cannot be read in place");
        return;
    }
    std::memset(guestTarget.Data(), static_cast<int>(Kept), GuestBlock::BlockBytes);
    std::memset(guestOther.Data(), static_cast<int>(Kept), GuestBlock::BlockBytes);
    // One constant per slot: A and B for the two draws, C for a hit at a third place, P behind a
    // pending GPU write, L under a label store.
    enum Slot : std::size_t { SlotA, SlotB, SlotC, SlotP, SlotL, SlotCount };
    constexpr std::array<float, SlotCount> Values{0.2f, 0.6f, 1.0f, 0.4f, 0.0f};
    constexpr float LabelValue = 0.8f;
    const auto slotAddress = [&](std::size_t slot) { return guestConstants.Address() + slot * SlotBytes; };
    for (std::size_t slot = 0; slot < SlotCount; ++slot) std::memcpy(guestConstants.Data() + slot * SlotBytes, &Values[slot], sizeof(float));

    auto* triangle = reinterpret_cast<std::array<float, 4>*>(guestVertices.Data());
    triangle[0] = {-1.0f, -1.0f, 0.5f, 1.0f};
    triangle[1] = {3.0f, -1.0f, 0.5f, 1.0f};
    triangle[2] = {-1.0f, 3.0f, 0.5f, 1.0f};
    const auto target = device.Target();

    std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexBuffer = BufferDescriptor(triangle, 16u, 3u);
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {WaveSize, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    const auto vertexResult = ShaderRecompiler::Recompile(vertex);
    const auto vertexPush = static_cast<std::uint32_t>(vertexResult.pushConstants.size());

    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.inputAddr = ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionX) | ShaderRecompiler::PixelInputBit(ShaderRecompiler::PixelInput::PositionY);
    pixel.posX = true;
    pixel.posY = true;
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    std::vector<std::uint32_t> pixelUserData(8, 0u);
    const auto constant = ConstantDescriptor(slotAddress(SlotA), ConstantBytes);
    std::copy(constant.begin(), constant.end(), pixelUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(ConstantPixelCode.data()), std::as_bytes(std::span(ConstantPixelCode))}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(ConstantPixelCode.data()), ConstantPixelCode, 0, {}},
        {WaveSize, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory},
        target,
        PixelPushLayout(vertexPush, target)
    };
    fragment.useCache = false;
    const auto pixelResult = ShaderRecompiler::Recompile(fragment);
    const auto guestBuffers = std::ranges::find_if(pixelResult.bindings, [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers && binding.count == 1u && binding.guestDescriptor.size() == 4u; });
    Require(guestBuffers != pixelResult.bindings.end() && !guestBuffers->bufferWritten.empty() && !guestBuffers->bufferWritten.front(), "the constant pixel shader does not bind its V# as one read-only guest buffer");
    const auto bindingIndex = static_cast<std::size_t>(guestBuffers - pixelResult.bindings.begin());
    // The same compiled variant with the V# moved to each slot (or to the other target's memory),
    // as the driver hands a draw whose descriptor words moved: only the address bits change.
    std::vector<ShaderRecompiler::RecompileResult> pixels;
    const auto moveTo = [&](std::uint64_t address) {
        auto moved = pixelResult;
        auto& words = moved.bindings[bindingIndex].guestDescriptor;
        words[0] = static_cast<std::uint32_t>(address);
        words[1] = (words[1] & 0xffff0000u) | static_cast<std::uint32_t>((address >> 32u) & 0xffffu);
        return moved;
    };
    pixels.reserve(SlotCount + 1);
    for (std::size_t slot = 0; slot < SlotCount; ++slot) pixels.push_back(moveTo(slotAddress(slot)));
    pixels.push_back(moveTo(guestOther.Address()));
    const auto shadersAt = [&](std::size_t index) {
        return std::array<AgcDriver::Graphics::CompiledShader, 2>{{
            {ShaderStage::Vertex, &vertexResult, 0},
            {ShaderStage::Fragment, &pixels[index], PixelPushOffset(vertexPush, target)}
        }};
    };
    constexpr std::size_t AtOther = SlotCount;

    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, WaveSize, WaveSize, std::nullopt, std::nullopt};
    state.color = {guestTarget.Address(), {TargetWidth, TargetHeight}, VK_FORMAT_R8G8B8A8_UNORM, colorLayout.Bytes(), 0xe4u, AgcDriver::Graphics::ColorTileMode::RenderTarget};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.renderExtent = {TargetWidth, TargetHeight};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(TargetHeight), static_cast<float>(TargetWidth), -static_cast<float>(TargetHeight), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {TargetWidth, TargetHeight}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    auto otherState = state;
    otherState.color.address = guestOther.Address();
    otherState.colors = {otherState.color};
    const AgcDriver::Pm4::DrawParameters draw{0, 3u, 0, 1, 0, false};

    // The target is one color everywhere, so its bytes read the same in any tiling.
    const auto expect = [&](float value, const std::string& what) {
        AgcDriver::Graphics::StorageTexture::FlushPending(guestTarget.Address(), colorLayout.Bytes(), nullptr, "test");
        device.WaitIdle();
        const auto expected = static_cast<int>(value * 255.0f + 0.5f);
        for (std::size_t offset = 0; offset < colorLayout.Bytes(); offset += 509) {
            const auto actual = static_cast<int>(guestTarget.Data()[offset]);
            Require(actual + 1 >= expected && actual <= expected + 1, what + ": the target holds " + std::to_string(actual) + " at byte " + std::to_string(offset) + ", expected " + std::to_string(expected));
        }
    };
    const auto hit = [&](const std::shared_ptr<const AgcDriver::DrawRecipe>& recipe, std::size_t index) {
        return device.DrawFromRecipe(state, draw, shadersAt(index), {}, recipe);
    };
    // The refusals go to DrawWithRecipe itself, which names its miss (DrawFromRecipe only answers
    // Rebuild): a failed template proof or a lost target must not pass for a refused buffer.
    const auto refused = [&](const std::shared_ptr<const AgcDriver::DrawRecipe>& recipe, std::size_t index) {
        return AgcDriver::Graphics::DrawWithRecipe(context, state, draw, shadersAt(index), {}, *recipe).miss == AgcDriver::Graphics::DrawRecipeMiss::MovedRefused;
    };

    std::shared_ptr<const AgcDriver::DrawRecipe> recipeA;
    device.Draw(state, draw, shadersAt(SlotA), {}, &recipeA);
    expect(Values[SlotA], "the draw that built the template");
    if (recipeA == nullptr) {
        std::puts("moved-buffer draw recipe test skipped: the template is not reusable on this device");
        return;
    }
    std::shared_ptr<const AgcDriver::DrawRecipe> recipeB;
    device.Draw(state, draw, shadersAt(SlotB), {}, &recipeB);
    expect(Values[SlotB], "the template hit whose buffer moved");
    // A recipe over the first draw's template is the moved hit's; a build of its own (the hit
    // refused) would carry another template.
    const auto sharesTemplate = [&] {
        const auto shared = recipeA->templateRef.lock();
        return shared != nullptr && recipeB != nullptr && recipeB->templateRef.lock() == shared;
    };
    if (!movedRecipes) {
        Require(!sharesTemplate(), "a template hit whose read-only buffer moved built a draw recipe without APS5_EXPERIMENT_MOVED_RECIPES");
        return;
    }
    Require(sharesTemplate(), "a template hit whose read-only buffer moved built no draw recipe over the template");

    Require(hit(recipeA, SlotC) == RecipeOutcome::Recorded, "the first draw's recipe refused a moved buffer");
    expect(Values[SlotC], "a recipe hit whose buffer moved from where the recipe's draw read it");
    Require(hit(recipeB, SlotA) == RecipeOutcome::Recorded, "the moved draw's recipe refused the template's own buffer");
    expect(Values[SlotA], "a recipe hit whose buffer is back where the template read it");

    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        auto* recorder = AgcDriver::Graphics::Recorder::Active();
        Require(recorder != nullptr, "no recorder is active");
        recorder->NotePendingWrite(slotAddress(SlotP), ConstantBytes);
    }
    Require(refused(recipeA, SlotP), "a recipe hit did not refuse a moved buffer behind a pending GPU write");
    device.WaitIdle();
    Require(hit(recipeA, SlotP) == RecipeOutcome::Recorded, "a recipe hit refused a moved buffer whose GPU write completed");
    expect(Values[SlotP], "a recipe hit after the moved buffer's GPU write completed");

    Require(hit(recipeA, SlotB) == RecipeOutcome::Recorded, "a recipe hit refused a settled moved buffer");
    const auto label = std::bit_cast<std::array<std::byte, 4>>(LabelValue);
    Require(device.WriteLabelOnGpu(slotAddress(SlotL), label, 1, 0, false) == 0, "the label store over the moved buffer was not recorded on the GPU");
    Require(refused(recipeA, SlotL), "a recipe hit did not refuse a moved buffer under a queued label store");
    device.WaitIdle();
    Require(hit(recipeA, SlotL) == RecipeOutcome::Recorded, "a recipe hit refused a moved buffer whose label store landed");
    expect(LabelValue, "a recipe hit reading the label's value");

    device.Draw(otherState, draw, shadersAt(SlotA));
    Require(refused(recipeA, AtOther), "a recipe hit did not refuse a moved buffer under pending storage-image results");
    AgcDriver::Graphics::StorageTexture::FlushPending(guestOther.Address(), GuestBlock::BlockBytes, nullptr, "test");
    device.WaitIdle();
    Require(hit(recipeA, AtOther) == RecipeOutcome::Recorded, "a recipe hit refused a moved buffer whose storage-image results were stored");
    // The other target's stored bytes (A's color in every byte) read as a float far below 1/255.
    expect(0.0f, "a recipe hit reading the stored results");

    const AgcDriver::Graphics::DepthTarget overColor{guestTarget.Address(), 0, {TargetWidth, TargetHeight}, VK_FORMAT_D32_SFLOAT, 1.0f, 0};
    AgcDriver::Graphics::DepthSurfaceView(context, overColor);
    Require(AgcDriver::Graphics::DepthSurfaceOverlaps(device.Device(), guestTarget.Address(), colorLayout.Bytes()), "no depth surface was made over the color target");
    Require(hit(recipeA, SlotB) == RecipeOutcome::Recorded, "a recipe hit refused after a depth surface was made over its target");
    Require(!AgcDriver::Graphics::DepthSurfaceOverlaps(device.Device(), guestTarget.Address(), colorLayout.Bytes()), "a recipe hit kept the depth surface made over its color target");
    expect(Values[SlotB], "the recipe hit that retired the depth surface");
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Draw(*device, -Bias, Bias);
        Check(true, false, "front faces pulled closer, back faces pushed away");
        Draw(*device, Bias, -Bias);
        Check(false, true, "front faces pushed away, back faces pulled closer");
        Draw(*device, -Bias, -Bias);
        Check(true, true, "both faces pulled closer");
        Draw(*device, -Bias, 0.0f);
        Check(true, false, "only front faces biased");
        // Before DepthDrawRecipe, which leaves its target's results pending over the blocks it
        // frees: a block allocated after it can be given the same memory, and the moved-buffer
        // test reads its blocks' bytes. It stores its own results before its blocks go.
        MovedBufferDrawRecipe(*device);
        DepthDrawRecipe(*device);
        std::puts("per-face depth bias tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
