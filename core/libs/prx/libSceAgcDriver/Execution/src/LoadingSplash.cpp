#include "prx/libSceAgcDriver/Execution/include/LoadingSplash.hpp"
#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace AgcDriver::LoadingSplash {

namespace {

using Clock = std::chrono::steady_clock;

Clock::time_point startedAt() {
    static const Clock::time_point started = Clock::now();
    return started;
}

std::chrono::seconds maxDuration() {
    static const std::chrono::seconds limit = [] {
        const char* value = std::getenv("APS5_LOADING_SPLASH_MAX_SECONDS");
        if (value == nullptr) return std::chrono::seconds(900);
        const long parsed = std::strtol(value, nullptr, 10);
        return std::chrono::seconds(parsed > 0 ? parsed : 900);
    }();
    return limit;
}

std::atomic<bool> finished{false};
std::mutex pictureMutex;
Picture picture;

float halfToFloat(std::uint16_t half) {
    const std::uint32_t exponent = (half >> 10) & 0x1fu;
    const std::uint32_t mantissa = half & 0x3ffu;
    const float sign = (half & 0x8000u) != 0 ? -1.0f : 1.0f;
    if (exponent == 0) return sign * std::ldexp(static_cast<float>(mantissa), -24);
    if (exponent == 31) return mantissa == 0 ? sign * 65536.0f : 0.0f;
    return sign * std::ldexp(static_cast<float>(mantissa | 0x400u), static_cast<int>(exponent) - 25);
}

// A texel with a color channel at 40/255 or more: sRGB 40/255 is 2.1% of white, 4.3 nits at the
// 203-nit reference white, PQ code 244 of 1023 on an HDR10 swapchain, and 0.021 in linear scRGB.
bool litTexel(const std::byte* texel, VkFormat format) {
    switch (format) {
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
            // Alpha is the fourth byte of every listed format (the packed ones are little endian).
            return std::max({std::to_integer<unsigned>(texel[0]), std::to_integer<unsigned>(texel[1]), std::to_integer<unsigned>(texel[2])}) >= 40u;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32: {
            std::uint32_t word = 0;
            std::memcpy(&word, texel, sizeof(word));
            return std::max({word & 0x3ffu, (word >> 10) & 0x3ffu, (word >> 20) & 0x3ffu}) >= 244u;
        }
        case VK_FORMAT_R16G16B16A16_SFLOAT: {
            std::uint16_t channels[3] = {};
            std::memcpy(channels, texel, sizeof(channels));
            return std::max({halfToFloat(channels[0]), halfToFloat(channels[1]), halfToFloat(channels[2])}) >= 0.021f;
        }
        default:
            return true;
    }
}

}

bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_LOADING_SPLASH");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool Showing() {
    if (!Enabled() || finished.load(std::memory_order_acquire)) return false;
    if (Clock::now() - startedAt() <= maxDuration()) return true;
    Finish("time limit");
    return false;
}

void Finish(const char* reason) {
    if (finished.exchange(true, std::memory_order_acq_rel)) return;
    const auto seconds = std::chrono::duration<double>(Clock::now() - startedAt()).count();
    std::fprintf(stderr, "[splash] loading splash ended after %.1f s (%s)\n", seconds, reason);
    std::lock_guard lock(pictureMutex);
    picture = {};
}

std::uint32_t ProbeTexelBytes(VkFormat format) {
    switch (format) {
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
            return 4;
        case VK_FORMAT_R16G16B16A16_SFLOAT:
            return 8;
        default:
            return 0;
    }
}

VkOffset3D ProbeOffset(std::uint32_t sample, VkExtent2D extent) {
    const auto column = sample % ProbeColumns;
    const auto row = sample / ProbeColumns;
    const auto x = static_cast<std::uint64_t>(2 * column + 1) * extent.width / (2 * ProbeColumns);
    const auto y = static_cast<std::uint64_t>(2 * row + 1) * extent.height / (2 * ProbeRows);
    return {static_cast<std::int32_t>(std::min<std::uint64_t>(x, extent.width - 1)), static_cast<std::int32_t>(std::min<std::uint64_t>(y, extent.height - 1)), 0};
}

bool ContentFrame(std::span<const std::byte> texels, VkFormat format) {
    const auto bytes = ProbeTexelBytes(format);
    if (bytes == 0) return true;
    const auto samples = texels.size() / bytes;
    std::size_t lit = 0;
    for (std::size_t i = 0; i < samples; ++i) {
        if (litTexel(texels.data() + i * bytes, format)) ++lit;
    }
    return lit >= 2;
}

void NoteProbe(std::span<const std::byte> texels, VkFormat format) {
    if (finished.load(std::memory_order_acquire)) return;
    if (ContentFrame(texels, format)) Finish("the game shows a frame with content");
}

void SetPicture(std::uint32_t width, std::uint32_t height, std::vector<std::uint8_t> rgba) {
    if (width == 0 || height == 0 || rgba.size() != static_cast<std::size_t>(width) * height * 4) return;
    std::lock_guard lock(pictureMutex);
    if (finished.load(std::memory_order_acquire)) return;
    picture = {width, height, std::make_shared<const std::vector<std::uint8_t>>(std::move(rgba))};
}

Picture CurrentPicture() {
    std::lock_guard lock(pictureMutex);
    return picture;
}

OverlayContent Content(Clock::time_point now) {
    const auto elapsed = std::max<std::int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(now - startedAt()).count());
    OverlayContent content;
    content.above.push_back("LOADING" + std::string(static_cast<std::size_t>((elapsed / 400) % 4), '.'));
    // A block of 24 bars sweeping across the 120-bar graph, over a low base line.
    constexpr std::int64_t bars = static_cast<std::int64_t>(OverlayStatistics::FrameHistory);
    constexpr std::int64_t block = 24;
    const auto head = (elapsed / 10) % (bars + block - 1);
    content.frameTimes.resize(OverlayStatistics::FrameHistory);
    for (std::int64_t i = 0; i < bars; ++i) content.frameTimes[static_cast<std::size_t>(i)] = i <= head && i > head - block ? 1.0 : 0.12;
    content.below.push_back(std::to_string(elapsed / 1000) + " S");
    return content;
}

}

extern "C" bool AgcDriverLoadingSplashShowing_nid_postfix() {
    return AgcDriver::LoadingSplash::Showing();
}

extern "C" void AgcDriverFinishLoadingSplash_nid_postfix(const char* reason) {
    AgcDriver::LoadingSplash::Finish(reason != nullptr ? reason : "ended by the host");
}

extern "C" void AgcDriverSetLoadingSplashPicture_nid_postfix(std::uint32_t width, std::uint32_t height, const std::uint8_t* rgba, std::size_t bytes) {
    if (rgba == nullptr) return;
    AgcDriver::LoadingSplash::SetPicture(width, height, std::vector<std::uint8_t>(rgba, rgba + bytes));
}
