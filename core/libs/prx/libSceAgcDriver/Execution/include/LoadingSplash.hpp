#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_LOADINGSPLASH_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_LOADINGSPLASH_HPP

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "prx/libSceAgcDriver/Execution/include/PerformanceOverlay.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

// APS5_LOADING_SPLASH=1: while a title boots and shows nothing but black frames, the window shows
// its key art (sce_sys/pic0.png, handed over by VideoOut) with a loading panel instead. Each
// presented frame is sampled on a grid of swapchain texels (read back when its slot retires); the
// first frame with content ends the splash for good, and so do the first key, button or click and
// APS5_LOADING_SPLASH_MAX_SECONDS. Presents with no game frame (before the first flip, black flips)
// show it too. Off by default.
namespace AgcDriver::LoadingSplash {

inline constexpr std::uint32_t ProbeColumns = 32;
inline constexpr std::uint32_t ProbeRows = 18;
inline constexpr std::uint32_t ProbeSamples = ProbeColumns * ProbeRows;

struct Picture {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // RGBA8, tightly packed rows.
    std::shared_ptr<const std::vector<std::uint8_t>> rgba;
};

bool Enabled();
// Enabled, not finished, and within APS5_LOADING_SPLASH_MAX_SECONDS (default 300) of the first call.
bool Showing();
void Finish(const char* reason);

// The bytes one probe texel takes in the swapchain's format, or 0 for a format the probe cannot
// read (the splash then ends at the first game frame).
std::uint32_t ProbeTexelBytes(VkFormat format);
// The probe's sample positions in a destination of `extent`: centers of a ProbeColumns x ProbeRows grid.
VkOffset3D ProbeOffset(std::uint32_t sample, VkExtent2D extent);
// True when at least two sampled texels have a color channel at 40/255 or more (for HDR10, the PQ
// code of that level at 203 nits). Dithered near-black loading frames stay well below it; a line of
// text on black reaches it.
bool ContentFrame(std::span<const std::byte> texels, VkFormat format);
// Classifies a frame's probe read back after its submission; content ends the splash.
void NoteProbe(std::span<const std::byte> texels, VkFormat format);

void SetPicture(std::uint32_t width, std::uint32_t height, std::vector<std::uint8_t> rgba);
Picture CurrentPicture();

// The loading panel: "LOADING" with cycling dots, a moving bar in the graph area and the seconds
// since the splash began.
OverlayContent Content(std::chrono::steady_clock::time_point now);

}

#endif
