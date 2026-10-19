#include "prx/libSceAgcDriver/Execution/include/LoadingSplash.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace Splash = AgcDriver::LoadingSplash;

void require(bool condition, const std::string& reason) {
    if (!condition) throw std::runtime_error(reason);
}

// A probe of `format` whose first `lit` samples hold `bright` and the rest a dithered near-black.
std::vector<std::byte> probe(VkFormat format, std::size_t lit, std::uint32_t bright) {
    const auto bytes = Splash::ProbeTexelBytes(format);
    std::vector<std::byte> texels(static_cast<std::size_t>(Splash::ProbeSamples) * bytes);
    for (std::size_t i = 0; i < Splash::ProbeSamples; ++i) {
        const std::uint32_t level = i < lit ? bright : static_cast<std::uint32_t>(i % 13);
        auto* texel = texels.data() + i * bytes;
        if (bytes == 4 && format != VK_FORMAT_A2B10G10R10_UNORM_PACK32) {
            for (int channel = 0; channel < 3; ++channel) texel[channel] = static_cast<std::byte>(level);
            texel[3] = std::byte{0xff};
        } else if (bytes == 4) {
            const std::uint32_t word = level | (level << 10) | (level << 20) | (3u << 30);
            std::memcpy(texel, &word, sizeof(word));
        } else {
            // Half floats: `level` 0 is 0.0, anything else is 0.5 (0x3800) when lit and 0.001 (0x1419) when not.
            const std::uint16_t half = level == 0 ? 0 : (i < lit ? 0x3800 : 0x1419);
            const std::uint16_t channels[4] = {half, half, half, 0x3c00};
            std::memcpy(texel, channels, sizeof(channels));
        }
    }
    return texels;
}

void classificationTests() {
    require(!Splash::ContentFrame(probe(VK_FORMAT_B8G8R8A8_UNORM, 0, 0), VK_FORMAT_B8G8R8A8_UNORM), "a dithered black frame reads as content");
    require(!Splash::ContentFrame(probe(VK_FORMAT_B8G8R8A8_UNORM, 1, 255), VK_FORMAT_B8G8R8A8_UNORM), "one bright sample reads as content");
    require(Splash::ContentFrame(probe(VK_FORMAT_B8G8R8A8_UNORM, 2, 255), VK_FORMAT_B8G8R8A8_UNORM), "two bright samples (a line of text) read as black");
    require(!Splash::ContentFrame(probe(VK_FORMAT_R8G8B8A8_SRGB, Splash::ProbeSamples, 39), VK_FORMAT_R8G8B8A8_SRGB), "a uniform 39/255 frame reads as content");
    require(Splash::ContentFrame(probe(VK_FORMAT_R8G8B8A8_SRGB, Splash::ProbeSamples, 40), VK_FORMAT_R8G8B8A8_SRGB), "a uniform 40/255 frame reads as black");
    require(!Splash::ContentFrame(probe(VK_FORMAT_A2B10G10R10_UNORM_PACK32, 0, 0), VK_FORMAT_A2B10G10R10_UNORM_PACK32), "a black HDR10 frame reads as content");
    require(!Splash::ContentFrame(probe(VK_FORMAT_A2B10G10R10_UNORM_PACK32, Splash::ProbeSamples, 243), VK_FORMAT_A2B10G10R10_UNORM_PACK32), "an HDR10 frame below 4.3 nits reads as content");
    require(Splash::ContentFrame(probe(VK_FORMAT_A2B10G10R10_UNORM_PACK32, Splash::ProbeSamples, 244), VK_FORMAT_A2B10G10R10_UNORM_PACK32), "an HDR10 frame at 4.3 nits reads as black");
    require(!Splash::ContentFrame(probe(VK_FORMAT_R16G16B16A16_SFLOAT, 0, 0), VK_FORMAT_R16G16B16A16_SFLOAT), "a near-black scRGB frame reads as content");
    require(Splash::ContentFrame(probe(VK_FORMAT_R16G16B16A16_SFLOAT, 50, 1), VK_FORMAT_R16G16B16A16_SFLOAT), "a lit scRGB frame reads as black");
    require(Splash::ProbeTexelBytes(VK_FORMAT_R5G6B5_UNORM_PACK16) == 0, "an unreadable format has a probe size");
    require(Splash::ContentFrame({}, VK_FORMAT_R5G6B5_UNORM_PACK16), "a frame the probe cannot read keeps the splash");
}

void probeGridTests() {
    for (const VkExtent2D extent : {VkExtent2D{1920, 1080}, VkExtent2D{3840, 2160}, VkExtent2D{1, 1}, VkExtent2D{17, 5}}) {
        std::set<std::pair<std::int32_t, std::int32_t>> distinct;
        for (std::uint32_t i = 0; i < Splash::ProbeSamples; ++i) {
            const auto offset = Splash::ProbeOffset(i, extent);
            require(offset.x >= 0 && offset.y >= 0 && offset.x < static_cast<std::int32_t>(extent.width) && offset.y < static_cast<std::int32_t>(extent.height) && offset.z == 0, "a probe sample lies outside " + std::to_string(extent.width) + "x" + std::to_string(extent.height));
            distinct.insert({offset.x, offset.y});
        }
        if (extent.width >= Splash::ProbeColumns && extent.height >= Splash::ProbeRows) require(distinct.size() == Splash::ProbeSamples, "probe samples repeat in " + std::to_string(extent.width) + "x" + std::to_string(extent.height));
    }
    const auto first = Splash::ProbeOffset(0, {1920, 1080});
    const auto last = Splash::ProbeOffset(Splash::ProbeSamples - 1, {1920, 1080});
    require(first.x == 30 && first.y == 30 && last.x == 1890 && last.y == 1050, "the probe grid is not centered in its cells");
}

void panelTests() {
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < 400; ++step) {
        const auto content = Splash::Content(start + std::chrono::milliseconds(step * 37));
        require(content.above.size() == 1 && content.above.front().rfind("LOADING", 0) == 0 && content.above.front().size() <= 10, "the panel's first line is not LOADING with up to three dots");
        require(content.below.size() == 1 && content.below.front().size() >= 3 && content.below.front().substr(content.below.front().size() - 2) == " S", "the panel's last line is not the seconds");
        require(content.frameTimes.size() == AgcDriver::OverlayStatistics::FrameHistory, "the panel's bar does not span the graph");
        std::size_t full = 0;
        for (const auto value : content.frameTimes) full += value == 1.0 ? 1 : 0;
        require(full >= 1 && full <= 24, "the moving bar is " + std::to_string(full) + " bars wide");
        // Every glyph exists and the panel fits a small window.
        static_cast<void>(AgcDriver::LayOutOverlay(content, {640, 360}, AgcDriver::OverlayCorner::BottomRight));
    }
}

void lifecycleTests() {
    require(Splash::Enabled() && Splash::Showing(), "APS5_LOADING_SPLASH=1 does not show the splash");
    require(Splash::CurrentPicture().rgba == nullptr, "a picture is shown before one was set");
    Splash::SetPicture(2, 1, std::vector<std::uint8_t>(7));
    require(Splash::CurrentPicture().rgba == nullptr, "a picture of the wrong size was taken");
    Splash::SetPicture(2, 1, std::vector<std::uint8_t>(8, 9));
    const auto picture = Splash::CurrentPicture();
    require(picture.width == 2 && picture.height == 1 && picture.rgba != nullptr && picture.rgba->size() == 8, "the picture was not kept");
    Splash::NoteProbe(probe(VK_FORMAT_B8G8R8A8_UNORM, 0, 0), VK_FORMAT_B8G8R8A8_UNORM);
    require(Splash::Showing(), "a black frame ended the splash");
    Splash::NoteProbe(probe(VK_FORMAT_B8G8R8A8_UNORM, Splash::ProbeSamples, 200), VK_FORMAT_B8G8R8A8_UNORM);
    require(!Splash::Showing() && Splash::CurrentPicture().rgba == nullptr, "a frame with content did not end the splash and drop its picture");
    Splash::SetPicture(2, 1, std::vector<std::uint8_t>(8, 9));
    require(Splash::CurrentPicture().rgba == nullptr, "a picture set after the end was kept");
}

}

int main() {
#ifdef _WIN32
    _putenv("APS5_LOADING_SPLASH=1");
#else
    setenv("APS5_LOADING_SPLASH", "1", 1);
#endif
    try {
        classificationTests();
        probeGridTests();
        panelTests();
        lifecycleTests();
        std::cout << "loading splash classification, probe grid, panel and lifecycle tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
