#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <thread>
#include <utility>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "SDL.h"
#include "SDL_vulkan.h"
#include "Decoder/Png.hpp"
#include "prx/libSceVideoOut/include/PadInput.hpp"
#include "prx/libSceVideoOut/include/MouseInput.hpp"
#include "prx/libSceAgcDriver/Execution/include/AspectFit.hpp"
#include "prx/libSceVideoOut/include/KeyboardInput.hpp"
#include "prx/libSceVideoOut/include/ImeTextInput.hpp"
#include "prx/libScePad/include/PadState.hpp"
#include "prx/libkernel/Equeue/Equeue.hpp"
#include "prx/libkernel/Time/include/Time.hpp"
#include "prx/libSceVideoOut/include/VideoOutDriver.hpp"
#if APS5_ENABLE_TIMING_LOG
#include "prx/libSceVideoOut/include/FrameTimingLog.hpp"
#endif
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostHdr.hpp"
#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/PreciseWait.hpp"
#include "prx/libc/include/Shutdown.hpp"

namespace {

constexpr std::uint64_t FLIP_ROOM_WAIT_MICROS = 60000000;
constexpr std::uint64_t PRESENT_POLL_MICROS = 10000;
// APS5_LOADING_SPLASH: while no flip came for this long, the present loop shows the splash itself.
constexpr auto SPLASH_IDLE_AFTER = std::chrono::milliseconds(100);
constexpr auto SPLASH_IDLE_INTERVAL = std::chrono::milliseconds(33);
// A covered flip this recent when the splash ended is the frame whose probe ended it, and is shown
// again; after a later end (input, time limit) the window is cleared instead.
constexpr auto SPLASH_COVERED_RECENT = std::chrono::milliseconds(1000);

// APS5_LOADING_SPLASH: Enter, Space, Escape (without modifiers, not repeated) or a controller button
// pressed while the window has focus ends the splash.
bool splashDismissal(const SDL_Event& event, SDL_Window* window) {
    if (window == nullptr || SDL_GetKeyboardFocus() != window) return false;
    if (event.type == SDL_CONTROLLERBUTTONDOWN) return true;
    if (event.type != SDL_KEYDOWN || event.key.repeat != 0 || (event.key.keysym.mod & (KMOD_CTRL | KMOD_ALT | KMOD_GUI)) != 0) return false;
    const auto scancode = event.key.keysym.scancode;
    return scancode == SDL_SCANCODE_RETURN || scancode == SDL_SCANCODE_SPACE || scancode == SDL_SCANCODE_ESCAPE;
}

bool sameDisplayBuffer(const AgcDriver::DisplayBuffer& a, const AgcDriver::DisplayBuffer& b) {
    return a.address == b.address && a.pixelFormat == b.pixelFormat && a.width == b.width && a.height == b.height && a.tilingMode == b.tilingMode && a.pitchInPixel == b.pitchInPixel && a.dccAddress == b.dccAddress && a.dccClearColor == b.dccClearColor && a.alignment == b.alignment;
}

void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(std::string("VideoOut: ") + reason);
}

std::vector<const char*> windowExtensions(SDL_Window* window) {
    unsigned extensionCount = 0;
    if (!SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, nullptr)) throw std::runtime_error(std::string("SDL_Vulkan_GetInstanceExtensions failed: ") + SDL_GetError());
    std::vector<const char*> extensions(extensionCount);
    if (!SDL_Vulkan_GetInstanceExtensions(window, &extensionCount, extensions.data())) throw std::runtime_error(std::string("SDL_Vulkan_GetInstanceExtensions failed: ") + SDL_GetError());
    extensions.resize(extensionCount);
    return extensions;
}

VkSurfaceKHR createWindowSurface(void* context, VkInstance instance) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(static_cast<SDL_Window*>(context), instance, &surface)) throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface failed: ") + SDL_GetError());
    return surface;
}

// APS5_LOADING_SPLASH (libSceAgcDriver LoadingSplash.hpp): hands the title's key art,
// sce_sys/pic0.png halved while it is wider than 1920, to the loading splash. Without it the splash
// shows its panel alone.
void loadSplashPicture() {
    try {
        std::ifstream file(ResolvePath_nid_no_patch("/app0/sce_sys/pic0.png"), std::ios::binary);
        if (!file) return;
        const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        auto image = Decoder::Png::Decode(bytes);
        if (!image || image->width == 0 || image->height == 0 || image->pixels.size() != static_cast<std::size_t>(image->width) * image->height * 4) {
            std::fprintf(stderr, "[splash] sce_sys/pic0.png could not be decoded\n");
            return;
        }
        auto width = image->width;
        auto height = image->height;
        auto pixels = std::move(image->pixels);
        while (width > 1920 && width % 2 == 0 && height % 2 == 0) {
            const auto halfWidth = width / 2;
            const auto halfHeight = height / 2;
            std::vector<std::uint8_t> half(static_cast<std::size_t>(halfWidth) * halfHeight * 4);
            for (std::uint32_t y = 0; y < halfHeight; ++y) {
                const auto* top = pixels.data() + static_cast<std::size_t>(2 * y) * width * 4;
                const auto* bottom = top + static_cast<std::size_t>(width) * 4;
                auto* out = half.data() + static_cast<std::size_t>(y) * halfWidth * 4;
                for (std::uint32_t x = 0; x < halfWidth * 4; ++x) {
                    const auto column = (x / 4) * 8 + x % 4;
                    out[x] = static_cast<std::uint8_t>((top[column] + top[column + 4] + bottom[column] + bottom[column + 4] + 2) / 4);
                }
            }
            pixels = std::move(half);
            width = halfWidth;
            height = halfHeight;
        }
        AgcDriverSetLoadingSplashPicture_nid_postfix(width, height, pixels.data(), pixels.size());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[splash] sce_sys/pic0.png is unavailable: %s\n", error.what());
    }
}

void windowDrawableSize(void* context, std::uint32_t* width, std::uint32_t* height) {
    if ((SDL_GetWindowFlags(static_cast<SDL_Window*>(context)) & SDL_WINDOW_MINIMIZED) != 0) {
        *width = 0;
        *height = 0;
        return;
    }
    int drawableWidth = 0;
    int drawableHeight = 0;
    SDL_Vulkan_GetDrawableSize(static_cast<SDL_Window*>(context), &drawableWidth, &drawableHeight);
    *width = drawableWidth > 0 ? static_cast<std::uint32_t>(drawableWidth) : 0;
    *height = drawableHeight > 0 ? static_cast<std::uint32_t>(drawableHeight) : 0;
}

void checkConfig(const VideoOutConfig& cfg) {
    cfg.Check();
}

void resolveFlipBuffer(FlipRequest& request, const VideoOutConfig& cfg) {
    if (request.index < 0) {
        request.width = cfg.width;
        request.height = cfg.height;
        return;
    }
    request.buffer = cfg.buffers[request.index];
    if (!request.buffer.Occupied()) {
        static std::array<std::atomic<bool>, VIDEO_OUT_BUFFER_NUM_MAX> reported{};
        if (!reported[request.index].exchange(true)) std::fprintf(stderr, "[videoout] GPU flip of unregistered buffer %d on handle %u presents black\n", request.index, request.outputHandle);
        request.unregistered = true;
        request.width = cfg.width;
        request.height = cfg.height;
        return;
    }
    require(request.buffer.groupIndex < VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX, "invalid buffer group");
    if (!request.groupCaptured) request.group = cfg.groups[request.buffer.groupIndex];
    require(request.group.occupied, "buffer group is not registered");
    require(request.buffer.dataAddress != 0, "null registered buffer address");
    static_cast<void>(DescribeVideoOutBuffer(request.buffer, request.group));
    request.width = request.group.attribute.width;
    request.height = request.group.attribute.height;
}

class RenderingWait final : public AgcDriver::IRenderingWait {
    std::shared_ptr<VideoOutConfig> _config;
    std::uint32_t _index;
    std::uint64_t _ticket;
public:
    RenderingWait(std::shared_ptr<VideoOutConfig> config, std::uint32_t index, std::uint64_t ticket)
        : _config(std::move(config)), _index(index), _ticket(ticket) {}
    void Wait() override {
        std::unique_lock lock(_config->mutex);
        _config->vblankCond.wait(lock, _config->shutdownToken, [&] {
            return _config->failure || _config->Closed() ||
                _config->bufferReuse[_index].IsComplete(_ticket);
        });
        _config->CheckAlive();
    }
};

class VideoOutput final : public AgcDriver::IVideoOutput {
public:
    VideoOutput(std::shared_ptr<VideoOutConfig> config, std::shared_ptr<FlipQueue> requests) : cfg(std::move(config)), queue(std::move(requests)) {}

    std::shared_ptr<AgcDriver::IRenderingWait> CaptureRenderingWait(std::uint32_t index) override {
        std::lock_guard lock(cfg->mutex);
        checkConfig(*cfg);
        require(index < VIDEO_OUT_BUFFER_NUM_MAX && cfg->buffers[index].Occupied(), "wait buffer is not registered");
        return std::make_shared<RenderingWait>(cfg, index, cfg->bufferReuse[index].Capture());
    }

    void WaitForFlipRoom() override {
        std::unique_lock queueLock(queue->mutex);
        const bool room = queue->changed.WaitUntil(queueLock, TimedWait::DeadlineNanos(FLIP_ROOM_WAIT_MICROS), [&] {
            return queue->failure || queue->stopping || cfg->shutdownToken.stop_requested() || queue->reservations.load() < VIDEO_OUT_FLIP_QUEUE_CAPACITY;
        });
        if (queue->failure) std::rethrow_exception(queue->failure);
        if (queue->stopping || cfg->shutdownToken.stop_requested()) throw ProcessShutdown{};
        require(room, "flip queue stayed full for 60 s: the presenter is not completing flips");
    }

    std::shared_ptr<AgcDriver::IFlipRequest> Reserve(const AgcDriver::FlipInfo& info) override { return Reserve(info, false); }
    std::shared_ptr<AgcDriver::IFlipRequest> TryReserve(const AgcDriver::FlipInfo& info) override { return Reserve(info, true); }

    std::shared_ptr<AgcDriver::IFlipRequest> Reserve(const AgcDriver::FlipInfo& info, bool limited) {
        // All flip modes are presented at the next vsync.
        require(info.mode >= VIDEO_OUT_FLIP_MODE_VSYNC && info.mode <= 6, "unsupported flip mode");
        require(info.index >= VIDEO_OUT_BUFFER_INDEX_BLACK && info.index < VIDEO_OUT_BUFFER_NUM_MAX, "invalid flip index");
        auto request = std::make_shared<FlipRequest>();
        request->cfg = cfg;
        request->queue = queue;
        request->index = info.index;
        request->outputHandle = info.handle;
        request->flipMode = static_cast<int>(info.mode);
        request->flipArg = info.argument;
        std::lock_guard queueLock(queue->mutex);
        if (queue->failure) std::rethrow_exception(queue->failure);
        if (queue->stopping || cfg->shutdownToken.stop_requested()) throw ProcessShutdown{};
        if (limited && queue->reservations.load() >= VIDEO_OUT_FLIP_QUEUE_CAPACITY) return {};
        std::lock_guard lock(cfg->mutex);
        checkConfig(*cfg);
        request->generation = cfg->generation;
        request->flipRate = cfg->flipRate;
        if (info.index >= 0) {
            if (limited) require(cfg->buffers[info.index].Occupied(), "EOP flip buffer is not registered");
            request->reuseTicket = cfg->bufferReuse[info.index].Reserve();
        }
        ++queue->reservations;
        ++cfg->flipStatus.flipPendingNum;
        if (info.index >= 0) ++cfg->bufferPending[info.index];
        if (info.index >= 0 && cfg->buffers[info.index].Occupied() && cfg->buffers[info.index].groupIndex < VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX && cfg->groups[cfg->buffers[info.index].groupIndex].occupied) {
            request->group = cfg->groups[cfg->buffers[info.index].groupIndex];
            request->groupCaptured = true;
        }
        request->reserved = true;
        return request;
    }

    void Fail(std::exception_ptr error) noexcept override {
        if (!error) std::terminate();
        {
            std::lock_guard lock(queue->mutex);
            if (!queue->failure) queue->failure = error;
        }
        {
            std::lock_guard lock(cfg->mutex);
            if (!cfg->failure) cfg->failure = error;
            cfg->vblankCond.notify_all();
        }
        queue->changed.NotifyAll();
    }

private:
    std::shared_ptr<VideoOutConfig> cfg;
    std::shared_ptr<FlipQueue> queue;
};

}

FlipRequest::~FlipRequest() {
    if (!reserved) return;
    std::lock_guard lock(cfg->mutex);
    ReleaseLocked();
}

void FlipRequest::ReleaseLocked() noexcept {
    if (!reserved || terminal) return;
    --cfg->flipStatus.flipPendingNum;
    --queue->reservations;
    if (index >= 0) {
        --cfg->bufferPending[index];
        cfg->bufferReuse[index].Complete(reuseTicket);
    }
    terminal = true;
    cfg->vblankCond.notify_all();
}

void FlipRequest::Cancel() noexcept {
    {
        std::lock_guard lock(cfg->mutex);
        ReleaseLocked();
    }
    queue->changed.NotifyAll();
}

void FlipRequest::GpuReady(const std::shared_ptr<AgcDriver::FrameTiming>& frameTiming) {
    require(frameTiming != nullptr, "missing frame timing");
    timing = frameTiming;
    {
        AgcDriver::PerformanceContext timingContext(timing.get());
        AgcDriver::PerformanceTimer readiness("VideoOut.Readiness");
        std::lock_guard queueLock(queue->mutex);
        if (queue->failure) std::rethrow_exception(queue->failure);
        if (queue->stopping || cfg->shutdownToken.stop_requested()) throw ProcessShutdown{};
        std::lock_guard lock(cfg->mutex);
        cfg->CheckAlive();
        if (cfg->Closed()) {
            ReleaseLocked();
            return;
        }
        require(reserved && !ready && !terminal && cfg->generation == generation, "invalid flip readiness transition");
        resolveFlipBuffer(*this, *cfg);
        readiness.Mark("locks_validate");
        queuedAt = AgcDriver::FrameTiming::Clock::now();
        queue->requests.push_back(shared_from_this());
        ready = true;
    }
    queue->changed.NotifyAll();
    // The worker is done once the request is queued: hardware does not stall the command processor
    // on a flip, and the title observes completion through flipPendingNum, which processFlip drops
    // once the frame's GPU work and its presentation have completed (see Driver::Present). A
    // presentation failure reaches the worker through ReportFailure at its next packet. Debug aid:
    // APS5_SYNC_FLIP=1 parks the worker until the presenter is done, as before.
    static const bool syncFlip = std::getenv("APS5_SYNC_FLIP") != nullptr;
    if (!syncFlip) return;
    std::unique_lock lock(cfg->mutex);
    cfg->vblankCond.wait(lock, cfg->shutdownToken, [&] { return gpuComplete || cfg->failure || cfg->closing; });
    checkConfig(*cfg);
    require(gpuComplete, "flip preparation did not complete on GPU");
}

void FlipRequest::Fail(std::exception_ptr error) noexcept {
    if (!error) std::terminate();
    std::lock_guard lock(cfg->mutex);
    if (!cfg->failure) cfg->failure = error;
    ReleaseLocked();
    queue->changed.NotifyAll();
}

VideoOutDriver& VideoOutDriver::Get() {
    static VideoOutDriver instance;
    return instance;
}

VideoOutDriver::VideoOutDriver() {
    SDL_SetHint(SDL_HINT_WINDOWS_DPI_AWARENESS, "permonitorv2");
#ifdef _WIN32
    SDL_SetHintWithPriority(SDL_HINT_DIRECTINPUT_ENABLED, "0", SDL_HINT_DEFAULT);
#endif
#ifdef __linux__
    SDL_SetHint(SDL_HINT_VIDEODRIVER, "wayland,x11");
#endif
    if (SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) < 0) {
        throw std::runtime_error(std::string("SDL_InitSubSystem(VIDEO | GAMECONTROLLER) failed: ") + SDL_GetError());
    }
    try {
        AgcDriverWaitIdle_nid_postfix();
        std::promise<void> started;
        auto attached = started.get_future();
        presentThread = std::jthread([this, &started](std::stop_token token) {
            try {
                AgcDriverRaiseWorkerThreadPriority_nid_postfix("presenter");
            } catch (...) {
                started.set_exception(std::current_exception());
                return;
            }
            presentLoop(token, started);
        });
        attached.get();
        vblankThread = std::jthread([this](std::stop_token token) { vblankLoop(token); });
        LibcRegisterShutdown_nid_postfix([] { VideoOutDriver::Get().Shutdown(); });
    } catch (...) {
        if (vblankThread.joinable()) {
            vblankThread.request_stop();
            flipQueue->changed.NotifyAll();
            vblankThread.join();
        }
        if (presentThread.joinable()) {
            presentThread.request_stop();
            flipQueue->changed.NotifyAll();
            presentThread.join();
        }
        SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER);
        throw;
    }
}

VideoOutDriver::~VideoOutDriver() {
    if (!stopped) Shutdown();
}

void VideoOutDriver::Shutdown() {
    require(std::this_thread::get_id() != presentThread.get_id(), "presentation thread cannot stop itself");
    std::lock_guard shutdownLock(shutdownMutex);
    if (stopped) return;
    LibcRequestShutdown_nid_postfix();
    {
        std::lock_guard queueLock(flipQueue->mutex);
        flipQueue->stopping = true;
    }
    presentThread.request_stop();
    vblankThread.request_stop();
    flipQueue->changed.NotifyAll();
    presentThread.join();
    vblankThread.join();
    {
        std::lock_guard lock(mutex);
        for (int handle = 1; handle < VIDEO_OUT_NUM_MAX; ++handle) {
            if (outputs[handle]) close(handle);
        }
    }
    SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER);
    stopped = true;
    std::lock_guard lock(flipQueue->mutex);
    if (flipQueue->failure) std::rethrow_exception(flipQueue->failure);
}

int VideoOutDriver::Open(int busType) {
    std::lock_guard lock(mutex);
    {
        std::lock_guard queueLock(flipQueue->mutex);
        if (flipQueue->failure) std::rethrow_exception(flipQueue->failure);
        if (flipQueue->stopping || LibcShutdownToken_nid_postfix().stop_requested()) throw ProcessShutdown{};
    }
    const int handle = busType + 1;
    require(handle > 0 && handle < VIDEO_OUT_NUM_MAX, "invalid output bus");
    auto previous = contexts[handle];
    uint64_t generation = 1;
    if (previous) {
        std::lock_guard cfgLock(previous->mutex);
        if (previous->opened) return -1;
        require(previous->generation != std::numeric_limits<uint64_t>::max(), "port generation overflow");
        generation = previous->generation + 1;
    }
    auto cfg = std::make_shared<VideoOutConfig>();
    cfg->generation = generation;
    cfg->busType = busType;
    cfg->opened = true;
    cfg->flipStatus.flipArg = -1;
    cfg->flipStatus.currentBuffer = -1;
    auto output = std::make_shared<VideoOutput>(cfg, flipQueue);
    AgcDriverRegisterVideoOutput_nid_postfix(static_cast<uint32_t>(handle), output);
    contexts[handle] = std::move(cfg);
    outputs[handle] = std::move(output);
    return handle;
}

bool VideoOutDriver::Close(int handle) {
    std::lock_guard lock(mutex);
    if (LibcShutdownToken_nid_postfix().stop_requested()) throw ProcessShutdown{};
    return close(handle);
}

bool VideoOutDriver::close(int handle) {
    require(handle > 0 && handle < VIDEO_OUT_NUM_MAX && outputs[handle] != nullptr, "invalid close handle");
    AgcDriverUnregisterVideoOutput_nid_postfix(static_cast<uint32_t>(handle), outputs[handle]);
    outputs[handle].reset();
    auto cfg = contexts[handle];
    std::lock_guard cfgLock(cfg->mutex);
    cfg->opened = false;
    cfg->closing = true;
    const auto removeEvents = [](const auto& events, int kind) {
        for (const auto& event : events) {
            const auto result = EqueueDeleteEvent_nid_postfix(event.eq, static_cast<uintptr_t>(kind), EVFILT_VIDEO_OUT);
            require(result == EQUEUE_OK || result == SCE_KERNEL_ERROR_EBADF || result == SCE_KERNEL_ERROR_ENOENT, "event removal during close failed");
        }
    };
    removeEvents(cfg->flipEvents, VIDEO_OUT_EVENT_FLIP);
    removeEvents(cfg->vblankEvents, VIDEO_OUT_EVENT_VBLANK);
    removeEvents(cfg->preVblankEvents, VIDEO_OUT_EVENT_PRE_VBLANK_START);
    removeEvents(cfg->outputModeEvents, VIDEO_OUT_EVENT_SET_MODE);
    removeEvents(cfg->vrrStatusEvents, VIDEO_OUT_EVENT_VRR_STATUS);
    cfg->flipEvents.clear();
    cfg->vblankEvents.clear();
    cfg->preVblankEvents.clear();
    cfg->outputModeEvents.clear();
    cfg->vrrStatusEvents.clear();
    cfg->vblankCond.notify_all();
    flipQueue->changed.NotifyAll();
    return true;
}

std::shared_ptr<VideoOutConfig> VideoOutDriver::GetConfig(int handle) {
    {
        std::lock_guard lock(flipQueue->mutex);
        if (flipQueue->failure) std::rethrow_exception(flipQueue->failure);
        if (flipQueue->stopping || LibcShutdownToken_nid_postfix().stop_requested()) throw ProcessShutdown{};
    }
    std::lock_guard lock(mutex);
    require(handle > 0 && handle < VIDEO_OUT_NUM_MAX && contexts[handle] != nullptr, "invalid output handle");
    auto cfg = contexts[handle];
    std::lock_guard cfgLock(cfg->mutex);
    checkConfig(*cfg);
    return cfg;
}

bool VideoOutDriver::IsOpen(int handle) {
    return GetConfig(handle) != nullptr;
}

bool VideoOutDriver::HasConfig(int handle) {
    std::shared_ptr<VideoOutConfig> cfg;
    {
        std::lock_guard lock(mutex);
        if (handle <= 0 || handle >= VIDEO_OUT_NUM_MAX || contexts[handle] == nullptr) return false;
        cfg = contexts[handle];
    }
    std::lock_guard cfgLock(cfg->mutex);
    return cfg->opened && !cfg->closing;
}

int VideoOutDriver::SubmitFlip(int handle, int index, int flipMode, int64_t flipArg) {
    if (LibcShutdownToken_nid_postfix().stop_requested()) throw ProcessShutdown{};
    // A title that does not pace on flipPendingNum can run ahead of the presenter now that the queue
    // worker no longer waits per flip; a full queue is the documented error, not a failure.
    if (flipQueue->reservations.load() >= VIDEO_OUT_FLIP_QUEUE_CAPACITY) return VIDEO_OUT_ERROR_FLIP_QUEUE_FULL;
    AgcDriverSubmitFlip_nid_postfix(AgcDriver::FlipInfo{static_cast<std::uint32_t>(handle), index, static_cast<std::uint32_t>(flipMode), flipArg});
    return 0;
}

int VideoOutDriver::SubmitEopFlip(int handle, int index, int flipMode, int64_t flipArg, std::uint32_t* token) {
    std::lock_guard lock(mutex);
    if (handle <= 0 || handle >= VIDEO_OUT_NUM_MAX || !outputs[handle]) return VIDEO_OUT_ERROR_INVALID_HANDLE;
    if (flipMode < VIDEO_OUT_FLIP_MODE_VSYNC || flipMode > 6) return VIDEO_OUT_ERROR_INVALID_VALUE;
    if (index < VIDEO_OUT_BUFFER_INDEX_BLACK || index >= VIDEO_OUT_BUFFER_NUM_MAX) return VIDEO_OUT_ERROR_INVALID_INDEX;
    const auto cfg = contexts[handle];
    {
        std::lock_guard cfgLock(cfg->mutex);
        checkConfig(*cfg);
        if (index >= 0 && !cfg->buffers[index].Occupied()) return VIDEO_OUT_ERROR_INVALID_INDEX;
    }
    const AgcDriver::FlipInfo info{static_cast<std::uint32_t>(handle), index, static_cast<std::uint32_t>(flipMode), flipArg};
    const auto reserved = AgcDriverReserveEopFlip_nid_postfix(info);
    if (reserved == 0) return VIDEO_OUT_ERROR_FLIP_QUEUE_FULL;
    std::memcpy(token, &reserved, sizeof(reserved));
    return 0;
}

void VideoOutDriver::triggerEvents(VideoOutConfig& cfg, int eventKind, void* triggerData) {
    std::vector<EventRegistration>* events = nullptr;
    if (eventKind == VIDEO_OUT_EVENT_FLIP) events = &cfg.flipEvents;
    else if (eventKind == VIDEO_OUT_EVENT_VBLANK) events = &cfg.vblankEvents;
    else if (eventKind == VIDEO_OUT_EVENT_PRE_VBLANK_START) events = &cfg.preVblankEvents;
    else if (eventKind == VIDEO_OUT_EVENT_SET_MODE) events = &cfg.outputModeEvents;
    else throw std::runtime_error("VideoOut: unknown event kind");
    for (auto it = events->begin(); it != events->end();) {
        require(it->generation == cfg.generation, "stale event registration");
        const auto result = EqueueTriggerEvent_nid_postfix(it->eq, static_cast<uintptr_t>(eventKind), EVFILT_VIDEO_OUT, triggerData);
        if (result == SCE_KERNEL_ERROR_EBADF || result == SCE_KERNEL_ERROR_ENOENT) it = events->erase(it);
        else {
            require(result == EQUEUE_OK, "event delivery failed");
            ++it;
        }
    }
}

bool VideoOutDriver::SupportsHighFrameRate() {
    return GetDisplayRefreshRate() >= 120;
}

int VideoOutDriver::GetDisplayRefreshRate() {
    return displayRefreshRate.load();
}

void VideoOutDriver::updateDisplayRefreshRate() {
    const int display = SDL_GetWindowDisplayIndex(window.Handle());
    if (display < 0) throw std::runtime_error(std::string("SDL_GetWindowDisplayIndex failed: ") + SDL_GetError());
    SDL_DisplayMode mode{};
    if (SDL_GetCurrentDisplayMode(display, &mode) != 0) throw std::runtime_error(std::string("SDL_GetCurrentDisplayMode failed: ") + SDL_GetError());
    displayRefreshRate.store(mode.refresh_rate);
}

void VideoOutDriver::pollHostHdr() {
    hdrProbe.Poll(window.Handle());
    const auto generation = AgcDriver::CurrentHostHdr().generation;
    if (generation == hdrGeneration) return;
    hdrGeneration = generation;
    std::lock_guard lock(mutex);
    for (const auto& cfg : contexts) {
        if (!cfg) continue;
        std::lock_guard cfgLock(cfg->mutex);
        if (!cfg->opened || cfg->closing || cfg->failure) continue;
        triggerEvents(*cfg, VIDEO_OUT_EVENT_SET_MODE, reinterpret_cast<void*>(cfg->outputMode));
    }
}

void VideoOutDriver::vblankEnd(std::uint64_t tick) {
    std::lock_guard lock(mutex);
    for (const auto& cfg : contexts) {
        if (!cfg) continue;
        std::lock_guard cfgLock(cfg->mutex);
        if (!cfg->opened || cfg->failure) continue;
        if (cfg->outputMode != VIDEO_OUT_OUTPUT_MODE_119_88HZ && tick % 2 != 0) continue;
        require(cfg->vblankStatus.count != std::numeric_limits<uint64_t>::max(), "vblank counter overflow");
        ++cfg->preVblankStatus.count;
        cfg->preVblankStatus.processTime = sceKernelGetProcessTime();
        cfg->preVblankStatus.processTimeCounter = sceKernelGetProcessTimeCounter();
        triggerEvents(*cfg, VIDEO_OUT_EVENT_PRE_VBLANK_START, reinterpret_cast<void*>(cfg->preVblankStatus.count));
        ++cfg->vblankStatus.count;
        cfg->vblankStatus.processTime = sceKernelGetProcessTime();
        cfg->vblankStatus.processTimeCounter = sceKernelGetProcessTimeCounter();
        triggerEvents(*cfg, VIDEO_OUT_EVENT_VBLANK, reinterpret_cast<void*>(cfg->vblankStatus.count));
        cfg->vblankCond.notify_all();
    }
}

void VideoOutDriver::processFlip(FlipRequest& req) {
    AgcDriver::PerformanceContext timingContext(req.timing.get());
    AgcDriver::PerformanceTimer timing("VideoOut.Flip");
    {
        std::unique_lock lock(req.cfg->mutex);
        timing.Mark("config_mutex_wait");
        req.cfg->CheckAlive();
        if (req.cfg->Closed()) {
            req.ReleaseLocked();
            return;
        }
        require(req.cfg->outputMode != VIDEO_OUT_OUTPUT_MODE_119_88HZ || SupportsHighFrameRate(), "display no longer supports 119.88 Hz output");
        require(req.ready && !req.terminal && req.generation == req.cfg->generation, "stale or incomplete flip request");
        const auto interval = static_cast<uint64_t>(req.flipRate + 1);
        require(req.cfg->lastFlipVblank <= std::numeric_limits<uint64_t>::max() - interval, "flip interval overflow");
        const auto target = req.cfg->lastFlipVblank + interval;
        timing.Mark("validate");
        req.cfg->vblankCond.wait(lock, req.cfg->shutdownToken, [&] { return req.cfg->vblankStatus.count >= target || req.cfg->failure || req.cfg->closing; });
        timing.Mark("vblank_wait");
        req.cfg->CheckAlive();
        if (req.cfg->Closed()) {
            req.ReleaseLocked();
            return;
        }
        req.cfg->lastFlipVblank = req.cfg->vblankStatus.count;
    }
    require(req.width != 0 && req.height != 0 && req.width <= static_cast<uint32_t>(std::numeric_limits<int>::max()) && req.height <= static_cast<uint32_t>(std::numeric_limits<int>::max()), "invalid window dimensions");
    if (req.cfg->busType == VIDEO_OUT_BUS_TYPE_OVERLAY) {
        std::lock_guard lock(req.cfg->mutex);
        req.cfg->CheckAlive();
        if (!req.cfg->Closed()) {
            require(!req.terminal && !req.gpuComplete, "invalid GPU completion transition");
            req.gpuComplete = true;
        }
    } else {
        window.Ensure(req.width, req.height);
        const auto extensions = windowExtensions(window.Handle());
        const AgcDriver::PresentationWindow target{window.Handle(), extensions, &createWindowSurface, &windowDrawableSize, req.width, req.height, req.timing};
        timing.Mark("window_prepare");
        const auto gpuReady = [](void* context) {
            auto& request = *static_cast<FlipRequest*>(context);
            std::lock_guard lock(request.cfg->mutex);
            request.cfg->CheckAlive();
            if (request.cfg->Closed()) return;
            require(!request.terminal && !request.gpuComplete, "invalid GPU completion transition");
            request.gpuComplete = true;
            request.cfg->vblankCond.notify_all();
        };
        if (req.index >= 0 && !req.unregistered) {
            const auto display = DescribeVideoOutBuffer(req.buffer, req.group);
            AgcDriverPresentBuffer_nid_postfix(target, display, gpuReady, &req);
        } else {
            AgcDriverPresentClear_nid_postfix(target, req.unregistered || req.index == VIDEO_OUT_BUFFER_INDEX_BLACK, gpuReady, &req);
        }
        timing.Mark("present");
        window.UpdateTitle();
        timing.Mark("window_title");
    }
    std::lock_guard lock(req.cfg->mutex);
    timing.Mark("completion_mutex_wait");
    req.cfg->CheckAlive();
    if (req.cfg->Closed()) {
        req.ReleaseLocked();
        return;
    }
    require(!req.terminal && req.cfg->generation == req.generation, "flip cancelled during presentation");
    require(req.gpuComplete, "flip submitted before GPU completion");
    require(req.cfg->flipStatus.count != std::numeric_limits<uint64_t>::max(), "flip counter overflow");
    triggerEvents(*req.cfg, VIDEO_OUT_EVENT_FLIP, reinterpret_cast<void*>(req.flipArg));
    ++req.cfg->flipStatus.count;
    req.cfg->flipStatus.processTime = sceKernelGetProcessTime();
    req.cfg->flipStatus.processTimeCounter = sceKernelGetProcessTimeCounter();
    req.cfg->flipStatus.flipArg = req.flipArg;
    const auto previous = req.cfg->flipStatus.currentBuffer;
    if (previous >= 0) {
        auto& label = req.cfg->bufferLabels[previous];
        AgcDriverStoreGuestLabel_nid_postfix(&label, 0);
    }
    req.cfg->flipStatus.currentBuffer = req.index;
    req.cfg->width = req.width;
    req.cfg->height = req.height;
    --req.cfg->flipStatus.flipPendingNum;
    --req.queue->reservations;
    if (req.index >= 0) {
        --req.cfg->bufferPending[req.index];
        req.cfg->bufferReuse[req.index].Complete(req.reuseTicket);
    }
    req.terminal = true;
    req.cfg->vblankCond.notify_all();
    timing.Mark("notify_game");
}

void VideoOutDriver::presentLoop(std::stop_token token, std::promise<void>& started) {
    try {
        window.Ensure(VIDEO_OUT_DEFAULT_WIDTH, VIDEO_OUT_DEFAULT_HEIGHT);
        updateDisplayRefreshRate();
        hdrProbe.Poll(window.Handle());
        const auto extensions = windowExtensions(window.Handle());
        AgcDriverAttachWindow_nid_postfix({window.Handle(), extensions, &createWindowSurface, &windowDrawableSize, VIDEO_OUT_DEFAULT_WIDTH, VIDEO_OUT_DEFAULT_HEIGHT, nullptr});
        hdrGeneration = AgcDriver::CurrentHostHdr().generation;
    } catch (...) {
        hdrProbe.Release();
        window.Destroy();
        started.set_exception(std::current_exception());
        return;
    }
    started.set_value();
    std::shared_ptr<FlipRequest> current;
#if APS5_ENABLE_TIMING_LOG
    std::unique_ptr<FrameTimingLog> timingLog;
#endif
    try {
#if APS5_ENABLE_TIMING_LOG
        timingLog = std::make_unique<FrameTimingLog>();
#endif
        PadInput padInput;
        MouseInput mouseInput;
        KeyboardInput keyboardInput;
        ImeTextInput imeTextInput;
        // APS5_LOADING_SPLASH: the picture loads beside the loop; until the splash ends, a gap in the
        // game's flips is filled with presents of the splash alone (black flips, see LoadingSplash.hpp).
        // The splash covers the frame whose probe ends it. `splashShown`: the window's last present
        // was the splash (an idle present or a flip covered by it). Once the splash ended and no flip
        // follows, `covered` (the last buffer flipped under the splash) is presented again if it is
        // recent and still registered as it was, and the window is cleared otherwise.
        std::jthread splashLoader;
        if (AgcDriverLoadingSplashShowing_nid_postfix()) splashLoader = std::jthread([] { loadSplashPicture(); });
        const auto splashExtensions = windowExtensions(window.Handle());
        auto lastFlip = std::chrono::steady_clock::now();
        auto lastSplash = std::chrono::steady_clock::time_point{};
        struct CoveredFlip {
            std::shared_ptr<VideoOutConfig> cfg;
            int index = 0;
            AgcDriver::DisplayBuffer display{};
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::chrono::steady_clock::time_point at;
        };
        std::optional<CoveredFlip> covered;
        bool splashShown = false;
        while (!token.stop_requested()) {
            {
                std::unique_lock lock(flipQueue->mutex);
                flipQueue->changed.WaitUntil(lock, TimedWait::DeadlineNanos(PRESENT_POLL_MICROS), [&] { return token.stop_requested() || flipQueue->failure || !flipQueue->requests.empty(); });
                if (flipQueue->failure) std::rethrow_exception(flipQueue->failure);
                if (token.stop_requested()) break;
                if (!flipQueue->requests.empty()) {
                    current = std::move(flipQueue->requests.front());
                    flipQueue->requests.pop_front();
                }
            }

            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    LibcRequestExit_nid_postfix(0);
                    throw ProcessShutdown{};
                }
                if (splashDismissal(event, window.Handle()) && AgcDriverLoadingSplashShowing_nid_postfix()) AgcDriverFinishLoadingSplash_nid_postfix("input");
                padInput.HandleEvent(event, window);
                if (window.Handle() != nullptr) {
                    mouseInput.HandleEvent(event, SDL_GetWindowID(window.Handle()));
                    keyboardInput.HandleEvent(event, SDL_GetWindowID(window.Handle()));
                    imeTextInput.HandleEvent(event, SDL_GetWindowID(window.Handle()));
                }
            }
            padInput.Update();
            updateDisplayRefreshRate();
            pollHostHdr();
            mouseInput.UpdateCapture(!padInput.MouseModeEnabled());
            if (current) {
                require(current->timing != nullptr, "missing presentation timing");
                const auto dequeued = AgcDriver::FrameTiming::Clock::now();
                current->timing->Add(current->timing->Get("VideoOut", "queue"), dequeued - current->queuedAt);
                const bool underSplash = AgcDriverLoadingSplashShowing_nid_postfix();
                processFlip(*current);
                if (current->cfg->busType != VIDEO_OUT_BUS_TYPE_OVERLAY) {
                    splashShown = underSplash;
                    covered.reset();
                    if (underSplash && current->index >= 0 && !current->unregistered) covered = CoveredFlip{current->cfg, current->index, DescribeVideoOutBuffer(current->buffer, current->group), current->width, current->height, std::chrono::steady_clock::now()};
                }
                int windowWidth = 0;
                int windowHeight = 0;
                if (window.Handle() != nullptr) SDL_GetWindowSize(window.Handle(), &windowWidth, &windowHeight);
                if (windowWidth > 0 && windowHeight > 0) {
                    const auto presented = AgcDriver::ComputeContainSize_nid_postfix(current->width, current->height, static_cast<std::uint32_t>(windowWidth), static_cast<std::uint32_t>(windowHeight), true);
                    mouseInput.SetPresentation(current->width, current->height, presented.width, presented.height);
                }
                const auto finished = AgcDriver::FrameTiming::Clock::now();
                AgcDriver::FrameTiming::Clock::duration interval{};
                {
                    std::lock_guard lock(current->cfg->mutex);
                    const auto previous = current->cfg->lastTimingFlip;
                    if (previous != AgcDriver::FrameTiming::Clock::time_point{}) interval = finished - previous;
                    current->cfg->lastTimingFlip = finished;
                }
#if APS5_ENABLE_TIMING_LOG
                timingLog->Enqueue(current->timing->Capture(current->outputHandle, current->index, current->flipArg, finished, interval));
#endif
            }
            if (current) {
                current.reset();
                flipQueue->changed.NotifyAll();
                lastFlip = std::chrono::steady_clock::now();
            } else if (window.Handle() != nullptr && std::chrono::steady_clock::now() - lastFlip >= SPLASH_IDLE_AFTER) {
                const auto now = std::chrono::steady_clock::now();
                static int splashContext = 0;
                const AgcDriver::PresentationWindow splash{window.Handle(), splashExtensions, &createWindowSurface, &windowDrawableSize, VIDEO_OUT_DEFAULT_WIDTH, VIDEO_OUT_DEFAULT_HEIGHT, nullptr};
                if (AgcDriverLoadingSplashShowing_nid_postfix()) {
                    if (now - lastSplash >= SPLASH_IDLE_INTERVAL && AgcDriverTryPresentSplash_nid_postfix(splash)) {
                        lastSplash = now;
                        splashShown = true;
                    }
                } else if (splashShown) {
                    const auto flip = std::exchange(covered, std::nullopt);
                    bool live = false;
                    if (flip && now - flip->at <= SPLASH_COVERED_RECENT) {
                        std::lock_guard lock(flip->cfg->mutex);
                        if (!flip->cfg->Closed() && !flip->cfg->failure) {
                            const auto& buffer = flip->cfg->buffers[static_cast<std::size_t>(flip->index)];
                            live = buffer.Occupied() && static_cast<std::size_t>(buffer.groupIndex) < flip->cfg->groups.size() && flip->cfg->groups[static_cast<std::size_t>(buffer.groupIndex)].occupied && sameDisplayBuffer(DescribeVideoOutBuffer(buffer, flip->cfg->groups[static_cast<std::size_t>(buffer.groupIndex)]), flip->display);
                        }
                    }
                    if (live) {
                        window.Ensure(flip->width, flip->height);
                        const AgcDriver::PresentationWindow target{window.Handle(), splashExtensions, &createWindowSurface, &windowDrawableSize, flip->width, flip->height, nullptr};
                        AgcDriverPresentBuffer_nid_postfix(target, flip->display, [](void*) {}, &splashContext);
                        splashShown = false;
                    } else {
                        splashShown = !AgcDriverTryPresentSplash_nid_postfix(splash);
                    }
                }
            }
        }
    } catch (const ProcessShutdown&) {
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[videoout] presentation failed: %s\n", error.what());
        std::fflush(stderr);
        std::terminate();
    } catch (...) {
        std::fprintf(stderr, "[videoout] presentation failed with a non-standard exception\n");
        std::fflush(stderr);
        std::terminate();
    }
#if APS5_ENABLE_TIMING_LOG
    try {
        timingLog->Finish();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[videoout] frame timing log failed: %s\n", error.what());
        std::fflush(stderr);
        std::terminate();
    }
#endif
    current.reset();
    {
        std::list<std::shared_ptr<FlipRequest>> cancelled;
        {
            std::lock_guard lock(flipQueue->mutex);
            cancelled.swap(flipQueue->requests);
        }
    }
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (...) {
        std::lock_guard lock(flipQueue->mutex);
        if (!flipQueue->failure) flipQueue->failure = std::current_exception();
    }
    hdrProbe.Release();
    window.Destroy();
}

void VideoOutDriver::vblankLoop(std::stop_token token) {
    using Frame = std::chrono::duration<int64_t, std::ratio<1001, 120000>>;
    const auto start = std::chrono::steady_clock::now();
    try {
        for (int64_t frame = 1; !token.stop_requested(); ++frame) {
            const auto next = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(Frame(frame));
            const auto remaining = std::chrono::duration_cast<std::chrono::microseconds>(next - std::chrono::steady_clock::now()).count();
            if (remaining > 0) PreciseSleepUs(static_cast<unsigned long long>(remaining));
            {
                std::lock_guard lock(flipQueue->mutex);
                if (token.stop_requested() || flipQueue->failure) return;
            }
            vblankEnd(static_cast<std::uint64_t>(frame));
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[videoout] vblank failed: %s\n", error.what());
        std::fflush(stderr);
        std::terminate();
    } catch (...) {
        std::fprintf(stderr, "[videoout] vblank failed with a non-standard exception\n");
        std::fflush(stderr);
        std::terminate();
    }
}
