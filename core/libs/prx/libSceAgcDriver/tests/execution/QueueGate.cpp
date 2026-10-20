#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/QueueGate.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <SDL_loadso.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// LOCAL EXPERIMENT (PARALLEL-COMMIT-PLAN A3, A4): the queue gate (Graphics::QueueGate). Unit tests
// on gates of their own (fake queue handles where no Vulkan call is made through them; a bare
// device of the test's own for the submitter), then a headless device's device-wide wait under the
// switches the run sets. Arguments run one death test each, or U1 (PARALLEL-COMMIT-PLAN section 5:
// a never-submitted fence's status and wait on this driver, --never-submitted-fence; the pool
// stress, one recording thread against the submitter, --pool-stress=<batches> with the recorder's
// shared pool and --pool-stress-own-pools=<batches> with APS5_EXPERIMENT_SUBMIT_POOLS' own pools;
// the plan's full run is 1000000 batches each).
namespace {

using AgcDriver::Graphics::QueueGate;
using Queue = QueueGate::Queue;

int failures = 0;

void Check(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

bool Asserted() {
    return std::getenv("APS5_ASSERT_GPU_LOCK") != nullptr;
}

QueueGate::Modes PresentModes(int present) {
    QueueGate::Modes modes;
    modes.presentQueue = present;
    return modes;
}

const auto FakePresentQueue = reinterpret_cast<VkQueue>(std::uintptr_t{0x10});

// A family with too few queues keeps the present on the graphics queue, inside the hold.
void PresentQueueFallback() {
    Check(QueueGate::PresentQueueIndex(8, 2) == 2, "a family of 8 queues gives the present index 2 after graphics and settle");
    Check(QueueGate::PresentQueueIndex(3, 2) == 2, "a family of 3 queues gives the present index 2");
    Check(QueueGate::PresentQueueIndex(2, 2) == 0, "a family of 2 queues (graphics, settle) has no present queue");
    Check(QueueGate::PresentQueueIndex(2, 1) == 1, "without a settle queue the present queue is index 1");
    Check(QueueGate::PresentQueueIndex(1, 1) == 0, "a family of 1 queue has no present queue");
    QueueGate after(PresentModes(1));
    Check(after.Gated() && !after.PresentAfterHold(), "without a present queue the present stays in the hold");
    after.SetPresentQueue(FakePresentQueue);
    Check(after.PresentAfterHold(), "=1 with a present queue presents after the hold");
    QueueGate inside(PresentModes(2));
    inside.SetPresentQueue(FakePresentQueue);
    Check(inside.Gated() && !inside.PresentAfterHold(), "=2 presents inside the hold");
    QueueGate off(PresentModes(0));
    off.SetPresentQueue(FakePresentQueue);
    Check(!off.Gated() && !off.PresentAfterHold(), "off, the gate is inert");
}

// A device-wide wait (AllQueues) never overlaps a present (or a graphics submit) of another thread,
// and the gate counts no violation for operations made under their queue's mutex.
void DeviceWideWaitExcludesQueues() {
    QueueGate gate(PresentModes(1));
    gate.SetPresentQueue(FakePresentQueue);
    const auto before = QueueGate::CountsSoFar();
    std::atomic<int> inside{0};
    std::atomic<bool> overlap{false};
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> presents{0}, submits{0}, waits{0};
    const auto operate = [&](Queue queue, std::atomic<std::uint64_t>& count) {
        while (!stop.load(std::memory_order_relaxed)) {
            const auto hold = gate.Lock(queue);
            gate.CheckOp(queue, queue == Queue::Present ? "test present" : "test submit");
            inside.fetch_add(1, std::memory_order_acq_rel);
            for (volatile int spin = 0; spin < 200; spin = spin + 1) {
            }
            inside.fetch_sub(1, std::memory_order_acq_rel);
            count.fetch_add(1, std::memory_order_relaxed);
        }
    };
    std::thread presenter(operate, Queue::Present, std::ref(presents));
    std::thread submitter(operate, Queue::Graphics, std::ref(submits));
    std::thread waiter([&] {
        for (int i = 0; i < 2000; ++i) {
            const auto all = gate.AllQueues();
            if (inside.load(std::memory_order_acquire) != 0) overlap = true;
            for (volatile int spin = 0; spin < 200; spin = spin + 1) {
            }
            if (inside.load(std::memory_order_acquire) != 0) overlap = true;
            waits.fetch_add(1, std::memory_order_relaxed);
        }
    });
    waiter.join();
    stop = true;
    presenter.join();
    submitter.join();
    const auto after = QueueGate::CountsSoFar();
    Check(!overlap, "a queue operation ran inside another thread's device-wide wait");
    Check(presents.load() != 0 && submits.load() != 0 && waits.load() == 2000, "presents, submits and device-wide waits all ran");
    Check(after.unheld == before.unheld && after.overlapped == before.overlapped && after.misordered == before.misordered, "operations under their queue's mutex counted a violation");
    Check(after.deviceWaits - before.deviceWaits == 2000, "every device-wide wait was counted");
    Check(AgcDriver::GuestMemory::QueueMutexesHeld() == 0, "the thread's held-queue bits were left set");
}

// The out-of-date flag a present leaves (from any thread) is taken once, by the next Resize; a
// successful present leaves none, a failure is the caller's to throw.
void OutOfDateTakenOnce() {
    QueueGate gate(PresentModes(1));
    Check(!gate.TakePresentOutOfDate(), "a fresh gate reports an out-of-date swapchain");
    bool noted = false;
    std::thread presenter([&] { noted = gate.NotePresentResult(VK_SUBOPTIMAL_KHR); });
    presenter.join();
    Check(noted, "a suboptimal present reads as a failure");
    Check(gate.TakePresentOutOfDate(), "a suboptimal present left no flag");
    Check(!gate.TakePresentOutOfDate(), "the flag was taken twice");
    Check(gate.NotePresentResult(VK_ERROR_OUT_OF_DATE_KHR) && gate.TakePresentOutOfDate(), "an out-of-date present left no flag");
    Check(gate.NotePresentResult(VK_SUCCESS) && !gate.TakePresentOutOfDate(), "a successful present left a flag");
    Check(!gate.NotePresentResult(VK_ERROR_DEVICE_LOST) && !gate.TakePresentOutOfDate(), "a failed present was not left to the caller");
}

// GpuMutex before a queue mutex is the allowed order; with the switch on and without the assertion
// switch, an operation without its mutex and a mutex out of the fixed order are counted.
void LockOrderAndCounts() {
    QueueGate gate(PresentModes(1));
    gate.SetPresentQueue(FakePresentQueue);
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        const auto hold = gate.Lock(Queue::Present);
        gate.CheckOp(Queue::Present, "present under GpuMutex");
        Check(AgcDriver::GuestMemory::QueueMutexesHeld() == (1u << static_cast<unsigned>(Queue::Present)), "the present bit is set while its mutex is held");
    }
    Check(AgcDriver::GuestMemory::QueueMutexesHeld() == 0, "the present bit outlived its hold");
    {
        // Graphics then settle is the fixed order.
        const auto graphics = gate.Lock(Queue::Graphics);
        const auto settle = gate.Lock(Queue::Settle);
    }
    if (Asserted()) return;
    const auto before = QueueGate::CountsSoFar();
    gate.CheckOp(Queue::Graphics, "a submit without the graphics mutex");
    {
        const auto settle = gate.Lock(Queue::Settle);
        const auto graphics = gate.Lock(Queue::Graphics);
    }
    const auto after = QueueGate::CountsSoFar();
    Check(after.unheld == before.unheld + 1, "an operation without its queue's mutex was not counted");
    Check(after.misordered == before.misordered + 1, "a queue mutex out of the fixed order was not counted");
}

// The device's own device-wide wait (VulkanDevice::WaitIdle, under GpuMutex) under the switches
// the run sets: counted by the gate when it is on, never a violation.
void DeviceWaitIdle() {
    const auto device = OpenVulkanTestDevice();
    if (device == nullptr) return;
    const auto before = QueueGate::CountsSoFar();
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        device->WaitIdle();
    }
    const auto after = QueueGate::CountsSoFar();
    const bool gated = QueueGate::PresentQueueMode() != 0 || QueueGate::SubmitThreadMode() != 0;
    Check(gated ? after.deviceWaits > before.deviceWaits : after.deviceWaits == before.deviceWaits, gated ? "the device's WaitIdle took no device-wide hold" : "the device's WaitIdle took a device-wide hold with the switches unset");
    Check(after.unheld == before.unheld && after.overlapped == before.overlapped && after.misordered == before.misordered, "the device's WaitIdle counted a violation");
}

// A bare Vulkan device: one queue of the first graphics family (a discrete GPU first), its command
// pool, and a Graphics::Context over them; VulkanDevice is not used, its gate would follow the
// switches. The constructor throws without a usable device.
class RawDevice {
public:
    RawDevice() {
#ifdef _WIN32
        library = SDL_LoadObject("vulkan-1.dll");
#else
        library = SDL_LoadObject("libvulkan.so.1");
#endif
        if (library == nullptr) throw std::runtime_error("cannot load Vulkan");
        try {
            instanceProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_LoadFunction(library, "vkGetInstanceProcAddr"));
            if (instanceProc == nullptr) throw std::runtime_error("missing vkGetInstanceProcAddr");
            VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            application.apiVersion = VK_API_VERSION_1_1;
            VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            info.pApplicationInfo = &application;
            AgcDriver::Graphics::Check(function<PFN_vkCreateInstance>("vkCreateInstance")(&info, nullptr, &instance), "vkCreateInstance");
            std::uint32_t count = 0;
            const auto enumerate = function<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
            AgcDriver::Graphics::Check(enumerate(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
            if (count == 0) throw std::runtime_error("no Vulkan device");
            std::vector<VkPhysicalDevice> devices(count);
            AgcDriver::Graphics::Check(enumerate(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
            const auto describe = function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
            context.physical = devices.front();
            for (const auto candidate : devices) {
                VkPhysicalDeviceProperties candidateProperties{};
                describe(candidate, &candidateProperties);
                if (candidateProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                    context.physical = candidate;
                    break;
                }
            }
            describe(context.physical, &properties);
            const auto families = function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
            families(context.physical, &count, nullptr);
            std::vector<VkQueueFamilyProperties> queues(count);
            families(context.physical, &count, queues.data());
            std::uint32_t family = 0;
            while (family < count && (queues[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) ++family;
            if (family == count) throw std::runtime_error("no graphics queue family");
            const float priority = 1.0f;
            VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue.queueFamilyIndex = family;
            queue.queueCount = 1;
            queue.pQueuePriorities = &priority;
            VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            device.queueCreateInfoCount = 1;
            device.pQueueCreateInfos = &queue;
            AgcDriver::Graphics::Check(function<PFN_vkCreateDevice>("vkCreateDevice")(context.physical, &device, nullptr, &context.device), "vkCreateDevice");
            context.deviceProc = function<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
            function<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(context.physical, &context.memory);
            context.limits = properties.limits;
            context.Function<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(context.device, family, 0, &context.queue);
            context.queueFamily = family;
            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool.queueFamilyIndex = family;
            AgcDriver::Graphics::Check(context.Function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(context.device, &pool, nullptr, &context.pool), "vkCreateCommandPool");
        } catch (...) {
            release();
            throw;
        }
    }
    ~RawDevice() { release(); }
    RawDevice(const RawDevice&) = delete;
    RawDevice& operator=(const RawDevice&) = delete;
    const AgcDriver::Graphics::Context& GetContext() const { return context; }
    const VkPhysicalDeviceProperties& Properties() const { return properties; }
    // A gate of the test's own attached to this device.
    void Attach(QueueGate& gate) const {
        gate.Attach(context.device, context.queue, context.Function<PFN_vkQueueSubmit>("vkQueueSubmit"), context.Function<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle"));
    }

private:
    template<typename TFunction>
    TFunction function(const char* name) const {
        const auto result = reinterpret_cast<TFunction>(instanceProc(instance, name));
        if (result == nullptr) throw std::runtime_error(std::string("missing ") + name);
        return result;
    }
    void release() noexcept {
        if (context.device != VK_NULL_HANDLE) {
            context.Function<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(context.device);
            if (context.pool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(context.device, context.pool, nullptr);
            function<PFN_vkDestroyDevice>("vkDestroyDevice")(context.device, nullptr);
            context.device = VK_NULL_HANDLE;
        }
        if (instance != VK_NULL_HANDLE) function<PFN_vkDestroyInstance>("vkDestroyInstance")(instance, nullptr);
        instance = VK_NULL_HANDLE;
        if (library != nullptr) SDL_UnloadObject(library);
        library = nullptr;
    }

    void* library = nullptr;
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    AgcDriver::Graphics::Context context{};
};

std::unique_ptr<RawDevice> OpenRawDevice() {
    try {
        return std::make_unique<RawDevice>();
    } catch (const std::exception& error) {
        if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) throw;
        std::printf("no usable Vulkan device for the submitter tests: %s\n", error.what());
        return nullptr;
    }
}

QueueGate::Modes SubmitModes(int mode, bool pools) {
    QueueGate::Modes modes;
    modes.submitThread = mode;
    modes.submitPools = pools;
    return modes;
}

// An ended, empty command buffer and an unsignaled fence of the raw device, freed at the end.
struct EmptyBatch {
    explicit EmptyBatch(const AgcDriver::Graphics::Context& context) : context(context) {
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = context.pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        AgcDriver::Graphics::Check(context.Function<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(context.device, &allocation, &commands), "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        AgcDriver::Graphics::Check(context.Function<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(commands, &begin), "vkBeginCommandBuffer");
        AgcDriver::Graphics::Check(context.Function<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(commands), "vkEndCommandBuffer");
        VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        AgcDriver::Graphics::Check(context.Function<PFN_vkCreateFence>("vkCreateFence")(context.device, &info, nullptr, &fence), "vkCreateFence");
    }
    ~EmptyBatch() {
        context.Function<PFN_vkWaitForFences>("vkWaitForFences")(context.device, 1, &fence, VK_TRUE, 5'000'000'000ull);
        context.Function<PFN_vkDestroyFence>("vkDestroyFence")(context.device, fence, nullptr);
        context.Function<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(context.device, context.pool, 1, &commands);
    }
    VkResult Status() const { return context.Function<PFN_vkGetFenceStatus>("vkGetFenceStatus")(context.device, fence); }
    QueueGate::Sealed Seal(std::uint64_t serial) { return {&issued, commands, fence, VK_NULL_HANDLE, serial}; }
    AgcDriver::Graphics::Context context;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    std::atomic<bool> issued{false};
};

// The submitter on a bare device: a held FIFO keeps seals unissued (their flags unset, a drain
// waits), a release issues them in serial order, and the submitter thread is joined by
// StopSubmitter. A foreign hold finds the FIFO issued.
void SubmitterBasics(RawDevice& raw) {
    QueueGate gate(SubmitModes(1, false));
    raw.Attach(gate);
    const auto running = QueueGate::RunningSubmitters();
    EmptyBatch first(raw.GetContext()), second(raw.GetContext());
    gate.PauseSubmitter(true);
    gate.Seal(first.Seal(1));
    gate.Seal(second.Seal(2));
    Check(QueueGate::RunningSubmitters() == running + 1, "the first seal started no submitter");
    Check(!first.issued.load() && !second.issued.load() && gate.SubmittedSerial() == 0, "a held submitter issued a seal");
    Check(first.Status() == VK_NOT_READY, "a sealed, unissued batch's fence is signaled");
    std::atomic<bool> released{false};
    std::thread release([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        released = true;
        gate.PauseSubmitter(false);
    });
    {
        const auto foreign = gate.LockForeign("test foreign submit");
        Check(released.load() && gate.SubmittedSerial() == 2 && first.issued.load() && second.issued.load(), "a foreign hold was given before every seal was issued");
    }
    release.join();
    gate.AwaitSubmitted(2);
    gate.StopSubmitter();
    Check(QueueGate::RunningSubmitters() == running, "StopSubmitter left the submitter running");
    // A seal after the stop starts the submitter again.
    EmptyBatch third(raw.GetContext());
    gate.Seal(third.Seal(3));
    gate.AwaitSubmitted(3);
    Check(third.issued.load() && gate.SubmittedSerial() == 3, "a seal after StopSubmitter was not issued");
    gate.StopSubmitter();
}

// U1, the fence: a fence no submission ever signaled, polled and waited (zero and 2 ms) on this
// driver. The poll rule and the await rule keep the driver from ever seeing either; this records
// what it would do (it must not report the fence signaled).
int NeverSubmittedFence() {
    const auto raw = OpenRawDevice();
    if (raw == nullptr) return VulkanTestSkipped;
    const auto& context = raw->GetContext();
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    AgcDriver::Graphics::Check(context.Function<PFN_vkCreateFence>("vkCreateFence")(context.device, &info, nullptr, &fence), "vkCreateFence");
    const auto status = context.Function<PFN_vkGetFenceStatus>("vkGetFenceStatus")(context.device, fence);
    const auto waitFences = context.Function<PFN_vkWaitForFences>("vkWaitForFences");
    const auto zero = waitFences(context.device, 1, &fence, VK_TRUE, 0);
    const auto start = std::chrono::steady_clock::now();
    const auto timed = waitFences(context.device, 1, &fence, VK_TRUE, 2'000'000ull);
    const double waitedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    context.Function<PFN_vkDestroyFence>("vkDestroyFence")(context.device, fence, nullptr);
    const auto& properties = raw->Properties();
    std::printf("[queue] U1: a never-submitted fence on %s (driver version 0x%x): vkGetFenceStatus %d, vkWaitForFences(0) %d, vkWaitForFences(2 ms) %d after %.2f ms\n", properties.deviceName, properties.driverVersion, static_cast<int>(status), static_cast<int>(zero), static_cast<int>(timed), waitedMs);
    Check(status == VK_NOT_READY, "a never-submitted fence did not poll VK_NOT_READY");
    Check(zero == VK_TIMEOUT && timed == VK_TIMEOUT, "a wait on a never-submitted fence did not time out");
    return failures == 0 ? 0 : 1;
}

// U1, the pools: one recording thread (this one, under GpuMutex, a recorder batch per iteration)
// against the submitter issuing the previous ones, with the recorder's shared pool or a pool per
// batch (APS5_EXPERIMENT_SUBMIT_POOLS). Every batch must complete, in order, without a violation.
int PoolStress(std::uint64_t batches, bool pools) {
    const auto raw = OpenRawDevice();
    if (raw == nullptr) return VulkanTestSkipped;
    QueueGate gate(SubmitModes(1, pools));
    raw->Attach(gate);
    auto context = raw->GetContext();
    context.queueGate = &gate;
    const auto before = QueueGate::CountsSoFar();
    std::uint64_t completed = 0;
    std::uint64_t outOfOrder = 0;
    const auto start = std::chrono::steady_clock::now();
    {
        std::optional<AgcDriver::Graphics::Recorder> recorder;
        {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            recorder.emplace(context);
            recorder->Activate();
        }
        const auto barrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        const VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
        for (std::uint64_t index = 0; index < batches; ++index) {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            const auto commands = recorder->Commands();
            barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &memory, 0, nullptr, 0, nullptr);
            recorder->OnComplete([&completed, &outOfOrder, index] {
                if (completed != index) ++outOfOrder;
                ++completed;
            });
            recorder->Submit();
            recorder->Reap();
        }
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        recorder->Sync();
        recorder.reset();
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto after = QueueGate::CountsSoFar();
    std::printf("[queue] U1 pool stress: %llu batches with %s in %.2f s (%.1f us per batch), %llu issued by the submitter, %llu polls found a batch unissued, %llu awaits waited\n", static_cast<unsigned long long>(batches), pools ? "a pool per batch (APS5_EXPERIMENT_SUBMIT_POOLS=1)" : "the shared pool", seconds,
        batches != 0 ? seconds * 1e6 / static_cast<double>(batches) : 0.0, static_cast<unsigned long long>(gate.SubmittedSerial()), static_cast<unsigned long long>(after.unissuedPolls - before.unissuedPolls), static_cast<unsigned long long>(after.waitedAwaits - before.waitedAwaits));
    Check(completed == batches && outOfOrder == 0, "a stressed batch did not complete, or completed out of order");
    Check(gate.SubmittedSerial() == batches, "the submitter did not issue every stressed batch");
    Check(after.unheld == before.unheld && after.overlapped == before.overlapped && after.misordered == before.misordered && after.ordering == before.ordering && after.submitFailures == before.submitFailures, "the pool stress counted a violation or a failed submit");
    return failures == 0 ? 0 : 1;
}

// Death tests (see Locks.cpp's PrepareScopeDeath for why abort() must end in exit code 3).
[[maybe_unused]] void ExitOnAbort(int) {
    std::_Exit(3);
}

int Death(const char* mode) {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#else
    std::signal(SIGABRT, ExitOnAbort);
#endif
    if (std::strcmp(mode, "--lock-order") == 0) {
        // A queue mutex, then GpuMutex: aborts under APS5_ASSERT_GPU_LOCK=1.
        QueueGate gate(PresentModes(1));
        gate.SetPresentQueue(FakePresentQueue);
        const auto hold = gate.Lock(Queue::Present);
        AgcDriver::GuestMemory::GpuMutex().lock();
        AgcDriver::GuestMemory::GpuMutex().unlock();
    } else if (std::strcmp(mode, "--unheld-verify") == 0) {
        // =2 makes the first violation fatal with or without the assertion switch.
        QueueGate gate(PresentModes(2));
        gate.CheckOp(Queue::Graphics, "a submit without the graphics mutex");
    } else if (std::strcmp(mode, "--misordered") == 0) {
        QueueGate gate(PresentModes(1));
        const auto settle = gate.Lock(Queue::Settle);
        const auto graphics = gate.Lock(Queue::Graphics);
    } else {
        std::fprintf(stderr, "unknown argument %s\n", mode);
        return 2;
    }
    std::printf("%s: the call returned (APS5_ASSERT_GPU_LOCK %s)\n", mode, Asserted() ? "set: the assertion did not fire" : "unset");
    return Asserted() ? 1 : 0;
}

}

int main(int argc, char** argv) {
    if (argc > 1) {
        const std::string_view argument(argv[1]);
        if (argument == "--never-submitted-fence") return NeverSubmittedFence();
        for (const auto [prefix, pools] : {std::pair<std::string_view, bool>{"--pool-stress=", false}, std::pair<std::string_view, bool>{"--pool-stress-own-pools=", true}}) {
            if (argument.starts_with(prefix)) return PoolStress(std::strtoull(argv[1] + prefix.size(), nullptr, 10), pools);
        }
        return Death(argv[1]);
    }
    PresentQueueFallback();
    DeviceWideWaitExcludesQueues();
    OutOfDateTakenOnce();
    LockOrderAndCounts();
    if (const auto raw = OpenRawDevice()) SubmitterBasics(*raw);
    DeviceWaitIdle();
    if (failures != 0) {
        std::fprintf(stderr, "%d queue gate check(s) failed\n", failures);
        return 1;
    }
    std::printf("queue gate tests passed\n");
    return 0;
}
