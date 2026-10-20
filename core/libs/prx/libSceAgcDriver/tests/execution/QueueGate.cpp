#include "VulkanTestDevice.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/QueueGate.hpp"
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
#include <thread>
#include <vector>

// LOCAL EXPERIMENT (PARALLEL-COMMIT-PLAN A3, A4): the queue gate (Graphics::QueueGate). Unit tests
// on gates of their own (fake queue handles; no Vulkan call is made through them), then a headless
// device's device-wide wait under the switches the run sets. Arguments run one death test each.
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
    const bool gated = QueueGate::PresentQueueMode() != 0;
    Check(gated ? after.deviceWaits > before.deviceWaits : after.deviceWaits == before.deviceWaits, gated ? "the device's WaitIdle took no device-wide hold" : "the device's WaitIdle took a device-wide hold with the switches unset");
    Check(after.unheld == before.unheld && after.overlapped == before.overlapped && after.misordered == before.misordered, "the device's WaitIdle counted a violation");
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
    if (argc > 1) return Death(argv[1]);
    PresentQueueFallback();
    DeviceWideWaitExcludesQueues();
    OutOfDateTakenOnce();
    LockOrderAndCounts();
    DeviceWaitIdle();
    if (failures != 0) {
        std::fprintf(stderr, "%d queue gate check(s) failed\n", failures);
        return 1;
    }
    std::printf("queue gate tests passed\n");
    return 0;
}
