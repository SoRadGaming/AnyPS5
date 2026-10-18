#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// The draw pipeline's hand-off without a device: plain commits stand in for draws and labels, so
// the waits for room and for an empty queue, failures, the stop, the commit runs and the labels
// appended to queued items are checked on their own. Run with and without the committer
// experiments (APS5_EXPERIMENT_ROOM_WAKE, APS5_EXPERIMENT_COMMIT_RUNS); only what a commit sees of
// the GPU-mutex hold may differ. The tests append labels themselves, with their own committer in
// place of the driver's, so the label tests run whether or not APS5_EXPERIMENT_LABEL_COALESCE is set.

namespace {

using AgcDriver::DriverDetail::DrawPipeline;
namespace GuestMemory = AgcDriver::GuestMemory;

constexpr std::string_view InjectedFailure = "injected commit failure";
constexpr auto Draw = DrawPipeline::ItemKind::Draw;

bool CommitRuns() {
    return std::getenv("APS5_EXPERIMENT_COMMIT_RUNS") != nullptr;
}

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// A wait that never ends is a lost wakeup: it is reported and the process ends, as the threads that
// still wait cannot be joined.
template<typename TDone>
void WaitUntil(TDone&& done, const char* what) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) {
            std::fprintf(stderr, "draw pipeline commits: timed out waiting for %s\n", what);
            std::fflush(stderr);
            std::_Exit(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

std::size_t RequireDepth() {
    const auto depth = DrawPipeline::Depth();
    Require(depth >= 4, "the draw pipeline depth is " + std::to_string(depth) + ", these tests need at least 4");
    return depth;
}

// The committer's record of what it ran, read by the test once the pipeline drained.
struct Log {
    std::mutex mutex;
    std::vector<int> entries;

    void Add(int entry) {
        std::lock_guard lock(mutex);
        entries.push_back(entry);
    }

    std::vector<int> Take() {
        std::lock_guard lock(mutex);
        return std::move(entries);
    }
};

std::vector<int> Sequence(int first, int last) {
    std::vector<int> values;
    for (int value = first; value <= last; ++value) values.push_back(value);
    return values;
}

// A commit that holds the committer until opened, so the test queues items behind it.
struct Gate {
    std::promise<void> opened;
    std::shared_future<void> released = opened.get_future().share();
    std::atomic<bool> entered{false};

    void Enqueue(DrawPipeline& pipeline) {
        pipeline.Enqueue([this, released = released] {
            entered = true;
            released.wait();
        }, {});
        WaitUntil([this] { return entered.load(); }, "the committer to reach the gate");
    }

    void Open() { opened.set_value(); }
};

// The label committer the test registers in place of the driver's: it logs the label's value, plus
// EndOfPipeFlag when the label raises an end-of-pipe interrupt.
constexpr int EndOfPipeFlag = 0x10000;
Log* labelLog = nullptr;

void LogLabel(const DrawPipeline::LabelPart& label) {
    std::uint32_t value = 0;
    std::memcpy(&value, label.bytes.data(), std::min(label.bytes.size(), sizeof(value)));
    labelLog->Add(static_cast<int>(value) | (label.endOfPipeInterrupt ? EndOfPipeFlag : 0));
}

DrawPipeline::LabelPart LabelAt(std::uint64_t address, std::uint32_t value, bool endOfPipeInterrupt = false) {
    DrawPipeline::LabelPart label;
    label.opcode = 0x49;
    label.address = address;
    label.bytes.resize(sizeof(value));
    std::memcpy(label.bytes.data(), &value, sizeof(value));
    label.endOfPipeInterrupt = endOfPipeInterrupt;
    return label;
}

bool Append(DrawPipeline& pipeline, std::uint64_t address, std::uint32_t value, bool endOfPipeInterrupt = false) {
    auto label = LabelAt(address, value, endOfPipeInterrupt);
    return pipeline.AppendLabel(label);
}

// A producer waiting for room and a drain waiting for an empty queue at the same time: both wake,
// and the commits run in the order they were queued.
void WaitersBothWake() {
    auto& pipeline = DrawPipeline::Queue0();
    const auto depth = static_cast<int>(RequireDepth());
    Log log;
    Gate gate;
    gate.Enqueue(pipeline);
    for (int i = 1; i < depth; ++i) pipeline.Enqueue([&log, i] { log.Add(i); }, {});
    Require(pipeline.Outstanding() == static_cast<std::size_t>(depth), "the pipeline is not full behind the gate");
    std::atomic<bool> enqueued{false};
    std::atomic<bool> drained{false};
    std::thread producer([&] {
        pipeline.Enqueue([&log, depth] { log.Add(depth); }, {});
        enqueued = true;
    });
    std::thread drainer([&] {
        pipeline.Drain(DrawPipeline::DrainReason::Flush);
        drained = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Require(!enqueued && !drained, "a producer or a drain returned while the full pipeline was held at the gate");
    gate.Open();
    WaitUntil([&] { return enqueued && drained; }, "the producer and the drain behind a full pipeline");
    producer.join();
    drainer.join();
    pipeline.Drain(DrawPipeline::DrainReason::Flush);
    Require(log.Take() == Sequence(1, depth), "the commits did not run in the order they were queued");
}

// A commit that throws stops the pipeline: the commits behind it and the labels appended to it are
// dropped (with commit runs, the failing draw is in the middle of a run), and the failure is
// rethrown once, to the producer that waited for room or to the next producer.
void FailureReachesWaiters() {
    auto& pipeline = DrawPipeline::Queue0();
    const auto depth = static_cast<int>(RequireDepth());
    constexpr std::uint64_t Label = 0x700000002000;
    std::atomic<int> ranBeforeFailure{0};
    std::atomic<int> ranAfterFailure{0};
    Log labels;
    labelLog = &labels;
    pipeline.SetLabelCommit(&LogLabel);
    Gate gate;
    gate.Enqueue(pipeline);
    pipeline.Enqueue([&ranBeforeFailure] { ++ranBeforeFailure; }, {}, 0, {}, Draw);
    Require(Append(pipeline, Label, 1), "a label behind the draw before the failing one was not appended");
    pipeline.Enqueue([] { throw std::runtime_error(std::string(InjectedFailure)); }, {}, 0, {}, Draw);
    Require(Append(pipeline, Label, 2), "a label behind the failing draw was not appended");
    for (int i = 3; i < depth; ++i) pipeline.Enqueue([&ranAfterFailure] { ++ranAfterFailure; }, {}, 0, {}, Draw);
    std::string producerError;
    std::atomic<bool> producerDone{false};
    std::thread producer([&] {
        try {
            pipeline.Enqueue([&ranAfterFailure] { ++ranAfterFailure; }, {}, 0, {}, Draw);
        } catch (const std::exception& error) {
            producerError = error.what();
        }
        producerDone = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    gate.Open();
    WaitUntil([&] { return producerDone.load(); }, "the producer behind the failing commit");
    producer.join();
    WaitUntil([&] { return pipeline.Outstanding() == 0; }, "the stopped committer to drop the queued commits");
    std::string laterError;
    try {
        pipeline.Enqueue([&ranAfterFailure] { ++ranAfterFailure; }, {});
    } catch (const std::exception& error) {
        laterError = error.what();
    }
    pipeline.Drain(DrawPipeline::DrainReason::Flush);
    Require((producerError == InjectedFailure) != (laterError == InjectedFailure), "the failure was not rethrown exactly once (waiting producer: '" + producerError + "', next producer: '" + laterError + "')");
    Require(ranBeforeFailure == 1, "the commit queued before the failed commit did not run");
    Require(ranAfterFailure == 0, "a commit queued behind the failed commit ran");
    Require(labels.Take() == std::vector<int>{1}, "a label appended to the failed commit ran, or the one appended before it did not");
    labelLog = nullptr;
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception&) {
        // The driver reports the failure again at its shutdown.
    }
}

// Draws queued behind a gate, each taking the GPU mutex as a draw commit does. With commit runs they
// share the committer's hold: the hold serial repeats and a commit's own lock nests at depth 2.
// Without, every commit is its own hold at depth 1. The order is the queue's either way.
void RunsShareOneHold() {
    auto& pipeline = DrawPipeline::Queue0();
    const auto commits = static_cast<int>(std::min<std::size_t>(RequireDepth() - 1, 6));
    struct Seen {
        std::uint64_t hold = 0;
        std::uint32_t depth = 0;
    };
    std::vector<Seen> seen(static_cast<std::size_t>(commits));
    Log log;
    Gate gate;
    gate.Enqueue(pipeline);
    for (int i = 0; i < commits; ++i) {
        pipeline.Enqueue([&seen, &log, i] {
            auto& mutex = GuestMemory::GpuMutex();
            std::lock_guard gpuLock(mutex);
            seen[static_cast<std::size_t>(i)] = {GuestMemory::GpuHoldSerial(), mutex.DepthOnThisThread()};
            log.Add(i);
        }, {}, 0, {}, Draw);
    }
    gate.Open();
    pipeline.Drain(DrawPipeline::DrainReason::Flush);
    Require(log.Take() == Sequence(0, commits - 1), "the commits did not run in the order they were queued");
    bool shared = false;
    for (int i = 0; i < commits; ++i) {
        const auto& commit = seen[static_cast<std::size_t>(i)];
        Require(commit.depth == (CommitRuns() ? 2u : 1u), "a draw commit held the GPU mutex at depth " + std::to_string(commit.depth));
        if (i != 0 && commit.hold == seen[static_cast<std::size_t>(i - 1)].hold) shared = true;
    }
    // A run also ends after about 150 us, so a preempted committer may split it; one shared pair is
    // enough to show that the hold carries across commits.
    Require(shared == CommitRuns(), CommitRuns() ? "no two draw commits shared the committer's GPU-mutex hold" : "two draw commits shared a GPU-mutex hold without commit runs");
    Require(!GuestMemory::GpuMutex().HeldByThisThread(), "the test thread holds the GPU mutex");
}

// A thread that waits for the GPU mutex ends a run at the next commit: the commit after the one
// that saw the waiter runs under a new hold, and the waiter gets the mutex.
void WaiterEndsRun() {
    auto& pipeline = DrawPipeline::Queue0();
    auto& mutex = GuestMemory::GpuMutex();
    std::uint64_t first = 0;
    std::uint64_t second = 0;
    std::atomic<bool> waiterLocked{false};
    std::thread waiter;
    Gate gate;
    gate.Enqueue(pipeline);
    pipeline.Enqueue([&] {
        std::lock_guard gpuLock(mutex);
        first = GuestMemory::GpuHoldSerial();
        waiter = std::thread([&] {
            std::lock_guard lock(mutex);
            waiterLocked = true;
        });
        if (CommitRuns()) WaitUntil([&] { return mutex.Waiters() != 0; }, "the second thread to wait for the GPU mutex");
        else std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }, {}, 0, {}, Draw);
    pipeline.Enqueue([&] {
        std::lock_guard gpuLock(mutex);
        second = GuestMemory::GpuHoldSerial();
    }, {}, 0, {}, Draw);
    gate.Open();
    pipeline.Drain(DrawPipeline::DrainReason::Flush);
    waiter.join();
    Require(waiterLocked, "the thread waiting for the GPU mutex never got it");
    Require(first != second, "the draw commit after a GPU-mutex waiter appeared ran in the same hold");
    Require(mutex.Waiters() == 0, "a thread is still counted as waiting for the free GPU mutex");
}

// The after-commit checks of a run: each commit leaves its check to the run's end, which runs it
// once, under the run's own hold (depth 1), before the drain returns; only the run's first label
// reaps. Without commit runs every commit runs its own check and every label reaps, and an item
// that is neither a draw nor a label never runs in a run.
int runEndChecks = 0;
std::uint32_t runEndCheckDepth = 0;

void RunEndCheck() {
    ++runEndChecks;
    runEndCheckDepth = GuestMemory::GpuMutex().DepthOnThisThread();
}

void ChecksAtRunEnd() {
    auto& pipeline = DrawPipeline::Queue0();
    const auto commits = std::min<std::size_t>(RequireDepth() - 2, 4);
    std::vector<int> deferred(commits, -1);
    std::vector<int> reaped(commits, -1);
    bool otherDeferred = true;
    runEndChecks = 0;
    runEndCheckDepth = 0;
    Gate gate;
    gate.Enqueue(pipeline);
    for (std::size_t i = 0; i < commits; ++i) {
        pipeline.Enqueue([&deferred, &reaped, i] {
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            deferred[i] = DrawPipeline::DeferToRunEnd(&RunEndCheck) ? 1 : 0;
            reaped[i] = DrawPipeline::LabelReapDue() ? 1 : 0;
        }, {}, 0, {}, i % 2 == 0 ? Draw : DrawPipeline::ItemKind::Label);
    }
    pipeline.Enqueue([&otherDeferred] { otherDeferred = DrawPipeline::DeferToRunEnd(&RunEndCheck); }, {});
    gate.Open();
    pipeline.Drain(DrawPipeline::DrainReason::Flush);
    Require(!otherDeferred, "a commit that is neither a draw nor a label left its check to a run");
    const auto reaps = static_cast<int>(std::count(reaped.begin(), reaped.end(), 1));
    if (!CommitRuns()) {
        Require(std::count(deferred.begin(), deferred.end(), 0) == static_cast<std::ptrdiff_t>(commits) && reaps == static_cast<int>(commits) && runEndChecks == 0, "without commit runs a commit left its check to a run or a label did not reap");
        return;
    }
    Require(std::count(deferred.begin(), deferred.end(), 1) == static_cast<std::ptrdiff_t>(commits), "a commit inside a run did not leave its check to the run's end");
    // A run may end early (its window, a preempted committer): every run reaps once and runs the
    // checks once.
    Require(reaped[0] == 1 && reaps >= 1 && runEndChecks == reaps, "the runs reaped " + std::to_string(reaps) + " times and ran their checks " + std::to_string(runEndChecks) + " times, expected once each per run");
    Require(runEndCheckDepth == 1, "the run's check ran at GPU-mutex depth " + std::to_string(runEndCheckDepth) + ", expected under the run's own hold");
}

// Labels appended to queued draws: they answer PendingLabel as their item's newest writes and count
// for the overlap checks until they ran, they are refused where they could run out of order or
// grow an item past the cap, and they run right after their item's commit, in queue order, with
// their end-of-pipe flag.
void AppendedLabels() {
    auto& pipeline = DrawPipeline::Queue0();
    constexpr std::uint64_t Label = 0x700000001000;
    constexpr std::uint64_t Other = Label + 0x100;
    constexpr std::uint64_t Target = Label + 0x1000;
    constexpr auto Cap = static_cast<std::uint32_t>(DrawPipeline::MaxAppendedLabels);
    Log log;
    labelLog = &log;
    pipeline.SetLabelCommit(&LogLabel);
    Gate gate;
    gate.Enqueue(pipeline);
    Require(!Append(pipeline, Label, 1), "a label was appended to the item the committer runs");
    pipeline.Enqueue([&log] { log.Add(100); }, {});
    Require(!Append(pipeline, Label, 2), "a label was appended to an item that is neither a draw nor a label");
    pipeline.Enqueue([&log] { log.Add(200); }, {{Target, Target + 64}}, 0, {}, Draw);
    Require(Append(pipeline, Label, 0x11), "a label behind a queued draw was not appended");
    Require(pipeline.PendingLabel(Label, 4) == 0x11u, "PendingLabel does not answer with an appended label's value");
    Require(!pipeline.PendingLabel(Label, 8).has_value(), "PendingLabel answers a wait wider than the appended label");
    Require(!pipeline.PendingLabel(Label + 2, 4).has_value(), "PendingLabel answers a wait the appended label covers in part");
    Require(!pipeline.PendingLabel(Target, 4).has_value(), "PendingLabel answers a wait on the draw's own writes");
    const std::array<DrawPipeline::Range, 2> runs{{{Other, Other + 4}, {Label + 3, Label + 4}}};
    Require(pipeline.Overlaps(Label, 4) && pipeline.OverlapsAny(runs) && !pipeline.Overlaps(Label + 4, 4) && !pipeline.OverlapsAny(std::span(runs).first(1)), "an appended label's bytes do not count as a queued write, exactly");
    Require(Append(pipeline, Label, 0x22, true), "a second label behind the queued draw was not appended");
    Require(pipeline.PendingLabel(Label, 4) == 0x22u, "PendingLabel does not answer with the newest appended label");
    pipeline.Enqueue([&log] { log.Add(300); }, {{Label, Label + 4}}, 0, {}, Draw);
    Require(!pipeline.PendingLabel(Label, 4).has_value(), "PendingLabel answers with a label that a later draw overwrites");
    Require(Append(pipeline, Label, 0x33), "a label behind the second draw was not appended");
    Require(pipeline.PendingLabel(Label, 4) == 0x33u, "PendingLabel does not answer with the label appended after the later draw");
    for (std::uint32_t i = 1; i < Cap; ++i) Require(Append(pipeline, Other, 0x40 + i), "a label within the cap was not appended");
    Require(!Append(pipeline, Other, 0x80), "an item took more than MaxAppendedLabels labels");
    Require(pipeline.PendingLabel(Other, 4) == std::uint64_t{0x40 + Cap - 1}, "PendingLabel does not answer with the newest label at another address");
    gate.Open();
    pipeline.Drain(DrawPipeline::DrainReason::Flush);
    std::vector<int> expected{100, 200, 0x11, 0x22 | EndOfPipeFlag, 300, 0x33};
    for (std::uint32_t i = 1; i < Cap; ++i) expected.push_back(static_cast<int>(0x40 + i));
    Require(log.Take() == expected, "the appended labels did not run right after their draws in the order they were queued");
    Require(!pipeline.Overlaps(Label, 4) && !pipeline.PendingLabel(Label, 4).has_value(), "a label that ran still counts as queued");
    labelLog = nullptr;
}

// The stop with a drain waiting: the committer still runs every queued commit, the drain wakes, and
// a commit queued after the stop runs on its caller.
void ClosingWakesDrain() {
    auto& pipeline = DrawPipeline::Queue0();
    const auto depth = static_cast<int>(RequireDepth());
    Log log;
    Gate gate;
    gate.Enqueue(pipeline);
    for (int i = 1; i < depth; ++i) pipeline.Enqueue([&log, i] { log.Add(i); }, {});
    std::atomic<bool> drained{false};
    std::atomic<bool> stopped{false};
    std::thread drainer([&] {
        pipeline.Drain(DrawPipeline::DrainReason::Flush);
        drained = true;
    });
    std::thread stopper([&] {
        pipeline.Stop();
        stopped = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    Require(!drained && !stopped, "a drain or the stop returned while the committer was held at the gate");
    gate.Open();
    WaitUntil([&] { return drained && stopped; }, "the drain and the stop");
    drainer.join();
    stopper.join();
    Require(log.Take() == Sequence(1, depth - 1), "the stop dropped or reordered queued commits");
    const auto caller = std::this_thread::get_id();
    bool ranHere = false;
    pipeline.Enqueue([&] { ranHere = std::this_thread::get_id() == caller; }, {});
    Require(ranHere, "a commit queued after the stop did not run on its caller");
}

}

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc > 1 ? argv[1] : "";
        if (mode == "--failure") {
            FailureReachesWaiters();
        } else if (mode == "--closing") {
            ClosingWakesDrain();
        } else {
            WaitersBothWake();
            RunsShareOneHold();
            WaiterEndsRun();
            ChecksAtRunEnd();
            AppendedLabels();
        }
        std::cout << "draw pipeline commit tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
