#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// The draw pipeline's hand-off without a device: plain commits stand in for draws and labels, so
// the waits for room and for an empty queue, failures and the stop are checked on their own. Run
// with and without the committer experiments (APS5_EXPERIMENT_ROOM_WAKE); the results must match.

namespace {

using AgcDriver::DriverDetail::DrawPipeline;

constexpr std::string_view InjectedFailure = "injected commit failure";

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

// A commit that throws stops the pipeline: the commits behind it are dropped, and the failure is
// rethrown once, to the producer that waited for room or to the next producer.
void FailureReachesWaiters() {
    auto& pipeline = DrawPipeline::Queue0();
    const auto depth = static_cast<int>(RequireDepth());
    std::atomic<int> ranAfterFailure{0};
    Gate gate;
    gate.Enqueue(pipeline);
    pipeline.Enqueue([] { throw std::runtime_error(std::string(InjectedFailure)); }, {});
    for (int i = 2; i < depth; ++i) pipeline.Enqueue([&ranAfterFailure] { ++ranAfterFailure; }, {});
    std::string producerError;
    std::atomic<bool> producerDone{false};
    std::thread producer([&] {
        try {
            pipeline.Enqueue([&ranAfterFailure] { ++ranAfterFailure; }, {});
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
    Require(ranAfterFailure == 0, "a commit queued behind the failed commit ran");
    try {
        AgcDriverShutdown_nid_postfix();
    } catch (const std::exception&) {
        // The driver reports the failure again at its shutdown.
    }
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
        }
        std::cout << "draw pipeline commit tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
