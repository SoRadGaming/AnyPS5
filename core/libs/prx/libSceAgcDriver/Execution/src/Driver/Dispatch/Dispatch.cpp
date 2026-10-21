#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/BufferCopy.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DwordPatternFill.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/StreamStats.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <cstdlib>
#include <optional>
#include <stdexcept>

namespace AgcDriver::DriverDetail {

namespace {

bool DispatchItemsEnabled() {
    static const bool enabled = [] {
        const char* text = std::getenv("APS5_PIPELINE_DISPATCH");
        return text != nullptr && text[0] != '\0' && text[0] != '0';
    }();
    return enabled;
}

enum class ItemFallback : std::size_t { Miss, Reads, NoRecipe, Writes, Verify, Count };
constexpr std::array<const char*, static_cast<std::size_t>(ItemFallback::Count)> ItemFallbackNames{"miss", "reads", "no recipe", "writes", "verify"};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ItemFallback::Count)> itemFallbacks{};
std::atomic<std::uint64_t> itemCommits{0};
std::atomic<std::uint64_t> itemRebuilds{0};

void NoteItemFallback(ItemFallback reason) {
    itemFallbacks[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void ReportDispatchItems(bool profile) {
    static auto lastReport = std::chrono::steady_clock::now();
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    std::string drained;
    for (std::size_t i = 0; i < ItemFallbackNames.size(); ++i) {
        char item[48];
        std::snprintf(item, sizeof(item), " %s %llu", ItemFallbackNames[i], static_cast<unsigned long long>(itemFallbacks[i].load(std::memory_order_relaxed)));
        drained += item;
    }
    std::fprintf(stderr, "[dispatch-item] %llu commit items (%llu recipes rebuilt on the committer); drained instead, cumulative:%s\n", static_cast<unsigned long long>(itemCommits.load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemRebuilds.load(std::memory_order_relaxed)), drained.c_str());
}

// LOCAL DEBUG (APS5_DEBUG_STREAM_STATS=1, PARALLEL-COMMIT-PLAN S0): the shadow dispatch classifier.
// With APS5_PIPELINE_DISPATCH off, every queue-0 dispatch (not a dispatch-ahead resolution) is
// classified the way A1 would treat it (Stream::DispatchClass; the classes and their order are
// documented there), once, and the drain episode its own drain opened is named after the class,
// so the [stream] line gives each class the committer idle its drains caused; a later dispatch of
// a class A1 drains for cuts an eligible episode there. The reads-overlap test runs against the
// queued writes copied before the drain (SnapshotWriters, or before the gate drain when one came
// first). The shadow's own work after the drain is timed (`workNs`, filed beside the idle it
// lengthens), and a miss keeps the point A1 would drain at, before the lookup. Nothing changes
// what the dispatch does.
struct StreamShadow {
    bool on = false;
    bool classified = false;
    std::uint64_t episode = 0;
    std::uint64_t workNs = 0;
    DrawPipeline::StreamPoint drain;

    void Classify(Stream::DispatchClass value) {
        if (!on || classified) return;
        classified = true;
        Stream::Add(Stream::counters.dispatches[static_cast<std::size_t>(value)]);
        if (workNs != 0) Stream::Add(Stream::counters.shadowNs, workNs);
        DrawPipeline::Queue0().NoteDrainClass(episode, static_cast<std::uint8_t>(value), workNs, drain);
    }
};

// Set while a dispatch repeats itself (dispatch() calls itself after recording queued labels, or
// after a dispatch-ahead adoption failed), so the repeat is not classified a second time.
struct ShadowRepeatTag {};
using ShadowRepeat = HostThreadSlot<bool, ShadowRepeatTag>;

class ShadowRepeatScope {
public:
    explicit ShadowRepeatScope(bool active) : active(active), previous(active && ShadowRepeat::Get()) {
        if (active) ShadowRepeat::Set(true);
    }
    ~ShadowRepeatScope() {
        if (active) ShadowRepeat::Set(previous);
    }
    ShadowRepeatScope(const ShadowRepeatScope&) = delete;
    ShadowRepeatScope& operator=(const ShadowRepeatScope&) = delete;

private:
    bool active;
    bool previous;
};

struct ShadowWritersTag {};

// By what the shader can write, the widest first: Gds, Pagetable (stores through guest addresses;
// a page table always comes with a fault buffer), FaultOnly (a fault buffer alone: rect-list
// validation), Image. The Bda bit only repeats the fault buffer's (ShaderAddressResources'
// contract), so it is a cross-check, not a class.
Stream::DispatchClass unboundedClass(std::uint32_t roles) {
    if ((roles & DispatchUnbounded::Gds) != 0) return Stream::DispatchClass::UnboundedGds;
    if ((roles & DispatchUnbounded::Pagetable) != 0) return Stream::DispatchClass::UnboundedPagetable;
    if ((roles & DispatchUnbounded::Fault) != 0) return Stream::DispatchClass::UnboundedFaultOnly;
    return Stream::DispatchClass::UnboundedImage;
}

}

bool Driver::pipelineDispatch() {
    return DispatchItemsEnabled();
}

void Driver::dispatch(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::uint64_t indirectArguments) {
    PerformanceTimer timing("Driver.Dispatch");
    const auto address = (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20c)) << 8u) | (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20d) & 0xffu) << 40u);
    if (address == SkippedShader() && address != 0) return;
    auto registeredShader = RegisteredProgram(*submission.shaders, address, {0});
    if (!registeredShader) registeredShader = ReadRawShader(address);
    const auto& snapshot = *registeredShader;
    require(snapshot.type == 0, "compute program refers to a non-compute shader");
    const auto userCount = (readRegister(queue.shader, 0x213) >> 1u) & 0x1fu;
    std::vector<std::uint32_t> userData;
    for (std::uint32_t i = 0; i < userCount; ++i) {
        userData.push_back(readUserData(queue.shader, 0x240 + i));
    }
    auto compute = Graphics::DecodeComputeStageInfo(queue.shader, snapshot.header);
    const bool resolving = resolvingAhead();
    // LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): the shadow classifier (StreamShadow), from the drain
    // episode open before this dispatch drained anything.
    StreamShadow shadow;
    if (Stream::On() && !resolving && DrawPipeline::Active() && !ShadowRepeat::Get()) {
        if (DispatchItemsEnabled()) {
            Stream::Add(Stream::counters.unshadowed);
        } else {
            shadow.on = true;
            shadow.episode = DrawPipeline::Queue0().DrainEpisode();
        }
    }
    std::vector<ShaderRecompiler::MemoryRegion> memory{{snapshot.codeAddress, std::as_bytes(std::span(snapshot.code))}};
    if (!snapshot.header.empty()) memory.push_back({snapshot.headerAddress, snapshot.header});

    static const bool unlockedDevice = std::getenv("APS5_NO_UNLOCKED_DEVICE") == nullptr;
    std::shared_ptr<VulkanDevice> localDevice = unlockedDevice ? device.Load() : nullptr;
    if (localDevice == nullptr) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    const auto codeOffset = static_cast<std::size_t>((address - snapshot.codeAddress) / sizeof(std::uint32_t));
    std::array<std::uint32_t, 5> resolved{};
    ResolvedDispatch ahead;
    bool fromAhead = false;
    auto& groupStats = groupCaptureStats();
    if (!resolving) {
        auto& parked = resolvedAhead();
        if (const auto found = parked.find(currentPacketOffset()); found != parked.end()) {
            ahead = std::move(found->second);
            parked.erase(found);
            fromAhead = ahead.address == address;
            if (!fromAhead) ++groupStats.addressMismatches;
        }
    }
    if (fromAhead) {
        resolved = ahead.packet;
        packet = resolved;
        indirectArguments = ahead.indirectArguments;
    } else if (indirectArguments != 0 && !resolving && matchesFillKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute)) {
        indirectSource().path = IndirectFillKernel;
        const auto readStart = std::chrono::steady_clock::now();
        resolved = readDispatchArguments(submission.queue, indirectArguments, packet[4]);
        countIndirect(IndirectFillKernel, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        packet = resolved;
        indirectArguments = 0;
    }
    std::array<std::uint32_t, 3> partialGroups{};
    if (indirectArguments == 0 && (packet[4] & 0x2u) != 0) {
        partialGroups = Pm4::PartialGroupThreads({packet[1], packet[2], packet[3]}, {readRegister(queue.shader, 0x207), readRegister(queue.shader, 0x208), readRegister(queue.shader, 0x209)});
    }
    const bool pipelineCandidate = !resolving && !fromAhead && DispatchItemsEnabled() && DrawPipeline::Active() && indirectArguments == 0 && deferredLabels().labels.empty() && !matchesCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) && !MatchesStreamCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute);
    const bool partialLastGroup = partialGroups != std::array<std::uint32_t, 3>{};
    // LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): whether the shadow treats this dispatch as A1's
    // candidate (pipelineCandidate without the switch), and how many items were queued when its
    // queued writes were copied.
    bool shadowCandidate = false;
    std::size_t shadowQueued = 0;
    if (resolving) {
        if (!partialLastGroup && matchesFillKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute)) return;
        if (!partialLastGroup && indirectArguments == 0 && (matchesCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchesStreamCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute))) return;
    } else {
        if (indirectArguments == 0 && (packet[4] & 0x20u) == 0) {
            if (const auto fill = Graphics::MatchMemoryValueFill(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads, {packet[1], packet[2], packet[3]})) Graphics::NoteMemoryValueFill(*fill);
        }
        if (!partialLastGroup && fillBuffer(queue, submission.queue, packet, std::span(snapshot.code).subspan(codeOffset), userData, compute, localDevice)) {
            pendingDispatchPhases().outcome = DispatchOutcome::FillHle;
            shadow.Classify(Stream::DispatchClass::Fill);
            return;
        }
        // LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): the classes A1's candidate test refuses are known
        // here and named after the drain below (so its episode takes the class); a candidate's
        // queued writes are copied before that drain empties the queue.
        std::optional<Stream::DispatchClass> shadowRefused;
        if (shadow.on) {
            if (fromAhead) shadowRefused = Stream::DispatchClass::Ahead;
            else if (indirectArguments != 0) shadowRefused = Stream::DispatchClass::Indirect;
            else if (!deferredLabels().labels.empty()) shadowRefused = Stream::DispatchClass::Labels;
            else if (matchesCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchesStreamCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute)) shadowRefused = Stream::DispatchClass::Copy;
            shadowCandidate = !shadowRefused.has_value();
            if (shadowCandidate) shadowQueued = DrawPipeline::Queue0().SnapshotWriters(ThreadScratch<std::vector<DrawPipeline::QueuedWriter>, ShadowWritersTag>(), shadow.episode);
        }
        if (DrawPipeline::Active() && !pipelineCandidate) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
        if (shadowRefused.has_value()) shadow.Classify(*shadowRefused);
        if (const auto htile = Graphics::HtileDepthClearAddress(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads); htile != 0) {
            const auto threads = (packet[4] & 0x20u) != 0 ? static_cast<std::uint64_t>(packet[1]) : static_cast<std::uint64_t>(packet[1]) * compute.numThreads[0];
            const auto nativeBytes = Graphics::HtileDepthClearBytes_nid_no_patch(std::span(snapshot.code).subspan(codeOffset), userData, compute, packet, indirectArguments);
            Graphics::NoteHtileDepthClear(htile, static_cast<std::size_t>(threads * 16u), nativeBytes);
        }
        if (!partialLastGroup && indirectArguments == 0 && copyBuffer(queue, submission.queue, packet, std::span(snapshot.code).subspan(codeOffset), userData, compute, localDevice, address)) {
            pendingDispatchPhases().outcome = DispatchOutcome::CopyHle;
            return;
        }
    }
    if (indirectArguments == 0 && (packet[4] & 0x20u) != 0) {
        const std::array<std::uint32_t, 3> threads{packet[1], packet[2], packet[3]};
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            if (threads[axis] % compute.numThreads[axis] != 0) compute.partialThreads = threads;
        }
    }
    if (partialLastGroup) compute.partialThreads = partialGroups;
    std::array<std::uint32_t, 3> threadGroup{};
    if (indirectArguments != 0 && (packet[4] & 0x20u) != 0) {
        for (std::uint32_t axis = 0; axis < 3; ++axis) threadGroup[axis] = std::max(readRegister(queue.shader, 0x207 + axis) & 0xffffu, 1u);
        compute.partialThreads = {~0u, ~0u, ~0u};
    }
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, address, std::span(snapshot.code).subspan(codeOffset), snapshot.headerAddress, snapshot.header},
        {(packet[4] & 0x8000u) != 0 ? 32u : 64u, 0, userData, compute, std::nullopt, std::nullopt, memory, DispatchFloatMode(snapshot, queue.shader)},
        localDevice->ComputeTarget((packet[4] & 0x8000u) != 0 ? 32u : 64u),
        {0, 0, 0, threadGroup != std::array<std::uint32_t, 3>{} ? 0u : 128u}
    };
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static double captureMs = 0, keyMs = 0, recompileMs = 0, deviceMs = 0;
    static std::uint64_t cacheHits = 0;
    static std::uint64_t dispatches = 0;
    auto lap = std::chrono::steady_clock::now();

    std::array<double, DriverPhaseCount> phaseMs{};
    auto phaseLap = lap;
    DispatchPhaseTiming phaseTiming{profile, lap, phaseMs, phaseLap};
    if (profile && packetStartedAt() != std::chrono::steady_clock::time_point{}) phaseMs[PhasePrologue] = std::chrono::duration<double, std::milli>(lap - packetStartedAt()).count();

    static const bool noDispatchCacheEnv = std::getenv("APS5_NO_DISPATCH_CACHE") != nullptr;

    static const std::pair<std::uint64_t, std::uint64_t> probeDispatch = [] {
        const char* text = std::getenv("APS5_PROBE_DISPATCH");
        if (text == nullptr) return std::pair<std::uint64_t, std::uint64_t>{0, 0};
        char* end = nullptr;
        const auto probeAddress = std::strtoull(text, &end, 16);
        const auto index = end != nullptr && *end == ':' ? std::strtoull(end + 1, nullptr, 10) : 0ull;
        return std::pair<std::uint64_t, std::uint64_t>{probeAddress, index};
    }();
    bool probeThis = false;

    if (probeDispatch.first != 0 && (address & 0xfffffffffull) == (probeDispatch.first & 0xfffffffffull)) {
        static std::atomic<std::uint64_t> dispatchesSeen{0};
        probeThis = dispatchesSeen.fetch_add(1) == probeDispatch.second;
        if (probeThis) std::fprintf(stderr, "[gpu] probing dispatch %llu of 0x%llx\n", static_cast<unsigned long long>(probeDispatch.second), static_cast<unsigned long long>(address));
    }

    struct ProbeScope {
        bool active;
        explicit ProbeScope(bool active) : active(active) { if (active) ShaderRecompiler::SetDebugProbeActive(true); }
        ~ProbeScope() { if (active) ShaderRecompiler::SetDebugProbeActive(false); }
    } probeScope{probeThis};
    const bool noDispatchCache = noDispatchCacheEnv || probeThis;
    std::uint64_t key = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
    };
    mix(address);
    mix(packet[4] & 0x8000u);
    for (const auto threads : compute.partialThreads) mix(threads);
    for (const auto word : userData) mix(word);

    static const bool keyHygiene = std::getenv("APS5_NO_DISPATCH_KEY_HYGIENE") == nullptr;
    if (keyHygiene) {
        for (const auto offset : {0x207u, 0x208u, 0x209u, 0x212u, 0x213u}) {
            const auto found = queue.shader.find(offset);

            mix(found == queue.shader.end() ? (1ull << 32u) : found->second);
        }
    } else {
        for (const auto& [offset, value] : queue.shader) {
            mix(offset);
            mix(value);
        }
    }
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiledResult;

    std::shared_ptr<DispatchVariant> keepVariant;

    std::shared_ptr<DispatchVariant> attachVariant;

    std::shared_ptr<ShaderMemory> shaderMemory;
    std::vector<ShaderRecompiler::MemoryRegion> captured;

    std::vector<std::uint32_t> liveWords;
    bool dataHit = false;
    bool cached = false;
    bool validated = false;

    std::shared_ptr<DispatchEntry> missedEntry;
    bool missedDiffering = false;
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;

    static const bool traceCache = std::getenv("APS5_TRACE_DISPATCH_CACHE") != nullptr;
    if (traceCache) {
        std::lock_guard traceLock(dispatchCacheMutex);
        struct Last { std::vector<std::uint32_t> userData; std::map<std::uint32_t, std::uint32_t> shader; std::uint64_t key; };
        static std::map<std::uint64_t, Last> last;
        static int reports = 0;
        auto& previous = last[address];
        if (previous.key != 0 && previous.key != key && reports < 200) {
            std::string what;
            for (std::size_t i = 0; i < userData.size(); ++i) {
                if (i >= previous.userData.size() || previous.userData[i] != userData[i]) {
                    char text[48];
                    std::snprintf(text, sizeof(text), " user[%zu] %08x->%08x", i, i < previous.userData.size() ? previous.userData[i] : 0u, userData[i]);
                    what += text;
                }
            }
            for (const auto& [offset, value] : queue.shader) {
                const auto old = previous.shader.find(offset);
                if (old == previous.shader.end() || old->second != value) {
                    char text[48];
                    std::snprintf(text, sizeof(text), " sh[%x] %08x->%08x", offset, old == previous.shader.end() ? 0u : old->second, value);
                    what += text;
                }
            }
            ++reports;
            std::fprintf(stderr, "[dispatch-cache] 0x%llx key changed:%s\n", static_cast<unsigned long long>(address), what.c_str());
        }
        previous.userData = userData;
        previous.shader = std::map<std::uint32_t, std::uint32_t>(queue.shader.begin(), queue.shader.end());
        previous.key = key;
    }

    mix(reinterpret_cast<std::uintptr_t>(registeredShader.get()));
    if (fromAhead && key != ahead.key) {
        fromAhead = false;
        ++groupStats.keyMismatches;
    }
    phaseTiming.Phase(PhaseKey);
    timing.Mark("decode_key_device_setup");
    // LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): the shadow of A1's cache and reads tests below (the
    // same entry regions, against the writes queued before the drain): a miss (A1 drains it here,
    // before its lookup), or the items queued behind the newest writer the reads overlap. The
    // entry's regions are collected only when writes were queued.
    bool shadowMiss = false;
    std::optional<std::size_t> shadowBehind;
    if (shadowCandidate) {
        const auto shadowStartNs = Stream::NowNs();
        const auto& writers = ThreadScratch<std::vector<DrawPipeline::QueuedWriter>, ShadowWritersTag>();
        std::vector<ShaderRecompiler::MemoryRegion> reads;
        bool entryFound = false;
        {
            std::lock_guard cacheLock(dispatchCacheMutex);
            if (const auto found = dispatchCache.find(key); found != dispatchCache.end()) {
                entryFound = true;
                if (!writers.empty()) {
                    for (const auto& variant : found->second->variants) appendEntryRegions(*variant, reads);
                }
            }
        }
        shadowMiss = !entryFound || noDispatchCache || stampValidate() || verifyDataHits();
        if (!shadowMiss) {
            std::optional<std::uint32_t> newest;
            for (const auto& region : reads) {
                const auto end = region.guestAddress + region.bytes.size();
                for (const auto& writer : writers) {
                    if (region.guestAddress < writer.end && writer.begin < end && (!newest.has_value() || writer.position > *newest)) newest = writer.position;
                }
            }
            if (newest.has_value() && *newest < shadowQueued) shadowBehind = shadowQueued - 1 - *newest;
        }
        const auto shadowEndNs = Stream::NowNs();
        shadow.workNs += shadowEndNs - shadowStartNs;
        if (shadowMiss) shadow.drain = {shadowEndNs, Stream::ThreadGpuWait::Get()};
    }
    bool readsClear = false;
    if (pipelineCandidate) {
        std::vector<ShaderRecompiler::MemoryRegion> reads;
        bool entryFound = false;
        {
            std::lock_guard cacheLock(dispatchCacheMutex);
            if (const auto found = dispatchCache.find(key); found != dispatchCache.end()) {
                entryFound = true;
                for (const auto& variant : found->second->variants) appendEntryRegions(*variant, reads);
            }
        }
        if (!entryFound || noDispatchCache || stampValidate()) {
            NoteItemFallback(ItemFallback::Miss);
        } else if (verifyDataHits()) {
            NoteItemFallback(ItemFallback::Verify);
        } else {
            readsClear = std::none_of(reads.begin(), reads.end(), [](const auto& region) { return DrawPipeline::Queue0().Overlaps(region.guestAddress, region.bytes.size()); });
            if (!readsClear) NoteItemFallback(ItemFallback::Reads);
        }
        if (!readsClear && DrawPipeline::Queue0().Busy()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
    }
    if (fromAhead) {
        compiledResult = std::move(ahead.compiledResult);
        keepVariant = std::move(ahead.keepVariant);
        attachVariant = std::move(ahead.attachVariant);
        shaderMemory = std::move(ahead.shaderMemory);
        captured = std::move(ahead.captured);
        liveWords = std::move(ahead.liveWords);
        capture = std::move(ahead.capture);
        dataHit = ahead.dataHit;
        cached = ahead.cached;
        validated = ahead.validated;
        ++groupStats.adopted;
        ++dispatchesAdoptedAhead;
        const auto age = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ahead.resolvedAt).count();
        groupStats.ageMs += age;
        groupStats.ageMaxMs = std::max(groupStats.ageMaxMs, age);
        phaseTiming.Phase(PhaseLookup);
    } else {
        lookupDispatch(address, submission, key, noDispatchCache, traceCache, profile, memory, phaseTiming, phaseMs, compiledResult, keepVariant, captured, liveWords, dataHit, cached, validated, missedEntry, missedDiffering);
    }
    timing.Mark("lookup_cache");
    // LOCAL DEBUG (APS5_DEBUG_STREAM_STATS): the candidate's class, in A1's order: a miss drains,
    // an unbounded hit drains, a bounded hit whose reads overlap drains through the newest writer,
    // a bounded clear hit is an item with or without a recipe. A miss found after the lookup, and an
    // unbounded hit, drain here (StreamShadow::drain stays unset).
    if (shadowCandidate) {
        const auto classifyStartNs = Stream::NowNs();
        auto shadowClass = Stream::DispatchClass::Miss;
        if (!shadowMiss && cached && keepVariant != nullptr && compiledResult != nullptr) {
            bool bounded = false;
            std::uint32_t roles = 0;
            static_cast<void>(DispatchWriteRanges(*compiledResult, bounded, &roles));
            if (((roles & DispatchUnbounded::Bda) != 0) != ((roles & DispatchUnbounded::Fault) != 0)) Stream::Add(Stream::counters.bdaRoleMismatches);
            if (!bounded) {
                shadowClass = unboundedClass(roles);
            } else if (shadowBehind.has_value()) {
                Stream::Add(Stream::counters.readsBehind[Stream::BehindBucket(*shadowBehind)]);
                shadowClass = Stream::DispatchClass::Reads;
            } else {
                shadowClass = keepVariant->recipe.load(std::memory_order_acquire) != nullptr ? Stream::DispatchClass::Item : Stream::DispatchClass::NoRecipe;
            }
        }
        shadow.workNs += Stream::NowNs() - classifyStartNs;
        shadow.Classify(shadowClass);
    }
    std::vector<DrawPipeline::Range> itemWrites;
    bool pipelined = false;
    if (readsClear) {
        if (cached && keepVariant != nullptr && keepVariant->recipe.load(std::memory_order_acquire) != nullptr) {
            bool bounded = false;
            itemWrites = DispatchWriteRanges(*compiledResult, bounded);
            pipelined = bounded;
            if (!bounded) NoteItemFallback(ItemFallback::Writes);
        } else {
            NoteItemFallback(ItemFallback::NoRecipe);
        }
        if (!pipelined && DrawPipeline::Queue0().Busy()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
    }
    if (cached) {
        require(keepVariant != nullptr && keepVariant->shader == registeredShader, "dispatch cache belongs to another registered shader");
        captureMs += phaseTiming.Elapsed();
    } else if (!fromAhead) {
        shaderMemory = std::make_shared<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter(), &QueryPendingSpan);
        std::uint64_t forgetAtCapture = 0;

        static const bool dumpShaders = std::getenv("APS5_DUMP_SHADERS") != nullptr;
        try {
            auto invocation = InvocationFor(snapshot, codeOffset, request);
            timing.Mark("prepared_invocation");

            const auto waitedBefore = traceCapSync() ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
            forgetAtCapture = GuestMemory::ForgetSerial();
            capture = [&] {
                const SampledReadScope sampling(evidenceReads);
                return CaptureInvocation(*shaderMemory, snapshot, codeOffset, request, invocation);
            }();
            captured = shaderMemory->Regions();
            request.context.memory = captured;
            if (traceCapSync()) traceCapture("dispatch-capture", address, submission.queue, captured, Graphics::Recorder::ThreadWaitedMs() - waitedBefore);
            captureMs += phaseTiming.Elapsed();
            phaseTiming.Phase(PhaseCapture);
            if (dumpShaders) static_cast<void>(dumpRequest(address, request));
            const auto started = std::chrono::steady_clock::now();


            timing.Mark("capture_resources");
            compiledResult = invocation.Materialize(*capture);
            timing.Mark("materialize");
            if (compiledResult->cacheHit) ++cacheHits;
            const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            static double totalMs = 0;
            totalMs += elapsed;
            if (profile && elapsed > 200) std::fprintf(stderr, "[gpu] compute shader 0x%llx recompile took %.0f ms (%zu SPIR-V words, %zu captured regions, total %.1f s)\n", static_cast<unsigned long long>(address), elapsed, compiledResult->spirv.size(), captured.size(), totalMs / 1000);
        } catch (const std::exception& error) {
            std::string dump;
            if (dumpShaders) {
                const auto regions = shaderMemory->Regions();
                auto failed = request;
                failed.context.memory = regions;
                dump = dumpRequest(address, failed, "_failed");
            }

            std::string reason = error.what();
            if (const auto newline = reason.find('\n'); newline != std::string::npos) reason.resize(newline);
            char where[96];
            if (dump.empty()) std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
            else std::snprintf(where, sizeof(where), "compute shader 0x%llx (%s): ", static_cast<unsigned long long>(address), dump.c_str());
            throw std::runtime_error(where + reason);
        }
        recompileMs += phaseTiming.Elapsed();
        phaseTiming.Phase(PhaseRecompile);
        insertDispatch(address, key, noDispatchCache || !CacheableResult(*compiledResult), profile, registeredShader, forgetAtCapture, memory, shaderMemory, captured, capture, compiledResult, missedEntry, missedDiffering, attachVariant, phaseTiming);
    }
    if (resolving) {
        auto& parked = resolvedAhead()[currentPacketOffset()];
        parked = ResolvedDispatch{};
        parked.address = address;
        parked.key = key;
        parked.indirectArguments = indirectArguments;
        std::copy_n(packet.begin(), std::min(packet.size(), parked.packet.size()), parked.packet.begin());
        parked.registeredShader = registeredShader;
        parked.compiledResult = compiledResult;
        parked.keepVariant = keepVariant;
        parked.attachVariant = attachVariant;
        parked.shaderMemory = shaderMemory;
        parked.captured = std::move(captured);
        parked.liveWords = std::move(liveWords);
        parked.capture = capture;
        parked.dataHit = dataHit;
        parked.cached = cached;
        parked.validated = validated;
        parked.resolvedAt = std::chrono::steady_clock::now();
        ++groupStats.resolved;
        if (cached) ++groupStats.resolvedHits;
        return;
    }
    if (verifyDataHits() && dataHit) verifyDataHit(snapshot, codeOffset, request, memory, address, *keepVariant, liveWords, *compiledResult);

    if (recordQueuedLabelsAfterCapture(submission.queue, captured)) {
        const ShadowRepeatScope repeat(Stream::On());
        dispatch(queue, packet, submission, indirectArguments);
        return;
    }
    phaseTiming.Phase(PhaseQueuedLabels);
    const auto& compiled = *compiledResult;
    std::vector<Graphics::GuestMemorySnapshot> snapshots;
    for (const auto& region : captured) snapshots.push_back({region.guestAddress, region.bytes});
    std::array<std::uint32_t, 3> groups{packet[1], packet[2], packet[3]};
    if (indirectArguments == 0 && (packet[4] & 0x20u) != 0) {

        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            const auto threads = std::max(readRegister(queue.shader, 0x207 + axis) & 0xffffu, 1u);
            groups[axis] = (groups[axis] + threads - 1) / threads;
        }
    }
    const std::uint64_t dispatchThreads = indirectArguments != 0 ? 0 : std::uint64_t{groups[0]} * groups[1] * groups[2] * compute.numThreads[0] * compute.numThreads[1] * compute.numThreads[2];
    static const bool traceIo = std::getenv("APS5_TRACE_DISPATCH_IO") != nullptr;
    if (traceIo) {

        std::string words;
        if (std::getenv("APS5_TRACE_DISPATCH_IO")[0] == '2') {
            for (const auto word : userData) {
                char text[12];
                std::snprintf(text, sizeof(text), " %08x", word);
                words += text;
            }
        }
        std::fprintf(stderr, "[dispatch-io] queue=0x%x submission=%llu shader 0x%llx%s\n", submission.queue, static_cast<unsigned long long>(submission.serial), static_cast<unsigned long long>(address), words.c_str());
    }

    const auto rethrow = [&](const std::exception& error) {
        if (fromAhead) {
            ++groupStats.deviceRejects;
            const ShadowRepeatScope repeat(Stream::On());
            dispatch(queue, packet, submission, indirectArguments);
            return true;
        }
        char where[64];
        std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
        throw std::runtime_error(where + std::string(error.what()) + describeIndirectSource());
    };
    phaseTiming.Phase(PhaseSnapshots);

    if (pipelined) {
        const auto recipe = keepVariant->recipe.load(std::memory_order_acquire);
        const auto epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
        const auto queueId = submission.queue;
        itemCommits.fetch_add(1, std::memory_order_relaxed);
        DrawPipeline::Queue0().Enqueue([this, localDevice = std::move(localDevice), result = compiledResult, snapshots = std::move(snapshots), liveWords = std::move(liveWords), groups, address, queueId, recipe, keepVariant, dataHit, epoch] {
            DrawPipeline::FollowEpoch(epoch);
            GuestMemory::SetCurrentPacket(0x15, queueId);
            const auto& compiledCommit = *result;
            const bool noteWrites = writeEvidenceEnabled() || traceCapSync();
            try {
                std::shared_ptr<RecipeHit> recipeHit = localDevice->PrepareRecipe(recipe, false);
                std::shared_ptr<const Recipe> builtRecipe;
                bool writersNoted = false;
                for (;;) {
                    std::shared_ptr<PreparedDispatch> prepared;
                    if (recipeHit == nullptr || VulkanDevice::VerifyRecipes()) prepared = localDevice->PrepareDispatch(compiledCommit, snapshots);
                    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
                    std::lock_guard gpuLock(GuestMemory::GpuMutex());
                    recordLabelsForPacket(localDevice.get(), queueId);
                    if (noteWrites && writerKeyedEvidence() && !writersNoted) {
                        noteWrittenBuffers(address, queueId, compiledCommit);
                        writersNoted = true;
                    }
                    if (recipeHit != nullptr) {
                        VulkanDevice::IndirectOutcome outcome{0, 0};
                        const auto recipeResult = localDevice->DispatchRecipe(compiledCommit, groups[0], groups[1], groups[2], 0, address, recipeHit, outcome, VulkanDevice::VerifyRecipes() ? prepared : nullptr, dataHit);
                        if (recipeResult == RecipeOutcome::Rebuild) {
                            recipeHit = nullptr;
                            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, false);
                            itemRebuilds.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                    } else {
                        localDevice->Dispatch(compiledCommit, groups[0], groups[1], groups[2], snapshots, address, std::move(prepared), &builtRecipe);
                    }
                    break;
                }
                if (noteWrites && !writerKeyedEvidence()) noteWrittenBuffers(address, queueId, compiledCommit);
                if (builtRecipe != nullptr) {
                    if (dataHit) {
                        auto own = std::make_shared<Recipe>(*builtRecipe);
                        own->dataWordsHash = Graphics::ShaderResources::DataWordsHash({ShaderRecompiler::ShaderStage::Compute, keepVariant->compiled.get(), 0});
                        builtRecipe = std::move(own);
                    }
                    keepVariant->recipe.store(std::move(builtRecipe), std::memory_order_release);
                    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, false);
                }
            } catch (const std::exception& error) {
                char where[64];
                std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
                throw std::runtime_error(where + std::string(error.what()));
            }
        }, std::move(itemWrites));
        ReportDispatchItems(profile);
        return;
    }

    std::shared_ptr<RecipeHit> recipeHit;
    if (cached && keepVariant != nullptr && !stampValidate()) {
        recipeHit = localDevice->PrepareRecipe(keepVariant->recipe.load(std::memory_order_acquire), indirectArguments != 0);
        phaseTiming.Phase(PhaseRecipePrecheck);
    }

    std::shared_ptr<const Recipe> builtRecipe;
    auto* const attachTo = stampValidate() ? nullptr : keepVariant != nullptr ? keepVariant.get() : attachVariant.get();
    bool writersNoted = false;
    const bool noteWrites = writeEvidenceEnabled() || traceCapSync();
    const auto keysFill = indirectArguments == 0 && !partialLastGroup ? MatchSgprPatternFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute) : std::nullopt;
    const auto filledKeys = keysFill ? KeysFillMemo(keysFill->pattern, false, true) : std::nullopt;

    std::optional<Graphics::DepthPlaneClear> depthPlaneClear;
    if (indirectArguments == 0 && !partialLastGroup) {
        if (const auto clear = Graphics::MatchStorageImageClear(std::span(snapshot.code).subspan(codeOffset), userData, compute, groups)) {
            const auto clearAddress = Graphics::StorageImageClearAddress(*clear);
            if (clearAddress != 0 && Graphics::DepthSurfaceAt(clearAddress)) {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
                std::lock_guard clearLock(GuestMemory::GpuMutex());
                recordLabelsForPacket(localDevice.get(), submission.queue);
                depthPlaneClear.emplace(*clear);
            }
        }
    }

    for (;;) {

        std::shared_ptr<PreparedDispatch> prepared;
        if (recipeHit == nullptr || VulkanDevice::VerifyRecipes()) {
            try {
                prepared = localDevice->PrepareDispatch(compiled, snapshots, dispatchThreads, address);
            } catch (const std::exception& error) {
                if (rethrow(error)) return;
            }
            if (profile) {
                const auto now = std::chrono::steady_clock::now();
                const auto prepareMs = std::chrono::duration<double, std::milli>(now - phaseLap).count();
                phaseLap = now;
                double parts = 0;
                if (prepared != nullptr) {
                    const auto phases = VulkanDevice::PreparePhaseMs(*prepared);
                    phaseMs[PhasePrepareKey] += phases[0];
                    phaseMs[PhasePrepareFind] += phases[1];
                    phaseMs[PhasePreparePrecollect] += phases[2];
                    phaseMs[PhasePreparePresync] += phases[3];
                    phaseMs[PhasePrepareStageA] += phases[4];
                    for (const auto part : phases) parts += part;
                }
                phaseMs[PhasePrepareOther] += std::max(0.0, prepareMs - parts);
            }
        }
        GuestMemory::TagGpuLockSite(indirectArguments != 0 ? GuestMemory::GpuLockSite::Indirect : GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        phaseTiming.Phase(PhaseLockWait);

        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(PhaseLabels);
        if (noteWrites && writerKeyedEvidence() && !writersNoted) {
            noteWrittenBuffers(address, submission.queue, compiled);
            writersNoted = true;
        }
        phaseTiming.Phase(PhaseNoteWriters);
        auto* const keysRecorder = filledKeys ? Graphics::Recorder::Active() : nullptr;
        const auto keysNoteBefore = keysRecorder != nullptr ? keysRecorder->LastWriteNote(keysFill->base, keysFill->bytes) : 0;
        try {
            if (recipeHit != nullptr) {
                VulkanDevice::IndirectOutcome outcome{0, 0};
                const auto result = localDevice->DispatchRecipe(compiled, groups[0], groups[1], groups[2], indirectArguments, address, recipeHit, outcome, VulkanDevice::VerifyRecipes() ? prepared : nullptr, dataHit, threadGroup);
                if (result == RecipeOutcome::Rebuild) {

                    recipeHit = nullptr;
                    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, indirectArguments != 0);
                    continue;
                }
                if (indirectArguments != 0) countIndirect(outcome.cpuReason, outcome.argumentReadMs);
            } else if (indirectArguments != 0) {
                const auto outcome = localDevice->DispatchIndirect(compiled, indirectArguments, snapshots, address, std::move(prepared), attachTo != nullptr ? &builtRecipe : nullptr, threadGroup);
                countIndirect(outcome.cpuReason, outcome.argumentReadMs);
            } else {
                localDevice->Dispatch(compiled, groups[0], groups[1], groups[2], snapshots, address, std::move(prepared), attachTo != nullptr ? &builtRecipe : nullptr);
            }
        } catch (const std::exception& error) {
            if (rethrow(error)) return;
        }
        if (keysRecorder != nullptr && keysRecorder == Graphics::Recorder::Active()) {
            const auto note = keysRecorder->LastWriteNote(keysFill->base, keysFill->bytes);
            if (note != 0 && note != keysNoteBefore) Graphics::NoteKeysFillOnGpu(keysFill->base, keysFill->bytes, *filledKeys);
        }
        if (builtRecipe != nullptr) {
            if (dataHit) {

                auto own = std::make_shared<Recipe>(*builtRecipe);
                own->dataWordsHash = Graphics::ShaderResources::DataWordsHash({ShaderRecompiler::ShaderStage::Compute, keepVariant->compiled.get(), 0});
                builtRecipe = std::move(own);
            }
            attachTo->recipe.store(std::move(builtRecipe), std::memory_order_release);
            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, indirectArguments != 0);
        }
        break;
    }
    phaseTiming.Phase(PhaseDevice);
    if (noteWrites && !writerKeyedEvidence()) noteWrittenBuffers(address, submission.queue, compiled);
    deviceMs += phaseTiming.Elapsed();
    phaseTiming.Phase(PhaseTail);
    if (profile) {
        auto& pending = pendingDispatchPhases();
        pending.outcome = DispatchOutcome::Real;
        pending.phases = true;
        pending.hit = cached;
        pending.validated = validated;
        pending.ms = phaseMs;
        pending.tailAt = phaseLap;
    }
    if (profile && ++dispatches % 100 == 0) AgcDriver::ProfilePrint_nid_no_patch( "[gpu] %llu dispatches (%llu dispatch cache hits, %llu evictions, %llu recompile cache hits): capture %.1f s, cache key %.1f s, recompile %.1f s, device %.1f s\n", static_cast<unsigned long long>(dispatches), static_cast<unsigned long long>(dispatchCacheHits), static_cast<unsigned long long>(dispatchCacheEvictions), static_cast<unsigned long long>(cacheHits), captureMs / 1000, keyMs / 1000, recompileMs / 1000, deviceMs / 1000);
}

}
