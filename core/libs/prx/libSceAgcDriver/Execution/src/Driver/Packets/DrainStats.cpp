#include "prx/libSceAgcDriver/Execution/include/DrainStats.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GdsStream.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libc/include/HostThreadSlot.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace AgcDriver::Graphics {
bool RecipeRefusalStats();
}

namespace AgcDriver::Drain {

namespace {

using Counter = Stream::Counter;
using Stream::Add;
using Stream::Take;

// C6's totals at the last [dispatch-guards] line (the reporter is queue 0's thread, in Enqueue).
std::array<std::uint64_t, Guards> lastHits{};
std::array<std::uint64_t, Guards> lastDrains{};

// LOCAL DEBUG (DRAIN-PLAN section 6): see DrainStats.hpp. The queue of each per-queue slot (the
// queue id + 1, 0 free).
std::array<std::atomic<std::uint32_t>, QueueSlots> slotQueues{};

std::size_t queueSlot(std::uint32_t queue) {
    const auto tagged = queue + 1u;
    for (std::size_t slot = 0; slot + 1 < QueueSlots; ++slot) {
        auto current = slotQueues[slot].load(std::memory_order_acquire);
        if (current == tagged) return slot;
        if (current == 0) {
            if (slotQueues[slot].compare_exchange_strong(current, tagged, std::memory_order_acq_rel)) return slot;
            if (current == tagged) return slot;
        }
    }
    return QueueSlots - 1;
}

std::string queueName(std::size_t slot) {
    const auto tagged = slotQueues[slot].load(std::memory_order_acquire);
    if (slot + 1 == QueueSlots || tagged == 0) return slot + 1 == QueueSlots ? "other" : "-";
    char text[16];
    std::snprintf(text, sizeof(text), "0x%x", tagged - 1u);
    return text;
}

double ms(std::uint64_t ns) {
    return static_cast<double>(ns) / 1e6;
}

unsigned long long count(std::uint64_t value) {
    return static_cast<unsigned long long>(value);
}

// Formatted with printf's checks; a piece is cut at 191 characters (as the [stream] line's).
__attribute__((format(gnu_printf, 2, 3))) void append(std::string& text, const char* format, ...) {
    char item[192];
    std::va_list args;
    va_start(args, format);
    const auto size = std::vsnprintf(item, sizeof(item), format, args);
    va_end(args);
    if (size > 0) text.append(item, std::min(static_cast<std::size_t>(size), sizeof(item) - 1));
}

constexpr std::array<const char*, static_cast<std::size_t>(Window::Count)> WindowNames{"menu", "entry", "hold"};
constexpr std::array<const char*, DmaKinds> DmaKindNames{"imm->mem", "mem->mem", "imm->gds", "mem->gds", "gds->mem", "gds->gds", "prefetch"};
constexpr std::array<const char*, DmaPaths> DmaPathNames{"item", "nothing", "gpu-store", "gpu-store-after-cpu-read", "gpu-copy", "gpu-fill", "cpu-after-wait", "cpu-idle", "device-drain", "fallback"};
constexpr std::size_t GdsTargets = static_cast<std::size_t>(GdsTarget::Count);
constexpr std::size_t Touches = static_cast<std::size_t>(Touch::Count);
constexpr std::array<const char*, Touches> TouchNames{"wait", "dispatch-args", "cond-exec", "predication", "dma-source", "hook-read", "overwritten", "aged-out"};
constexpr std::size_t WaitHows = static_cast<std::size_t>(WaitHow::Count);
constexpr std::array<const char*, WaitHows> WaitHowNames{"entry", "held", "stored", "table", "polled", "queued"};
constexpr std::size_t GateRefusals = static_cast<std::size_t>(GateRefusal::Count);
constexpr std::array<const char*, GateRefusals> GateRefusalNames{"would-skip", "deferred-labels", "label-batch-submit", "A1", "group-capture", "unordered-fills", "hidden-write"};
constexpr std::size_t GateExits = static_cast<std::size_t>(GateExit::Count);
constexpr std::array<const char*, GateExits> GateExitNames{"fill-item", "return", "msaa-guard", "value-fill-guard", "capture", "non-fill"};
constexpr std::array<const char*, Guards> GuardNames{"msaa", "value-fill", "hidden-shader", "hidden-fill"};
constexpr std::array<const char*, RecipeOutcomes> RecipeOutcomeNames{"recipe-hit", "rebuild", "precheck-refused", "stamp-validate", "no-attach", "built", "recipes-off", "uncacheable", "moved-overlay", "not-reusable", "needs-completion", "no-objects", "no-content-key"};
constexpr std::size_t UnreusableReasons = static_cast<std::size_t>(Graphics::ShaderResources::Unreusable::Count);
enum IndirectMiss : std::size_t { MissFirst, MissSameKey, MissUserData, MissPartialThreads, MissUserDataPartial, MissRegisters, MissShader, MissWave, MissSeveral, IndirectMisses };
constexpr std::array<const char*, IndirectMisses> IndirectMissNames{"first", "same-key", "user-data", "partial-threads", "user-data+partial-threads", "registers", "shader", "wave", "several"};

// C2: queue-0 DMA_DATA by kind and path, and C7's dumps (row 0: on the GPU, 1: on the CPU).
struct PacketRow {
    Counter n{};
    Counter drained{};
    Counter drainWaitNs{};
    Counter packetNs{};
    Counter gpuWaitNs{};
    Counter episodes{};
    Counter idleNs{};
    Counter serialNs{};
};
std::array<std::array<PacketRow, DmaPaths>, DmaKinds> dma0Rows{};
std::array<Counter, FallbackReasons> dma0Fallbacks{};
std::array<PacketRow, 2> zpassRows{};
Counter zpassOtherQueues{};

// C1: DMA_DATA touching the GDS by queue, kind and path: count, CPU wait, the GDS's unfinished
// writers when it started (GdsTarget: idle / in flight / in the open batch, split under a targeted
// store's own hold / pending, the lock-free snapshot).
struct GdsRow {
    Counter n{};
    Counter waitNs{};
    std::array<Counter, GdsTargets> target{};
};
std::array<std::array<std::array<GdsRow, DmaPaths>, DmaKinds>, QueueSlots> gdsRows{};
std::array<std::array<Counter, FallbackReasons>, QueueSlots> gdsFallbacks{};
// Per queue, the GPU waits of its GDS DMA_DATAs so far, added as each wait ends (Stream::GpuWaitScope
// through Stream::NoteGdsGpuWait, and noteDeviceDrain): the 0x93 blame's clock.
std::array<Counter, QueueSlots> gdsPacketWaitNs{};

// C1's readback destinations, per producer queue, under ringMutex (a leaf lock); [ringLow, ringHigh)
// bounds the live entries so most accesses are refused without the lock.
constexpr std::size_t RingEntries = 64;
struct Ring {
    std::array<Range, RingEntries> entries{};
    std::size_t next = 0;
};
std::mutex ringMutex;
std::array<Ring, QueueSlots> rings{};
std::size_t ringLive = 0;
std::atomic<std::uint64_t> ringLow{~0ull};
std::atomic<std::uint64_t> ringHigh{0};
std::array<Counter, QueueSlots> readbacksPushed{};
std::array<std::array<Counter, Touches>, QueueSlots> firstTouches{};
std::array<std::array<std::array<Counter, WaitHows>, QueueSlots>, QueueSlots> waitTouches{};
// The 0x93 blame: queue-0 waits that polled, by the producer queue of the satisfying label.
Counter blameWaits{};
Counter blameNs{};
std::array<Counter, QueueSlots> blameByProducer{};
std::array<Counter, QueueSlots> blameByProducerNs{};
std::array<Counter, QueueSlots> blameInGds{};
std::array<Counter, QueueSlots> blameInGdsNs{};
Counter blameUnknown{};
Counter blameUnknownNs{};

// C3.
Counter gateFirst{};
std::array<Counter, GateRefusals> gateRefusals{};
std::array<Counter, 2> gateWaitNs{};
std::array<std::array<Counter, GateExits>, 2> gateExits{};
std::array<std::array<Counter, GateExits>, 2> gateExitNs{};
Counter gateOverlapFillNs{};
Counter gateOverlapNonFillNs{};

// C6, cumulative (GuardTotals); the report keeps what it printed last.
std::array<Counter, Guards> guardHits{};
std::array<Counter, Guards> guardDrains{};

// C5, by template (1) or built (0).
std::array<std::array<Counter, RecipeOutcomes>, 2> recipeOutcomes{};
std::array<std::array<Counter, UnreusableReasons>, 2> recipeUnreusable{};
Counter indirectDispatches{};
Counter indirectHits{};
std::array<Counter, IndirectMisses> indirectMisses{};

// The thread's open PacketRecord path note: Armed with the kind (bits 24-31), then Noted with the
// path (bits 8-15) and the fallback reason (bits 16-23), and a GDS target sampled under a lock
// (noteGdsTarget: bits 2-4, the GdsTarget + 1; 0 none); and the queue-0 drain episode the packet
// opened (0: none), tagged with the path as soon as it is known, since a fill's own push closes it
// before the packet ends. The device drains' unlocked waits on this thread (noteDeviceDrain), a
// clock the records read as they read Stream::ThreadGpuWait.
struct DmaPathTag {};
using DmaPathSlot = HostThreadSlot<std::uint32_t, DmaPathTag>;
struct DmaEpisodeTag {};
using DmaEpisodeSlot = HostThreadSlot<std::uint64_t, DmaEpisodeTag>;
struct DeviceDrainTag {};
using DeviceDrainSlot = HostThreadSlot<std::uint64_t, DeviceDrainTag>;
constexpr std::uint32_t PathArmed = 1u;
constexpr std::uint32_t PathNoted = 2u;
constexpr std::uint32_t PathTargetShift = 2u;
constexpr std::uint32_t PathTargetMask = 7u << PathTargetShift;

// C5's device-call note (RecipeNote): Armed, then Noted, FromCache, the outcome (bits 8-15) and the
// unreusable reason (bits 16-23).
struct RecipeNoteTag {};
using RecipeNoteSlot = HostThreadSlot<std::uint32_t, RecipeNoteTag>;
constexpr std::uint32_t NoteArmed = 1u;
constexpr std::uint32_t NoteNoted = 2u;
constexpr std::uint32_t NoteFromCache = 4u;

// C3's window, on queue 0's thread.
struct GateWindowState {
    bool open = false;
    bool wouldSkip = false;
    bool hidden = false;
    std::uint64_t waitNs = 0;
    std::uint64_t startNs = 0;
    std::vector<Range> writes;
};
struct GateWindowTag {};
GateWindowState& gateWindow() {
    return ThreadScratch<GateWindowState, GateWindowTag>();
}

// C5's indirect keys by shader address (queue 0 only; the lock for dispatch-ahead's other threads).
struct IndirectParts {
    std::vector<std::uint32_t> userData;
    std::array<std::uint32_t, 3> partialThreads{};
    std::array<std::uint32_t, 5> registers{};
    const void* shader = nullptr;
    std::uint32_t wave = 0;
};
std::mutex indirectMutex;
std::unordered_map<std::uint64_t, IndirectParts> indirectParts;

bool memorySource(std::span<const std::uint32_t> packet) {
    const auto source = Pm4::DmaSource(packet);
    return source == 0 || source == 3;
}

// Takes the live readback entries overlapping the range out of their rings (their first touch).
// Each producer slot taken is passed to `taken`. Under nothing; takes ringMutex.
template<typename TTaken>
void takeOverlapping(std::uint64_t address, std::size_t bytes, TTaken&& taken) {
    if (bytes == 0) return;
    const auto end = address + bytes;
    if (address >= ringHigh.load(std::memory_order_relaxed) || end <= ringLow.load(std::memory_order_relaxed)) return;
    std::lock_guard lock(ringMutex);
    for (std::size_t slot = 0; slot < QueueSlots; ++slot) {
        for (auto& entry : rings[slot].entries) {
            if (entry.second == 0 || address >= entry.second || entry.first >= end) continue;
            entry = {0, 0};
            --ringLive;
            taken(slot);
        }
    }
    if (ringLive == 0) {
        ringLow.store(~0ull, std::memory_order_relaxed);
        ringHigh.store(0, std::memory_order_relaxed);
    }
}

void pushReadback(std::size_t slot, std::uint64_t address, std::size_t bytes) {
    if (bytes == 0) return;
    std::lock_guard lock(ringMutex);
    auto& ring = rings[slot];
    auto& entry = ring.entries[ring.next];
    ring.next = (ring.next + 1) % RingEntries;
    if (entry.second != 0) Add(firstTouches[slot][static_cast<std::size_t>(Touch::AgedOut)]);
    else ++ringLive;
    entry = {address, address + bytes};
    Add(readbacksPushed[slot]);
    if (address < ringLow.load(std::memory_order_relaxed)) ringLow.store(address, std::memory_order_relaxed);
    if (address + bytes > ringHigh.load(std::memory_order_relaxed)) ringHigh.store(address + bytes, std::memory_order_relaxed);
}

std::size_t liveReadbacks(std::size_t slot) {
    std::lock_guard lock(ringMutex);
    return static_cast<std::size_t>(std::count_if(rings[slot].entries.begin(), rings[slot].entries.end(), [](const Range& entry) { return entry.second != 0; }));
}

void appendPacketRow(std::string& line, const char* name, PacketRow& row, bool& first) {
    const auto n = Take(row.n);
    const auto drained = Take(row.drained);
    const auto drainWaitNs = Take(row.drainWaitNs);
    const auto packetNs = Take(row.packetNs);
    const auto gpuWaitNs = Take(row.gpuWaitNs);
    const auto episodes = Take(row.episodes);
    const auto idleNs = Take(row.idleNs);
    const auto serialNs = Take(row.serialNs);
    if (n == 0 && episodes == 0 && idleNs == 0) return;
    if (!first) line += ',';
    first = false;
    append(line, " %s %llu %llu %.1f %.1f %.1f %llu %.1f %.1f", name, count(n), count(drained), ms(drainWaitNs), ms(packetNs), ms(gpuWaitNs), count(episodes), ms(idleNs), ms(serialNs));
}

void reportGds(const char* window, bool print = true) {
    std::string line;
    // Pending: in flight or open, from the lock-free snapshot (split only under a targeted store's hold).
    append(line, "[gds] window=%s DMA_DATA touching the GDS (queue kind/path n, CPU wait ms, target idle/in-flight/open/pending):", window);
    bool first = true;
    for (std::size_t slot = 0; slot < QueueSlots; ++slot) {
        for (std::size_t kind = 0; kind < DmaKinds; ++kind) {
            for (std::size_t path = 0; path < DmaPaths; ++path) {
                auto& row = gdsRows[slot][kind][path];
                const auto n = Take(row.n);
                const auto waitNs = Take(row.waitNs);
                std::array<std::uint64_t, GdsTargets> target{};
                for (std::size_t state = 0; state < GdsTargets; ++state) target[state] = Take(row.target[state]);
                if (n == 0) continue;
                if (!first) line += ',';
                first = false;
                append(line, " %s %s/%s %llu %.1f %llu/%llu/%llu/%llu", queueName(slot).c_str(), DmaKindNames[kind], DmaPathNames[path], count(n), ms(waitNs), count(target[0]), count(target[1]), count(target[2]), count(target[3]));
            }
        }
    }
    append(line, "; fallback reasons (queue rN n):");
    first = true;
    for (std::size_t slot = 0; slot < QueueSlots; ++slot) {
        for (std::size_t reason = 0; reason < FallbackReasons; ++reason) {
            const auto n = Take(gdsFallbacks[slot][reason]);
            if (n == 0) continue;
            if (!first) line += ',';
            first = false;
            append(line, " %s r%zu %llu", queueName(slot).c_str(), reason, count(n));
        }
    }
    // Readback destinations: pushed, still live, and every first touch (waits by waiter and how).
    // Gate (c)'s share: CPU uses (every wait but a queued-label answer, and every read) of the first
    // touches whose outcome was seen (aged-out entries apart).
    append(line, "; GDS readbacks by producer queue (pushed, live: first touches n):");
    std::uint64_t firstAll = 0, cpuFirst = 0, agedOut = 0;
    first = true;
    for (std::size_t slot = 0; slot < QueueSlots; ++slot) {
        const auto pushed = Take(readbacksPushed[slot]);
        std::array<std::uint64_t, Touches> touches{};
        for (std::size_t touch = 0; touch < Touches; ++touch) touches[touch] = Take(firstTouches[slot][touch]);
        std::array<std::array<std::uint64_t, WaitHows>, QueueSlots> waits{};
        for (std::size_t waiter = 0; waiter < QueueSlots; ++waiter) {
            for (std::size_t how = 0; how < WaitHows; ++how) waits[waiter][how] = Take(waitTouches[slot][waiter][how]);
        }
        std::uint64_t touchSum = 0;
        for (const auto value : touches) touchSum += value;
        const auto live = liveReadbacks(slot);
        if (pushed == 0 && touchSum == 0 && live == 0) continue;
        if (!first) line += ',';
        first = false;
        append(line, " %s %llu %zu:", queueName(slot).c_str(), count(pushed), live);
        for (std::size_t waiter = 0; waiter < QueueSlots; ++waiter) {
            for (std::size_t how = 0; how < WaitHows; ++how) {
                if (waits[waiter][how] == 0) continue;
                append(line, " wait/%s/%s %llu", queueName(waiter).c_str(), WaitHowNames[how], count(waits[waiter][how]));
                // A wait answered from the recorder's label table still used the readback's value
                // (filed there by its store); one answered from a label queued after it did not.
                if (how != static_cast<std::size_t>(WaitHow::Queued)) cpuFirst += waits[waiter][how];
            }
        }
        for (std::size_t touch = 1; touch < Touches; ++touch) {
            if (touches[touch] == 0) continue;
            append(line, " %s %llu", TouchNames[touch], count(touches[touch]));
            if (touch != static_cast<std::size_t>(Touch::Overwritten) && touch != static_cast<std::size_t>(Touch::AgedOut)) cpuFirst += touches[touch];
        }
        agedOut += touches[static_cast<std::size_t>(Touch::AgedOut)];
        firstAll += touchSum;
    }
    const auto seen = firstAll - agedOut;
    append(line, "; CPU uses first %llu of %llu first touches seen (%.1f%%), aged out %llu (not counted)", count(cpuFirst), count(seen), seen == 0 ? 0.0 : 100.0 * static_cast<double>(cpuFirst) / static_cast<double>(seen), count(agedOut));
    // The 0x93 blame.
    const auto waits = Take(blameWaits);
    const auto waitNs = Take(blameNs);
    append(line, "; queue-0 waits that polled %llu, %.1f ms; by producer queue (n, ms; with its GDS DMA_DATA GPU waits ending inside n, their ms capped at each wait):", count(waits), ms(waitNs));
    first = true;
    for (std::size_t slot = 0; slot < QueueSlots; ++slot) {
        const auto n = Take(blameByProducer[slot]);
        const auto nNs = Take(blameByProducerNs[slot]);
        const auto inGds = Take(blameInGds[slot]);
        const auto inGdsNs = Take(blameInGdsNs[slot]);
        if (n == 0) continue;
        if (!first) line += ',';
        first = false;
        append(line, " %s %llu %.1f %llu %.1f", queueName(slot).c_str(), count(n), ms(nNs), count(inGds), ms(inGdsNs));
    }
    append(line, "; producer unknown %llu, %.1f ms", count(Take(blameUnknown)), ms(Take(blameUnknownNs)));
    if (print) AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
}

// [gds] without stream stats: every 10 s, from the queue worker that files a GDS packet.
void maybeReportGdsAlone() {
    static std::atomic<std::int64_t> last{0};
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto previous = last.load(std::memory_order_relaxed);
    if (previous == 0) {
        last.compare_exchange_strong(previous, now);
        return;
    }
    if (now - previous < 10000 || !last.compare_exchange_strong(previous, now)) return;
    reportGds("-");
}

void reportRecipes(const char* window) {
    std::string line;
    append(line, "[recipe] dispatch window=%s queue-0 direct dispatch-cache hits by the device call's outcome (template/built):", window);
    bool first = true;
    for (std::size_t outcome = 0; outcome < RecipeOutcomes; ++outcome) {
        const auto templated = Take(recipeOutcomes[1][outcome]);
        const auto built = Take(recipeOutcomes[0][outcome]);
        if (!first) line += ',';
        first = false;
        append(line, " %s %llu/%llu", RecipeOutcomeNames[outcome], count(templated), count(built));
    }
    append(line, "; not-reusable by the first reason (template/built):");
    first = true;
    for (std::size_t reason = 1; reason < UnreusableReasons; ++reason) {
        const auto templated = Take(recipeUnreusable[1][reason]);
        const auto built = Take(recipeUnreusable[0][reason]);
        if (templated == 0 && built == 0) continue;
        if (!first) line += ',';
        first = false;
        append(line, " %s %llu/%llu", Graphics::ShaderResources::UnreusableNames[reason], count(templated), count(built));
    }
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
}

void maybeReportRecipesAlone() {
    static std::atomic<std::int64_t> last{0};
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto previous = last.load(std::memory_order_relaxed);
    if (previous == 0) {
        last.compare_exchange_strong(previous, now);
        return;
    }
    if (now - previous < 10000 || !last.compare_exchange_strong(previous, now)) return;
    reportRecipes("-");
}

}

namespace {

int experimentMode(const char* name) {
    const char* text = std::getenv(name);
    if (text == nullptr || text[0] == '\0' || std::strcmp(text, "0") == 0) return 0;
    return std::strcmp(text, "2") == 0 ? 2 : 1;
}

}

int noteGuardMode() {
    return experimentMode("APS5_EXPERIMENT_DISPATCH_NOTE_GUARDS");
}

int hiddenWriteMode() {
    return experimentMode("APS5_EXPERIMENT_HIDDEN_WRITE_DRAIN");
}

bool GdsOn() {
    return Stream::GdsStatsOn();
}

Window ClassifyWindow(std::uint64_t presents, std::uint64_t draws, std::uint64_t leaseQueue0Ns, Window previous) {
    if (presents == 0 || draws / presents < LightDrawsPerPresent) return Window::Menu;
    if (previous == Window::Menu || leaseQueue0Ns >= EntryLeaseWaitNs) return Window::Entry;
    return Window::Hold;
}

Window TagWindow(std::uint64_t presents, std::uint64_t draws, std::uint64_t leaseQueue0Ns) {
    // One report per window (DrawPipeline::reportStream), possibly from different threads.
    static std::atomic<std::uint8_t> previous{static_cast<std::uint8_t>(Window::Menu)};
    const auto window = ClassifyWindow(presents, draws, leaseQueue0Ns, static_cast<Window>(previous.load(std::memory_order_relaxed)));
    previous.store(static_cast<std::uint8_t>(window), std::memory_order_relaxed);
    return window;
}

const char* WindowName(Window window) {
    return WindowNames[static_cast<std::size_t>(window)];
}

DmaKind ClassifyDma(std::span<const std::uint32_t> packet) {
    if (packet.size() < 7) return DmaKind::Prefetch;
    const auto source = Pm4::DmaSource(packet);
    const auto destination = Pm4::DmaDestination(packet);
    if (destination == Pm4::DmaDestinationNowhere) return DmaKind::Prefetch;
    const bool toGds = destination == Pm4::DmaSelectGds;
    if (source == Pm4::DmaSelectGds) return toGds ? DmaKind::GdsToGds : DmaKind::GdsToMem;
    if (source == Pm4::DmaSourceData) return toGds ? DmaKind::ImmToGds : DmaKind::ImmToMem;
    return toGds ? DmaKind::MemToGds : DmaKind::MemToMem;
}

void noteDmaPath(DmaPath path, int reason) {
    const auto slot = DmaPathSlot::Get();
    if ((slot & PathArmed) == 0) return;
    const auto kind = (slot >> 24u) & 0xffu;
    DmaPathSlot::Set(PathArmed | PathNoted | (slot & PathTargetMask) | (static_cast<std::uint32_t>(path) << 8u) | (static_cast<std::uint32_t>(std::clamp(reason, 0, static_cast<int>(FallbackReasons) - 1)) << 16u) | (kind << 24u));
    if (const auto episode = DmaEpisodeSlot::Get(); episode != 0) DriverDetail::DrawPipeline::Queue0().TagStreamEpisode(episode, static_cast<std::uint16_t>(TagDma0 | (kind * DmaPaths + static_cast<std::uint32_t>(path))));
}

void noteGdsTarget(GdsTarget target) {
    const auto slot = DmaPathSlot::Get();
    if ((slot & PathArmed) == 0) return;
    DmaPathSlot::Set((slot & ~PathTargetMask) | ((static_cast<std::uint32_t>(target) + 1u) << PathTargetShift));
}

void noteDeviceDrain(std::uint64_t unlockedWaitNs) {
    if (unlockedWaitNs != 0) {
        DeviceDrainSlot::Set(DeviceDrainSlot::Get() + unlockedWaitNs);
        if (const auto gds = Stream::ThreadGdsSlot::Get(); gds != 0 && gds <= QueueSlots) Add(gdsPacketWaitNs[gds - 1u], unlockedWaitNs);
    }
    const auto slot = DmaPathSlot::Get();
    if ((slot & PathArmed) != 0 && (slot & PathNoted) == 0) noteDmaPath(DmaPath::DeviceDrain, 0);
}

// A record armed but not finished (the packet threw).
void PacketRecord::release() {
    DmaPathSlot::Set(0);
    DmaEpisodeSlot::Set(0);
    if (gds) Stream::ThreadGdsSlot::Set(0);
    armed = false;
}

void PacketRecord::Start(std::span<const std::uint32_t> packet, std::uint32_t opcode, std::uint32_t queueId) {
    queue = queueId;
    if (opcode == 0x50) {
        if (packet.size() < 7) return;
        kind = ClassifyDma(packet);
        // C1: a later DMA_DATA's memory source is a readback destination's first touch.
        if (GdsCounting() && memorySource(packet)) noteTouch(Touch::DmaSource, packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u), Pm4::DmaBytes(packet));
        stream = Stream::On() && queue == 0;
        gds = GdsCounting() && TouchesGds(kind);
        if (!stream && !gds) return;
        destination = packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u);
        bytes = Pm4::DmaBytes(packet);
        active = true;
        return;
    }
    if (opcode == 0x46 && Stream::On() && Pm4::SampleDump(packet)) {
        if (queue != 0) {
            Add(zpassOtherQueues);
            return;
        }
        zpass = true;
        stream = true;
        active = true;
    }
}

void PacketRecord::NoteDrain(std::uint64_t waitNs, std::uint64_t episodeBefore, std::uint64_t episodeAfter) {
    drained = true;
    drainWaitNs = waitNs;
    episode = episodeAfter != 0 && episodeAfter != episodeBefore ? episodeAfter : 0;
}

void PacketRecord::Arm() {
    if (!active) return;
    if (!zpass) {
        armed = true;
        DmaPathSlot::Set(PathArmed | (static_cast<std::uint32_t>(kind) << 24u));
        DmaEpisodeSlot::Set(stream ? episode : 0);
    }
    if (gds) {
        // The GDS's unfinished writers as the packet starts, from the published pending-write
        // snapshot (no lock: GpuMutex here would be a hold today's packet does not make). A targeted
        // GDS store splits it under its own hold later (noteGdsTarget).
        target = Graphics::Recorder::SnapshotWriteOverlaps(Pm4::GdsAddress(), Pm4::GdsBytes) ? GdsTarget::Pending : GdsTarget::Idle;
        // This thread's GPU waits from here on run the queue's GDS-wait clock (the 0x93 blame).
        Stream::ThreadGdsSlot::Set(static_cast<std::uint32_t>(queueSlot(queue)) + 1u);
    }
    startNs = Stream::NowNs();
    gpuWaitNs = Stream::ThreadGpuWait::Get();
    deviceDrainNs = DeviceDrainSlot::Get();
}

void PacketRecord::Finish(bool wroteOnGpu) {
    if (!active) return;
    active = false;
    const auto packetNs = Stream::NowNs() - startNs;
    const auto waitedNs = (Stream::ThreadGpuWait::Get() - gpuWaitNs) + (DeviceDrainSlot::Get() - deviceDrainNs);
    if (zpass) {
        auto& row = zpassRows[wroteOnGpu ? 0 : 1];
        Add(row.n);
        if (drained) {
            Add(row.drained);
            Add(row.drainWaitNs, drainWaitNs);
        }
        Add(row.packetNs, packetNs);
        Add(row.gpuWaitNs, waitedNs);
        if (episode != 0) DriverDetail::DrawPipeline::Queue0().TagStreamEpisode(episode, static_cast<std::uint16_t>(TagZpass | (wroteOnGpu ? 0u : 1u)));
        return;
    }
    auto path = DmaPath::Nothing;
    std::size_t reason = 0;
    const auto note = DmaPathSlot::Get();
    if ((note & PathNoted) != 0) {
        path = static_cast<DmaPath>((note >> 8u) & 0xffu);
        reason = (note >> 16u) & 0xffu;
    } else if (!wroteOnGpu) {
        path = waitedNs != 0 ? DmaPath::CpuAfterWait : DmaPath::CpuIdle;
    }
    if (const auto sampled = (note & PathTargetMask) >> PathTargetShift; sampled != 0 && sampled <= GdsTargets) target = static_cast<GdsTarget>(sampled - 1u);
    if (armed) {
        armed = false;
        DmaPathSlot::Set(0);
        DmaEpisodeSlot::Set(0);
    }
    const auto kindIndex = static_cast<std::size_t>(kind);
    const auto pathIndex = static_cast<std::size_t>(path);
    if (stream) {
        auto& row = dma0Rows[kindIndex][pathIndex];
        Add(row.n);
        if (drained) {
            Add(row.drained);
            Add(row.drainWaitNs, drainWaitNs);
        }
        Add(row.packetNs, packetNs);
        Add(row.gpuWaitNs, waitedNs);
        if (path == DmaPath::Fallback) Add(dma0Fallbacks[reason]);
        if (episode != 0) DriverDetail::DrawPipeline::Queue0().TagStreamEpisode(episode, static_cast<std::uint16_t>(TagDma0 | (kindIndex * DmaPaths + pathIndex)));
    }
    if (gds) {
        const auto slot = queueSlot(queue);
        auto& row = gdsRows[slot][kindIndex][pathIndex];
        Add(row.n);
        Add(row.waitNs, waitedNs);
        Add(row.target[static_cast<std::size_t>(target)]);
        if (path == DmaPath::Fallback) Add(gdsFallbacks[slot][reason]);
        Stream::ThreadGdsSlot::Set(0);
        gds = false;
        // The destination of a GDS readback, from now on until its first touch.
        if (kind == DmaKind::GdsToMem && path != DmaPath::Nothing) pushReadback(slot, destination, bytes);
        if (!Stream::On()) maybeReportGdsAlone();
    }
}

void CountDmaItem(std::span<const std::uint32_t> packet, std::uint32_t queue) {
    if (packet.size() < 7) return;
    const auto kind = ClassifyDma(packet);
    if (GdsCounting() && memorySource(packet)) noteTouch(Touch::DmaSource, packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u), Pm4::DmaBytes(packet));
    if (Stream::On() && queue == 0) Add(dma0Rows[static_cast<std::size_t>(kind)][static_cast<std::size_t>(DmaPath::Item)].n);
    if (GdsCounting() && TouchesGds(kind)) Add(gdsRows[queueSlot(queue)][static_cast<std::size_t>(kind)][static_cast<std::size_t>(DmaPath::Item)].n);
}

void FileTaggedEpisode(std::uint16_t tag, std::uint64_t idleNs, std::uint64_t serialNs, bool closed) {
    PacketRow* row = nullptr;
    const auto slot = static_cast<std::size_t>(tag & 0xffu);
    if ((tag & 0xff00u) == TagDma0 && slot < DmaKinds * DmaPaths) row = &dma0Rows[slot / DmaPaths][slot % DmaPaths];
    else if ((tag & 0xff00u) == TagZpass && slot < zpassRows.size()) row = &zpassRows[slot];
    if (row == nullptr) return;
    if (closed) Add(row->episodes);
    Add(row->idleNs, idleNs);
    Add(row->serialNs, serialNs);
}

void noteTouch(Touch touch, std::uint64_t address, std::size_t bytes) {
    takeOverlapping(address, bytes, [touch](std::size_t slot) { Add(firstTouches[slot][static_cast<std::size_t>(touch)]); });
}

void noteGuestAccess(std::uint64_t address, std::size_t bytes, bool store) {
    noteTouch(store ? Touch::Overwritten : Touch::HookRead, address, bytes);
}

void GuestAccess(std::uint64_t address, std::size_t bytes, bool store) {
    if (GdsCounting()) noteGuestAccess(address, bytes, store);
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_GDS_IN_STREAM, DRAIN-PLAN D2): armed by its first recorded
    // operation only, so the switch is not read here.
    if (GdsStream::Armed()) GdsStream::CheckGuestAccess(address, bytes);
}

namespace {
// GuestMemory's flush path calls the counter through a registered pointer (it links without this file).
const bool guestAccessCounterRegistered = [] {
    if (GdsCounting()) GuestMemory::SetGuestAccessCounter(&GuestAccess);
    return true;
}();
}

void noteWaitTouch(std::uint32_t queue, std::uint64_t address, std::size_t bytes, WaitHow how) {
    const auto waiter = queueSlot(queue);
    takeOverlapping(address, bytes, [&](std::size_t slot) {
        Add(waitTouches[slot][waiter][static_cast<std::size_t>(how)]);
        Add(firstTouches[slot][static_cast<std::size_t>(Touch::Wait)]);
    });
}

void WaitRecord::start(std::uint64_t awaited, std::size_t awaitedBytes) {
    startNs = Stream::NowNs();
    takeOverlapping(awaited, awaitedBytes, [this](std::size_t slot) {
        if (taken < producers.size()) producers[taken] = static_cast<std::uint8_t>(slot);
        ++taken;
    });
    if (queue != 0) return;
    for (std::size_t slot = 0; slot < QueueSlots; ++slot) gdsWaitNs[slot] = gdsPacketWaitNs[slot].load(std::memory_order_relaxed);
}

void WaitRecord::finish() {
    const auto waiter = queueSlot(queue);
    for (std::uint32_t index = 0; index < taken; ++index) {
        // Past the kept producers (more than four destinations under one wait): the first one's.
        const auto producerSlot = producers[std::min<std::size_t>(index, producers.size() - 1)];
        Add(waitTouches[producerSlot][waiter][static_cast<std::size_t>(how)]);
        Add(firstTouches[producerSlot][static_cast<std::size_t>(Touch::Wait)]);
    }
    if (queue != 0 || !polled) return;
    const auto ns = Stream::NowNs() - startNs;
    Add(blameWaits);
    Add(blameNs, ns);
    if (producer == ~0u) {
        Add(blameUnknown);
        Add(blameUnknownNs, ns);
        return;
    }
    const auto slot = queueSlot(producer);
    Add(blameByProducer[slot]);
    Add(blameByProducerNs[slot], ns);
    // The producer's GDS DMA_DATA GPU waits that ended during this wait, capped at the wait.
    const auto gdsNs = gdsPacketWaitNs[slot].load(std::memory_order_relaxed) - gdsWaitNs[slot];
    if (gdsNs != 0) {
        Add(blameInGds[slot]);
        Add(blameInGdsNs[slot], std::min(ns, gdsNs));
    }
}

std::vector<Range>& GateWrites() {
    auto& window = gateWindow();
    if (window.open) GateClose(GateExit::Return);
    window.writes.clear();
    return window.writes;
}

void GateOpen(GateRefusal refusal, std::uint64_t waitNs, bool hidden) {
    auto& window = gateWindow();
    if (window.open) GateClose(GateExit::Return);
    Add(gateFirst);
    Add(gateRefusals[static_cast<std::size_t>(refusal)]);
    window.wouldSkip = refusal == GateRefusal::None;
    Add(gateWaitNs[window.wouldSkip ? 0 : 1], waitNs);
    window.open = true;
    window.hidden = hidden;
    window.waitNs = waitNs;
    window.startNs = Stream::NowNs();
}

void GateCounted(GateRefusal refusal) {
    Add(gateFirst);
    Add(gateRefusals[static_cast<std::size_t>(refusal)]);
}

bool GateWindowOpen() {
    return gateWindow().open;
}

void GateCapture(std::uint64_t address, std::size_t bytes) {
    auto& window = gateWindow();
    if (!window.open || bytes == 0) return;
    const auto end = address + bytes;
    const bool overlaps = std::any_of(window.writes.begin(), window.writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
    if (window.hidden || overlaps) GateClose(GateExit::Capture);
}

void GateClose(GateExit exit) {
    auto& window = gateWindow();
    if (!window.open) return;
    window.open = false;
    const auto windowNs = Stream::NowNs() - window.startNs;
    const auto row = window.wouldSkip ? 0u : 1u;
    Add(gateExits[row][static_cast<std::size_t>(exit)]);
    Add(gateExitNs[row][static_cast<std::size_t>(exit)], windowNs);
    if (!window.wouldSkip) return;
    if (exit == GateExit::FillItem) Add(gateOverlapFillNs, std::min(window.waitNs, windowNs));
    else if (exit == GateExit::NonFill) Add(gateOverlapNonFillNs, std::min(window.waitNs, windowNs));
}

void NoteGuard(Guard guard, bool drained) {
    Add(guardHits[static_cast<std::size_t>(guard)]);
    if (drained) Add(guardDrains[static_cast<std::size_t>(guard)]);
}

GuardCounts GuardTotals() {
    GuardCounts totals;
    for (std::size_t guard = 0; guard < Guards; ++guard) {
        totals.hits[guard] = guardHits[guard].load(std::memory_order_relaxed);
        totals.drains[guard] = guardDrains[guard].load(std::memory_order_relaxed);
    }
    return totals;
}

bool recipeRefusalStats() {
    return Graphics::RecipeRefusalStats();
}

void RecipeNote::arm(bool on) {
    RecipeNoteSlot::Set(on ? NoteArmed : 0u);
}

bool RecipeNote::Taken(RecipeOutcome& outcome, std::uint8_t& unreusable, bool& fromCache) const {
    if (!armed) return false;
    const auto note = RecipeNoteSlot::Get();
    if ((note & NoteNoted) == 0) return false;
    outcome = static_cast<RecipeOutcome>((note >> 8u) & 0xffu);
    unreusable = static_cast<std::uint8_t>((note >> 16u) & 0xffu);
    fromCache = (note & NoteFromCache) != 0;
    return true;
}

bool RecipeNoteArmed() {
    return RecipeCounting() && (RecipeNoteSlot::Get() & NoteArmed) != 0;
}

void NoteRecipeBuild(RecipeOutcome outcome, std::uint8_t unreusable, bool fromCache) {
    if ((RecipeNoteSlot::Get() & NoteArmed) == 0) return;
    RecipeNoteSlot::Set(NoteArmed | NoteNoted | (fromCache ? NoteFromCache : 0u) | (static_cast<std::uint32_t>(outcome) << 8u) | (static_cast<std::uint32_t>(unreusable) << 16u));
}

void CountDispatchRecipe(RecipeOutcome outcome, std::uint8_t unreusable, bool fromCache) {
    const auto row = fromCache ? 1u : 0u;
    Add(recipeOutcomes[row][static_cast<std::size_t>(outcome)]);
    if (outcome == RecipeOutcome::NotReusable && unreusable < UnreusableReasons) Add(recipeUnreusable[row][unreusable]);
    if (!Stream::On()) maybeReportRecipesAlone();
}

void NoteIndirectKey(std::uint64_t address, std::span<const std::uint32_t> userData, const std::array<std::uint32_t, 3>& partialThreads, const std::array<std::uint32_t, 5>& registers, const void* shader, std::uint32_t wave, bool missed) {
    Add(indirectDispatches);
    if (!missed) Add(indirectHits);
    std::lock_guard lock(indirectMutex);
    const auto [found, inserted] = indirectParts.try_emplace(address);
    auto& parts = found->second;
    if (missed) {
        auto miss = MissFirst;
        if (!inserted) {
            const bool data = !std::equal(userData.begin(), userData.end(), parts.userData.begin(), parts.userData.end());
            const bool partial = partialThreads != parts.partialThreads;
            const bool registered = registers != parts.registers;
            const bool program = shader != parts.shader;
            const bool waveChanged = wave != parts.wave;
            const auto changed = (data ? 1u : 0u) | (partial ? 2u : 0u) | (registered ? 4u : 0u) | (program ? 8u : 0u) | (waveChanged ? 16u : 0u);
            miss = changed == 0 ? MissSameKey : changed == 1u ? MissUserData : changed == 2u ? MissPartialThreads : changed == 3u ? MissUserDataPartial : changed == 4u ? MissRegisters : changed == 8u ? MissShader : changed == 16u ? MissWave : MissSeveral;
        }
        Add(indirectMisses[miss]);
    }
    parts.userData.assign(userData.begin(), userData.end());
    parts.partialThreads = partialThreads;
    parts.registers = registers;
    parts.shader = shader;
    parts.wave = wave;
}

void ReportWindow(Window window, bool print) {
    if (!print) {
        // The first [stream] window starts here: what was counted before is dropped.
        for (auto& rows : dma0Rows) {
            for (auto& row : rows) {
                std::string ignored;
                bool first = true;
                appendPacketRow(ignored, "", row, first);
            }
        }
        for (auto& counter : dma0Fallbacks) Take(counter);
        for (auto& row : zpassRows) {
            std::string ignored;
            bool first = true;
            appendPacketRow(ignored, "", row, first);
        }
        Take(zpassOtherQueues);
        Take(gateFirst);
        for (auto& counter : gateRefusals) Take(counter);
        for (auto& counter : gateWaitNs) Take(counter);
        for (auto& rows : gateExits) for (auto& counter : rows) Take(counter);
        for (auto& rows : gateExitNs) for (auto& counter : rows) Take(counter);
        Take(gateOverlapFillNs);
        Take(gateOverlapNonFillNs);
        for (auto& rows : recipeOutcomes) for (auto& counter : rows) Take(counter);
        for (auto& rows : recipeUnreusable) for (auto& counter : rows) Take(counter);
        Take(indirectDispatches);
        Take(indirectHits);
        for (auto& counter : indirectMisses) Take(counter);
        Take(Stream::counters.segmentSyncs);
        Take(Stream::counters.segmentSyncNs);
        if (GdsCounting()) reportGds("-", false);
        GdsStream::ReportWindow("-", false);
        const auto totals = GuardTotals();
        lastHits = totals.hits;
        lastDrains = totals.drains;
        return;
    }
    const auto* name = WindowName(window);
    // C2.
    {
        std::string line;
        append(line, "[dma0] window=%s queue-0 DMA_DATA (kind/path n, drained, drain wait ms, packet ms, GPU wait ms, episodes, episode idle ms, serial ms):", name);
        bool first = true;
        for (std::size_t kind = 0; kind < DmaKinds; ++kind) {
            for (std::size_t path = 0; path < DmaPaths; ++path) {
                char cell[40];
                std::snprintf(cell, sizeof(cell), "%s/%s", DmaKindNames[kind], DmaPathNames[path]);
                appendPacketRow(line, cell, dma0Rows[kind][path], first);
            }
        }
        append(line, "; fallback reasons:");
        first = true;
        for (std::size_t reason = 0; reason < FallbackReasons; ++reason) {
            const auto n = Take(dma0Fallbacks[reason]);
            if (n == 0) continue;
            if (!first) line += ',';
            first = false;
            append(line, " r%zu %llu", reason, count(n));
        }
        AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
    }
    // C3.
    {
        std::string line;
        std::array<std::uint64_t, GateRefusals> refusals{};
        for (std::size_t refusal = 0; refusal < GateRefusals; ++refusal) refusals[refusal] = Take(gateRefusals[refusal]);
        const auto gates = Take(gateFirst);
        append(line, "[gate-skip] window=%s gate-first 0x15 with items queued %llu: would skip %llu (%.1f%%), refused", name, count(gates), count(refusals[0]), gates == 0 ? 0.0 : 100.0 * static_cast<double>(refusals[0]) / static_cast<double>(gates));
        for (std::size_t refusal = 1; refusal < GateRefusals; ++refusal) append(line, "%s %s %llu", refusal == 1 ? "" : ",", GateRefusalNames[refusal], count(refusals[refusal]));
        const auto skipWaitNs = Take(gateWaitNs[0]);
        const auto refusedWaitNs = Take(gateWaitNs[1]);
        append(line, "; W would-skip %.1f ms, refused %.1f ms", ms(skipWaitNs), ms(refusedWaitNs));
        for (std::size_t row = 0; row < 2; ++row) {
            append(line, "; exits %s (n/window ms):", row == 0 ? "would-skip" : "refused");
            for (std::size_t exit = 0; exit < GateExits; ++exit) append(line, "%s %s %llu/%.1f", exit == 0 ? "" : ",", GateExitNames[exit], count(Take(gateExits[row][exit])), ms(Take(gateExitNs[row][exit])));
        }
        append(line, "; would-skip overlap: sum min(W,G) fills %.1f ms, sum min(W,prologue) non-fills %.1f ms", ms(Take(gateOverlapFillNs)), ms(Take(gateOverlapNonFillNs)));
        AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
    }
    // C6: the window's share of the cumulative counts.
    {
        const auto totals = GuardTotals();
        std::string line;
        append(line, "[dispatch-guards] window=%s D1 guard conditions with items queued (hits/drains):", name);
        for (std::size_t guard = 0; guard < Guards; ++guard) {
            append(line, "%s %s %llu/%llu", guard == 0 ? "" : ",", GuardNames[guard], count(totals.hits[guard] - lastHits[guard]), count(totals.drains[guard] - lastDrains[guard]));
        }
        lastHits = totals.hits;
        lastDrains = totals.drains;
        AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
    }
    // C7.
    {
        std::string line;
        append(line, "[zpass] window=%s queue-0 ZPASS_DONE dumps (dump n, drained, drain wait ms, packet ms, GPU wait ms, episodes, episode idle ms, serial ms):", name);
        bool first = true;
        appendPacketRow(line, "gpu", zpassRows[0], first);
        appendPacketRow(line, "cpu", zpassRows[1], first);
        append(line, "; other queues %llu; CountSamples segment syncs %llu, %.1f ms", count(Take(zpassOtherQueues)), count(Take(Stream::counters.segmentSyncs)), ms(Take(Stream::counters.segmentSyncNs)));
        AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
    }
    // C5.
    reportRecipes(name);
    {
        std::string line;
        const auto dispatches = Take(indirectDispatches);
        const auto hits = Take(indirectHits);
        std::array<std::uint64_t, IndirectMisses> misses{};
        std::uint64_t missed = 0;
        for (std::size_t miss = 0; miss < IndirectMisses; ++miss) {
            misses[miss] = Take(indirectMisses[miss]);
            missed += misses[miss];
        }
        const auto carried = misses[MissUserData] + misses[MissPartialThreads] + misses[MissUserDataPartial];
        append(line, "[dispatch-cache] indirect window=%s queue-0 dispatches of 0x16 packets %llu, cache hits %llu, misses %llu by the key parts changed since the shader's previous one:", name, count(dispatches), count(hits), count(missed));
        for (std::size_t miss = 0; miss < IndirectMisses; ++miss) append(line, "%s %s %llu", miss == 0 ? "" : ",", IndirectMissNames[miss], count(misses[miss]));
        append(line, "; validation could carry (user data, partial threads) %llu (%.1f%% of misses)", count(carried), missed == 0 ? 0.0 : 100.0 * static_cast<double>(carried) / static_cast<double>(missed));
        AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
    }
    // C1.
    if (GdsCounting()) reportGds(name);
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_GDS_IN_STREAM, DRAIN-PLAN D2): its [gds-stream] line, tagged too.
    GdsStream::ReportWindow(name, true);
}

}

namespace AgcDriver::Stream {

// LOCAL DEBUG (APS5_DEBUG_GDS_STATS, DRAIN-PLAN C1): see StreamStats.hpp (ThreadGdsSlot).
void NoteGdsGpuWait(std::uint32_t slot, std::uint64_t ns) {
    if (slot < Drain::QueueSlots) Add(Drain::gdsPacketWaitNs[slot], ns);
}

}
