#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawTicketBuild.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawTicket.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

// LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS, COMMITTER-SPLIT-PLAN S0): the [q0prep] line. Queue 0 prints
// it every 10 s after a draw enqueue: the depth its draws found the pipeline at and how long they
// waited for room (the time a ticket could use), the committer's write-watch walks by draw step and
// enqueue depth (GuestMemory::WalkStats), and how many committed draws used a draw-ahead
// PreparedDraw against the draw-ahead compiles and drops.
namespace {

std::atomic<std::uint64_t> committedDraws{0};

struct Q0PrepReport {
    bool started = false;
    std::chrono::steady_clock::time_point last{};
    std::uint64_t preparedUses = 0;
    std::uint64_t compiled = 0;
    std::uint64_t dropped = 0;
};

double percent(std::uint64_t part, std::uint64_t whole) {
    return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

// A ticket part's time on queue 0 (APS5_DEBUG_Q0_PREP_STATS).
struct PartTimer {
    Graphics::DrawTicket::Part part;
    bool timed = GuestMemory::Q0PrepStats() != 0;
    std::chrono::steady_clock::time_point start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ~PartTimer() {
        if (timed) Graphics::NoteTicketPartNs(part, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()));
    }
};

// LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS=3): queue 0 collects the draw's index range (at least 1024
// indices) and the hint's predicted small slices, then reads the registries' serials after them
// (COMMITTER-MAP 3.2: collect first); no byte is read or copied. The committer only counts
// (NoteTicketIndex, NoteTicketSlice).
void fillByteShadow(Graphics::DrawTicket& ticket, const TicketInputs& inputs, const DrawPipeline::PrepareBudget& budget) {
    constexpr std::uint32_t IndexMinimum = 1024;
    const PartTimer timer{Graphics::DrawTicket::ByteShadow};
    const auto& draw = inputs.parameters;
    if (draw.indexed && draw.indexCount >= IndexMinimum) {
        const auto bytes = static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize;
        if (const auto generation = GuestMemory::CollectWrites(draw.indexAddress, static_cast<std::size_t>(bytes)); generation != 0) ticket.indexShadow = Graphics::DrawTicket::IndexShadow{draw.indexAddress, bytes, generation};
    }
    if (ticket.resources != nullptr) {
        std::vector<std::pair<std::uint64_t, std::size_t>> ranges;
        ticket.resources->PredictDrawSlices(ticket.memo, ranges);
        for (const auto& [begin, bytes] : ranges) {
            if (budget.Stop()) break;
            if (const auto generation = GuestMemory::CollectWrites(begin, bytes); generation != 0) ticket.sliceShadows.push_back({begin, bytes, generation});
        }
    }
    ticket.pendingSerial = Graphics::StorageTexture::PendingSerial();
    ticket.allocationsGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    ticket.forgetSerial = GuestMemory::ForgetSerial();
    ticket.parts |= Graphics::DrawTicket::ByteShadow;
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_RECEIPTS, COMMITTER-SPLIT-PLAN Q3): queue 0's receipt walks for
// the draw it is enqueuing, in its own thread's list (one draw at a time: planned by prepareTicket,
// walked on by WalkTicketReceipts after the push, dropped by the next plan), at most 64 MiB walked
// per draw.
constexpr std::uint64_t ReceiptBudgetBytes = std::uint64_t{64} << 20;

GuestMemory::ReceiptWalks& receiptWalks() {
    struct Tag {};
    return ThreadScratch<GuestMemory::ReceiptWalks, Tag>();
}

bool prepareStops(const void* budget) {
    return static_cast<const DrawPipeline::PrepareBudget*>(budget)->Stop();
}

bool postPushStops(const void* post) {
    return DrawPipeline::Queue0().Outstanding() <= *static_cast<const std::size_t*>(post);
}

// Plans the ranges from the closure's objects and the ticket (Graphics::PlanDrawTicketWalks) into
// queue 0's list; walks them now only inside a full pipeline wait (every draw with a 0 pre depth,
// the tests' setting), stopping at the ticket's bail depth: collects first and only, no result
// derived from guest bytes, so nothing here needs a re-check at the commit beyond the receipt's own.
void planReceipts(const Graphics::DrawTicket& ticket, TicketInputs& inputs, const DrawPipeline::PrepareBudget& budget) {
    const bool timed = GuestMemory::Q0PrepStats() != 0;
    const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    struct PlannedTag {};
    auto& planned = ThreadScratch<std::vector<std::pair<std::uint64_t, std::uint64_t>>, PlannedTag>();
    planned.clear();
    Graphics::PlanDrawTicketWalks(ticket, *inputs.state, inputs.parameters, inputs.stages, planned);
    auto& walks = receiptWalks();
    GuestMemory::PlanReceiptWalks(planned, walks);
    if (timed) {
        std::uint64_t bytes = 0;
        for (const auto& [begin, end] : walks.ranges) bytes += end - begin;
        GuestMemory::NoteReceiptPlan(walks.ranges.size(), bytes, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()));
    }
    inputs.receipts = !walks.Done();
    if (inputs.receipts && (budget.full || Graphics::Q0PrepDepths().pre == 0) && !budget.Stop()) GuestMemory::WalkReceipts(inputs.token, walks, ReceiptBudgetBytes, &prepareStops, &budget, false);
}

// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_TICKET, COMMITTER-SPLIT-PLAN Q1): queue 0 builds the draw's
// ticket inside Enqueue, in a GuestMemory::PrepareScope and without GpuMutex. The context is the
// device's without a recorder or copied writers, so any recorder use on this path faults at once;
// nothing here reads guest bytes. A part whose checks throw is left out (the committer runs them).
void prepareTicket(void* self, const DrawPipeline::PrepareBudget& budget) {
    auto& inputs = *static_cast<TicketInputs*>(self);
    // A stage without a compiled program fails VulkanDevice::Draw's first check; the checks after
    // it assume programs, so such a draw gets no ticket.
    if (std::any_of(inputs.stages.begin(), inputs.stages.end(), [](const Graphics::CompiledShader& stage) { return stage.program == nullptr; })) return;
    try {
        auto ticket = std::make_unique<Graphics::DrawTicket>();
        auto context = inputs.device->graphicsContext();
        context.recorder = nullptr;
        context.copiedWriters = nullptr;
        ticket->device = context.device;
        ticket->pool = context.bufferPool;
        ticket->token = inputs.token;
        ticket->enqueueDepth = static_cast<std::uint32_t>(budget.depth);
        ticket->shadow = Graphics::Q0Ticket() == 0;
        {
            const PartTimer timer{Graphics::DrawTicket::AbiChecked};
            if (VulkanDevice::DrawStagesAbiCurrent(inputs.stages)) ticket->parts |= Graphics::DrawTicket::AbiChecked;
        }
        {
            const PartTimer timer{Graphics::DrawTicket::WriterRanges};
            if (AppendDrawWriterRanges(inputs.stages, ticket->writerRanges)) ticket->parts |= Graphics::DrawTicket::WriterRanges;
            else ticket->writerRanges.clear();
        }
        Graphics::FillDrawTicketChecks(*ticket, context, *inputs.state, inputs.parameters, inputs.stages);
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_TEMPLATE, COMMITTER-SPLIT-PLAN Q2): the template hint
        // and its parts, unless the committer is about to run dry.
        if (Graphics::Q0TemplateParts() && !budget.Stop()) Graphics::FillDrawTicketTemplate(*ticket, context, *inputs.state, inputs.parameters, inputs.stages, inputs.prepared);
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_WORD_BUFFERS, COMMITTER-SPLIT-PLAN Q2b).
        if (Graphics::Q0WordParts() && !budget.Stop()) Graphics::FillDrawTicketWords(*ticket, context, inputs.stages);
        // LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS=3): the byte shadow, in full pipeline waits only.
        if (Graphics::Q0ByteShadow() && budget.full && !budget.Stop()) fillByteShadow(*ticket, inputs, budget);
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_RECEIPTS, COMMITTER-SPLIT-PLAN Q3): the draw's
        // first-in-token walks, after the cheap parts (the rest after the push); a failure here
        // costs the receipts only, never the ticket.
        if (GuestMemory::Q0Receipts() != 0) {
            try {
                planReceipts(*ticket, inputs, budget);
            } catch (const std::exception&) {
                inputs.receipts = false;
            }
        }
        Graphics::NoteTicketBuilt(*ticket);
        *inputs.slot = std::move(ticket);
    } catch (const std::exception&) {
        // No ticket: the committer runs today's code for every part.
    }
}

void skippedTicket(void*, DrawPipeline::PrepareSkip why) {
    Graphics::NoteTicketSkip(why == DrawPipeline::PrepareSkip::Shallow ? Graphics::TicketSkip::Shallow : Graphics::TicketSkip::Locked);
}

}

DrawPipeline::Prepare DrawTicketPrepare(TicketInputs& inputs) {
    const auto& depths = Graphics::Q0PrepDepths();
    DrawPipeline::Prepare prepare;
    prepare.self = &inputs;
    prepare.run = &prepareTicket;
    prepare.skipped = &skippedTicket;
    prepare.pre = depths.pre;
    prepare.bail = depths.bail;
    return prepare;
}

void WalkTicketReceipts(const TicketInputs& inputs) {
    if (!inputs.receipts) return;
    auto& walks = receiptWalks();
    if (walks.Done() || GuestMemory::GpuMutex().HeldByThisThread()) return;
    const auto post = Graphics::Q0PrepDepths().post;
    const GuestMemory::PrepareScope scope;
    try {
        GuestMemory::WalkReceipts(inputs.token, walks, ReceiptBudgetBytes, &postPushStops, &post, true);
    } catch (const std::exception&) {
        // Only the receipts are lost: the committer walks as today.
        walks.next = walks.ranges.size();
    }
}

void Driver::noteQ0PrepCommit() {
    committedDraws.fetch_add(1, std::memory_order_relaxed);
}

void Driver::reportQ0Prep() {
    // Only queue 0 enqueues pipelined draws, so only queue 0 reports.
    static Q0PrepReport state;
    const auto now = std::chrono::steady_clock::now();
    if (!state.started) {
        state.started = true;
        state.last = now;
        state.preparedUses = Graphics::PreparedDrawUses();
        state.compiled = drawAheadCounts.compiled.load(std::memory_order_relaxed);
        state.dropped = drawAheadCounts.dropped.load(std::memory_order_relaxed);
        DrawPipeline::Queue0().TakePrepStats();
        GuestMemory::TakeWalkStats();
        Graphics::TakeTicketCounts();
        if (GuestMemory::Q0Receipts() != 0) GuestMemory::TakeReceiptStats();
        committedDraws.store(0, std::memory_order_relaxed);
        return;
    }
    if (now - state.last < std::chrono::seconds(10)) return;
    state.last = now;
    const auto count = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    const auto pipeline = DrawPipeline::Queue0().TakePrepStats();
    const auto depth = DrawPipeline::Depth();
    std::string depths;
    std::uint64_t atLeastTwo = 0, atLeastThree = 0;
    for (std::size_t bucket = 0; bucket < pipeline.drawDepths.size(); ++bucket) {
        char item[48];
        std::snprintf(item, sizeof(item), " %zu%s:%llu", bucket, bucket + 1 == pipeline.drawDepths.size() ? "+" : "", count(pipeline.drawDepths[bucket]));
        depths += item;
        if (bucket >= 2) atLeastTwo += pipeline.drawDepths[bucket];
        if (bucket >= 3) atLeastThree += pipeline.drawDepths[bucket];
    }
    // Full depth is counted at the enqueue (Depth() may exceed the last bucket).
    const auto full = pipeline.fullDraws;
    const auto preparedUses = Graphics::PreparedDrawUses();
    const auto compiled = drawAheadCounts.compiled.load(std::memory_order_relaxed);
    const auto dropped = drawAheadCounts.dropped.load(std::memory_order_relaxed);
    const auto committed = committedDraws.exchange(0, std::memory_order_relaxed);
    AgcDriver::ProfilePrint_nid_no_patch("[q0prep] (10 s, depth %zu) draws %llu enqueued behind:%s; d>=2 %.1f%%, d>=3 %.1f%%, full %.1f%%; room waits: draws %llu (%.1f%%) %.1f ms (%.1f us per waiting draw, %.1f us per draw), labels %llu of %llu %.1f ms; committed draws %llu, %llu used a draw-ahead PreparedDraw (%.1f%%); draw-ahead compiled %llu, dropped %llu\n", depth, count(pipeline.draws), depths.c_str(), percent(atLeastTwo, pipeline.draws), percent(atLeastThree, pipeline.draws), percent(full, pipeline.draws), count(pipeline.drawWaits), percent(pipeline.drawWaits, pipeline.draws), static_cast<double>(pipeline.drawWaitNs) / 1e6, pipeline.drawWaits == 0 ? 0.0 : static_cast<double>(pipeline.drawWaitNs) / 1e3 / static_cast<double>(pipeline.drawWaits), pipeline.draws == 0 ? 0.0 : static_cast<double>(pipeline.drawWaitNs) / 1e3 / static_cast<double>(pipeline.draws), count(pipeline.labelWaits), count(pipeline.labels), static_cast<double>(pipeline.labelWaitNs) / 1e6, count(committed), count(preparedUses - state.preparedUses), percent(preparedUses - state.preparedUses, committed), count(compiled - state.compiled), count(dropped - state.dropped));
    state.preparedUses = preparedUses;
    state.compiled = compiled;
    state.dropped = dropped;
    // The committer's walks: per draw step, the ms and the share of it in draws enqueued at d >= 2,
    // d >= 3 and at full depth (the predictable part a ticket or a receipt could move).
    static constexpr std::array<const char*, static_cast<std::size_t>(GuestMemory::WalkCaller::Count)> callers{"other", "revalidate", "index", "target", "depth", "bindings", "inputs"};
    const auto walks = GuestMemory::TakeWalkStats();
    std::string steps;
    std::uint64_t totalNs = 0, totalTwoNs = 0, totalWalks = 0;
    for (std::size_t caller = 0; caller < callers.size(); ++caller) {
        std::uint64_t n = 0, ns = 0, bytes = 0, twoNs = 0, threeNs = 0;
        for (std::size_t bucket = 0; bucket < GuestMemory::WalkDepthBuckets; ++bucket) {
            n += walks.walks[caller][bucket];
            ns += walks.ns[caller][bucket];
            bytes += walks.bytes[caller][bucket];
            if (bucket >= 2) twoNs += walks.ns[caller][bucket];
            if (bucket >= 3) threeNs += walks.ns[caller][bucket];
        }
        const auto fullNs = walks.fullNs[caller];
        totalNs += ns;
        totalTwoNs += twoNs;
        totalWalks += n;
        char item[160];
        std::snprintf(item, sizeof(item), " %s %llu walks %.1f ms %.0f MiB (d>=2 %.0f%%, d>=3 %.0f%%, full %.0f%%);", callers[caller], count(n), static_cast<double>(ns) / 1e6, static_cast<double>(bytes) / 1048576.0, percent(twoNs, ns), percent(threeNs, ns), percent(fullNs, ns));
        steps += item;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[q0prep] committer walks (10 s) by draw step:%s total %llu walks %.1f ms, %.1f ms of it in draws enqueued at d>=2\n", steps.c_str(), count(totalWalks), static_cast<double>(totalNs) / 1e6, static_cast<double>(totalTwoNs) / 1e6);
    if (!Graphics::Q0TicketsBuilt()) return;
    // Tickets (APS5_EXPERIMENT_Q0_TICKET, or shadow tickets under =2): built, skipped, refused at
    // the commit; per part how often queue 0 filled it, its queue-0 time, how often the committer
    // consumed it (used, verified or compared), found it clear in a ticket, and disagreements.
    const auto tickets = Graphics::TakeTicketCounts();
    std::string parts;
    for (std::size_t index = 0; index < Graphics::DrawTicket::PartCount; ++index) {
        char item[200];
        std::snprintf(item, sizeof(item), " %s filled %llu (%.2f us) used %llu clear %llu disagreed %llu;", Graphics::DrawTicket::PartNames[index], count(tickets.filled[index]), tickets.filled[index] == 0 ? 0.0 : static_cast<double>(tickets.buildNs[index]) / 1e3 / static_cast<double>(tickets.built), count(tickets.used[index]), count(tickets.clear[index]), count(tickets.disagreed[index]));
        parts += item;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[q0prep] tickets (10 s, %s): built %llu, skipped shallow %llu, skipped holding the GPU mutex %llu; refused at the commit: device %llu, pool %llu; parts:%s\n", Graphics::Q0Ticket() == 0 ? "shadow" : Graphics::Q0Ticket() == 2 ? "verify" : "used", count(tickets.built), count(tickets.shallow), count(tickets.locked), count(tickets.refusedDevice), count(tickets.refusedPool), parts.c_str());
    // LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_RECEIPTS): queue 0's receipt walks (the walk time moved off
    // the committer) and what the committer's collects made of them.
    if (GuestMemory::Q0Receipts() != 0) {
        const auto receipts = GuestMemory::TakeReceiptStats();
        std::string steps;
        for (std::size_t caller = 0; caller < callers.size(); ++caller) {
            char item[48];
            std::snprintf(item, sizeof(item), " %s %llu", callers[caller], count(receipts.hitsByCaller[caller]));
            steps += item;
        }
        static constexpr std::array<const char*, static_cast<std::size_t>(GuestMemory::ReceiptMiss::Count)> missNames{"token", "epoch", "serial", "uncovered", "busy"};
        std::string misses;
        for (std::size_t why = 0; why < missNames.size(); ++why) {
            char item[48];
            std::snprintf(item, sizeof(item), " %s %llu", missNames[why], count(receipts.misses[why]));
            misses += item;
        }
        const auto ms = [](std::uint64_t ns) { return static_cast<double>(ns) / 1e6; };
        AgcDriver::ProfilePrint_nid_no_patch("[q0prep] receipts (10 s, %s): queue 0 planned %llu draws, %llu ranges %.0f MiB (%.2f us per draw); walked %llu chunks %.0f MiB in %.1f ms before the push + %.1f ms after (%.1f + %.1f ms with the skips and publishing); runs skipped: noted by queue 0 (no lock) %llu, in the ring %llu, walked by queue 0 this epoch (published) %llu; ring locks %llu (%.1f per draw); chunks unwatched %llu, failed %llu; published %llu ranges, committer walks %llu; draws without the owner role %llu, unfinished after the push %llu; committer: hits %llu (%.0f MiB, by step:%s), count-only hits %llu (their walk stamped the requested range %llu, the walk widened to whole blocks %llu), misses:%s; dirty pages hidden by hits %llu\n", GuestMemory::Q0Receipts() == 2 ? "count only" : "used", count(receipts.draws), count(receipts.ranges), static_cast<double>(receipts.rangeBytes) / 1048576.0, receipts.draws == 0 ? 0.0 : static_cast<double>(receipts.planNs) / 1e3 / static_cast<double>(receipts.draws), count(receipts.chunks), static_cast<double>(receipts.walkedBytes) / 1048576.0, ms(receipts.walkNs[0]), ms(receipts.walkNs[1]), ms(receipts.spentNs[0]), ms(receipts.spentNs[1]), count(receipts.notedCovered), count(receipts.ringCovered), count(receipts.ownCovered), count(receipts.ringLocks), receipts.draws == 0 ? 0.0 : static_cast<double>(receipts.ringLocks) / static_cast<double>(receipts.draws), count(receipts.unwatched), count(receipts.failed), count(receipts.published), count(receipts.committerPublished), count(receipts.notOwner), count(receipts.unfinished), count(receipts.hits), static_cast<double>(receipts.hitBytes) / 1048576.0, steps.c_str(), count(receipts.wouldHits), count(receipts.wouldHitsStamped), count(receipts.wouldHitsWidened), misses.c_str(), count(receipts.hiddenPages));
    }
    if (!Graphics::Q0TemplateParts()) return;
    // What the committer did with the template parts (TicketEvent).
    std::string events;
    for (std::size_t index = 0; index < tickets.events.size(); ++index) {
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu", Graphics::TicketEventNames[index], count(tickets.events[index]));
        events += item;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[q0prep] ticket templates (10 s):%s; queue-0 word buffer cache %.2f MiB\n", events.c_str(), static_cast<double>(Graphics::TicketWordCacheBytes()) / 1048576.0);
}

}
