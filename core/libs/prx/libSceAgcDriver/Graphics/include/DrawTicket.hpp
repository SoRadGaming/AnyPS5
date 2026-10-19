#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWTICKET_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWTICKET_HPP

#include "prx/libSceAgcDriver/Graphics/include/ColorTargetLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

class BufferPool;
class Pipeline;
struct PreparedDraw;

// LOCAL EXPERIMENT (COMMITTER-SPLIT-PLAN, room-wait draw tickets): what queue 0 prepared for one
// pipelined draw while it waited for room in DrawPipeline::Enqueue, before the push. The ticket
// rides in the draw's commit closure (never in a thread slot) and is complete before the push,
// whose pipeline mutex publishes it to the committer; after the push only the committer touches
// it. Each part has its bit in `parts`, set only when queue 0 filled it and every check in it
// passed: the committer uses a part at the place today's code computes it, after the part's own
// check, and runs today's code for any part whose bit is clear, so a failure is raised at its
// own place after the same side effects. Every part is derived from the draw's own closure
// objects (its state, parameters and stages) and from build-fixed data only; queue 0 reads no
// guest bytes, calls no Recorder instance method and never takes GuestMemory::GpuMutex for it.
struct DrawTicket {
    enum Part : std::uint32_t {
        // Q1: pure checks and target descriptions (APS5_EXPERIMENT_Q0_TICKET).
        AbiChecked = 1u << 0,      // VulkanDevice::Draw's runtime ABI loop passed
        StencilChecked = 1u << 1,  // the stencil-replay Requires of Graphics::Draw passed
        Validated = 1u << 2,       // prepareDrawInputs' parameter, mesh and tessellation Requires and validators passed
        AliasesChecked = 1u << 3,  // CheckBufferAliases (a template hit's alias checks) passed
        Targets = 1u << 4,         // `targets`: each color target's layout and resident surface
        WriterRanges = 1u << 5,    // `writerRanges`: noteDrawWriters' ranges, noted at the commit
        // Q2: the template hint and what derives from it (APS5_EXPERIMENT_Q0_TEMPLATE). Every part
        // after Template is for `resources` and is used only when the committer's accepted
        // template is that same object.
        Template = 1u << 6,        // `cacheableShaders`, the content key and the hint `resources` (null: a miss, never trusted)
        Moved = 1u << 7,           // `memo`: the hint's moved-candidate walk for these stages
        Push = 1u << 8,            // `pushBytes` / `pushStages` with the hint's push patches
        Pipeline = 1u << 9,        // `pipeline`: the store's pipeline for the hint, `pipelinePrepared` and `proxiedMask`
        // LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS=3): the byte shadow, counted at the commit, never used.
        ByteShadow = 1u << 10,
        // Q2b: `wordBuffers`, the hint's moved data-word buffers (APS5_EXPERIMENT_Q0_WORD_BUFFERS).
        WordBuffers = 1u << 11,
    };
    static constexpr std::size_t PartCount = 12;
    static constexpr std::array<const char*, PartCount> PartNames{"abi", "stencil", "validated", "aliases", "targets", "writers", "template", "moved", "push", "pipeline", "bytes", "words"};
    // Identity: the device and buffer pool the ticket was built for (TicketFor refuses another).
    VkDevice device = VK_NULL_HANDLE;
    std::weak_ptr<BufferPool> pool;
    std::uint64_t token = 0;
    std::uint32_t parts = 0;
    // The queue depth the draw was enqueued behind (APS5_DEBUG_Q0_PREP_STATS).
    std::uint32_t enqueueDepth = 0;
    // APS5_DEBUG_Q0_PREP_STATS=2 without APS5_EXPERIMENT_Q0_TICKET: the committer runs today's code
    // for every part and only counts whether the ticket agreed.
    bool shadow = false;
    bool Has(Part part) const { return (parts & part) != 0; }
    // Q1
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writerRanges;
    struct Target {
        bool multisampled = false;
        // Single-sample targets: Graphics::Draw's ColorTargetLayout and the surface its resident
        // lookup asks for (SurfaceForTarget, or ArraySurfaceForTarget for a slice of a 2D array).
        std::optional<ColorTargetLayout> layout;
        GuestTextureResource surface{};
    };
    std::vector<Target> targets;
    // Q2. The content key is the PreparedDraw's (`preparedKey`, which the draw's closure keeps) or
    // `ownKey`, built with the ticket's context exactly as resolveDrawResources builds it.
    bool cacheableShaders = false;
    const std::vector<std::uint32_t>* preparedKey = nullptr;
    std::vector<std::uint32_t> ownKey;
    const std::vector<std::uint32_t>& Key() const { return preparedKey != nullptr ? *preparedKey : ownKey; }
    std::shared_ptr<ShaderResources> resources;
    ShaderResources::MovedWordsMemo memo;
    std::array<std::byte, PipelinePushConstantBytes> pushBytes{};
    VkShaderStageFlags pushStages = 0;
    std::shared_ptr<Graphics::Pipeline> pipeline;
    // The PreparedDraw whose vertex layout and fragment outputs the pipeline was looked up with,
    // and the targets predicted to render through an attachment proxy (bit per color target).
    const PreparedDraw* pipelinePrepared = nullptr;
    std::uint32_t proxiedMask = 0;
    // LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS=3, the byte shadow of COMMITTER-SPLIT-PLAN 1.8 without
    // any copy): in a full pipeline wait queue 0 collects the index range (at least 1024 indices)
    // and the hint's predicted small slices, then reads the pending, allocation and forget serials.
    // The committer counts at the index scan and at each slice how often a Q4 re-check (its own
    // collect, UnchangedSince that generation, the serials unchanged) would have passed.
    struct IndexShadow {
        std::uint64_t address = 0;
        std::uint64_t bytes = 0;
        std::uint64_t generation = 0;
    };
    std::optional<IndexShadow> indexShadow;
    std::vector<ShaderResources::TicketSlice> sliceShadows;
    std::uint64_t pendingSerial = 0;
    std::uint64_t allocationsGeneration = 0;
    std::uint64_t forgetSerial = 0;
    ShaderResources::TicketBindings ByteBindings() const { return {true, sliceShadows, pendingSerial, allocationsGeneration, forgetSerial, {}, false, false}; }
    // Q2b: one per data candidate of `memo`, made with the hint's data patches.
    std::vector<ShaderResources::TicketWordBuffer> wordBuffers;
    DrawTicket() = default;
    DrawTicket(const DrawTicket&) = delete;
    DrawTicket& operator=(const DrawTicket&) = delete;
    // Hands the strong references the committer did not take to the ticket release list
    // (ReleaseTicketReferences), so nothing a ticket holds is destroyed on queue 0 or the reaper.
    ~DrawTicket();
};

// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_TICKET): 0 off (today's code exactly), 1 queue 0 fills Q1 tickets and the
// committer uses them, 2 the same with every used part also computed inline and compared (a
// difference aborts with a [q0ticket] line). Read once.
int Q0Ticket();
// Whether queue 0 builds tickets at all: APS5_EXPERIMENT_Q0_TICKET, or the shadow tickets of
// APS5_DEBUG_Q0_PREP_STATS=2 (built and compared, never used).
bool Q0TicketsBuilt();
// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_TEMPLATE): 0 off; 1 queue 0 adds the template hint, the moved-candidate memo,
// the push block and the pipeline to its tickets (requires APS5_EXPERIMENT_Q0_TICKET, ignored
// without it), and the committer uses them after checking the template's identity; 2 the same
// with the committer also probing the cache, running today's CachedPipeline, push assembly and
// memo-less MovedReadOnlyBuffers, and aborting on a difference (template membership is counted,
// never aborted on). Shadow tickets (APS5_DEBUG_Q0_PREP_STATS=2) carry these parts too, built
// with lookup-only probes. Read once.
int Q0Template();
bool Q0TemplateParts();
// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_WORD_BUFFERS): 0 off; 1 queue 0 makes the hint's moved data-word buffers for
// its tickets (requires APS5_EXPERIMENT_Q0_TEMPLATE, ignored without it; a queue-0 cache of 64
// keeps them for reuse, keyed by template, allocation and words) and the committer binds them
// when its template is the hint and the element's allocation, words and size match; 2 the same
// with each bound buffer's bytes compared with today's (a difference aborts). Shadow tickets
// (APS5_DEBUG_Q0_PREP_STATS=2) carry the bytes only. Read once.
int Q0WordBuffers();
bool Q0WordParts();
// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_WORD_BUFFERS): queue 0's WordBuffers part (after the
// template parts, with the hint's memo).
void FillDrawTicketWords(DrawTicket& ticket, const Context& context, std::span<const CompiledShader> shaders);
// The bytes the queue-0 word buffer cache holds ([q0prep]).
std::uint64_t TicketWordCacheBytes();
// LOCAL DEBUG: APS5_DEBUG_Q0_PREP_STATS=3 builds the byte shadow; it needs
// APS5_EXPERIMENT_LAZY_GENERATION=1 (queue 0's extra walks would otherwise move the tracker
// generation and retire the committer's slice memos) and acts as =2 without it (said once).
bool Q0ByteShadow();
// The committer's counts of the byte shadow (TicketEvent Index* / Slice*): at the in-place index
// scan of a draw whose ticket collected its index range, and at each small slice
// PrepareDrawBindings takes (`copied`: today's path copied it, what Q4 would save).
void NoteTicketIndex(const DrawTicket& ticket, std::uint64_t address, std::uint64_t bytes);
void NoteTicketSlice(const ShaderResources::TicketBindings& ticket, std::uint64_t begin, std::size_t bytes, bool copied);
// LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_PREP_DEPTH=<pre>:<bail>, a comma also separates): queue 0 prepares a ticket
// only when it finds at least `pre` items queued (default 3), and stops between parts once
// `bail` or fewer are left (default 2). 0:0 prepares every draw (tests).
struct Q0PrepDepth {
    std::size_t pre = 3;
    std::size_t bail = 2;
};
const Q0PrepDepth& Q0PrepDepths();

// The ticket when it was built for this context's device and buffer pool, else null (the whole
// ticket is refused, counted under APS5_DEBUG_Q0_PREP_STATS).
DrawTicket* TicketForBuilt(DrawTicket* ticket, const Context& context);
inline DrawTicket* TicketFor(DrawTicket* ticket, const Context& context) {
    return ticket == nullptr ? nullptr : TicketForBuilt(ticket, context);
}
// How the committer treats one part of a ticket (null or a clear bit: None, today's code): a used
// part replaces today's code; Verify (APS5_EXPERIMENT_Q0_TICKET=2) uses it and runs today's code
// as well, aborting on a difference; a shadow ticket's part only counts agreement.
enum class TicketUse : std::uint8_t { None, Use, Verify, Shadow };
TicketUse TicketPartUseBuilt(const DrawTicket& ticket, DrawTicket::Part part);
inline TicketUse TicketPartUse(const DrawTicket* ticket, DrawTicket::Part part) {
    return ticket == nullptr ? TicketUse::None : TicketPartUseBuilt(*ticket, part);
}
// A difference between a ticket part and today's code: aborts under Verify, counts under Shadow.
void TicketDisagrees(DrawTicket::Part part, TicketUse use, const char* what);
// Runs today's `checks` unless the part was used: a Verify or Shadow part whose checks throw
// although the ticket passed them disagrees (and the exception continues as today's would).
template <typename TChecks>
void TicketChecks(TicketUse use, DrawTicket::Part part, TChecks&& checks) {
    if (use == TicketUse::Use) return;
    if (use == TicketUse::None) {
        checks();
        return;
    }
    try {
        checks();
    } catch (const std::exception& error) {
        TicketDisagrees(part, use, error.what());
        throw;
    }
}

// Strong references a ticket held that must not be dropped on queue 0 or on the reaper: the
// committer drains them under GuestMemory::GpuMutex (DrainTicketRelease, at a draw's start) and
// keeps a last reference with the open batch, so its destruction runs on the release thread once
// the batch completed; another device's are never destroyed (its objects went with it).
struct TicketReference {
    VkDevice device = VK_NULL_HANDLE;
    std::shared_ptr<ShaderResources> resources;
    std::shared_ptr<void> object;
};
void ReleaseTicketReference(TicketReference reference);
// Under GuestMemory::GpuMutex, on the thread that records: the release list's references, and
// the ones a committed draw's ticket still holds.
void DrainTicketRelease(const Context& context);
void ReleaseTicketReferences(DrawTicket& ticket, const Context& context);

// What the committer did with a ticket's template parts (APS5_DEBUG_Q0_PREP_STATS): the hint used,
// found uncached, or a miss the committer's Find found or not; Revalidate refusing the template;
// the memo kept or rebuilt; the pipeline accepted or refused by reason; the push block used.
enum class TicketEvent : std::uint8_t { HintUsed, HintUncached, MissFound, MissAbsent, HintRevalidateFailed, OtherTemplate, MemoKept, MemoRebuilt, PipelineAccepted, PipelineNotLean, PipelineStencil, PipelineOtherTemplate, PipelineOtherPrepared, PipelineProxies, PipelineAbandoned, PushUsed, PushOtherTemplate, WordsBuilt, WordsCacheHit, WordsUsed, WordsUnmatched, WordsOtherTemplate, IndexOffered, IndexOther, IndexPass, IndexStamp, IndexGuard, SliceOffered, SliceUnpredicted, SlicePass, SliceStamp, SliceGuard, SliceCopied, Count };
inline constexpr std::array<const char*, static_cast<std::size_t>(TicketEvent::Count)> TicketEventNames{"hint-used", "hint-uncached", "miss-found", "miss-absent", "revalidate-failed", "other-template", "memo-kept", "memo-rebuilt", "pipeline-accepted", "pipeline-not-lean", "pipeline-stencil", "pipeline-other-template", "pipeline-other-prepared", "pipeline-proxies", "pipeline-abandoned", "push-used", "push-other-template", "words-built", "words-cache-hit", "words-used", "words-unmatched", "words-other-template", "index-offered", "index-other-range", "index-pass", "index-stamp", "index-serials", "slice-offered", "slice-unpredicted", "slice-pass", "slice-stamp", "slice-serials", "slice-copied-today"};
void NoteTicketEvent(TicketEvent event);

// LOCAL DEBUG (APS5_DEBUG_Q0_PREP_STATS): the ticket counters of the [q0prep] line, reset by
// TakeTicketCounts. Queue 0 counts what it built (and its time per part), the committer what it
// used, refused and compared.
struct TicketCounts {
    std::uint64_t built = 0;
    std::uint64_t shallow = 0;
    std::uint64_t locked = 0;
    std::uint64_t refusedDevice = 0;
    std::uint64_t refusedPool = 0;
    std::array<std::uint64_t, DrawTicket::PartCount> filled{};
    std::array<std::uint64_t, DrawTicket::PartCount> buildNs{};
    std::array<std::uint64_t, DrawTicket::PartCount> used{};
    std::array<std::uint64_t, DrawTicket::PartCount> clear{};
    std::array<std::uint64_t, DrawTicket::PartCount> disagreed{};
    std::array<std::uint64_t, static_cast<std::size_t>(TicketEvent::Count)> events{};
};
enum class TicketSkip : std::uint8_t { Shallow, Locked };
void NoteTicketSkip(TicketSkip skip);
void NoteTicketBuilt(const DrawTicket& ticket);
void NoteTicketPartNs(DrawTicket::Part part, std::uint64_t ns);
TicketCounts TakeTicketCounts();
std::size_t TicketPartIndex(DrawTicket::Part part);

}

#endif
