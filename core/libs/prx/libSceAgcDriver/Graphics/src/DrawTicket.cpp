#include "prx/libSceAgcDriver/Graphics/include/DrawTicket.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawInputMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>

namespace AgcDriver::Graphics {

namespace {

struct Counters {
    std::atomic<std::uint64_t> built{0};
    std::atomic<std::uint64_t> shallow{0};
    std::atomic<std::uint64_t> locked{0};
    std::atomic<std::uint64_t> refusedDevice{0};
    std::atomic<std::uint64_t> refusedPool{0};
    std::array<std::atomic<std::uint64_t>, DrawTicket::PartCount> filled{};
    std::array<std::atomic<std::uint64_t>, DrawTicket::PartCount> buildNs{};
    std::array<std::atomic<std::uint64_t>, DrawTicket::PartCount> used{};
    std::array<std::atomic<std::uint64_t>, DrawTicket::PartCount> clear{};
    std::array<std::atomic<std::uint64_t>, DrawTicket::PartCount> disagreed{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(TicketEvent::Count)> events{};
};

Counters& counters() {
    static auto* instance = new Counters();
    return *instance;
}

bool counting() {
    return GuestMemory::Q0PrepStats() != 0;
}

// The release list (TicketReference): producers are queue 0 (its caches' evictions), the reaper
// (tickets of items that never committed) and, under APS5_EXPERIMENT_PREP_HELPERS, the prep helpers
// (the word cache's evictions, tickets a commit closed the slot before); the committer drains it.
// Never destroyed: a reference left at exit belongs to a device that is going.
struct ReleaseList {
    std::mutex mutex;
    std::vector<TicketReference> references;
    std::atomic<bool> pending{false};
};

ReleaseList& releaseList() {
    static auto* list = new ReleaseList();
    return *list;
}

// A reference of another device than the one recording now: its objects went with that device
// (or go with it), so it is forgotten, never destroyed (as the pipeline store abandons them).
void abandonReference(TicketReference&& reference) {
    static auto* graveyard = new std::vector<TicketReference>();
    static std::mutex graveyardMutex;
    std::lock_guard lock(graveyardMutex);
    graveyard->push_back(std::move(reference));
}

// Under GuestMemory::GpuMutex: a last reference goes with the open batch (destroyed on the release
// thread once the batch completed, as the dispatch path's evicted templates); a shared one is just
// dropped, and without an open batch a last one is destroyed here, under the mutex, as the draw
// path's own cache evictions are.
void settleReference(TicketReference&& reference, const Context& context) {
    if (reference.device != context.device) {
        abandonReference(std::move(reference));
        return;
    }
    auto* recorder = Recorder::Active();
    const bool recording = recorder != nullptr && recorder->Recording();
    if (reference.resources != nullptr) {
        if (recording && reference.resources.use_count() == 1) {
            reference.resources->KeepStorageImageBytes(*recorder);
            recorder->Keep(std::move(reference.resources));
        }
        reference.resources.reset();
    }
    if (reference.object != nullptr) {
        if (recording && reference.object.use_count() == 1) recorder->Keep(std::move(reference.object));
        reference.object.reset();
    }
}

// Every strong reference the ticket holds, moved out of it: the template hint and the pipeline.
void takeReferences(DrawTicket& ticket, std::vector<TicketReference>& out) {
    if (ticket.resources != nullptr) out.push_back({ticket.device, std::move(ticket.resources), nullptr});
    if (ticket.pipeline != nullptr) out.push_back({ticket.device, nullptr, std::move(ticket.pipeline)});
    for (auto& words : ticket.wordBuffers) {
        if (words.buffer != nullptr) out.push_back({ticket.device, nullptr, std::move(words.buffer)});
    }
    // The memo names the template only by address; it must not outlive the reference.
    ticket.memo.owner = nullptr;
}

}

int Q0Ticket() {
    static const int mode = [] {
        const char* text = std::getenv("APS5_EXPERIMENT_Q0_TICKET");
        if (text == nullptr) return 0;
        return std::strtol(text, nullptr, 10) == 2 ? 2 : 1;
    }();
    return mode;
}

bool Q0TicketsBuilt() {
    static const bool built = Q0Ticket() != 0 || GuestMemory::Q0PrepStats() >= 2;
    return built;
}

int Q0Template() {
    static const int mode = [] {
        const char* text = std::getenv("APS5_EXPERIMENT_Q0_TEMPLATE");
        if (text == nullptr || Q0Ticket() == 0) return 0;
        return std::strtol(text, nullptr, 10) == 2 ? 2 : 1;
    }();
    return mode;
}

int Q0WordBuffers() {
    static const int mode = [] {
        const char* text = std::getenv("APS5_EXPERIMENT_Q0_WORD_BUFFERS");
        if (text == nullptr || Q0Template() == 0) return 0;
        return std::strtol(text, nullptr, 10) == 2 ? 2 : 1;
    }();
    return mode;
}

bool Q0WordParts() {
    static const bool parts = Q0WordBuffers() != 0 || (Q0Ticket() == 0 && GuestMemory::Q0PrepStats() >= 2);
    return parts;
}

namespace {

// Queue 0's moved data-word buffers (Q2b), for reuse by later tickets of the same template with
// the same words, as the template's own movedWordBuffers serve today's path. A hit needs the very
// template (owner equality with the live hint, so a recycled address never matches, and no
// temporary strong reference that could end up the last one on a ticket producer). The producers
// are queue 0, or under APS5_EXPERIMENT_PREP_HELPERS (PARALLEL-COMMIT-PLAN B0) the prep helpers, any
// number at once: the mutex serves all of them. An evicted buffer goes to the ticket release list:
// a batch may still read it.
struct WordCacheEntry {
    VkDevice device = VK_NULL_HANDLE;
    std::weak_ptr<ShaderResources> owner;
    std::size_t allocation = 0;
    std::vector<std::uint32_t> words;
    std::size_t size = 0;
    std::shared_ptr<Buffer> buffer;
};

struct WordCache {
    std::mutex mutex;
    std::vector<WordCacheEntry> entries;
    std::size_t next = 0;
    std::atomic<std::uint64_t> bytes{0};
};

WordCache& wordCache() {
    static auto* cache = new WordCache();
    return *cache;
}

constexpr std::size_t WordCacheSlots = 64;

std::shared_ptr<Buffer> ticketWordBuffer(const Context& context, const std::shared_ptr<ShaderResources>& owner, std::size_t allocation, std::span<const std::uint32_t> words, std::size_t size) {
    auto& cache = wordCache();
    std::lock_guard lock(cache.mutex);
    for (const auto& entry : cache.entries) {
        if (entry.allocation != allocation || entry.size != size || entry.device != context.device || !std::ranges::equal(entry.words, words) || entry.owner.owner_before(owner) || owner.owner_before(entry.owner)) continue;
        NoteTicketEvent(TicketEvent::WordsCacheHit);
        if (DebugDrawInputBytes()) CountDrawInputBytes(DrawInputBytes::MovedReused, size);
        return entry.buffer;
    }
    // The properties of today's buffers (PrepareDrawBindings), so the VRAM A/Bs stay comparable.
    auto buffer = std::make_shared<Buffer>(context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, DrawSnapshotChunkProperties());
    if (DebugDrawInputBytes()) {
        NoteDrawInputBuffer(context, DrawInputSite::MovedWords, *buffer, size);
        CountDrawInputBytes(DrawInputBytes::MovedCopied, size);
    }
    owner->MovedWordBytes(allocation, words, size, buffer->Bytes());
    NoteTicketEvent(TicketEvent::WordsBuilt);
    WordCacheEntry entry{context.device, owner, allocation, {words.begin(), words.end()}, size, buffer};
    cache.bytes.fetch_add(size, std::memory_order_relaxed);
    if (cache.entries.size() < WordCacheSlots) {
        cache.entries.push_back(std::move(entry));
    } else {
        auto& slot = cache.entries[cache.next++ % WordCacheSlots];
        cache.bytes.fetch_sub(slot.size, std::memory_order_relaxed);
        ReleaseTicketReference({slot.device, nullptr, std::move(slot.buffer)});
        slot = std::move(entry);
    }
    return buffer;
}

}

void FillDrawTicketWords(DrawTicket& ticket, const Context& context, std::span<const CompiledShader> shaders) {
    // Only on the hint's memo for these stages: every data candidate gets the bytes today's path
    // would write (the words with the hint's data patches).
    if (!ticket.Has(DrawTicket::Moved) || ticket.resources == nullptr || ticket.memo.owner != ticket.resources.get() || ticket.memo.refused || ticket.memo.builtFor != shaders.data()) return;
    const bool timed = counting();
    const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    try {
        for (const auto& candidate : ticket.memo.candidates) {
            if (!candidate.data) continue;
            const auto& words = shaders[candidate.shader].program->bindings[candidate.binding].guestDescriptor;
            ShaderResources::TicketWordBuffer entry;
            entry.allocation = candidate.allocation;
            entry.words = words.data();
            entry.size = candidate.size;
            if (ticket.shadow) {
                entry.bytes.resize(candidate.size);
                ticket.resources->MovedWordBytes(candidate.allocation, words, candidate.size, entry.bytes);
            } else {
                entry.buffer = ticketWordBuffer(context, ticket.resources, candidate.allocation, words, candidate.size);
            }
            ticket.wordBuffers.push_back(std::move(entry));
        }
        ticket.parts |= DrawTicket::WordBuffers;
    } catch (const std::exception&) {
        for (auto& entry : ticket.wordBuffers) {
            if (entry.buffer != nullptr) ReleaseTicketReference({ticket.device, nullptr, std::move(entry.buffer)});
        }
        ticket.wordBuffers.clear();
    }
    if (timed) NoteTicketPartNs(DrawTicket::WordBuffers, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()));
}

std::uint64_t TicketWordCacheBytes() {
    return wordCache().bytes.load(std::memory_order_relaxed);
}

bool Q0ByteShadow() {
    static const bool shadow = [] {
        if (GuestMemory::Q0PrepStats() < 3) return false;
        if (std::getenv("APS5_EXPERIMENT_LAZY_GENERATION") != nullptr) return true;
        std::fprintf(stderr, "[q0prep] APS5_DEBUG_Q0_PREP_STATS=3 needs APS5_EXPERIMENT_LAZY_GENERATION=1: no byte shadow (as =2)\n");
        return false;
    }();
    return shadow;
}

namespace {

// The serials a Q4 re-check requires unchanged (COMMITTER-MAP 3.2, commit steps 3 and 4).
bool byteSerialsHold(std::uint64_t pendingSerial, std::uint64_t allocations, std::uint64_t forgetSerial) {
    return (forgetSerial & 1u) == 0 && StorageTexture::PendingSerial() == pendingSerial && GuestAllocations::GuestAllocationsGeneration_nid_postfix() == allocations && GuestMemory::ForgetSerial() == forgetSerial;
}

}

void NoteTicketIndex(const DrawTicket& ticket, std::uint64_t address, std::uint64_t bytes) {
    if (!ticket.indexShadow.has_value()) return;
    NoteTicketEvent(TicketEvent::IndexOffered);
    const auto& shadow = *ticket.indexShadow;
    if (shadow.address != address || shadow.bytes != bytes) {
        NoteTicketEvent(TicketEvent::IndexOther);
        return;
    }
    // The committer's own collect first (a memo hit after InPlaceIndexRange's), then the stamp.
    GuestMemory::CollectWrites(address, static_cast<std::size_t>(bytes));
    if (!GuestMemory::UnchangedSince(address, static_cast<std::size_t>(bytes), shadow.generation)) NoteTicketEvent(TicketEvent::IndexStamp);
    else if (!byteSerialsHold(ticket.pendingSerial, ticket.allocationsGeneration, ticket.forgetSerial)) NoteTicketEvent(TicketEvent::IndexGuard);
    else NoteTicketEvent(TicketEvent::IndexPass);
}

void NoteTicketSlice(const ShaderResources::TicketBindings& ticket, std::uint64_t begin, std::size_t bytes, bool copied) {
    NoteTicketEvent(TicketEvent::SliceOffered);
    const auto found = std::find_if(ticket.slices.begin(), ticket.slices.end(), [&](const ShaderResources::TicketSlice& slice) { return slice.begin == begin && slice.bytes == bytes; });
    if (found == ticket.slices.end()) {
        NoteTicketEvent(TicketEvent::SliceUnpredicted);
        return;
    }
    GuestMemory::CollectWrites(begin, bytes);
    if (!GuestMemory::UnchangedSince(begin, bytes, found->generation)) NoteTicketEvent(TicketEvent::SliceStamp);
    else if (!byteSerialsHold(ticket.pendingSerial, ticket.allocations, ticket.forgetSerial)) NoteTicketEvent(TicketEvent::SliceGuard);
    else NoteTicketEvent(TicketEvent::SlicePass);
    if (copied) NoteTicketEvent(TicketEvent::SliceCopied);
}

void PlanDrawTicketWalks(const DrawTicket& ticket, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) {
    const auto add = [&](std::uint64_t address, std::uint64_t bytes) {
        if (address == 0 || bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address) return;
        ranges.emplace_back(address, address + bytes);
    };
    // Each group stands alone: a descriptor that does not describe throws, and only its own ranges
    // are left out.
    if (ticket.resources != nullptr) {
        try {
            ticket.resources->AppendDescribedRanges(ranges);
            if (ticket.Has(DrawTicket::Moved) && ticket.memo.owner == ticket.resources.get()) {
                struct SlicesTag {};
                auto& slices = ThreadScratch<std::vector<std::pair<std::uint64_t, std::size_t>>, SlicesTag>();
                slices.clear();
                ticket.resources->PredictDrawSlices(ticket.memo, slices);
                for (const auto& [begin, bytes] : slices) add(begin, bytes);
            }
        } catch (const std::exception&) {
        }
    }
    if (ticket.Has(DrawTicket::Targets) && ticket.targets.size() == state.colors.size()) {
        for (std::size_t index = 0; index < state.colors.size(); ++index) {
            const auto& target = ticket.targets[index];
            const auto& color = state.colors[index];
            if (target.multisampled) continue;
            try {
                const auto guestBytes = DescribeSurface(target.surface).guestBytes;
                add(target.surface.baseAddress, guestBytes);
                // StorageTexture::refreshProved's key collect: one key byte per 256 surface bytes.
                if (target.surface.dccAddress != 0) add(target.surface.dccAddress, guestBytes / 256);
            } catch (const std::exception&) {
            }
            if (color.cmaskAddress != 0) add(color.cmaskAddress, color.cmaskBytes);
        }
    }
    if (state.depth && state.depth->samples == 1) {
        const auto& depth = *state.depth;
        try {
            const bool d16 = depth.format == VK_FORMAT_D16_UNORM || depth.format == VK_FORMAT_D16_UNORM_S8_UINT;
            add(depth.address, DepthSliceBytes(depth.extent, d16 ? 2u : 4u));
            if (depth.stencilAddress != 0) add(depth.stencilAddress, DepthSliceBytes(depth.extent, 1u));
            if (depth.htileAddress != 0) add(depth.htileAddress, HtileMetadataBytes(depth.extent));
        } catch (const std::exception&) {
        }
    }
    const bool inputCopies = DrawInputsVram() != 0;
    if (draw.indexed && draw.indexCount != 0 && (inputCopies || draw.indexCount >= Recorder::KeptIndexRangeIndices)) add(draw.indexAddress, static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize);
    if (!inputCopies || shaders.empty() || shaders.front().program == nullptr || draw.indexCount == 0 || draw.instanceCount == 0) return;
    // The vertex fetches as planVertexFetches makes them (from each descriptor's base), with the
    // extent of a vertex-rate fetch in place of the highest index an indexed draw scans for, merged
    // as PlanVertexCopies merges them.
    try {
        if (!draw.indexed && draw.firstVertex > std::numeric_limits<std::uint32_t>::max() - (draw.indexCount - 1u)) return;
        const auto maxIndex = draw.indexed ? 0u : draw.firstVertex + draw.indexCount - 1u;
        struct FetchesTag {};
        auto& fetches = ThreadScratch<std::vector<VertexFetch>, FetchesTag>();
        fetches.clear();
        for (const auto& attribute : shaders.front().program->vertexAttributes) {
            if (NullVertexDescriptor(attribute) || VertexFetchOutOfRange(attribute)) continue;
            const auto& fields = attribute.resource.fields;
            const auto address = fields[0] | (static_cast<std::uint64_t>(fields[1] & 0xffffu) << 32u);
            const auto stride = (fields[1] >> 16u) & 0x3fffu;
            const auto bytes = draw.indexed && attribute.fetchIndex == 0 ? VertexBufferExtent(attribute) : VertexBufferReadSize(attribute, maxIndex, draw.instanceCount, draw.firstInstance);
            fetches.push_back({address, address + bytes, stride, attribute.fetchIndex, DecodeVertexFormat(attribute).alignment});
        }
        for (const auto& [begin, end] : PlanVertexCopies(fetches).copies) add(begin, end - begin);
    } catch (const std::exception&) {
    }
}

bool Q0TemplateParts() {
    // Used tickets with the switch, and the shadow tickets of APS5_DEBUG_Q0_PREP_STATS=2.
    static const bool parts = Q0Template() != 0 || (Q0Ticket() == 0 && GuestMemory::Q0PrepStats() >= 2);
    return parts;
}

const Q0PrepDepth& Q0PrepDepths() {
    static const Q0PrepDepth depths = [] {
        Q0PrepDepth parsed;
        const char* text = std::getenv("APS5_EXPERIMENT_Q0_PREP_DEPTH");
        if (text == nullptr) return parsed;
        char* end = nullptr;
        parsed.pre = static_cast<std::size_t>(std::strtoull(text, &end, 10));
        if (end != nullptr && (*end == ':' || *end == ',')) parsed.bail = static_cast<std::size_t>(std::strtoull(end + 1, &end, 10));
        else parsed.bail = parsed.pre == 0 ? 0 : parsed.pre - 1;
        // LOCAL EXPERIMENT (APS5_EXPERIMENT_Q0_RECEIPTS): the post-push walks' stop depth.
        if (end != nullptr && (*end == ':' || *end == ',')) parsed.post = static_cast<std::size_t>(std::strtoull(end + 1, nullptr, 10));
        else parsed.post = std::min<std::size_t>(parsed.post, parsed.bail);
        return parsed;
    }();
    return depths;
}

DrawTicket::~DrawTicket() {
    // A committed ticket gave its references up already (ReleaseTicketReferences); one that never
    // committed (a stopped pipeline, a commit that threw) hands them to the release list here.
    std::vector<TicketReference> left;
    takeReferences(*this, left);
    for (auto& reference : left) ReleaseTicketReference(std::move(reference));
}

DrawTicket* TicketForBuilt(DrawTicket* ticket, const Context& context) {
    if (ticket->device != context.device) {
        if (counting()) counters().refusedDevice.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    // Owner equivalence: the same pool object, without locking the weak reference.
    if (ticket->pool.owner_before(context.bufferPool) || context.bufferPool.owner_before(ticket->pool)) {
        if (counting()) counters().refusedPool.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    return ticket;
}

std::size_t TicketPartIndex(DrawTicket::Part part) {
    return static_cast<std::size_t>(std::countr_zero(static_cast<std::uint32_t>(part)));
}

TicketUse TicketPartUseBuilt(const DrawTicket& ticket, DrawTicket::Part part) {
    const auto index = TicketPartIndex(part);
    if (!ticket.Has(part)) {
        if (counting()) counters().clear[index].fetch_add(1, std::memory_order_relaxed);
        return TicketUse::None;
    }
    if (counting()) counters().used[index].fetch_add(1, std::memory_order_relaxed);
    if (ticket.shadow) return TicketUse::Shadow;
    // The template parts verify under APS5_EXPERIMENT_Q0_TEMPLATE=2, the others under =2 of
    // APS5_EXPERIMENT_Q0_TICKET.
    const bool templatePart = part == DrawTicket::Template || part == DrawTicket::Moved || part == DrawTicket::Push || part == DrawTicket::Pipeline;
    const int mode = part == DrawTicket::WordBuffers ? Q0WordBuffers() : templatePart ? Q0Template() : Q0Ticket();
    return mode == 2 ? TicketUse::Verify : TicketUse::Use;
}

void TicketDisagrees(DrawTicket::Part part, TicketUse use, const char* what) {
    const auto index = TicketPartIndex(part);
    if (counting()) counters().disagreed[index].fetch_add(1, std::memory_order_relaxed);
    if (use != TicketUse::Verify) return;
    std::fprintf(stderr, "[q0ticket] APS5_EXPERIMENT_Q0_TICKET=2: the ticket's %s part disagrees with today's code: %s\n", index < DrawTicket::PartCount ? DrawTicket::PartNames[index] : "?", what);
    std::fflush(stderr);
    std::abort();
}

void ReleaseTicketReference(TicketReference reference) {
    if (reference.resources == nullptr && reference.object == nullptr) return;
    auto& list = releaseList();
    std::lock_guard lock(list.mutex);
    list.references.push_back(std::move(reference));
    list.pending.store(true, std::memory_order_release);
}

void DrainTicketRelease(const Context& context) {
    auto& list = releaseList();
    if (!list.pending.load(std::memory_order_acquire)) return;
    std::vector<TicketReference> drained;
    {
        std::lock_guard lock(list.mutex);
        drained.swap(list.references);
        list.pending.store(false, std::memory_order_relaxed);
    }
    for (auto& reference : drained) settleReference(std::move(reference), context);
}

void ReleaseTicketReferences(DrawTicket& ticket, const Context& context) {
    std::vector<TicketReference> left;
    takeReferences(ticket, left);
    for (auto& reference : left) settleReference(std::move(reference), context);
}

void NoteTicketEvent(TicketEvent event) {
    if (counting()) counters().events[static_cast<std::size_t>(event)].fetch_add(1, std::memory_order_relaxed);
}

void NoteTicketSkip(TicketSkip skip) {
    if (!counting()) return;
    (skip == TicketSkip::Shallow ? counters().shallow : counters().locked).fetch_add(1, std::memory_order_relaxed);
}

void NoteTicketBuilt(const DrawTicket& ticket) {
    if (!counting()) return;
    auto& all = counters();
    all.built.fetch_add(1, std::memory_order_relaxed);
    for (std::size_t index = 0; index < DrawTicket::PartCount; ++index) {
        if ((ticket.parts & (1u << index)) != 0) all.filled[index].fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteTicketPartNs(DrawTicket::Part part, std::uint64_t ns) {
    counters().buildNs[TicketPartIndex(part)].fetch_add(ns, std::memory_order_relaxed);
}

TicketCounts TakeTicketCounts() {
    auto& all = counters();
    TicketCounts counts;
    counts.built = all.built.exchange(0, std::memory_order_relaxed);
    counts.shallow = all.shallow.exchange(0, std::memory_order_relaxed);
    counts.locked = all.locked.exchange(0, std::memory_order_relaxed);
    counts.refusedDevice = all.refusedDevice.exchange(0, std::memory_order_relaxed);
    counts.refusedPool = all.refusedPool.exchange(0, std::memory_order_relaxed);
    for (std::size_t index = 0; index < DrawTicket::PartCount; ++index) {
        counts.filled[index] = all.filled[index].exchange(0, std::memory_order_relaxed);
        counts.buildNs[index] = all.buildNs[index].exchange(0, std::memory_order_relaxed);
        counts.used[index] = all.used[index].exchange(0, std::memory_order_relaxed);
        counts.clear[index] = all.clear[index].exchange(0, std::memory_order_relaxed);
        counts.disagreed[index] = all.disagreed[index].exchange(0, std::memory_order_relaxed);
    }
    for (std::size_t index = 0; index < counts.events.size(); ++index) counts.events[index] = all.events[index].exchange(0, std::memory_order_relaxed);
    return counts;
}

}
