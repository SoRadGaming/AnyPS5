#include "prx/libSceAgcDriver/Graphics/include/DrawTicket.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include <bit>
#include <cstdio>
#include <cstdlib>
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
};

Counters& counters() {
    static auto* instance = new Counters();
    return *instance;
}

bool counting() {
    return GuestMemory::Q0PrepStats() != 0;
}

// The release list (TicketReference): producers are queue 0 (its caches' evictions) and the
// reaper (tickets of items that never committed); the committer drains it. Never destroyed: a
// reference left at exit belongs to a device that is going.
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

// Every strong reference the ticket holds, moved out of it (none in Q1: its parts are values).
void takeReferences(DrawTicket& ticket, std::vector<TicketReference>& out) {
    static_cast<void>(ticket);
    static_cast<void>(out);
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

const Q0PrepDepth& Q0PrepDepths() {
    static const Q0PrepDepth depths = [] {
        Q0PrepDepth parsed;
        const char* text = std::getenv("APS5_EXPERIMENT_Q0_PREP_DEPTH");
        if (text == nullptr) return parsed;
        char* end = nullptr;
        parsed.pre = static_cast<std::size_t>(std::strtoull(text, &end, 10));
        if (end != nullptr && (*end == ':' || *end == ',')) parsed.bail = static_cast<std::size_t>(std::strtoull(end + 1, nullptr, 10));
        else parsed.bail = parsed.pre == 0 ? 0 : parsed.pre - 1;
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
    return Q0Ticket() == 2 ? TicketUse::Verify : TicketUse::Use;
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
    return counts;
}

}
