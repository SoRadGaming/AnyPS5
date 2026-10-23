#include "prx/libSceAgcDriver/Graphics/include/InputLease.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>

namespace AgcDriver::Graphics {

namespace {

constexpr std::size_t AdditionSites = static_cast<std::size_t>(PendingAdditionSite::Count);

// The I0 counts (see InputLease.hpp): relaxed atomics, written by the committer (and, for the
// additions, by whichever thread adds to the pending registry), taken by the 10 s report.
struct Counters {
    std::atomic<std::uint64_t> indexCalls{0}, vertexCalls{0}, inPlace{0}, reused{0}, copied{0}, padding{0};
    std::atomic<std::uint64_t> present{0}, inserted{0}, evicted{0}, withoutGpuMutex{0};
    std::atomic<std::uint64_t> pMemo{0}, pQuery{0}, pPending{0}, pFlushing{0}, pShadow{0}, pMoved{0}, pNoSerial{0};
    std::array<std::atomic<std::uint64_t>, AdditionSites> unlocked{};
    std::atomic<std::uint64_t> cWould{0}, cNoKey{0}, cOwner{0}, cEpoch{0}, cUnwatch{0}, cMapping{0}, cNotCovered{0}, cWrong{0};
    std::atomic<std::uint64_t> mWould{0}, mNoLease{0}, mMoved{0}, mWrong{0};
    std::atomic<std::uint64_t> held{0};
    // Cumulative (PendingAdditions), and the values the last report saw of it and of PendingSerial.
    std::atomic<std::uint64_t> additions{0};
    std::atomic<std::uint64_t> lastAdditions{0};
    std::atomic<std::uint64_t> lastSerial{0};
    std::atomic<std::int64_t> lastReportMs{0};
    // The first few W and M `wrong` cases are printed with their input.
    std::atomic<std::uint32_t> printed{0};
};

Counters& counters() {
    static auto* counts = new Counters();
    return *counts;
}

// LOCAL EXPERIMENT (L1): the switch's cumulative counts (InputLeaseExperimentCounts), what the last =2
// report printed of them, and how many void cases were printed.
struct Experiment {
    std::atomic<std::uint64_t> cHeld{0}, cPredicted{0}, cVerified{0}, cVoided{0};
    std::atomic<std::uint64_t> mHeld{0}, mPredicted{0}, mVerified{0};
    InputLeaseExperimentCounts reported{};
    std::atomic<std::int64_t> lastReportMs{0};
    std::atomic<std::uint32_t> printed{0};
};

Experiment& experiment() {
    static auto* counts = new Experiment();
    return *counts;
}

void add(std::atomic<std::uint64_t>& counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
}

std::int64_t steadyMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template<typename Read>
InputLeaseCounts readCounts(Counters& c, Read read) {
    InputLeaseCounts out;
    out.indexCalls = read(c.indexCalls);
    out.vertexCalls = read(c.vertexCalls);
    out.inPlace = read(c.inPlace);
    out.reused = read(c.reused);
    out.copied = read(c.copied);
    out.padding = read(c.padding);
    out.present = read(c.present);
    out.inserted = read(c.inserted);
    out.evicted = read(c.evicted);
    out.withoutGpuMutex = read(c.withoutGpuMutex);
    out.pMemo = read(c.pMemo);
    out.pQuery = read(c.pQuery);
    out.pPending = read(c.pPending);
    out.pFlushing = read(c.pFlushing);
    out.pShadow = read(c.pShadow);
    out.pMoved = read(c.pMoved);
    out.pNoSerial = read(c.pNoSerial);
    for (std::size_t site = 0; site < AdditionSites; ++site) out.unlocked[site] = read(c.unlocked[site]);
    out.cWould = read(c.cWould);
    out.cNoKey = read(c.cNoKey);
    out.cOwner = read(c.cOwner);
    out.cEpoch = read(c.cEpoch);
    out.cUnwatch = read(c.cUnwatch);
    out.cMapping = read(c.cMapping);
    out.cNotCovered = read(c.cNotCovered);
    out.cWrong = read(c.cWrong);
    out.mWould = read(c.mWould);
    out.mNoLease = read(c.mNoLease);
    out.mMoved = read(c.mMoved);
    out.mWrong = read(c.mWrong);
    out.held = read(c.held);
    return out;
}

bool printWrong() {
    return counters().printed.fetch_add(1, std::memory_order_relaxed) < 8;
}

double ratio(std::uint64_t part, std::uint64_t whole) {
    return whole == 0 ? 0.0 : static_cast<double>(part) / static_cast<double>(whole);
}

// The [input-lease] line, every 10 s from the first counted input on. The projection (REVALIDATE-PLAN
// 7) weighs each row's share of its predictions by the points it would save: 1.5 per P memo hold, 0.85
// per P query nothing-over, -0.5 per P overlap (the query's cost for nothing), 0.4 per C hold, 0.8
// per M hold, plus row E's 0.31 and minus 0.2 of overhead.
void reportInputLeases() {
    auto& c = counters();
    const auto nowMs = steadyMs();
    auto last = c.lastReportMs.load(std::memory_order_relaxed);
    if (last == 0) {
        if (c.lastReportMs.compare_exchange_strong(last, nowMs, std::memory_order_relaxed)) {
            c.lastSerial.store(StorageTexture::PendingSerial(), std::memory_order_relaxed);
            c.lastAdditions.store(c.additions.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return;
    }
    if (nowMs - last < 10000 || !c.lastReportMs.compare_exchange_strong(last, nowMs, std::memory_order_relaxed)) return;
    const auto n = readCounts(c, [](std::atomic<std::uint64_t>& counter) { return counter.exchange(0, std::memory_order_relaxed); });
    const auto serial = StorageTexture::PendingSerial();
    const auto serialBumps = serial - c.lastSerial.exchange(serial, std::memory_order_relaxed);
    const auto additions = c.additions.load(std::memory_order_relaxed);
    const auto additionBumps = additions - c.lastAdditions.exchange(additions, std::memory_order_relaxed);
    const auto pTotal = n.pMemo + n.pQuery + n.pPending + n.pFlushing + n.pShadow + n.pMoved + n.pNoSerial;
    const auto cTotal = n.cWould + n.cNoKey + n.cOwner + n.cEpoch + n.cUnwatch + n.cMapping + n.cNotCovered;
    const auto mTotal = n.mWould + n.mNoLease + n.mMoved;
    const double projection = 1.5 * ratio(n.pMemo, pTotal) + 0.85 * ratio(n.pQuery, pTotal) - 0.5 * ratio(n.pPending + n.pFlushing + n.pShadow, pTotal) + 0.4 * ratio(n.cWould, cTotal) + 0.8 * ratio(n.mWould, mTotal) + 0.31 - 0.2;
    using U = unsigned long long;
    AgcDriver::ProfilePrint_nid_no_patch("[input-lease] (10 s): calls index %llu vertex %llu (in place %llu): reused %llu, copied %llu, padding refusals %llu; probes: present %llu, inserted %llu, evicted %llu, without the GpuMutex %llu; P: memo would-hold %llu, query nothing-over %llu, overlap pending %llu flushing %llu shadow %llu, query refused: serial moved %llu, serial 0 %llu, PendingSerial bumps %llu, PendingAdditions bumps %llu, U mark %llu lent %llu reconcile %llu shadow %llu; C: would-hold %llu, misses: no key %llu, owner %llu, epoch %llu, unwatch %llu, mapping %llu, not covered %llu; W %llu; M: would-hold %llu, misses: no lease %llu, set serial moved %llu; wrong %llu; held together %llu; projection %.2f points\n",
        static_cast<U>(n.indexCalls), static_cast<U>(n.vertexCalls), static_cast<U>(n.inPlace), static_cast<U>(n.reused), static_cast<U>(n.copied), static_cast<U>(n.padding),
        static_cast<U>(n.present), static_cast<U>(n.inserted), static_cast<U>(n.evicted), static_cast<U>(n.withoutGpuMutex),
        static_cast<U>(n.pMemo), static_cast<U>(n.pQuery), static_cast<U>(n.pPending), static_cast<U>(n.pFlushing), static_cast<U>(n.pShadow), static_cast<U>(n.pMoved), static_cast<U>(n.pNoSerial), static_cast<U>(serialBumps), static_cast<U>(additionBumps),
        static_cast<U>(n.unlocked[0]), static_cast<U>(n.unlocked[1]), static_cast<U>(n.unlocked[2]), static_cast<U>(n.unlocked[3]),
        static_cast<U>(n.cWould), static_cast<U>(n.cNoKey), static_cast<U>(n.cOwner), static_cast<U>(n.cEpoch), static_cast<U>(n.cUnwatch), static_cast<U>(n.cMapping), static_cast<U>(n.cNotCovered), static_cast<U>(n.cWrong),
        static_cast<U>(n.mWould), static_cast<U>(n.mNoLease), static_cast<U>(n.mMoved), static_cast<U>(n.mWrong), static_cast<U>(n.held), projection);
}

// LOCAL EXPERIMENT (L1): the =2 line every 10 s (from the first input on), with or without
// APS5_PROFILE_DRAW: the gates G3 and G4 judge the void share of the predictions from it.
void reportInputLeaseExperiment() {
    auto& e = experiment();
    const auto nowMs = steadyMs();
    auto last = e.lastReportMs.load(std::memory_order_relaxed);
    if (last == 0) {
        e.lastReportMs.compare_exchange_strong(last, nowMs, std::memory_order_relaxed);
        return;
    }
    if (nowMs - last < 10000 || !e.lastReportMs.compare_exchange_strong(last, nowMs, std::memory_order_relaxed)) return;
    const auto now = InputLeaseExperimentNow();
    const auto was = e.reported;
    e.reported = now;
    using U = unsigned long long;
    const auto cPredicted = now.cPredicted - was.cPredicted;
    const auto cVoided = now.cVoided - was.cVoided;
    AgcDriver::ProfilePrint_nid_no_patch("[input-lease] APS5_EXPERIMENT_INPUT_LEASES=2 (10 s): C predicted %llu, verified %llu, void %llu (%.3f%%); M predicted %llu, verified %llu; E checked at every gate read\n",
        static_cast<U>(cPredicted), static_cast<U>(now.cVerified - was.cVerified), static_cast<U>(cVoided), 100.0 * ratio(cVoided, cPredicted), static_cast<U>(now.mPredicted - was.mPredicted), static_cast<U>(now.mVerified - was.mVerified));
}

// Row C's lease rule (GuestMemory.hpp) for I0's counts: a hold, or the first part that fails.
bool countCollectRow(Counters& c, const Recorder::InputLease& lease, const GuestMemory::CollectMemoKey& key, std::uint64_t owner) {
    if (!key.Valid()) add(c.cNoKey);
    else if (!lease.memo.Valid()) add(c.cNotCovered);
    else if (lease.owner != owner) add(c.cOwner);
    else if (lease.memo.epoch != key.epoch) add(c.cEpoch);
    else if (lease.memo.unwatched != key.unwatched) add(c.cUnwatch);
    else if (lease.memo.mapping != key.mapping) add(c.cMapping);
    else {
        add(c.cWould);
        return true;
    }
    return false;
}

int readInputLeases() {
    const char* text = std::getenv("APS5_EXPERIMENT_INPUT_LEASES");
    if (text == nullptr) return 0;
    const auto value = std::strtol(text, nullptr, 10);
    return value == 0 ? 0 : value == 2 ? 2 : 1;
}

// APS5_EXPERIMENT_INPUT_LEASE_ROWS: the letters of PCME present (either case); unset: every row.
std::uint8_t readInputLeaseRows() {
    const char* text = std::getenv("APS5_EXPERIMENT_INPUT_LEASE_ROWS");
    if (text == nullptr) return 0xf;
    std::uint8_t rows = 0;
    for (const char* at = text; *at != '\0'; ++at) {
        switch (*at) {
            case 'P': case 'p': rows |= static_cast<std::uint8_t>(InputLeaseRow::P); break;
            case 'C': case 'c': rows |= static_cast<std::uint8_t>(InputLeaseRow::C); break;
            case 'M': case 'm': rows |= static_cast<std::uint8_t>(InputLeaseRow::M); break;
            case 'E': case 'e': rows |= static_cast<std::uint8_t>(InputLeaseRow::E); break;
            default: break;
        }
    }
    return rows;
}

// Test only: see InputLeaseSuppressBumpsForTest.
std::atomic<bool> suppressBumps{false};

}

bool InputLeaseCounting() {
    static const bool counting = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return counting;
}

int InputLeases() {
    static const int mode = readInputLeases();
    return mode;
}

bool InputLeaseRowOn(InputLeaseRow row) {
    static const std::uint8_t rows = InputLeases() != 0 ? readInputLeaseRows() : 0;
    return (rows & static_cast<std::uint8_t>(row)) != 0;
}

bool InputLeaseViewed() {
    static const bool viewed = InputLeaseCounting() || InputLeases() != 0;
    return viewed;
}

void InputLeaseSuppressBumpsForTest(bool suppress) {
    suppressBumps.store(suppress, std::memory_order_relaxed);
}

void NotePendingAddition(PendingAdditionSite site) {
    auto& c = counters();
    add(c.additions);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) add(c.unlocked[static_cast<std::size_t>(site)]);
}

std::uint64_t PendingAdditions() {
    return counters().additions.load(std::memory_order_relaxed);
}

InputLeaseCounts InputLeaseCountsNow() {
    return readCounts(counters(), [](std::atomic<std::uint64_t>& counter) { return counter.load(std::memory_order_relaxed); });
}

InputLeaseExperimentCounts InputLeaseExperimentNow() {
    auto& e = experiment();
    const auto load = [](std::atomic<std::uint64_t>& counter) { return counter.load(std::memory_order_relaxed); };
    InputLeaseExperimentCounts counts;
    counts.cHeld = load(e.cHeld);
    counts.cPredicted = load(e.cPredicted);
    counts.cVerified = load(e.cVerified);
    counts.cVoided = load(e.cVoided);
    counts.mHeld = load(e.mHeld);
    counts.mPredicted = load(e.mPredicted);
    counts.mVerified = load(e.mVerified);
    return counts;
}

std::size_t Recorder::InputLeaseTable::SetOf(std::uint64_t address, SnapshotUse use) {
    const auto mixed = (address ^ (static_cast<std::uint64_t>(use) << 59u)) * 0x9E3779B97F4A7C15ull;
    return static_cast<std::size_t>(mixed >> 52u) & (Sets - 1);
}

void Recorder::InputLeaseTable::Bump(std::uint64_t address, SnapshotUse use) {
    if (suppressBumps.load(std::memory_order_relaxed)) return;
    ++sets[SetOf(address, use)].mapSerial;
}

Recorder::InputLease* Recorder::InputLeaseTable::Find(std::uint64_t address, std::size_t bytes, SnapshotUse use, Probe& probe) {
    const auto index = SetOf(address, use);
    auto& set = sets[index];
    const auto tag = static_cast<std::uint8_t>(static_cast<std::uint8_t>(use) + 1u);
    const auto narrow = static_cast<std::uint32_t>(bytes);
    std::size_t way = Ways;
    for (std::size_t candidate = 0; candidate < Ways; ++candidate) {
        if (set.use[candidate] == tag && set.address[candidate] == address && set.bytes[candidate] == narrow) {
            way = candidate;
            break;
        }
    }
    if (way == Ways) {
        way = static_cast<std::size_t>(std::max_element(set.age.begin(), set.age.end()) - set.age.begin());
        probe = set.use[way] != 0 ? Probe::Evicted : Probe::Inserted;
        set.address[way] = address;
        set.bytes[way] = narrow;
        set.use[way] = tag;
        leases[index * Ways + way] = InputLease{};
    } else {
        probe = Probe::Present;
    }
    const auto rank = set.age[way];
    for (auto& age : set.age) {
        if (age < rank) ++age;
    }
    set.age[way] = 0;
    return &leases[index * Ways + way];
}

Recorder::InputLease* Recorder::InputLeaseFor(std::uint64_t address, std::size_t bytes, SnapshotUse use) {
    if (!InputLeaseViewed()) return nullptr;
    const bool counting = InputLeaseCounting();
    auto& c = counters();
    if (!GuestMemory::GpuMutex().HeldByThisThread()) {
        if (counting) add(c.withoutGpuMutex);
        return nullptr;
    }
    // Inputs of 4 GiB or more get no lease (a way holds 32 bits of size).
    if (bytes == 0 || bytes > std::numeric_limits<std::uint32_t>::max()) return nullptr;
    if (inputLeases == nullptr) inputLeases = std::make_unique<InputLeaseTable>();
    auto probe = InputLeaseTable::Probe::Present;
    auto* lease = inputLeases->Find(address, bytes, use, probe);
    if (counting) {
        if (probe == InputLeaseTable::Probe::Present) {
            add(c.present);
        } else {
            add(c.inserted);
            if (probe == InputLeaseTable::Probe::Evicted) add(c.evicted);
        }
    }
    return lease;
}

// ReusableDrawSnapshot's lookup with a lease (REVALIDATE-PLAN 5.4). Row M holds while the lease found
// an entry of its (address, use) at the set serial the set still has: every key that falls between a
// Vertex request and the entry it returns shares that (address, use), so its erase or emplace bumped
// the serial, and collisions only cause misses; the entry's iterator is valid while no erase of the
// set happened. =1 takes the entry; =2 runs the lookup, which must find that very entry (the map is the
// committer's, changed under the GpuMutex only, so nothing races it); I0 records, with `wrong` as its
// shadow check. Everything after the lookup is ReusableDrawSnapshot's own.
auto Recorder::inputLeaseLookup(InputLease& lease, std::uint64_t address, std::size_t bytes, SnapshotUse use) -> std::map<DrawSnapshotKey, DrawSnapshot>::iterator {
    // A lease exists only with its table.
    const auto serial = inputLeases->Serial(address, use);
    const bool held = lease.mapValid && lease.mapSerial == serial;
    lease.mapHeld = held;
    const int mode = InputLeaseRowOn(InputLeaseRow::M) ? InputLeases() : 0;
    std::map<DrawSnapshotKey, DrawSnapshot>::iterator found;
    if (held && mode == 1) {
        found = lease.entry;
        add(experiment().mHeld);
    } else {
        found = use == SnapshotUse::Vertex ? drawSnapshots.lower_bound({address, use, bytes}) : drawSnapshots.find({address, use, bytes});
        if (held && mode == 2) {
            add(experiment().mPredicted);
            if (found != lease.entry) {
                const bool other = found != drawSnapshots.end() && std::get<0>(found->first) == address && std::get<1>(found->first) == use;
                std::fprintf(stderr, "[input-lease] APS5_EXPERIMENT_INPUT_LEASES=2: M: the lease of 0x%llx+0x%zx (use %u) holds its drawSnapshots entry (%zu bytes) at set serial %llu, which did not move, but today's lookup %s\n", static_cast<unsigned long long>(address), bytes, static_cast<unsigned>(use), std::get<2>(lease.entry->first), static_cast<unsigned long long>(serial), other ? "finds another entry of the range" : "finds no entry of the range");
                std::fflush(stderr);
                std::abort();
            }
            add(experiment().mVerified);
        }
    }
    const bool hit = found != drawSnapshots.end() && std::get<0>(found->first) == address && std::get<1>(found->first) == use;
    if (InputLeaseCounting()) {
        auto& c = counters();
        if (held) {
            add(c.mWould);
            if (mode == 0 && (!hit || found != lease.entry)) {
                add(c.mWrong);
                if (printWrong()) AgcDriver::ProfilePrint_nid_no_patch("[input-lease] M wrong: the lease of 0x%llx (use %u) held at set serial %llu, but today's lookup %s\n", static_cast<unsigned long long>(address), static_cast<unsigned>(use), static_cast<unsigned long long>(serial), hit ? "found another entry" : "found no entry");
            }
        } else {
            add(lease.mapValid ? c.mMoved : c.mNoLease);
        }
    }
    if (hit) {
        lease.entry = found;
        lease.mapValid = true;
        lease.mapSerial = serial;
    } else {
        lease.mapValid = false;
    }
    return found;
}

InputLeaseShadow::InputLeaseShadow(Recorder* recorder, std::uint64_t address, std::size_t bytes, Recorder::SnapshotUse use) {
    if (recorder == nullptr || bytes == 0 || !InputLeaseViewed()) return;
    active = true;
    counting = InputLeaseCounting();
    auto& c = counters();
    if (counting) add(use == Recorder::SnapshotUse::Vertex ? c.vertexCalls : c.indexCalls);
    lease = recorder->InputLeaseFor(address, bytes, use);
    if (lease == nullptr) return;
    lease->mapHeld = false;
    // I0's row P (PendingRowAt), a shadow prediction, only with the switch unset: the lease's answer
    // while PendingSerial is still the S it held at, else the query, whose answer the lease keeps.
    if (!counting || InputLeases() != 0) return;
    if (lease->pendingAt != 0 && StorageTexture::PendingSerial() == lease->pendingAt) {
        add(c.pMemo);
        pHeld = true;
        return;
    }
    auto over = StorageTexture::PendingOver::Nothing;
    lease->pendingAt = StorageTexture::NothingPendingOver(address, bytes, &over);
    switch (over) {
        case StorageTexture::PendingOver::Nothing:
            add(c.pQuery);
            pHeld = true;
            break;
        case StorageTexture::PendingOver::Pending: add(c.pPending); break;
        case StorageTexture::PendingOver::Flushing: add(c.pFlushing); break;
        case StorageTexture::PendingOver::Shadow: add(c.pShadow); break;
        case StorageTexture::PendingOver::Moved: add(c.pMoved); break;
        // PendingSerial still 0 (no storage image went pending yet): no proof, and no memo to keep.
        case StorageTexture::PendingOver::NoSerial: add(c.pNoSerial); break;
    }
}

InputLeaseShadow::~InputLeaseShadow() {
    if (!active) return;
    if (counting && finished && pHeld && cHeld && lease != nullptr && lease->mapHeld) add(counters().held);
    try {
        if (counting) reportInputLeases();
        if (InputLeases() == 2) reportInputLeaseExperiment();
    } catch (...) {
    }
}

void InputLeaseShadow::InPlace() {
    if (counting) add(counters().inPlace);
}

std::uint64_t InputLeaseShadow::countedCollect(std::uint64_t address, std::size_t bytes) {
    bool covered = false;
    if (lease == nullptr || !counting) return GuestMemory::CollectWritesCovering(address, bytes, covered);
    auto& c = counters();
    const auto key = GuestMemory::CurrentCollectMemoKey();
    const auto owner = GuestMemory::CollectCoverageOwner();
    if (countCollectRow(c, *lease, key, owner)) {
        cHeld = true;
        // The =2 check, as a counter (I0, the switch unset): the coverage is this thread's, so nothing
        // races it.
        if (InputLeases() == 0 && !GuestMemory::CollectCovered(address, bytes)) {
            add(c.cWrong);
            if (printWrong()) AgcDriver::ProfilePrint_nid_no_patch("[input-lease] W: the lease of 0x%llx+0x%llx held under collect memo key {epoch %llu, unwatch %llu, mapping %llu} but its range is not covered\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(key.epoch), static_cast<unsigned long long>(key.unwatched), static_cast<unsigned long long>(key.mapping));
        }
    }
    const auto generation = GuestMemory::CollectWritesCovering(address, bytes, covered);
    if (covered && GuestMemory::CurrentCollectMemoKey() == key) {
        lease->memo = key;
        lease->owner = owner;
    }
    return generation;
}

std::uint64_t InputLeaseShadow::Collect(std::uint64_t address, std::size_t bytes, std::uint64_t& registryGeneration) {
    const int mode = InputLeaseRowOn(InputLeaseRow::C) ? InputLeases() : 0;
    if (mode == 0 || lease == nullptr) {
        registryGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        return countedCollect(address, bytes);
    }
    // LOCAL EXPERIMENT (L1 row C, REVALIDATE-PLAN 5.3): the key read once, where today reads the
    // registry generation, before the collect.
    const auto key = GuestMemory::CurrentCollectMemoKey();
    registryGeneration = key.Valid() ? key.mapping : GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    const auto owner = GuestMemory::CollectCoverageOwner();
    const bool held = key.Valid() && lease->memo == key && lease->owner == owner;
    if (counting) countCollectRow(counters(), *lease, key, owner);
    cHeld = held;
    if (held) {
        auto& e = experiment();
        // =1: today's collect would take the epoch coverage hit (the lease rule), whose only effects
        // CoveredCollectGeneration repeats.
        if (mode == 1) {
            add(e.cHeld);
            return GuestMemory::CoveredCollectGeneration();
        }
        // =2, just before today's collect: the thread's coverage must hold the range under the
        // lease's key (the coverage is this thread's and changes only by its own collects, none of
        // which ran since the key was read, so this is its answer at =1's skip point; nothing races
        // it). Stricter than a check under the current key: it holds whether or not the key moved. A
        // key that moved since the read (only another thread's unwatch or mapping change can come
        // between: the epoch is this thread's own) makes the prediction void, counted: today's
        // collect then walks, where =1's skip took the move as landing just after it, a legal
        // interleaving.
        add(e.cPredicted);
        if (!GuestMemory::CollectCoveredUnder(key, address, bytes)) {
            std::fprintf(stderr, "[input-lease] APS5_EXPERIMENT_INPUT_LEASES=2: C: the lease of 0x%llx+0x%zx holds under collect memo key {epoch %llu, unwatch %llu, mapping %llu} and coverage owner %llu, both the thread's, but the thread's coverage does not hold the range under that key: today's collect would not be the coverage hit =1 replaces it with\n", static_cast<unsigned long long>(address), bytes, static_cast<unsigned long long>(key.epoch), static_cast<unsigned long long>(key.unwatched), static_cast<unsigned long long>(key.mapping), static_cast<unsigned long long>(owner));
            std::fflush(stderr);
            std::abort();
        }
        if (const auto now = GuestMemory::CurrentCollectMemoKey(); now != key) {
            add(e.cVoided);
            if (e.printed.fetch_add(1, std::memory_order_relaxed) < 8) AgcDriver::ProfilePrint_nid_no_patch("[input-lease] APS5_EXPERIMENT_INPUT_LEASES=2: C void: the lease of 0x%llx+0x%zx held under collect memo key {epoch %llu, unwatch %llu, mapping %llu}, which moved to {epoch %llu, unwatch %llu, mapping %llu} before today's collect\n", static_cast<unsigned long long>(address), bytes, static_cast<unsigned long long>(key.epoch), static_cast<unsigned long long>(key.unwatched), static_cast<unsigned long long>(key.mapping), static_cast<unsigned long long>(now.epoch), static_cast<unsigned long long>(now.unwatched), static_cast<unsigned long long>(now.mapping));
        } else {
            add(e.cVerified);
        }
    }
    bool covered = false;
    const auto generation = GuestMemory::CollectWritesCovering(address, bytes, covered);
    // Filled only when the key after the collect is the one read before it: every part only grows,
    // so it was that key throughout.
    if (covered && GuestMemory::CurrentCollectMemoKey() == key) {
        lease->memo = key;
        lease->owner = owner;
    }
    return generation;
}

void InputLeaseShadow::Result(bool reused, bool padding) {
    if (!active) return;
    finished = true;
    if (!counting) return;
    auto& c = counters();
    add(padding ? c.padding : reused ? c.reused : c.copied);
}

}
