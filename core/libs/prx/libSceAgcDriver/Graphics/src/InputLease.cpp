#include "prx/libSceAgcDriver/Graphics/include/InputLease.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
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

void add(std::atomic<std::uint64_t>& counter) {
    counter.fetch_add(1, std::memory_order_relaxed);
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
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
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

}

bool InputLeaseCounting() {
    static const bool counting = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return counting;
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

std::size_t Recorder::InputLeaseTable::SetOf(std::uint64_t address, SnapshotUse use) {
    const auto mixed = (address ^ (static_cast<std::uint64_t>(use) << 59u)) * 0x9E3779B97F4A7C15ull;
    return static_cast<std::size_t>(mixed >> 52u) & (Sets - 1);
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
    if (!InputLeaseCounting()) return nullptr;
    auto& c = counters();
    if (!GuestMemory::GpuMutex().HeldByThisThread()) {
        add(c.withoutGpuMutex);
        return nullptr;
    }
    // Inputs of 4 GiB or more get no lease (a way holds 32 bits of size).
    if (bytes == 0 || bytes > std::numeric_limits<std::uint32_t>::max()) return nullptr;
    if (inputLeases == nullptr) inputLeases = std::make_unique<InputLeaseTable>();
    auto probe = InputLeaseTable::Probe::Present;
    auto* lease = inputLeases->Find(address, bytes, use, probe);
    if (probe == InputLeaseTable::Probe::Present) {
        add(c.present);
    } else {
        add(c.inserted);
        if (probe == InputLeaseTable::Probe::Evicted) add(c.evicted);
    }
    return lease;
}

void Recorder::noteInputLeaseMap(InputLease& lease, std::uint64_t address, SnapshotUse use, std::map<DrawSnapshotKey, DrawSnapshot>::iterator found) {
    if (inputLeases == nullptr) return;
    auto& c = counters();
    const bool hit = found != drawSnapshots.end() && std::get<0>(found->first) == address && std::get<1>(found->first) == use;
    const auto serial = inputLeases->Serial(address, use);
    lease.mapHeld = lease.mapValid && lease.mapSerial == serial;
    if (lease.mapHeld) {
        add(c.mWould);
        if (!hit || found != lease.entry) {
            add(c.mWrong);
            if (printWrong()) AgcDriver::ProfilePrint_nid_no_patch("[input-lease] M wrong: the lease of 0x%llx (use %u) held at set serial %u, but today's lookup %s\n", static_cast<unsigned long long>(address), static_cast<unsigned>(use), static_cast<unsigned>(serial), hit ? "found another entry" : "found no entry");
        }
    } else {
        add(lease.mapValid ? c.mMoved : c.mNoLease);
    }
    if (hit) {
        lease.entry = found;
        lease.mapValid = true;
        lease.mapSerial = serial;
    } else {
        lease.mapValid = false;
    }
}

InputLeaseShadow::InputLeaseShadow(Recorder* recorder, std::uint64_t address, std::size_t bytes, Recorder::SnapshotUse use) {
    if (recorder == nullptr || bytes == 0 || !InputLeaseCounting()) return;
    active = true;
    auto& c = counters();
    add(use == Recorder::SnapshotUse::Vertex ? c.vertexCalls : c.indexCalls);
    lease = recorder->InputLeaseFor(address, bytes, use);
    if (lease == nullptr) return;
    lease->mapHeld = false;
    // Row P (PendingRowAt): the lease's answer while PendingSerial is still the S it held at, else
    // the query, whose answer the lease keeps.
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
    if (finished && pHeld && cHeld && lease != nullptr && lease->mapHeld) add(counters().held);
    try {
        reportInputLeases();
    } catch (...) {
    }
}

void InputLeaseShadow::InPlace() {
    if (active) add(counters().inPlace);
}

std::uint64_t InputLeaseShadow::Collect(std::uint64_t address, std::size_t bytes) {
    bool covered = false;
    if (lease == nullptr) return GuestMemory::CollectWritesCovering(address, bytes, covered);
    auto& c = counters();
    // Row C: the lease rule (GuestMemory.hpp), by the first part that fails.
    const auto key = GuestMemory::CurrentCollectMemoKey();
    const auto owner = GuestMemory::CollectCoverageOwner();
    if (!key.Valid()) {
        add(c.cNoKey);
    } else if (!lease->memo.Valid()) {
        add(c.cNotCovered);
    } else if (lease->owner != owner) {
        add(c.cOwner);
    } else if (lease->memo.epoch != key.epoch) {
        add(c.cEpoch);
    } else if (lease->memo.unwatched != key.unwatched) {
        add(c.cUnwatch);
    } else if (lease->memo.mapping != key.mapping) {
        add(c.cMapping);
    } else {
        add(c.cWould);
        cHeld = true;
        // The =2 check of row C, as a counter: the coverage is this thread's, so nothing races it.
        if (!GuestMemory::CollectCovered(address, bytes)) {
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

void InputLeaseShadow::Result(bool reused, bool padding) {
    if (!active) return;
    finished = true;
    auto& c = counters();
    add(padding ? c.padding : reused ? c.reused : c.copied);
}

}
