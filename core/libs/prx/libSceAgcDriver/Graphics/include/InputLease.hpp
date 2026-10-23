#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_INPUTLEASE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_INPUTLEASE_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

// LOCAL DEBUG (APS5_PROFILE_DRAW, REVALIDATE-PLAN section 7, I0; stage S1): the draw-input lease
// counters. No behaviour change: L1's lease table (section 5.1) is filled from today's results and
// never applied, and APS5_EXPERIMENT_INPUT_LEASES is not read yet (stages S3 and S4 apply the rows).
// Every draw input that reaches CopyDrawInput's collect (an index range, or a planned vertex copy
// range; InPlaceDrawInput runs after the probe) is looked up in its recorder's table, which exists
// only while profiling and only under the GpuMutex (InputLeaseFor), and its rows are predicted:
//   P: nothing pending, flushing or live-shadowed overlaps the range at PendingSerial S: the lease's
//      S still current (memo), else StorageTexture::NothingPendingOver (query) stores the answer;
//   C: the lease's collect memo key and coverage owner are the thread's now (GuestMemory's lease
//      rule), checked against CollectCovered before today's collect (W counts a predicted hold the
//      coverage does not show: must be 0); a collect reported covered under an unchanged key fills it;
//   M: the lease's drawSnapshots entry, while its set's serial (bumped by every erase and emplace of a
//      (address, use) hashing to the set) did not move; today's lookup must find that entry (else
//      `wrong`, must be 0); a found entry fills it.
// The [input-lease] line every 10 s prints the window's counts (see reportInputLeases).
namespace AgcDriver::Graphics {

// Whether the I0 counters run: APS5_PROFILE_DRAW, read once.
bool InputLeaseCounting();

// One table way's payload (REVALIDATE-PLAN 5.1): the drawSnapshots entry and the set serial it was
// found at (row M), the collect memo key and coverage owner a covered collect left (row C), the
// PendingSerial a nothing-over answer held at (row P, 0: none). `mapHeld`: row M's prediction for
// the input in progress (I0 only).
struct Recorder::InputLease {
    std::map<DrawSnapshotKey, DrawSnapshot>::iterator entry{};
    bool mapValid = false;
    bool mapHeld = false;
    std::uint32_t mapSerial = 0;
    GuestMemory::CollectMemoKey memo{};
    std::uint64_t owner = 0;
    std::uint64_t pendingAt = 0;
};

// 4096 sets of 4 ways (REVALIDATE-PLAN 5.1): a set header is one cache line holding the ways' keys
// (address, bytes, use), the set's map serial and the ways' recency; the payloads lie in a parallel
// array, so a probe touches two lines. A key miss takes the oldest way and resets its payload.
struct Recorder::InputLeaseTable {
    static constexpr std::size_t Sets = 4096;
    static constexpr std::size_t Ways = 4;
    struct alignas(64) Set {
        std::array<std::uint64_t, Ways> address{};
        std::array<std::uint32_t, Ways> bytes{};
        // SnapshotUse + 1; 0: an empty way.
        std::array<std::uint8_t, Ways> use{};
        std::uint32_t mapSerial = 0;
        // Recency rank per way: 0 the newest, Ways - 1 the oldest (the way a key miss takes).
        std::array<std::uint8_t, Ways> age{0, 1, 2, 3};
    };
    static_assert(sizeof(Set) == 64, "an input lease set header is one cache line");
    enum class Probe : std::uint8_t { Present, Inserted, Evicted };
    std::vector<Set> sets = std::vector<Set>(Sets);
    std::vector<InputLease> leases = std::vector<InputLease>(Sets * Ways);
    static std::size_t SetOf(std::uint64_t address, SnapshotUse use);
    InputLease* Find(std::uint64_t address, std::size_t bytes, SnapshotUse use, Probe& probe);
    void Bump(std::uint64_t address, SnapshotUse use) { ++sets[SetOf(address, use)].mapSerial; }
    std::uint32_t Serial(std::uint64_t address, SnapshotUse use) const { return sets[SetOf(address, use)].mapSerial; }
};

// The pending registry's addition sites (REVALIDATE-PLAN 5.2): markLayersPending's push, its lent
// re-mark bump, reconcilePending's push and MarkShadowed's units going live.
enum class PendingAdditionSite : std::uint8_t { Mark, Lent, Reconcile, Shadow, Count };
// Counts one addition at its site (PendingAdditions) and, when the calling thread does not hold the
// GpuMutex, one U for the site (row P's invariant: every addition runs under the GpuMutex, as every
// P skip would). Called under InputLeaseCounting() only, inside the site's own lock.
void NotePendingAddition(PendingAdditionSite site);
std::uint64_t PendingAdditions();

// The I0 counts since the last [input-lease] report (InputLeaseCountsNow reads them without taking
// them, for tests).
struct InputLeaseCounts {
    std::uint64_t indexCalls = 0, vertexCalls = 0, inPlace = 0, reused = 0, copied = 0, padding = 0;
    std::uint64_t present = 0, inserted = 0, evicted = 0, withoutGpuMutex = 0;
    std::uint64_t pMemo = 0, pQuery = 0, pPending = 0, pFlushing = 0, pShadow = 0, pMoved = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(PendingAdditionSite::Count)> unlocked{};
    std::uint64_t cWould = 0, cNoKey = 0, cOwner = 0, cEpoch = 0, cUnwatch = 0, cMapping = 0, cNotCovered = 0, cWrong = 0;
    std::uint64_t mWould = 0, mNoLease = 0, mMoved = 0, mWrong = 0;
    std::uint64_t held = 0;
};
InputLeaseCounts InputLeaseCountsNow();

// I0's view of one draw input, made before InPlaceDrawInput (or by CopyDrawInput for a direct
// caller) while InputLeaseCounting(): probes the recorder's table and predicts row P; CopyDrawInput
// hands it its collect (row C) and its ReusableDrawSnapshot the lease (row M), then its outcome.
// Inactive (counts nothing) without a recorder or for an empty range.
class InputLeaseShadow {
public:
    InputLeaseShadow(Recorder* recorder, std::uint64_t address, std::size_t bytes, Recorder::SnapshotUse use);
    ~InputLeaseShadow();
    InputLeaseShadow(const InputLeaseShadow&) = delete;
    InputLeaseShadow& operator=(const InputLeaseShadow&) = delete;
    // The input was bound in place (InPlaceDrawInput): no collect or lookup follows.
    void InPlace();
    // CopyDrawInput's collect: today's (CollectWritesCovering is CollectWrites), after row C's
    // prediction and its W check; a collect reported covered under the key read before it, which
    // still holds after it, fills the lease's key and owner.
    std::uint64_t Collect(std::uint64_t address, std::size_t bytes);
    // The lease for ReusableDrawSnapshot's row M (null: none).
    Recorder::InputLease* Lease() const { return lease; }
    // CopyDrawInput's outcome: a kept copy reused, or one refused for its padding (else a copy).
    void Result(bool reused, bool padding);

private:
    Recorder::InputLease* lease = nullptr;
    bool active = false;
    bool pHeld = false;
    bool cHeld = false;
    bool finished = false;
};

}

#endif
