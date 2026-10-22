#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PARALLELCOMPARE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PARALLELCOMPARE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace AgcDriver::Graphics {

// LOCAL EXPERIMENT (APS5_EXPERIMENT_SNAPSHOT_HASH, GPU-OWNED-PLAN L1-1): the hash memo of one
// snapshot block. While `known`, `lanes` is HashLanes of the snapshot's own bytes of the block.
struct SpanHash {
    std::array<std::uint64_t, 4> lanes{};
    std::uint8_t known = 0;
};

// `first` is the snapshot (S), `second` guest memory. A span with a `hash` memo is compared by
// hashing guest memory alone once the memo is known; CompareSpans reports such an equal span as 3
// (1 is equal by memcmp, 0 differs), and learns an unknown memo from S when memcmp finds it equal.
struct CompareSpan {
    const void* first;
    const void* second;
    std::size_t bytes;
    SpanHash* hash = nullptr;
};

void CompareSpans(std::span<const CompareSpan> spans, std::span<std::uint8_t> equal);
void CompareSpansFrom(std::span<const CompareSpan> spans, std::span<std::uint8_t> equal, std::size_t parallelBytes);
unsigned CompareHelpers();
void StopCompareHelpers();
unsigned CompareHelpersRunning();

// The four XXH64 lane states of HashPage (GuestMemory.cpp) over any length, without the finalizer:
// 32-byte stripes, then the remaining whole 8-byte words into lanes 0, 1, 2 in turn, then the last
// 1-7 bytes zero-padded into one word for the next lane. A change confined to one 8-byte word always
// changes the lanes (each round is a bijection of the word and of the lane).
std::array<std::uint64_t, 4> HashLanes(const void* data, std::size_t bytes);
// The switch, read once: 0 (unset) no memo; 1 memo for Storage snapshots of at least 1 MiB, with one
// memo-decided block memcmp-audited per refresh; 2 verify (memcmp beside every memo verdict, its
// verdict used, false-equals and stale memos counted); 3 as 2, aborting on the first of either.
int SnapshotHashMode();
// Totals since the last take ([snapshot-stats]): blocks the memo decided (and found equal), memos
// learned from a memcmp, blocks rehashed after their copy, audited blocks (and anomalies: the audit
// found the memo wrong, S written behind it or a lane collision), memo verdicts memcmp contradicted
// (false-equal: a lane collision, under =2/=3 or found by the audit; stale, under =2/=3: the memo no
// longer described S), and contradictions guest memory moving between the two reads explains
// (raced: not a fault of the memo).
struct SnapshotHashCounts {
    std::uint64_t decided = 0;
    std::uint64_t decidedEqual = 0;
    std::uint64_t learned = 0;
    std::uint64_t rehashed = 0;
    std::uint64_t audited = 0;
    std::uint64_t anomalies = 0;
    std::uint64_t falseEqual = 0;
    std::uint64_t stale = 0;
    std::uint64_t raced = 0;
};
void NoteSnapshotHashCounts(const SnapshotHashCounts& counts);
SnapshotHashCounts TakeSnapshotHashCounts();
// Why a known memo's verdict and a memcmp of the same span disagreed when each read guest memory at
// its own time (a guest CPU store may land between them, GPU-OWNED-PLAN 2.3). Stale: HashLanes of S
// is not the memo, so S was written behind it (exact: nothing writes S meanwhile); the memo is
// repaired from S and `equal` is left as passed. Otherwise one copy of the guest block decides both
// verdicts, and `equal` gets the copy's memcmp verdict: hashing to the memo while differing from S
// is a lane collision; anything else means guest memory moved between the reads (raced). The rare
// path: it allocates the copy.
enum class MemoDisagreement : std::uint8_t { Raced, Stale, Collision };
MemoDisagreement ClassifyMemoDisagreement(const CompareSpan& span, bool& equal);

}

#endif
