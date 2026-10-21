#include <cstdint>
#include <cstddef>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

namespace {

constexpr int NP_PARTY_ERROR_INVALID_ARGUMENT = static_cast<int>(0x80552504u);
constexpr std::uint16_t NP_PARTY_STATE_NOT_IN_PARTY = 2;

}

extern "C" {

int APS5_VABI sceNpPartyGetState(std::uint16_t* state) {
    if (!state) return NP_PARTY_ERROR_INVALID_ARGUMENT;
    *state = NP_PARTY_STATE_NOT_IN_PARTY;
    return 0;
}

}
