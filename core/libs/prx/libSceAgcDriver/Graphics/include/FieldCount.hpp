#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_FIELDCOUNT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_FIELDCOUNT_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

// LOCAL EXPERIMENT (GPU-OWNED-PLAN 9.4.1 D4-0, 9.4.2 D4-A): the merge tripwire of a key that
// compares every field of an aggregate by hand (the target lease key, sameTarget; the target memo
// keys, sameResource). AggregateFieldCount<T, N> holds when T is an aggregate of exactly N members:
// N initializers that convert to any type initialize it, and one more does not. A member (or a
// base) added by a merge makes it false wherever it lands, in a padding hole too, which a sizeof
// check misses. Each initializer converts to the member's own type, so a member that is itself an
// aggregate (VkExtent2D, std::array) takes one initializer: no brace elision. The extra one is
// tried as `{}` too, for a member type nothing converts to. Used in static_asserts only (unevaluated).
namespace AgcDriver::Graphics {
namespace FieldCountProbe {

struct AnyField {
    template <typename T>
    constexpr operator T() const;
};

template <typename T, std::size_t... I>
constexpr bool Takes(std::index_sequence<I...>) {
    return requires { T{(static_cast<void>(I), AnyField{})...}; };
}

template <typename T, std::size_t... I>
constexpr bool TakesOneMore(std::index_sequence<I...>) {
    return requires { T{(static_cast<void>(I), AnyField{})..., AnyField{}}; } || requires { T{(static_cast<void>(I), AnyField{})..., {}}; };
}

}

template <typename T, std::size_t N>
inline constexpr bool AggregateFieldCount = std::is_aggregate_v<T> && FieldCountProbe::Takes<T>(std::make_index_sequence<N>{}) && !FieldCountProbe::TakesOneMore<T>(std::make_index_sequence<N>{});

}

#endif
