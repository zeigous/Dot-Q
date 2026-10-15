#pragma once
#include "tensor.hpp"
#include <type_traits>

/* Accumulator -> storage */
template <typename T, typename Acc>
T accTo(const Acc& a) {
    if constexpr (std::is_same_v<Acc, T> || std::is_arithmetic_v<Acc>) {
        return static_cast<T>(a);
    } else if constexpr (requires { a.template convert_to<T>(); }) {
        return a.template convert_to<T>();      // universal quire API (53-bit intermediate)
    } else if constexpr (requires { static_cast<T>(a); }) {
        return static_cast<T>(a);
    } else {
        static_assert(sizeof(Acc) == 0, "accTo: no known quire -> value conversion; check universal's quire API");
    }
}

template <typename T, typename Acc>
Tensor<T> accTensorTo(const Tensor<Acc>& in) {
    Tensor<T> out(in.layout);
    for (size_t i = 0; i < in.data.size(); i++) out.data[i] = accTo<T>(in.data[i]);
    return out;
}