#pragma once

#include "tensor.hpp"
#include <cmath>
#include <float.h>
#include <limits>

template<typename T>
constexpr T epsilon() {
    return std::numeric_limits<T>::min(); // smallest positive normal value
}

template <typename T>
T eExp(T val) {
    return static_cast<T>(std::exp(static_cast<double>(val)));
}

template <typename T>
T eLog(T val) {
    return static_cast<T>(std::log(static_cast<double>(val) + std::numeric_limits<double>::min()));
}

template <typename T, typename S>
std::pair<Tensor<T>, T> softmaxCrossEntropyOH(Tensor<T> out, Tensor<T> expected) { /* Returns derivatives for passing back and overall loss */
    if (out.layout != expected.layout) 
        throw std::runtime_error("Err: Softmax Input and Expected Tensors Don't Match");

    if (out.layout.rows != 1 || out.layout.chan != 1) 
        throw std::runtime_error("Err: Softmax needs 1d arr");

    Tensor<T> normPerNode(out.layout);

    S rSum = static_cast<S>(0.0f); /* Running sum */
    T max = out[0, 0, 0];
    for (size_t i = 0; i < normPerNode.layout.cols; i++) {
        if (out[i, 0, 0] > max)
            max = out[i, 0, 0];
    }

    /* Find the one hot in the tensor */
    int OHIdx = 0;
    for (size_t i = 0; i < expected.layout.cols; i++) {
        if (expected[i, 0, 0] >= static_cast<T>(0.9) /* In case conversion does wierd shit */) {
            OHIdx = i;
            break;
        }
    }

    /* Calculate e^x per node and the running sum for normalization */
    for (size_t i = 0; i < normPerNode.layout.cols; i++) {
        T tmp = eExp(out[i, 0, 0] - max);
        rSum += tmp;
        normPerNode[i, 0, 0] = tmp;
    }

    normPerNode = normPerNode / static_cast<T>(rSum);

    /* Calculate loss (cross entropy loss) */
    Tensor<T> lossDeriv(normPerNode - expected);

    T loss = static_cast<T>(-1) * eLog<T>(normPerNode[OHIdx, 0, 0]);

    return std::make_pair(lossDeriv, loss);
}
