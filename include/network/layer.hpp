#pragma once

#include "tensor.hpp"

template <typename F, typename B>
class Layer {
    public:
    virtual ~Layer() = default;

    /* Forward / backward passes */
    virtual Tensor<F> forward(const Tensor<F>& input)  = 0;
    virtual Tensor<B> backward(const Tensor<B>& errIn, B learningRate) = 0;
    
    /* For caching and updating weights / biases */
    virtual void clearCache()   = 0; 
    virtual void updateValues() = 0;
};