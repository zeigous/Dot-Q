#pragma once
#include "network/layer.hpp"

template <typename F, typename B>
class Flatten : public Layer<F, B> {

    /* Dimenstions */
    Layout inDim;
    Layout outDim;

    public:
        Flatten(Layout in, Layout out) 
            : inDim(in), outDim(out) {
            if (outDim.chan != 1 || outDim.rows != 1) {
                throw std::runtime_error("Cannot Flatten to non-flat output");
            }
            if (outDim.cols != inDim.cols * inDim.rows * inDim.chan) {
                throw std::runtime_error("Flat size mismatch");
            }
        }

        /* Forward / backward passes */
        Tensor<F> forward(const Tensor<F>& input) noexcept override {
            Tensor<F> tmp(input);
            tmp.layout = outDim;
            return tmp;
        }

        Tensor<B> backward(const Tensor<B>& errIn, [[maybe_unused]] B learningRate) noexcept override {
            Tensor<B> tmp(errIn);
            tmp.layout = inDim;
            return tmp;
        }
        
        /* For caching and updating weights / biases */
        void clearCache() override {return;}; 
        void updateValues() override {return;};
};