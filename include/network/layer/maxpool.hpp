#pragma once
#include "network/layer.hpp"

template <typename F, typename B>
class MaxPool : public Layer<F, B> {

    /* Dimenstions */
    Layout inDim;
    Layout outDim;

    /* Cached MaxPool Indecies */
    Tensor<std::pair<int8_t, int8_t>> cachedIndicies;

    public:
        MaxPool(Layout in, Layout out) 
            : inDim(in), outDim(out),  cachedIndicies(out) {
            if (outDim.chan != inDim.chan || outDim.cols != inDim.cols / 2 || outDim.rows != inDim.rows / 2) {
                throw std::runtime_error("Dimension Mismatch");
            }
            if ( inDim.cols % 2 != 0 || inDim.rows % 2 != 0) {
                throw std::runtime_error("MaxPool cannot have Non-even inputs");
            }
        }

        /* Forward / backward passes */
        Tensor<F> forward(const Tensor<F>& input) noexcept override {
            Tensor<F> output(outDim);

            for (size_t c = 0; c < inDim.chan; c++) {
            for (size_t y = 0; y < inDim.rows; y += 2) {
            for (size_t x = 0; x < inDim.cols; x += 2) {
                int8_t bestDx = 0, bestDy = 0;
                F maxVal = input[x, y, c];
                
                for (int8_t dy = 0; dy < 2; ++dy) {
                for (int8_t dx = 0; dx < 2; ++dx) {
                    F val = input[x + dx, y + dy, c];
                    if (val > maxVal) {
                        maxVal = val;
                        bestDx = dx;
                        bestDy = dy;
                    }
                }
                }

                size_t outX = x / 2;
                size_t outY = y / 2;

                output[outX, outY, c] = maxVal;
                cachedIndicies[outX, outY, c] = {bestDx, bestDy};
            }
            }
            }

            return output;

        }

        Tensor<B> backward(const Tensor<B>& errIn, [[maybe_unused]] B learningRate) noexcept override {
            Tensor<B> errOut(inDim , static_cast<B>(0.0));

            for (size_t c = 0; c < inDim.chan; ++c) {
            for (size_t y = 0; y < outDim.rows; ++y) {
            for (size_t x = 0; x < outDim.cols; ++x) {
                auto [bestDx, bestDy] = cachedIndicies[x, y, c];

                size_t ix = (x * 2) + bestDx;
                size_t iy = (y * 2) + bestDy;

                errOut[ix, iy, c] = errIn[x, y, c];
            }
            }
            }
            
            return errOut;
        }
        
        /* For caching and updating weights / biases */
        void clearCache() override {return;}
        void updateValues() override {return;};
};