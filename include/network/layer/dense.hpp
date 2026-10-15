#pragma once

#include "tensor.hpp"
#include "network/layer.hpp"
#include "network/activation.hpp"
#include "accum.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

template <typename F, typename IF, typename B, typename IB, auto QMF, auto QMB, typename A = ReLU>
class Dense : public Layer<F, B> {
    /* Weights */
    Tensor<F>   weights;    /* Weights                              */
    Tensor<B>   sWeights;   /* Shadow Weights                       */
    Tensor<B>   wSum;       /* Running sum of weight updates        */

    /* Biases */
    Tensor<F>   biases;     /* Biases                               */
    Tensor<B>   sBiases;    /* Shadow Biases                        */
    Tensor<B>   bSum;       /* Running sum of bias updates          */

    size_t nCached = 0;     /* Images accumulated since last update */

    /* Cached input/outputs */
    Tensor<F> cachedInputs;
    Tensor<F> cachedPreActivation;

    /* Preallocated backward accumulator (reset per image) */
    std::vector<IB> errAcc;

    /* Dimenstions, rows/chan = 1 */
    Layout inDim;
    Layout outDim;

    static void zeroB(Tensor<B>& t) {
        std::fill(t.data.begin(), t.data.end(), static_cast<B>(0));
    }

    public:
        /* Constructor */
        Dense(Layout in, Layout out) :
            weights({out.cols, in.cols, 1}),
            sWeights({out.cols, in.cols, 1}),
            wSum({out.cols, in.cols, 1}, static_cast<B>(0.0f)),

            biases({out.cols, 1, 1},    static_cast<F>(0.0f)),
            sBiases({out.cols, 1, 1},   static_cast<B>(0.0f)),
            bSum({out.cols, 1, 1},      static_cast<B>(0.0f)),

            cachedInputs({in.cols, 1, 1}),
            cachedPreActivation({out.cols, 1, 1}),

            errAcc(in.cols),

            inDim(in),
            outDim(out)
        {
            if (in.chan != 1 || in.rows != 1)
                throw std::runtime_error("Dense layer input not 1d");
            if (out.chan != 1 || out.rows != 1)
                throw std::runtime_error("Dense layer output not 1d");

            kaimingInit();
        }

        /* Forward Pass */
        Tensor<F> forward(const Tensor<F>& input) noexcept override {
            cachedInputs = input;
            const F oneF = static_cast<F>(1);
            Tensor<F> outputs(outDim, static_cast<F>(0));

            for (size_t out = 0; out < outDim.cols; ++out) {
                IF sum{};
                for (size_t in = 0; in < inDim.cols; ++in)
                    sum += QMF(input[in, 0, 0], weights[out, in, 0]);
                sum += QMF(biases[out, 0, 0], oneF);

                F pre = accTo<F>(sum);
                cachedPreActivation[out, 0, 0] = pre;
                outputs[out, 0, 0] = A::template forward<F>(pre);
            }
            return outputs;
        }

        Tensor<B> backward(const Tensor<B>& errIn, B learningRate) noexcept override {
            /*
                Error signal (δ):
                δ = 2(a − y) (output layer)
                δ = Σ (w_{n+1} · σ'{n+1}(z{n+1}) · δ_{n+1}) (hidden layer, recursive)

                Weight update:
                w = w − η(a_{n−1} · σ'(z) · δ)

                Bias update:
                b = b − η(σ'(z) · δ)
            */

            std::fill(errAcc.begin(), errAcc.end(), IB{});

            for (size_t out = 0; out < outDim.cols; ++out) {
                B err = errIn[out, 0, 0];
                B activDeriv = A::template backward<B>(static_cast<B>(cachedPreActivation[out, 0, 0]));
                B delta = err * activDeriv;

                for (size_t in = 0; in < inDim.cols; in++) {
                    /* Update Weights (straight into the running sum) */
                    B upd = learningRate * static_cast<B>(cachedInputs[in, 0, 0]) * delta;
                    wSum[out, in, 0] = static_cast<B>(wSum[out, in, 0] + upd);

                    /* Calculate Err for Next Layer */
                    errAcc[in] += QMB(static_cast<B>(weights[out, in, 0]), delta);
                }

                /* Update Biases */
                B bUpd = learningRate * delta;
                bSum[out, 0, 0] = static_cast<B>(bSum[out, 0, 0] + bUpd);
            }
            nCached++;

            Tensor<B> errOut({inDim.cols, 1, 1});
            for (size_t i = 0; i < inDim.cols; i++) errOut[i, 0, 0] = accTo<B>(errAcc[i]);
            return errOut;
        }

        /* For caching and updating weights / biases */
        void clearCache() override {
            zeroB(wSum);
            zeroB(bSum);
            nCached = 0;
        }

        void updateValues() override {
            if (nCached == 0) return;
            const B n = static_cast<B>(nCached);

            sWeights = sWeights - (wSum / n);
            weights  = static_cast<Tensor<F>>(sWeights);

            sBiases = sBiases - (bSum / n);
            biases  = static_cast<Tensor<F>>(sBiases);
        }

    private:
        void kaimingInit() {
            /* Init weights via Kaiming Init */
            std::mt19937 gen(nextSeed());

            double stddev = std::sqrt(2.0 / inDim.cols);
            std::normal_distribution<double> dist(0.0, stddev);

            for (size_t oc = 0; oc < outDim.cols; ++oc) {
                for (size_t ic = 0; ic < inDim.cols; ++ic) {
                    sWeights[oc, ic, 0] = static_cast<B>(dist(gen));
                }
            }

            weights = static_cast<Tensor<F>>(sWeights);
        }
};