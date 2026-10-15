#pragma once

#include "tensor.hpp"
#include "network/layer.hpp"
#include "network/activation.hpp"
#include "accum.hpp"
#include <random>
#include <stdexcept>

template <typename F, typename IF, typename B, typename IB, auto QMF, auto QMB, typename A = ReLU>
class Dense : public Layer<F, B> {
    /* Weights */
    Tensor<F>               weights;    /* Weights                  */
    Tensor<B>               sWeights;   /* Shadow Weights           */
    std::vector<Tensor<B>>  cWeights;   /* Cached Weight Updates    */

    /* Biases */
    Tensor<F>               biases;     /* Biases                   */
    Tensor<B>               sBiases;    /* Shadow Biases            */
    std::vector<Tensor<B>>  cBiases;    /* Cached Bias Updates      */

    /* Cached input/outputs */
    Tensor<F> cachedInputs;
    Tensor<F> cachedPreActivation;

    /* Dimenstions, rows/chan = 1 */
    Layout inDim;
    Layout outDim;

    public:
        /* Constructor */
        Dense(Layout in, Layout out) :
            weights({out.cols, in.cols, 1}),
            sWeights({out.cols, in.cols, 1}),

            biases({out.cols, 1, 1},    static_cast<F>(0.0f)),
            sBiases({out.cols, 1, 1},   static_cast<B>(0.0f)),

            cachedInputs({in.cols, 1, 1}),
            cachedPreActivation({out.cols, 1, 1}),

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
            Tensor<F> preActivation(outDim, static_cast<F>(0));

            for (size_t out = 0; out < outDim.cols; ++out) {
                IF sum{};
                for (size_t in = 0; in < inDim.cols; ++in)
                    sum += QMF(input[in, 0, 0], weights[out, in, 0]);
                sum += QMF(biases[out, 0, 0], oneF);

                F pre = accTo<F>(sum);
                preActivation[out, 0, 0] = pre;
                outputs[out, 0, 0] = A::template forward<F>(pre);   // was passing `sum`
            }
            cachedPreActivation = preActivation;
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

            Tensor<B> cachedWeightUpdate({outDim.cols, inDim.cols, 1});
            Tensor<B> cachedBiasUpdate({outDim.cols, 1, 1});

            std::vector<IB> errAcc(inDim.cols);

            for (size_t out = 0; out < outDim.cols; ++out) {
                B err = errIn[out, 0, 0];
                B activDeriv = A::template backward<B>(static_cast<B>(cachedPreActivation[out, 0, 0]));
                B delta = err * activDeriv;

                for (size_t in = 0; in < inDim.cols; in++) {
                    /* Update Weights */
                    cachedWeightUpdate[out, in, 0] = learningRate * static_cast<B>(cachedInputs[in, 0, 0]) * delta;

                    /* Calculate Err for Next Layer */
                    errAcc[in] += QMB(static_cast<B>(weights[out, in, 0]), delta);
                }

                /* Update Biases */
                cachedBiasUpdate[out, 0, 0] = learningRate * delta;
            }

            Tensor<B> errOut({inDim.cols, 1, 1});
            for (size_t i = 0; i < inDim.cols; i++) errOut[i, 0, 0] = accTo<B>(errAcc[i]);
            cWeights.emplace_back(cachedWeightUpdate);
            cBiases.emplace_back(cachedBiasUpdate);
            return errOut;

            return static_cast<Tensor<B>>(errOut);
        }

        /* For caching and updating weights / biases */
        void clearCache() override {
            cWeights.clear();
            cBiases.clear();
        }

        void updateValues() override {
            if (cWeights.empty()) return;
            const B n = static_cast<B>(cWeights.size());
            Tensor<B> wAvg({outDim.cols, inDim.cols, 1}, static_cast<B>(0));
            for (const auto& cw : cWeights) wAvg = wAvg + cw;
            sWeights = sWeights - (wAvg / n);
            weights  = static_cast<Tensor<F>>(sWeights);

            Tensor<B> bAvg({outDim.cols, 1, 1}, static_cast<B>(0));
            for (const auto& cb : cBiases) bAvg = bAvg + cb;
            sBiases = sBiases - (bAvg / n);
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