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

template <typename F, typename IF, typename B, typename IB, auto QMF, auto QMB, typename A = ReLU, int S = 3>
class Conv : public Layer<F, B> {
    static_assert(S % 2 == 1, "S must be odd\n");
    static_assert(S > 1, "S must be > 1\n");

    static constexpr int R = S/2;

    /* Weights */
    std::vector<Tensor<F>>  weights;    /* Kernel Weights                       */
    std::vector<Tensor<B>>  sWeights;   /* Shadow Weights                       */
    std::vector<Tensor<B>>  wSum;       /* Running sum of weight updates        */

    /* Biases */
    Tensor<F>   biases;                 /* Biases                               */
    Tensor<B>   sBiases;                /* Shadow Biases                        */
    Tensor<B>   bSum;                   /* Running sum of bias updates          */

    size_t nCached = 0;                 /* Images accumulated since last update */

    /* Cached input/outputs */
    Tensor<F> cachedInputs;
    Tensor<F> cachedPreActivation;

    /* Preallocated backward accumulators (reset per image, never reallocated) */
    std::vector<Tensor<IB>> kGrad;
    Tensor<IB>              bGrad;
    Tensor<IB>              errAcc;

    /* Dimenstions */
    Layout inDim;
    Layout outDim;

    static void zeroAcc(Tensor<IB>& t) {
        std::fill(t.data.begin(), t.data.end(), IB{});
    }
    static void zeroB(Tensor<B>& t) {
        std::fill(t.data.begin(), t.data.end(), static_cast<B>(0));
    }

    public:
        /* Constructor */
        Conv(Layout in, Layout out) :
            weights(out.chan, Tensor<F>({S, S, in.chan})),
            sWeights(out.chan, Tensor<B>({S, S, in.chan})),
            wSum(out.chan, Tensor<B>({S, S, in.chan}, static_cast<B>(0))),

            biases({out.chan, 1, 1},    static_cast<F>(0.0f)),
            sBiases({out.chan, 1, 1},   static_cast<B>(0.0f)),
            bSum({out.chan, 1, 1},      static_cast<B>(0.0f)),

            cachedInputs({in.cols, in.rows, in.chan}),
            cachedPreActivation({out.cols, out.rows, out.chan}),

            kGrad(out.chan, Tensor<IB>({S, S, in.chan}, IB{})),
            bGrad({out.chan, 1, 1}, IB{}),
            errAcc(in, IB{}),

            inDim(in),
            outDim(out)
        {
            if (out.cols != in.cols - S + 1 || out.rows != in.rows - S + 1)
                throw std::runtime_error("Conv output layer must be S-1 less than input");

            kaimingInit();
        }

        /* Forward Pass */
        Tensor<F> forward(const Tensor<F>& input) noexcept override {
            cachedInputs = input;

            Tensor<F> output({outDim.cols, outDim.rows, outDim.chan}, static_cast<F>(0.0));

            /* Purely so C++ shuts up */
            const int nKer = static_cast<int>(outDim.chan);
            const int inH  = static_cast<int>(inDim.rows);
            const int inW  = static_cast<int>(inDim.cols);
            const int inC  = static_cast<int>(inDim.chan);

            const F oneF = static_cast<F>(1);

            /* Kernel loop */
            for (int ker = 0; ker < nKer; ker++) {
                for (int y = R; y < inH - R; y++) {
                    for (int x = R; x < inW - R; x++) {

                        /* Conv Sum */
                        IF rSum{};
                        for (int c = 0; c < inC; c++) {
                        for (int kr = -R; kr <= R; kr++) {
                        for (int kc = -R; kc <= R; kc++) {
                            rSum += QMF(input[x + kc, y + kr, c], weights[ker][kc + R, kr + R, c]);
                        }
                        }
                        }
                        rSum += QMF(biases[ker, 0, 0], oneF);
                        F pre = accTo<F>(rSum);
                        cachedPreActivation[x - R, y - R, ker] = pre;
                        output[x - R, y - R, ker] = A::template forward<F>(pre);
                    }
                }
            }

            return output;
        }

        Tensor<B> backward(const Tensor<B>& errIn, B learningRate) noexcept override {
            /*
                δ[ker,y,x]        = errIn[ker,y,x] * σ'(z[ker,y,x])
                ∂w[ker,c,kr,kc]   = Σ_{y,x} a_{n−1}[c, y+kr, x+kc] · δ[ker,y,x]
                ∂b[ker]           = Σ_{y,x} δ[ker,y,x]
                errOut[c,y+kr,x+kc] += δ[ker,y,x] * w[ker,c,kr,kc]
            */

            /* Purely so C++ shuts up */
            const int nKer = static_cast<int>(outDim.chan);
            const int inH  = static_cast<int>(inDim.rows);
            const int inW  = static_cast<int>(inDim.cols);
            const int inC  = static_cast<int>(inDim.chan);

            /* Reset the preallocated accumulators */
            const B oneB = static_cast<B>(1);
            for (auto& k : kGrad) zeroAcc(k);
            zeroAcc(bGrad);
            zeroAcc(errAcc);

            for (int ker = 0; ker < nKer; ker++) {
                Tensor<IB>& kg = kGrad[ker];
                for (int y = R; y < inH - R; y++) {
                    for (int x = R; x < inW - R; x++) {

                        B delta = errIn[x - R, y - R, ker] * A::template backward<B>(static_cast<B>(cachedPreActivation[x - R, y - R, ker]));

                        bGrad[ker, 0, 0] += QMB(delta, oneB);

                        for (int c = 0; c < inC; c++) {
                            for (int kr = -R; kr <= R; kr++) {
                                for (int kc = -R; kc <= R; kc++) {
                                    /* Weight grad */
                                    kg[kc + R, kr + R, c] += QMB(static_cast<B>(cachedInputs[x + kc, y + kr, c]), delta);

                                    /* Err for previous layer */
                                    errAcc[x + kc, y + kr, c] += QMB(static_cast<B>(weights[ker][kc + R, kr + R, c]), delta);
                                }
                            }
                        }
                    }
                }
            }

            for (int ker = 0; ker < nKer; ker++) {
                for (size_t i = 0; i < wSum[ker].data.size(); i++) {
                    B upd = static_cast<B>(accTo<B>(kGrad[ker].data[i]) * learningRate);
                    wSum[ker].data[i] = static_cast<B>(wSum[ker].data[i] + upd);
                }
                B bUpd = static_cast<B>(accTo<B>(bGrad[ker, 0, 0]) * learningRate);
                bSum[ker, 0, 0] = static_cast<B>(bSum[ker, 0, 0] + bUpd);
            }
            nCached++;

            return accTensorTo<B>(errAcc);
        }

        /* For caching and updating weights / biases */
        void clearCache() override {
            for (auto& w : wSum) zeroB(w);
            zeroB(bSum);
            nCached = 0;
        }

        void updateValues() override {
            if (nCached == 0) return;
            const int nKer = static_cast<int>(outDim.chan);
            const B n = static_cast<B>(nCached);

            for (int ker = 0; ker < nKer; ker++) {
                sWeights[ker] = sWeights[ker] - (wSum[ker] / n);
                weights[ker]  = static_cast<Tensor<F>>(sWeights[ker]);
            }
            sBiases = sBiases - (bSum / n);
            biases  = static_cast<Tensor<F>>(sBiases);
        }

    private:
        void kaimingInit() {
            /* Init weights via Kaiming Init */
            std::mt19937 gen(nextSeed());

            double stddev = std::sqrt(2.0 / (S * S * inDim.chan));
            std::normal_distribution<double> dist(0.0, stddev);
            for (auto& kernel : sWeights) {
                for (size_t ic = 0; ic < inDim.chan; ic++) {
                    for (size_t kr = 0; kr < S; kr++) {
                        for (size_t kc = 0; kc < S; kc ++) {
                            kernel[kc, kr, ic] = static_cast<B>(dist(gen));
                        }
                    }
                }
            }

            for (size_t i = 0; i < weights.size(); i++) {
                weights[i] = static_cast<Tensor<F>>(sWeights[i]);
            }
        }
};