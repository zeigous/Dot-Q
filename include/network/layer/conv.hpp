#pragma once

#include "tensor.hpp"
#include "network/layer.hpp"
#include "network/activation.hpp"
#include <random>
#include <stdexcept>

template <typename F, typename IF, typename B, typename IB, auto QMF, auto QMB, typename A = ReLU, int S = 3>
class Conv : public Layer<F, B> {
    static_assert(S % 2 == 1, "S must be odd\n");
    static_assert(S > 1, "S must be > 1\n");

    static constexpr int R = S/2;

    /* Weights */
    std::vector<Tensor<F>>                  weights;    /* Kernel Weights           */
    std::vector<Tensor<B>>                  sWeights;   /* Shadow Weights           */
    std::vector<std::vector<Tensor<B>>>     cWeights;   /* Cached Weight Updates    */

    /* Biases */
    Tensor<F>               biases;     /* Biases                   */
    Tensor<B>               sBiases;    /* Shadow Biases            */
    std::vector<Tensor<B>>  cBiases;    /* Cached Bias Updates      */

    /* Cached input/outputs */
    Tensor<F> cachedInputs;
    Tensor<F> cachedPreActivation;

    /* Dimenstions */
    Layout inDim;
    Layout outDim;

    public:
        /* Constructor */
        Conv(Layout in, Layout out) :
            weights(out.chan, Tensor<F>({S, S, in.chan})),
            sWeights(out.chan, Tensor<B>({S, S, in.chan})),

            biases({out.chan, 1, 1},    static_cast<F>(0.0f)),
            sBiases({out.chan, 1, 1},   static_cast<B>(0.0f)),

            cachedInputs({in.cols, in.rows, in.chan}),
            cachedPreActivation({out.cols, out.rows, out.chan}),

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

            /* Kernel loop */
            for (int ker = 0; ker < nKer; ker++) {
                for (int y = R; y < inH - R; y++) {
                    for (int x = R; x < inW - R; x++) {

                        /* Conv Sum */
                        IF rSum = static_cast<IF>(0.0);
                        for (int c = 0; c < inC; c++) {
                            for (int kr = -R; kr <= R; kr++) {
                                for (int kc = -R; kc <= R; kc++) {
                                    rSum += QMF( input[x + kc, y + kr, c], weights[ker][kc + R, kr + R, c] );
                                }
                            }
                        }

                        rSum += biases[ker, 0, 0];
                        F pre = static_cast<F>(rSum);
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
        
            /* Accumulators */
            std::vector<Tensor<IB>> kGrad(nKer, Tensor<IB>(weights[0].layout, static_cast<IB>(0.0)));
            Tensor<IB> bGrad({outDim.chan, 1, 1}, static_cast<IB>(0.0));
            Tensor<IB> errOut(inDim, static_cast<IB>(0.0));
        
            for (int ker = 0; ker < nKer; ker++) {
                for (int y = R; y < inH - R; y++) {
                    for (int x = R; x < inW - R; x++) {
        
                        B delta = errIn[x - R, y - R, ker] * A::template backward<B>(static_cast<B>(cachedPreActivation[x - R, y - R, ker]));
        
                        bGrad[ker, 0, 0] += delta;
        
                        for (int c = 0; c < inC; c++) {
                            for (int kr = -R; kr <= R; kr++) {
                                for (int kc = -R; kc <= R; kc++) {
                                    /* Weight grad */
                                    kGrad[ker][kc + R, kr + R, c] += QMB(static_cast<B>(cachedInputs[x + kc, y + kr, c]), delta);
        
                                    /* Err for previous layer */
                                    errOut[x + kc, y + kr, c] += QMB(static_cast<B>(weights[ker][kc + R, kr + R, c]), delta);
                                }
                            }
                        }
                    }
                }

               
            }

            /* apply learningRate and round and cache weights */
            std::vector<Tensor<B>> cachedWeightUpdate;
            cachedWeightUpdate.reserve(nKer);

            for (int ker = 0; ker < nKer; ker++) {

                Tensor<B> update(weights[ker].layout, static_cast<B>(0.0));
                for (size_t i = 0; i < update.data.size(); i++) {
                    update.data[i] = static_cast<B>(kGrad[ker].data[i]) * learningRate;
                }

                cachedWeightUpdate.emplace_back(update);
            }

            /* Cache Bias Updates */
            Tensor<B> cachedBiasUpdate(biases.layout, static_cast<B>(0.0));
            for (int ker = 0; ker < nKer; ker++)
                cachedBiasUpdate[ker, 0, 0] = static_cast<B>(bGrad[ker, 0, 0]) * learningRate;


            cWeights.emplace_back(std::move(cachedWeightUpdate));
            cBiases.emplace_back(std::move(cachedBiasUpdate));

            return static_cast<Tensor<B>>(errOut);
        }


        /* For caching and updating weights / biases */
        void clearCache() override {
            cWeights.clear();
            cBiases.clear();
        }

        void updateValues() override {
            if (cWeights.empty()) return;
            const int nKer = static_cast<int>(outDim.chan);
                
            for (int ker = 0; ker < nKer; ker++) {
                /* Weights (Kernels) */
                Tensor<IB> wAverage({S, S, inDim.chan} , static_cast<IB>(0.0));

                for (const auto& cWeight : cWeights) {
                    wAverage = wAverage + cWeight[ker];
                }

                sWeights[ker]   = sWeights[ker] - (static_cast<Tensor<B>>(wAverage) / cWeights.size());
                weights[ker]    = static_cast<Tensor<F>>(sWeights[ker]);

                
            }

            /* Bias */
            Tensor<IB> bAverage({outDim.chan, 1, 1}, static_cast<IB>(0.0));

            for (const auto& cBias : cBiases) {
                bAverage = bAverage + cBias;
            }
            sBiases = sBiases - (static_cast<Tensor<B>>(bAverage) / static_cast<B>(cBiases.size()));
            biases = static_cast<Tensor<F>>(sBiases);
            
        }
            
        

    private:
        void kaimingInit() {
            /* Init weights via Kaiming Init */
            std::random_device rd;
            std::mt19937 gen(rd());
            
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