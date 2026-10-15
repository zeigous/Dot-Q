#pragma once

#ifndef RANDOM_SEED
#define RANDOM_SEED 0
#endif

#include "network/layer.hpp"
#include "network/error/softmax.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <ranges>
#include <vector>

template<typename F, typename IF, typename B, typename IB>
class NeuralNetwork {
    std::vector<std::pair<Tensor<F>, int>> train;
    std::vector<std::pair<Tensor<F>, int>> test;

    std::vector<Layer<F, B>*> network;

    std::mt19937 gen;
    public:
        NeuralNetwork(std::string trainPath, std::string testPath, Layout layout, std::function<std::vector<std::pair<Tensor<F>, int>>(const std::string& filepath, Layout layout, size_t reserve)> loaderFn, std::vector<Layer<F, B>*> net) 
            : network(std::move(net))
            {
            train = loaderFn(trainPath, layout, 0);
            test = loaderFn(testPath, layout, 0);

            /* Random for choosing images */
            gen = std::mt19937(RANDOM_SEED);

            std::shuffle(test.begin(), test.end(), gen);
        }

        ~NeuralNetwork() {
            for (size_t i = 0; i < network.size(); i++) {
                delete network[i];
            }
        }

        void trainEpoch(int batchSize, B learningRate) {
            /* Every image per epoch */
            std::shuffle(train.begin(), train.end(), gen);

            /* Per batch */
            for (size_t batch = 0; batch < train.size() / batchSize; batch++) {
                float averageLoss = 0;

                for (int im = 0; im < batchSize; im++) {
                    /* Forward pass */
                    Tensor<F> layerOut(train[batch * batchSize + im].first);

                    for (auto* layer : network) {
                        layerOut = layer->forward(layerOut); 
                    }

                    /* Softmax final calc */
                    Tensor<B> expected({10, 1, 1}, static_cast<B>(0.0f));
                    expected[train[batch * batchSize + im].second, 0, 0] = 1;

                    auto soft = softmaxCrossEntropyOH<B, B>(static_cast<Tensor<B>>(layerOut), expected);
                    averageLoss += static_cast<float>(soft.second);

                    /* Backprop */
                    Tensor<B> gradient = soft.first;
                    for (auto* layer : network | std::views::reverse) {
                        gradient = layer->backward(gradient, learningRate);
                    }
                }

                averageLoss /= batchSize;

                if (std::isnan(averageLoss)) {
                    throw std::runtime_error("NaN loss detected");
                }

                for (auto& layer : network) {
                    layer->updateValues(); 
                    layer->clearCache(); 
                }

            }
        }

        std::pair<float, float> testEpoch(int numImages) {
            /* Test images */
            float averageLoss = 0.0f;
            float averageAccuracy = 0.0f;

            const size_t n = (numImages < 1) ? test.size() : std::min<size_t>(numImages, test.size());

            for (size_t im = 0; im < n; im++) {
                /* Forward pass */
                Tensor<F> layerOut(test[im].first);

                for (auto& layer : network) {
                    layerOut = layer->forward(layerOut); 
                }

                /* Softmax final calc */
                Tensor<B> expected({10, 1, 1}, static_cast<B>(0.0f));
                expected[test[im].second, 0, 0] = 1;

                auto soft = softmaxCrossEntropyOH<B, B>(static_cast<Tensor<B>>(layerOut), expected);

                /* See if its correct */
                size_t maxIdx = 0;
                for (size_t i = 0; i < layerOut.layout.cols; i++) {
                    if (layerOut[i, 0, 0] > layerOut[maxIdx, 0, 0])
                        maxIdx = i;
                }

                if (maxIdx == static_cast<size_t>(test[im].second))
                    averageAccuracy += 1.0f;
                
                averageLoss += static_cast<float>(soft.second);
            }

            averageLoss     /= static_cast<float>(n);
            averageAccuracy /= static_cast<float>(n);
            averageAccuracy *= 100;

            return std::make_pair(averageAccuracy, averageLoss);
        }
};