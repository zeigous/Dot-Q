#include "dataset/csv.hpp"
#include "network/layer/dense.hpp"
#include "network/layer/conv.hpp"
#include "network/layer/flatten.hpp"
#include "network/layer/maxpool.hpp"
#include "network/layer.hpp"
#include "network/network.hpp"

#include <iostream>
#include <vector>

#include <universal/number_systems.hpp>

#ifndef REAL_FWD
#define REAL_FWD float
#endif

#ifndef REAL_BWD
#define REAL_BWD double
#endif

#define NORM_MUL std::multiplies<>{}
#define QUIRE_MUL [](auto&&... args) { return sw::universal::quire_mul(std::forward<decltype(args)>(args)...); }

#ifndef MUL
#define MUL NORM_MUL
#endif

#ifndef DATASET_TEST
#define DATASET_TEST "datasets/CIFAR-10/cifar10_test.csv"
#endif

#ifndef DATASET_TRAIN
#define DATASET_TRAIN "datasets/CIFAR-10/cifar10_train.csv"
#endif

using RealBack = REAL_BWD;
using RealForward = REAL_FWD;

constexpr RealBack learningRate = static_cast<RealBack>(0.02);
constexpr int batchSize = 32;
constexpr int epochs = 8;


int main() {  

    /* Construct the Network */
    std::vector<Layer<RealForward, RealBack>*> network{
        new Conv<RealForward, RealForward, RealBack, RealBack, MUL, MUL, ReLU, 5>({32, 32, 3}, {28, 28, 6}),
        new MaxPool<RealForward, RealBack>({28, 28, 6}, {14, 14, 6}),
        new Conv<RealForward, RealForward, RealBack, RealBack, MUL, MUL, ReLU, 3>({14, 14, 6}, {12, 12, 8}),
        new MaxPool<RealForward, RealBack>({12, 12, 8}, {6, 6, 8}),
        new Conv<RealForward, RealForward, RealBack, RealBack, MUL, MUL, ReLU, 3>({6, 6, 8}, {4, 4, 10}),
        new Flatten<RealForward, RealBack>({4, 4, 10}, {4*4*10, 1, 1}),
        new Dense<RealForward, RealForward, RealBack, RealBack, MUL, MUL, ReLU>({4*4*10 , 1, 1}, {100, 1, 1}),
        new Dense<RealForward, RealForward, RealBack, RealBack, MUL, MUL, ReLU>({100, 1, 1}, {50, 1, 1}),
        new Dense<RealForward, RealForward, RealBack, RealBack, MUL, MUL, Linear>({50, 1, 1}, {10, 1, 1}),
    };

    /* Init the neural net */
    NeuralNetwork<RealForward, RealForward, RealBack, RealBack> net(DATASET_TRAIN, DATASET_TEST, {32, 32, 3}, parseImageCSV<RealForward>, std::move(network));
    

    /* Epochs */
    std::cout << "Loss,Accuracy" << "\n";
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::pair<float, float> preRes = net.testEpoch(2000);
        std::cout << preRes.second << "," << preRes.first << "\n";

        net.trainEpoch(batchSize, learningRate);
    }

    std::pair<float, float> res = net.testEpoch(-1);
    std::cout << res.second << "," << res.first << "\n";


    return 0;
}