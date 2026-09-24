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

#define QUIRE_MUL [](auto&&... args) { return sw::universal::quire_mul(std::forward<decltype(args)>(args)...); }

using half = _Float16;

using RealBack = double;
using RealForward = sw::universal::posit<32, 2>;

constexpr RealBack learningRate = static_cast<RealBack>(0.03f);
constexpr int batchSize = 32;
constexpr int epochs = 5;


int main() {  

    /* Construct the Network */
    std::vector<Layer<RealForward, RealBack>*> network{
        new Conv<RealForward, RealForward, RealBack, RealBack, std::multiplies<>{}, std::multiplies<>{}, ReLU, 3>({28, 28, 1}, {26, 26, 4}),
        new Conv<RealForward, RealForward, RealBack, RealBack, std::multiplies<>{}, std::multiplies<>{}, ReLU, 3>({26, 26, 4}, {24, 24, 6}),
        new MaxPool<RealForward, RealBack>({24, 24, 6}, {12, 12, 6}),
        new Conv<RealForward, RealForward, RealBack, RealBack, std::multiplies<>{}, std::multiplies<>{}, ReLU, 3>({12, 12, 6}, {10, 10, 8}),
        new MaxPool<RealForward, RealBack>({10, 10, 8}, {5, 5, 8}),
        new Flatten<RealForward, RealBack>({5, 5, 8}, {5 * 5 * 8, 1, 1}),
        new Dense<RealForward, RealForward, RealBack, RealBack, std::multiplies<>{}, std::multiplies<>{}, ReLU>({5 * 5 * 8 , 1, 1}, {100, 1, 1}),
        new Dense<RealForward, RealForward, RealBack, RealBack, std::multiplies<>{}, std::multiplies<>{}, ReLU>({100, 1, 1}, {50, 1, 1}),
        new Dense<RealForward, RealForward, RealBack, RealBack, std::multiplies<>{}, std::multiplies<>{}, Linear>({50, 1, 1}, {10, 1, 1}),
    };

    /* Init the neural net */
    NeuralNetwork<RealForward, RealForward, RealBack, RealBack> net("datasets/MNIST/mnist_train.csv", "datasets/MNIST/mnist_test.csv", parseMNIST<RealForward>, std::move(network));
    

    /* Epochs */
    std::cout << "Loss,Accuracy" << "\n";
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::pair<float, float> preRes = net.testEpoch();
        std::cout << preRes.second << "," << preRes.first << "\n";

        net.trainEpoch(batchSize, learningRate);
    }

    std::pair<float, float> res = net.testEpoch();
    std::cout << res.second << "," << res.first << "\n";


    return 0;
}