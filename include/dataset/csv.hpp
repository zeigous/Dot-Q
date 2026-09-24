#pragma once

#include "tensor.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

template <typename T>
std::vector<std::pair<Tensor<T>, int>> parseMNIST(const std::string& filepath) {
    std::ifstream file(filepath);

    // Check if the file opened successfully
    if (!file.is_open()) {
        throw std::runtime_error("Err: Could not find database file");
    }

    std::string line;

    std::vector<std::pair<Tensor<T>, int>> dataset;
    dataset.reserve(60000);

    // Read the file line-by-line
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        Tensor<double> tmpImage({28, 28, 1});
        int tmpType = 0;

        if (std::getline(ss, cell, ',')) {
            tmpType = std::stoi(cell);
        }

        for (int y = 0; y < 28; y++) { /* loop over every image pixel */
        for (int x = 0; x < 28; x++) { /* loop over every image pixel */
            if (std::getline(ss, cell, ',')) {
                tmpImage[x, y, 0] = std::stod(cell) / 255.0; /* Normalize between 0 and 1 */
            }
        }
        }
        

        dataset.emplace_back(std::make_pair(static_cast<Tensor<T>>(tmpImage), tmpType));
    }

    return dataset;
}