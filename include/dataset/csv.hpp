#pragma once

#include "tensor.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

template <typename T>
std::vector<std::pair<Tensor<T>, int>> parseImageCSV(
    const std::string& filepath, Layout layout /* Whc */, size_t reserve = 0)
{
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Err: Could not find database file: " + filepath);
    }

    std::vector<std::pair<Tensor<T>, int>> dataset;
    dataset.reserve(reserve);

    std::string line, cell;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        Tensor<T> img(layout);
        int label = 0;

        if (std::getline(ss, cell, ',')) label = std::stoi(cell);

        for (unsigned y = 0; y < layout.rows; y++) {
        for (unsigned x = 0; x < layout.cols; x++) {
        for (unsigned c = 0; c < layout.chan; c++) {
            if (std::getline(ss, cell, ','))
                img[x, y, c] = static_cast<T>(std::stod(cell) / 255.0);
        }
        }
        }
        dataset.emplace_back(std::move(img), label);
    }
    return dataset;
}
