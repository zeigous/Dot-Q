#pragma once

#include <stdexcept>
#include <vector>

struct Layout {
    size_t cols;
    size_t rows;
    size_t chan;
    auto operator<=>(const Layout& other) const = default;
};

template<typename T>
struct Tensor {
    /* Data */
    std::vector<T> data;
    
    /* Dims */
    Layout layout;

    /* Normal Constructor with a Size */
    Tensor(const Layout& l) 
        : data(l.cols * l.rows * l.chan), layout(l) {};

    /* Normal Constructor with a Size + Val */
    Tensor(const Layout& l, T val) 
        : data(l.cols * l.rows * l.chan, val), layout(l) {};

    /* For static_cast<U>(this) */
    template<typename U>
    explicit Tensor(const Tensor<U>& other) 
        : data(other.data.size()), layout(other.layout) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = static_cast<T>(other.data[i]);
        }

    }

    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;

    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    /* Accessing the Tensor using [x,y,z] */
    T& operator[] (size_t x, size_t y, size_t z) {
        return data[z * (layout.cols * layout.rows) + y * layout.cols + x];
    }
    const T& operator[] (size_t x, size_t y, size_t z) const {
        return data[z * (layout.cols * layout.rows) + y * layout.cols + x];
    }

    private:
        /* Generic */
        template <typename U, typename Op>
        auto oper(const Tensor<U>& other, Op op) const 
            -> Tensor<decltype(op(std::declval<T>(), std::declval<U>()))> {

            if (other.layout != layout) {
                throw std::runtime_error("Err: Tensor layout mismatch");
            }

            using R = decltype(op(std::declval<T>(), std::declval<U>()));

            Tensor<R> result(layout); /* Auto allocs */

            for (size_t i = 0; i < data.size(); i++) {
                result.data[i] = static_cast<R>(op(data[i], other.data[i]));
            }

            return result;
        }

        template <typename U, typename Op>
        auto oper(const U& other, Op op) const 
            -> Tensor<decltype(op(std::declval<T>(), std::declval<U>()))> {

            using R = decltype(op(std::declval<T>(), std::declval<U>()));

            Tensor<R> result(layout); /* Auto allocs */

            for (size_t i = 0; i < data.size(); i++) {
                result.data[i] = static_cast<T>(op(data[i], other));
            }

            return result;
        }

    public:
        template <typename U>
        auto operator+(const Tensor<U>& other) const 
            -> Tensor<decltype(std::declval<T>() + std::declval<U>())> {
            return oper(other, std::plus<>{});
        }

        template <typename U>
        auto operator-(const Tensor<U>& other) const 
            -> Tensor<decltype(std::declval<T>() - std::declval<U>())> {
            return oper(other, std::minus<>{});
        }

        template <typename U>
        auto operator*(const Tensor<U>& other) const 
            -> Tensor<decltype(std::declval<T>() * std::declval<U>())> {
            return oper(other, std::multiplies<>{});
        }

        template <typename U>
        auto operator/(const Tensor<U>& other) const 
            -> Tensor<decltype(std::declval<T>() / std::declval<U>())> {
            return oper(other, std::divides<>{});
        }

        /* Constants */
        template <typename U>
        auto operator*(const U& other) const 
            -> Tensor<decltype(std::declval<T>() * std::declval<U>())> {
            return oper(other, std::multiplies<>{});
        }

        template <typename U>
        auto operator/(const U& other) const 
            -> Tensor<decltype(std::declval<T>() / std::declval<U>())> {
            return oper(other, std::divides<>{});
        }
};