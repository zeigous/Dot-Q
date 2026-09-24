#pragma once


struct ReLU { 
    template <typename T>
    static T forward(const T& input) {
        if (input < 0) return static_cast<T>(0);
        return input;
    };
    template <typename T>
    static T backward(const T& input) {
        if (input < 0) return static_cast<T>(0);
        return static_cast<T>(1);
    };
};

struct Linear { 
    template <typename T>
    static T forward(const T& input) {
        return input;
    };
    template <typename T>
    static T backward([[maybe_unused]] const T& input) {
        return 1;
    };
};