#pragma once


#include <type_traits>
#include <random>
#include <cuda/std/complex>

// create a type trait for std::complex<>

template<typename T>
struct is_cuda_complex: public std::false_type {};

template<typename T>
struct is_cuda_complex<cuda::std::complex<T>>: public std::true_type {};


template<typename T>
T get_random_value(std::mt19937 & gen, T min, T max) {
    if constexpr (std::is_floating_point<T>::value) {
        return std::uniform_real_distribution<T>(min, max)(gen);
    } else if constexpr (std::is_integral<T>::value) {
        return std::uniform_int_distribution<T>(min, max)(gen);
    } else if constexpr (is_cuda_complex<T>::value) {
        std::uniform_real_distribution<double> dist_real(min.real(), max.real());
        std::uniform_real_distribution<double> dist_imag(min.imag(), max.imag());
        return T(dist_real(gen), dist_imag(gen));
    } else {
        static_assert(!sizeof(T), "Arithmetic type is not supported by random number generation (random.h)");
        return;
    }
}


