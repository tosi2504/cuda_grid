#pragma once

#include <random>
#include <type_traits>
#include <cuda/std/complex>

// metafunction for cuda::std::complex
template<class T> struct is_complex : public std::false_type {};
template<class T_basic> struct is_complex<cuda::std::complex<T_basic>> : public std::true_type {};
template<class T> inline constexpr bool is_complex_v = is_complex<T>::value;


template<class T, unsigned len>
void fill_buffer_random(std::mt19937 & gen, T * data, T min, T max) {
	static_assert(std::is_integral_v<T> or std::is_floating_point_v<T> or is_complex_v<T>, "T not allowed");
	if constexpr (std::is_integral_v<T>) {
		std::uniform_int_distribution<> dist(min, max);
		for (unsigned i = 0; i < len; i++) {
			data[i] = dist(gen);
		}
	} else if constexpr (std::is_floating_point_v<T>) {
		std::uniform_real_distribution<> dist(min, max);
		for (unsigned i = 0; i < len; i++) {
			data[i] = dist(gen);
		}
	} else if constexpr (is_complex_v<T>) {
		std::uniform_real_distribution<> dist_real(min.real(), max.real());
		std::uniform_real_distribution<> dist_imag(min.imag(), max.imag());
		for (unsigned i = 0; i < len; i++) {
			data[i].real(dist_real(gen));
			data[i].imag(dist_imag(gen));
		}
	}
}
