#pragma once

#include <random>
#include "datatypes.h"

template<class T, unsigned len>
void fill_buffer_random(std::mt19937 & gen, T * data, T min, T max) {
	if constexpr (is_real_v<T>) {
		std::uniform_real_distribution<> dist(min, max);
		for (unsigned i = 0; i < len; i++) {
			data[i] = dist(gen);
		}
	} else if constexpr (is_complex_v<T>) {
		std::uniform_real_distribution<> dist_real(min.real, max.real);
		std::uniform_real_distribution<> dist_imag(min.imag, max.imag);
		for (unsigned i = 0; i < len; i++) {
			data[i].real = dist_real(gen);
			data[i].imag = dist_imag(gen);
		}
	}
}
