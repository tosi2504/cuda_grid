#pragma once

#include <cuda/std/complex>
#include <type_traits>
#include <iostream>

using realF = float;
using realD = double;
using complexF = cuda::std::complex<float>;
using complexD = cuda::std::complex<double>;

template<class T> class is_real : public std::false_type {};
template<> class is_real<realF> : public std::true_type {};
template<> class is_real<realD> : public std::true_type {};
template<class T> constexpr bool is_real_v = is_real<T>::value;

template<class T> class is_complex : public std::false_type {};
template<> class is_complex<complexF> : public std::true_type {};
template<> class is_complex<complexD> : public std::true_type {};
template<class T> constexpr bool is_complex_v = is_complex<T>::value;

std::ostream & operator<<(std::ostream & left, const complexF & right) {
	return left << "(" << right.real() << "," << right.imag() << ")";
}
std::ostream & operator<<(std::ostream & left, const complexD & right) {
	return left << "(" << right.real() << "," << right.imag() << ")";
}

