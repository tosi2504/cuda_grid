#pragma once

#include <cuda/std/complex>
#include <type_traits>
#include <iostream>
#include <string>

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

template<class T> struct type_as_string;
template<> struct type_as_string<realF> {constexpr static char value[] = "realF";};
template<> struct type_as_string<realD> {constexpr static char value[] = "realD";};
template<> struct type_as_string<complexF> {constexpr static char value[] = "complexF";};
template<> struct type_as_string<complexD> {constexpr static char value[] = "complexD";};
