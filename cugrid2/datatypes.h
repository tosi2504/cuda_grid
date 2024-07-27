#pragma once

#include <type_traits>
#include <iostream>
#include <string>

using realF = float;
using realD = double;

template<class T>
struct __align__(2*sizeof(T)) complex {
    using T_base = T;
    T real, imag;
    __host__ __device__ inline
    complex() {
        real = 0;
        imag = 0;
    }
    __host__ __device__ inline
    complex(int real):
        real(real), imag(0) {}
    __host__ __device__ inline
    complex(T real, T imag):
        real(real), imag(imag) {}
    __host__ __device__ inline
    complex(const complex<T> & other):
        real(other.real), imag(other.imag) {}
    __host__ __device__ inline
    void operator += (const complex<T> & right) {
        real += right.real;
        imag += right.imag;
    }
    __host__ __device__ inline
    void mac(const complex<T> & left, const complex<T> & right) {
        real += left.real*right.real - left.imag*right.imag;
        imag += left.imag*right.real + left.real*right.imag;
    }
};
using complexF = complex<float>;
using complexD = complex<double>;
template<class T>
__host__ __device__ inline
complex<T> operator * (const complex<T> & left, const complex<T> & right) {
    return complex<T>(
        left.real*right.real - left.imag*right.imag,
        left.imag*right.real + left.real*right.imag
    );
}
template<class T>
__host__ __device__ inline
complex<T> operator + (const complex<T> & left, const complex<T> & right) {
    return complex<T>(
        left.real + right.real,
        left.imag + right.imag
    );
}
template<class T>
__host__ __device__ inline
complex<T> operator - (const complex<T> & left, const complex<T> & right) {
    return complex<T>(
        left.real - right.real,
        left.imag - right.imag
    );
}
std::ostream & operator<<(std::ostream & left, const complexF & right) {
	return left << "(" << right.real << "," << right.imag << ")";
}
std::ostream & operator<<(std::ostream & left, const complexD & right) {
	return left << "(" << right.real << "," << right.imag << ")";
}


template<class T> class is_real : public std::false_type {};
template<> class is_real<realF> : public std::true_type {};
template<> class is_real<realD> : public std::true_type {};
template<class T> constexpr bool is_real_v = is_real<T>::value;

template<class T> class is_complex : public std::false_type {};
template<> class is_complex<complexF> : public std::true_type {};
template<> class is_complex<complexD> : public std::true_type {};
template<class T> constexpr bool is_complex_v = is_complex<T>::value;


template<class T> struct type_as_string;
template<> struct type_as_string<realF> {constexpr static char value[] = "realF";};
template<> struct type_as_string<realD> {constexpr static char value[] = "realD";};
template<> struct type_as_string<complexF> {constexpr static char value[] = "complexF";};
template<> struct type_as_string<complexD> {constexpr static char value[] = "complexD";};
