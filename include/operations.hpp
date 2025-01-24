#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>

#include "tensor/tensor_base.hpp"

namespace tensor {

// Scalar in place operations
template <typename dtype, std::size_t N>
TensorBase<dtype, N>& TensorBase<dtype, N>::operator+=(const dtype& value) {
    std::ranges::for_each(*this, [&value](auto& x) { x += value; });
    return *this;
}
template <typename dtype, std::size_t N>
TensorBase<dtype, N>& TensorBase<dtype, N>::operator-=(const dtype& value) {
    std::ranges::for_each(*this, [&value](auto& x) { x -= value; });
    return *this;
}
template <typename dtype, std::size_t N>
TensorBase<dtype, N>& TensorBase<dtype, N>::operator*=(const dtype& value) {
    std::ranges::for_each(*this, [&value](auto& x) { x *= value; });
    return *this;
}
template <typename dtype, std::size_t N>
TensorBase<dtype, N>& TensorBase<dtype, N>::operator/=(const dtype& value) {
    std::ranges::for_each(*this, [&value](auto& x) { x /= value; });
    return *this;
}
template <typename dtype, std::size_t N>
TensorBase<dtype, N>& TensorBase<dtype, N>::operator%=(const dtype& value) {
    std::ranges::for_each(*this, [&value](auto& x) { x %= value; });
    return *this;
}

// Scalar operations
template <typename dtype, std::size_t N>
TensorBase<dtype, N> operator+(const TensorBase<dtype, N>& tensor,
                               const dtype& value) {
    TensorBase result = tensor;
    result += value;
    return result;
}
template <typename dtype, std::size_t N>
TensorBase<dtype, N> operator-(const TensorBase<dtype, N>& tensor,
                               const dtype& value) {
    TensorBase result = tensor;
    result -= value;
    return result;
}
template <typename dtype, std::size_t N>
TensorBase<dtype, N> operator*(const TensorBase<dtype, N>& tensor,
                               const dtype& value) {
    TensorBase result = tensor;
    result *= value;
    return result;
}
template <typename dtype, std::size_t N>
TensorBase<dtype, N> operator/(const TensorBase<dtype, N>& tensor,
                               const dtype& value) {
    TensorBase result = tensor;
    result /= value;
    return result;
}
template <typename dtype, std::size_t N>
TensorBase<dtype, N> operator%(const TensorBase<dtype, N>& tensor,
                               const dtype& value) {
    TensorBase result = tensor;
    result %= value;
    return result;
}

template <typename dtype, std::size_t N>
template <typename M>
    requires tensor_impl::TensorType<dtype, N, M>
inline TensorBase<dtype, N>& TensorBase<dtype, N>::operator+=(const M& other) {
    static_assert(TensorBase::order() == M::order(), "+= Order mismatch");
    if (this->size() != other.size()) {
        throw std::out_of_range("+= Size mismatch");
    }
    std::transform(this->begin(), this->end(), other.begin(), this->begin(),
                   std::plus<dtype>{});
    return *this;
}
template <typename dtype, std::size_t N>
template <typename M>
    requires tensor_impl::TensorType<dtype, N, M>
inline TensorBase<dtype, N>& TensorBase<dtype, N>::operator-=(const M& other) {
    static_assert(TensorBase::order() == M::order(), "-= Order mismatch");
    if (this->size() != other.size()) {
        throw std::out_of_range("-= Size mismatch");
    }
    std::transform(this->begin(), this->end(), other.begin(), this->begin(),
                   std::minus<dtype>{});
    return *this;
}
template <typename dtype, std::size_t N>
template <typename M>
    requires tensor_impl::TensorType<dtype, N, M>
inline TensorBase<dtype, N>& TensorBase<dtype, N>::operator*=(const M& other) {
    static_assert(TensorBase::order() == M::order(), "*= Order mismatch");
    if (this->size() != other.size()) {
        throw std::out_of_range("*= Size mismatch");
    }
    std::transform(this->begin(), this->end(), other.begin(), this->begin(),
                   std::multiplies<dtype>{});
    return *this;
}

template <typename dtype, std::size_t N>
template <typename M>
    requires tensor_impl::TensorType<dtype, N, M>
inline TensorBase<dtype, N>& TensorBase<dtype, N>::operator/=(const M& other) {
    static_assert(TensorBase::order() == M::order(), "/= Order mismatch");
    if (this->size() != other.size()) {
        throw std::out_of_range("/= Size mismatch");
    }
    std::transform(this->begin(), this->end(), other.begin(), this->begin(),
                   std::divides<dtype>{});
    return *this;
}
template <typename dtype, std::size_t N>
template <typename M>
    requires tensor_impl::TensorType<dtype, N, M>
inline TensorBase<dtype, N>& TensorBase<dtype, N>::operator%=(const M& other) {
    static_assert(TensorBase::order() == M::order(), "%= Order mismatch");
    if (this->size() != other.size()) {
        throw std::out_of_range("%= Size mismatch");
    }
    std::transform(this->begin(), this->end(), other.begin(), this->begin(),
                   std::modulus<dtype>{});
    return *this;
}

// Matrix operations
template <typename dtype, std::size_t N>
TensorBase<dtype, N> operator+(const TensorBase<dtype, N>& tensor,
                               const TensorBase<dtype, N>& other) {
    TensorBase result = tensor;
    result += other;
    return result;
}
template <typename dtype, std::size_t N>
TensorBase<dtype, N> operator-(const TensorBase<dtype, N>& tensor,
                               const TensorBase<dtype, N>& other) {
    TensorBase result = tensor;
    result -= other;
    return result;
}
template <typename dtype, std::size_t N>
TensorBase<dtype, N> operator*(const TensorBase<dtype, N>& tensor,
                               const TensorBase<dtype, N>& other) {
    TensorBase result = tensor;
    result *= other;
    return result;
}
template <typename dtype, std::size_t N>
TensorBase<dtype, N> operator/(const TensorBase<dtype, N>& tensor,
                               const TensorBase<dtype, N>& other) {
    TensorBase result = tensor;
    result /= other;
    return result;
}
template <typename dtype, std::size_t N>
TensorBase<dtype, N> operator%(const TensorBase<dtype, N>& tensor,
                               const TensorBase<dtype, N>& other) {
    TensorBase result = tensor;
    result %= other;
    return result;
}

}  // namespace tensor
