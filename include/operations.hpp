#pragma once

#include <cstddef>

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
TensorBase<dtype, N>& TensorBase<dtype, N>::operator+=(const M& other) {
    static_assert(this->order() == other.order(), "+= Order mismatch");
    assert(this->size() == other.size() && "+= Size mismatch");
    std::ranges::transform(*this, other, *this, std::plus<dtype>{});
    return *this;
}

}  // namespace tensor
