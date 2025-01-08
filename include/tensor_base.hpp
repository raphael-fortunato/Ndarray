#pragma once
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <tensor_impl.hpp>

namespace tensor {

template <typename dtype, std::size_t N>
class TensorRef;

template <typename dtype, std::size_t N>
class Tensor;

template <typename dtype, std::size_t N>
class TensorBase {
   public:
    TensorBase(const TensorBase& other) {
        printf("Base Copy constructor\n");
        // copy shape
        std::copy(other.m_shape.begin(), other.m_shape.end(),
                  std::back_inserter(m_shape));
        std::copy(other.m_strides.begin(), other.m_strides.end(),
                  std::back_inserter(m_strides));
        m_size = other.m_size;
        assert(m_size == other.m_size);
        // Copy data
        m_data = new dtype[other.m_size];
        m_end_itr = m_data + other.m_size;
        std::copy(other.m_data, other.m_end_itr, m_data);
        assert(m_data != nullptr);
    }
    TensorBase& operator=(const TensorBase& other) {
        printf("Base Copy assignment\n");
        if (this != &other) {
            // copy shape
            m_shape = other.m_shape;
            m_strides = other.m_strides;
            m_size = other.m_size;
            assert(m_size == other.m_size);
            // Copy data
            m_data = new dtype[other.m_size];
            m_end_itr = m_data + other.m_size;
            std::copy(other.m_data, other.m_end_itr, m_data);
        }
        return *this;
    }
    TensorBase(TensorBase&& other) noexcept {
        printf("Base Move constructor\n");
        m_data = other.m_data;
        m_end_itr = other.m_end_itr;
        m_shape = std::move(other.m_shape);
        m_size = other.m_size;

        other.m_size = 0;
        other.m_data = nullptr;
        other.m_end_itr = nullptr;
    }
    TensorBase& operator=(TensorBase&& other) noexcept {
        printf("Base Move assignment\n");
        if (this != &other) {
            // copy shape
            m_data = other.m_data;
            m_end_itr = other.m_end_itr;
            m_shape = std::move(other.m_shape);
            m_size = other.m_size;

            other.m_size = 0;
            other.m_data = nullptr;
            other.m_end_itr = nullptr;
        }
        return *this;
    }

    template <typename... Args>
        requires tensor_impl::ValidateElementReturn<N, Args...>
    dtype operator()(Args&&... args) {
        static_assert(sizeof...(args) == N, "Invalid number of arguments");
        const std::array<std::size_t, N> indices{
            static_cast<std::size_t>(args)...};
        std::size_t linear_index = 0;
        for (std::size_t i = 0; i < N; ++i) {
            if (indices[i] >= this->m_shape[i]) {
                throw std::out_of_range("Index out of range");
            }
            linear_index += indices[i] * this->m_strides[i];
        }

        return this->m_data[linear_index];
    }

    template <typename... Args>
        requires tensor_impl::ValidateTensorRefReturn<N, Args...>
    TensorRef<dtype, N - sizeof...(Args)> operator()(Args&&... args) {
        static_assert(sizeof...(args) < N, "Invalid number of arguments");
        std::array<std::size_t, sizeof...(args)> indices{
            static_cast<std::size_t>(args)...};

        std::vector<std::size_t> new_shape(
            this->m_shape.begin() + indices.size(), this->m_shape.end());
        std::vector<std::size_t> new_strides(
            this->m_strides.begin() + indices.size(), this->m_strides.end());

        std::size_t offset = 0;
        for (std::size_t i = 0; i < indices.size(); ++i) {
            assert(indices[i] < this->m_shape[i] &&
                   "Index out of bounds: index exceeds shape dimension");
            offset += indices[i] * this->m_strides[i];
        }
        return TensorRef<dtype, N - sizeof...(Args)>(this->m_data + offset,
                                                     std::move(new_shape),
                                                     std::move(new_strides));
    }

    using value_type = dtype;
    virtual ~TensorBase() {}
    dtype* cbegin() const { return m_data; }
    dtype* cend() const { return m_end_itr; }
    const dtype* begin() const { return m_data; }
    const dtype* end() const { return m_end_itr; }
    size_t size() const { return m_size; }
    const std::vector<std::size_t>& shape() const { return m_shape; }
    const std::vector<std::size_t>& strides() const { return m_strides; }
    const std::vector<dtype> data() const {
        return std::vector<dtype>(m_data, m_data + m_size);
    }

   protected:
    TensorBase() = default;
    dtype* m_data;
    dtype* m_end_itr;
    std::size_t m_size = 0;
    std::vector<std::size_t> m_shape;
    std::vector<std::size_t> m_strides;
};

}  // namespace tensor
