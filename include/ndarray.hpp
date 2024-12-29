/*
 * This is the header file for the Ndarray class.
 * This class is the base class of a multi-dimensional array.
 * Every ndarray array needs a data_type T, shape, and strides.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <iterator>
#include <utility>
#include <vector>

#include "tensor_impl.hpp"

namespace ndarray {
template <typename dtype, size_t N>
class Ndarray {
   public:
    using value_type = dtype;

   private:
    dtype* m_end_itr;
    dtype* m_data;
    std::array<std::size_t, N> m_shape;
    size_t m_size = 0;

    void m_allocate_data() {
        m_data = new dtype[m_size];
        m_end_itr = m_data + m_size;
    }

    void m_compute_size() {
        if (m_shape.empty()) {
            m_size = 0;
            return;
        }
        m_size = 1;
        for (const auto& dim : m_shape) {
            m_size *= dim;
        }
    }

   public:
    Ndarray(tensor_impl::tensor_initializer<dtype, N> init) {
        printf("Constructor\n");
        m_shape = tensor_impl::derive_shape<N>(init);
        m_compute_size();
        m_allocate_data();
        tensor_impl::insert_flat(init, m_data);
    }
    ~Ndarray() { delete[] m_data; }
    Ndarray(const Ndarray& other) {
        printf("Copy constructor\n");
        // copy shape
        std::copy(other.m_shape.begin(), other.m_shape.end(), m_shape.begin());
        m_compute_size();
        assert(m_size == other.m_size);
        // Copy data
        m_data = new dtype[other.m_size];
        m_end_itr = m_data + other.m_size;
        std::copy(other.m_data, other.m_end_itr, m_data);
    }
    Ndarray operator=(const Ndarray& other) {
        printf("Copy assignment\n");
        if (this != &other) {
            // copy shape
            std::copy(other.m_shape.begin(), other.m_shape.end(),
                      m_shape.begin());
            m_compute_size();
            assert(m_size == other.m_size);
            // Copy data
            m_data = new dtype[other.m_size];
            m_end_itr = m_data + other.m_size;
            std::copy(other.m_data, other.m_end_itr, m_data);
        }
        return *this;
    }

    Ndarray(Ndarray&& other) noexcept {
        printf("Move constructor\n");
        m_data = other.m_data;
        m_end_itr = other.m_end_itr;
        m_shape = std::move(other.m_shape);
        m_size = other.m_size;

        other.m_size = 0;
        other.m_data = nullptr;
        other.m_end_itr = nullptr;
    }

    dtype* cbegin() const { return m_data; }
    dtype* cend() const { return m_end_itr; }
    dtype* begin() { return m_data; }
    const dtype* begin() const { return m_data; }
    dtype* end() { return m_end_itr; }
    const dtype* end() const { return m_end_itr; }
    size_t size() const { return m_size; }
    const std::array<std::size_t, N>& shape() const { return m_shape; }
};

}  // namespace ndarray
