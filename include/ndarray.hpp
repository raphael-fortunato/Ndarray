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
    Ndarray(tensor_impl::tensor_initializer<dtype, N> init) {
        printf("Constructor\n");
        m_shape = tensor_impl::derive_shape<N>(init);
        m_compute_size();
        m_allocate_data();
        tensor_impl::insert_flat(init, m_data);
        m_compute_strides();
    }
    ~Ndarray() {
        printf("Destructor\n");
        delete[] m_data;
    }
    Ndarray(const Ndarray& other) {
        printf("Copy constructor\n");
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
    }
    Ndarray& operator=(const Ndarray& other) {
        printf("Copy assignment\n");
        if (this != &other) {
            // copy shape
            m_shape = other.m_shape;
            m_strides = other.m_strides;
            m_size = other.m_size;
            assert(m_size == other.m_size);
            // Copy data
            delete[] m_data;
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
    const std::vector<std::size_t>& shape() const { return m_shape; }
    const std::vector<std::size_t>& strides() const { return m_strides; }

   private:
    dtype* m_end_itr;
    dtype* m_data;
    std::vector<std::size_t> m_shape;
    std::vector<std::size_t> m_strides;
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
    void m_compute_strides() {
        m_strides.resize(m_shape.size());
        if (m_strides.empty()) {
            return;
        }
        m_strides.back() = 1;
        for (int i = m_strides.size() - 2; i >= 0; --i) {
            m_strides[i] = m_shape[i + 1] * m_strides[i + 1];
        }
    }
};

}  // namespace ndarray
