/*
 * This is the header file for the Ndarray class.
 * This class is the base class of a multi-dimensional array.
 * Every ndarray array needs a data_type T, shape, and strides.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

namespace ndarray {
template <typename dtype = double>
class Ndarray {
   private:
    dtype* m_end_itr;
    dtype* m_data;
    std::vector<std::size_t> m_shape;
    size_t m_size = 1;

    void allocate_data() { m_data = new dtype[m_size]; }

    void m_compute_size() {
        for (const auto& dim : m_shape) {
            m_size *= dim;
        }
    }

   public:
    explicit Ndarray(std::vector<std::size_t> shape) : m_shape(shape) {
        printf("Constructor\n");
        m_compute_size();
        allocate_data();
        m_end_itr = m_data + m_size;
    }
    ~Ndarray() {
        delete[] m_data;
        m_shape.clear();
    }
    Ndarray(const Ndarray& other) {
        printf("Copy constructor\n");
        // copy shape
        std::copy(other.m_shape.begin(), other.m_shape.end(),
                  std::back_inserter(m_shape));
        m_compute_size();
        assert(m_size == other.m_size);
        // Copy data
        m_data = new dtype[other.m_size];
        m_end_itr = m_data + other.m_size;
        std::copy(other.m_data, other.m_end_itr, m_data);
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

    size_t size() const { return m_size; }

    dtype* cbegin() const { return m_data; }
    dtype* cend() const { return m_end_itr; }
    dtype* begin() { return m_data; }
    dtype* end() { return m_end_itr; }
};
}  // namespace ndarray
