/*
 * This is the header file for the Ndarray class.
 * This class is the base class of a multi-dimensional array.
 * Every ndarray array needs a data_type T, shape, and strides.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

namespace ndarray {
template <typename dtype = double>
class Ndarray {
   private:
    dtype* m_end_itr;
    dtype* m_start_itr;
    std::vector<std::size_t> m_shape;
    size_t m_size = 1;

    void allocate_data() { m_start_itr = new dtype[m_size]; }

    void m_compute_size() {
        for (const auto& dim : m_shape) {
            m_size *= dim;
        }
    }

   public:
    explicit Ndarray(std::vector<std::size_t> shape) : m_shape(shape) {
        m_compute_size();
        allocate_data();
        m_end_itr = m_start_itr + m_size;
    }
    ~Ndarray() {
        delete[] m_start_itr;
        m_shape.clear();
    }
    Ndarray(const Ndarray& other) {
        // copy shape
        std::copy(other.m_shape.begin(), other.m_shape.end(),
                  std::back_inserter(m_shape));
        m_compute_size();
        assert(m_size == other.m_size);
        // Copy data
        m_start_itr = new dtype[other.m_size];
        m_end_itr = m_start_itr + other.m_size;
        std::copy(other.m_start_itr, other.m_end_itr, m_start_itr);
    }

    size_t size() const { return m_size; }

    dtype* get_start_itr() const { return m_start_itr; }
    dtype* get_end_itr() const { return m_end_itr; }
};
}  // namespace ndarray
