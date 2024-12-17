/*
 * This is the header file for the Ndarray class.
 * This class is the base class of a multi-dimensional array.
 * Every ndarray array needs a data_type T, shape, and strides.
 */
#pragma once

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
    std::vector<std::size_t> m_strides;
    size_t m_size = 1;

    void allocate_data() { m_start_itr = new dtype[m_size]; }

    void m_compute_size() {
        for (const auto& dim : m_shape) {
            m_size *= dim;
        }
    }

   public:
    Ndarray(std::vector<std::size_t> shape) : m_shape(shape) {
        m_compute_size();
        allocate_data();
        m_end_itr = m_start_itr + m_size;
    }
    ~Ndarray() { delete[] m_start_itr; }

    size_t size() const { return m_size; }

    dtype* get_start_itr() const { return m_start_itr; }
    dtype* get_end_itr() const { return m_end_itr; }
};
}  // namespace ndarray
