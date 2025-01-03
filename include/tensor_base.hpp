#pragma once
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace tensor {

template <typename dtype, size_t N>
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

    using value_type = dtype;
    virtual ~TensorBase() {}
    dtype* cbegin() const { return m_data; }
    dtype* cend() const { return m_end_itr; }
    const dtype* begin() const { return m_data; }
    const dtype* end() const { return m_end_itr; }
    size_t size() const { return m_size; }
    const std::vector<std::size_t>& shape() const { return m_shape; }
    const std::vector<std::size_t>& strides() const { return m_strides; }

   protected:
    TensorBase() = default;
    dtype* m_data;
    dtype* m_end_itr;
    std::vector<std::size_t> m_shape;
    std::vector<std::size_t> m_strides;
    std::size_t m_size = 0;
};

}  // namespace tensor
