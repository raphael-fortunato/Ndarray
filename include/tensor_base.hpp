#pragma once
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <shape.hpp>
#include <slice_impl.hpp>
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
        desc = other.desc;
        assert(size() == other.size());
        // Copy data
        m_data = new dtype[other.size()];
        m_end_itr = m_data + other.size();
        std::copy(other.m_data, other.m_end_itr, m_data);
        assert(m_data != nullptr);
    }
    TensorBase& operator=(const TensorBase& other) {
        printf("Base Copy assignment\n");
        if (this != &other) {
            // copy shape
            desc = other.desc;
            assert(size() == other.size());
            // Copy data
            m_data = new dtype[other.size()];
            m_end_itr = m_data + other.size();
            std::copy(other.m_data, other.m_end_itr, m_data);
        }
        return *this;
    }
    TensorBase(TensorBase&& other) noexcept {
        printf("Base Move constructor\n");
        desc = std::move(other.desc);
        m_data = other.m_data;
        m_end_itr = other.m_end_itr;

        other.m_data = nullptr;
        other.m_end_itr = nullptr;
    }
    TensorBase& operator=(TensorBase&& other) noexcept {
        printf("Base Move assignment\n");
        if (this != &other) {
            // copy shape
            desc = std::move(other.desc);
            m_data = other.m_data;
            m_end_itr = other.m_end_itr;

            other.m_data = nullptr;
            other.m_end_itr = nullptr;
        }
        return *this;
    }

    template <typename... Args>
        requires tensor_impl::ValidateElementReturn<N, Args...>
    inline dtype operator()(Args&&... args) {
        static_assert(sizeof...(args) == N, "Invalid number of arguments");
        const std::array<std::size_t, N> indices{
            static_cast<std::size_t>(args)...};
        std::size_t linear_index{0};
        for (std::size_t i = 0; i < N; ++i) {
            if (indices[i] >= this->shape()[i]) {
                throw std::out_of_range("Index out of range");
            }
            linear_index += indices[i] * this->strides()[i];
        }

        return this->m_data[linear_index];
    }

    template <typename... Args>
        requires tensor_impl::ValidateTensorRefReturn<N, Args...>
    inline TensorRef<dtype, N - sizeof...(Args)> operator()(
        const Args&... args) {
        static_assert(sizeof...(args) < N, "Invalid number of arguments");

        details::Descriptor new_desc{this->desc};
        std::array<bool, N> keep_dim;
        keep_dim.fill(true);
        assert(keep_dim.size() == this->shape().size() && "Size mismatch");

        std::size_t offset = slice_impl::do_slice<N>(
            this->desc, new_desc, keep_dim, std::size_t{0}, args...);

        std::vector<std::size_t> final_shape, final_strides;
        for (std::size_t i = 0; i < keep_dim.size(); ++i) {
            if (keep_dim[i]) {
                final_shape.push_back(new_desc.shape()[i]);
                final_strides.push_back(new_desc.strides()[i]);
            }
        }
        new_desc = details::Descriptor(std::move(final_shape),
                                       std::move(final_strides));

        return TensorRef<dtype, N - sizeof...(Args)>(this->m_data + offset,
                                                     std::move(new_desc));
    }

    virtual ~TensorBase() {}
    using value_type = dtype;
    dtype* cbegin() const { return m_data; }
    dtype* cend() const { return m_end_itr; }
    const dtype* begin() const { return m_data; }
    const dtype* end() const { return m_end_itr; }
    const size_t size() const { return desc.size(); }
    const std::vector<std::size_t>& shape() const { return desc.shape(); }
    const std::vector<std::size_t>& strides() const { return desc.strides(); }
    const std::vector<dtype> data() const {
        return std::vector<dtype>(m_data, m_data + size());
    }

   protected:
    TensorBase() = default;
    details::Descriptor desc;
    dtype* m_data;
    dtype* m_end_itr;
};

}  // namespace tensor
