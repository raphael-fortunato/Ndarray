/*
 * This is the header file for the Ndarray class.
 * This class is the base class of a multi-dimensional array.
 * Every ndarray array needs a data_type T, shape, and strides.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <tensor_ref.hpp>
#include <utility>

#include "tensor_base.hpp"
#include "tensor_impl.hpp"

namespace tensor {
template <typename dtype, std::size_t N>
class Tensor : public TensorBase<dtype, N> {
   public:
    using value_type = dtype;
    Tensor() = delete;
    explicit Tensor(tensor_impl::tensor_initializer<dtype, N> init) {
        printf("Constructor\n");
        this->m_shape = tensor_impl::derive_shape<N>(init);
        this->m_size = tensor_impl::m_compute_size(this->m_shape);
        m_allocate_data();
        tensor_impl::insert_flat(init, this->m_data);
        this->m_strides = tensor_impl::m_compute_strides(this->m_shape);
        assert(this->end() - this->begin() == this->size());
    }
    ~Tensor() override {
        printf("Destructor\n");
        delete[] this->m_data;
    }
    Tensor& operator=(const Tensor& other) {
        printf("Copy assignment\n");
        if (this != &other) {
            delete[] this->m_data;
            TensorBase<dtype, N>::operator=(std::move(other));
        }
        return *this;
    }

    Tensor(const Tensor& other) : TensorBase<dtype, N>(other) {
        printf("Copy constructor\n");
    }
    Tensor(Tensor&& other) noexcept : TensorBase<dtype, N>(std::move(other)) {
        printf("Move constructor\n");
    }
    Tensor& operator=(Tensor&& other) noexcept {
        printf("Move assignment\n");
        if (this != &other) {
            delete[] this->m_data;
            TensorBase<dtype, N>::operator=(std::move(other));
        }
        return *this;
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

   private:
    void m_allocate_data() {
        this->m_data = new dtype[this->m_size];
        this->m_end_itr = this->m_data + this->m_size;
    }
};

}  // namespace tensor
