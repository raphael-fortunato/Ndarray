/*
 * This is the header file for the TensorLib++ class.
 * This class is the base class of a multi-dimensional array.
 * Every tensor array needs a data_type T, shape, and strides.
 */
#pragma once

#include <algorithm>
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
        this->m_size = m_compute_size(this->m_shape);
        m_allocate_data();
        tensor_impl::insert_flat(init, this->m_data);
        this->m_strides = m_compute_strides(this->m_shape);
        assert(this->end() - this->begin() == this->size());
    }

    template <typename... Args>
        requires tensor_impl::AllConvertibleToSizeT<Args...>
    explicit Tensor(Args... args) {
        printf("Constructor\n");
        this->m_shape = {static_cast<std::size_t>(args)...};
        this->m_size = m_compute_size(this->m_shape);
        m_allocate_data();
        this->m_strides = m_compute_strides(this->m_shape);
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

    Tensor(Tensor&& other) noexcept
        : TensorBase<dtype, N>(std::forward<Tensor>(other)) {
        printf("Move constructor\n");
    }

    Tensor& operator=(Tensor&& other) noexcept {
        printf("Move assignment\n");
        if (this != &other) {
            delete[] this->m_data;
            TensorBase<dtype, N>::operator=(std::forward<Tensor>(other));
        }
        return *this;
    }

   private:
    void m_allocate_data();
};

template <typename dtype, std::size_t N>
void Tensor<dtype, N>::m_allocate_data() {
    this->m_data = new dtype[this->m_size];
    this->m_end_itr = this->m_data + this->m_size;
}

}  // namespace tensor
