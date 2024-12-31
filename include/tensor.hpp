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
#include <numeric>
#include <utility>
#include <vector>

#include "tensor_base.hpp"
#include "tensor_impl.hpp"

namespace tensor {
template <typename dtype, size_t N>
class Tensor : public TensorBase<dtype, N> {
   public:
    using value_type = dtype;
    explicit Tensor(tensor_impl::tensor_initializer<dtype, N> init) {
        printf("Constructor\n");
        this->m_shape = tensor_impl::derive_shape<N>(init);
        m_compute_size();
        m_allocate_data();
        tensor_impl::insert_flat(init, this->m_data);
        m_compute_strides();
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

   private:
    void m_allocate_data() {
        this->m_data = new dtype[this->m_size];
        this->m_end_itr = this->m_data + this->m_size;
    }

    void m_compute_size() {
        this->m_size =
            std::accumulate(this->m_shape.begin(), this->m_shape.end(), 1,
                            std::multiplies<std::size_t>());
    }
    void m_compute_strides() {
        this->m_strides.resize(this->m_shape.size());
        if (this->m_strides.empty()) {
            return;
        }
        this->m_strides.back() = 1;
        for (int i = this->m_strides.size() - 2; i >= 0; --i) {
            this->m_strides[i] = this->m_shape[i + 1] * this->m_strides[i + 1];
        }
    }
};

}  // namespace tensor
