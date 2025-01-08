#pragma once
#include "tensor_base.hpp"

namespace tensor {
template <typename dtype, std::size_t N>
class TensorRef : public TensorBase<dtype, N> {
   public:
    TensorRef() = delete;

    TensorRef(dtype* data, std::vector<std::size_t> shape,
              std::vector<std::size_t> strides) {
        this->m_data = data;
        this->m_shape = std::forward<std::vector<std::size_t>>(shape);
        this->m_strides = std::forward<std::vector<std::size_t>>(strides);
        this->m_size = m_compute_size(this->m_shape);
        this->m_end_itr = data + this->m_size;
    }
};
}  // namespace tensor
