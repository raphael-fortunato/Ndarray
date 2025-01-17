#pragma once
#include "tensor_base.hpp"

namespace tensor {
template <typename dtype, std::size_t N>
class TensorRef : public TensorBase<dtype, N> {
   public:
    TensorRef() = delete;

    TensorRef(dtype* data, std::vector<std::size_t>&& shape,
              std::vector<std::size_t>&& strides) {
        this->m_data = data;
        this->desc = details::Descriptor(std::move(shape), std::move(strides));
        this->m_end_itr = data + this->size();
    }
    TensorRef(dtype* data, details::Descriptor&& desc) {
        this->m_data = data;
        this->desc = std::forward<details::Descriptor>(desc);
        this->m_end_itr = data + this->size();
    }
};
}  // namespace tensor
