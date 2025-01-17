#pragma once

#include <vector>

#include "tensor_impl.hpp"

namespace tensor {
namespace details {
class Descriptor {
   public:
    Descriptor(std::vector<std::size_t>&& shape,
               std::vector<std::size_t>&& strides)
        : m_shape{std::move(shape)}, m_strides{std::move(strides)} {}

    explicit Descriptor(std::vector<std::size_t>&& init_shape)
        : m_shape{init_shape},
          m_strides{tensor_impl::compute_strides(m_shape)} {}

    template <typename... Args>
        requires tensor_impl::AllConvertibleToSizeT<Args...>
    explicit Descriptor(Args... args) {
        this->m_shape = {static_cast<std::size_t>(args)...};
        this->m_strides = tensor_impl::compute_strides(this->m_shape);
    }
    Descriptor() = default;

    Descriptor(Descriptor& other)
        : m_shape{other.shape()}, m_strides{other.strides()} {}

    Descriptor& operator=(const Descriptor& other) {
        if (this != &other) {
            m_shape = other.shape();
            m_strides = other.strides();
        }
        return *this;
    }

    Descriptor(Descriptor&& other)
        : m_shape{std::move(other.shape())},
          m_strides{std::move(other.strides())} {}

    Descriptor& operator=(Descriptor&& other) {
        if (this != &other) {
            m_shape = std::move(other.shape());
            m_strides = std::move(other.strides());
        }
        return *this;
    }
    ~Descriptor() {
        m_shape.clear();
        m_strides.clear();
    };
    const std::vector<std::size_t>& shape() const { return m_shape; }
    const std::vector<std::size_t>& strides() const { return m_strides; }
    std::vector<std::size_t>& shape() { return m_shape; }
    std::vector<std::size_t>& strides() { return m_strides; }
    const std::size_t size() const {
        return tensor_impl::compute_size(m_shape);
    }

   private:
    std::vector<std::size_t> m_shape;
    std::vector<std::size_t> m_strides;
};
}  // namespace details
}  // namespace tensor
