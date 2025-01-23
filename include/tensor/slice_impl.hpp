#include <cassert>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "tensor/shape.hpp"
#include "tensor/slice.hpp"

namespace tensor {

namespace slice_impl {

template <std::size_t N>
inline std::size_t do_slice(const details::Descriptor& old_desc,
                            details::Descriptor& new_desc,
                            std::array<bool, N>& keep_dim,
                            const std::size_t current_dim) {
    return 0;
}

template <std::size_t N>
inline std::size_t do_slice_dim(const details::Descriptor& old_desc,
                                details::Descriptor& new_desc,
                                std::array<bool, N>& keep_dim,
                                const std::size_t current_dim,
                                const std::size_t dim) {
    if (dim >= old_desc.shape()[current_dim]) {
        throw std::out_of_range("Index out of range");
    }
    std::size_t offset = dim * old_desc.strides()[current_dim];
    keep_dim[current_dim] = false;
    return offset;
}

template <std::size_t N>
inline std::size_t do_slice_dim(const details::Descriptor& old_desc,
                                details::Descriptor& new_desc,
                                std::array<bool, N>& keep_dim,
                                const std::size_t current_dim,
                                const slice::slice& dim) {
    std::size_t start =
        (dim.start == static_cast<std::size_t>(-1)) ? 0 : dim.start;
    std::size_t stop = (dim.stop == static_cast<std::size_t>(-1))
                           ? old_desc.shape()[current_dim]
                           : dim.stop;
    if (start > old_desc.shape()[current_dim] ||
        stop > old_desc.shape()[current_dim]) {
        throw std::out_of_range("Slice bounds out of range");
    }
    std::size_t offset = start * old_desc.strides()[current_dim];
    new_desc.shape()[current_dim] = stop - start;
    return offset;
}

template <std::size_t N, typename T, typename... Args>
inline std::size_t do_slice(const details::Descriptor& old_desc,
                            details::Descriptor& new_desc,
                            std::array<bool, N>& keep_dim,
                            std::size_t current_dim, const T& dim,
                            const Args&... args) {
    std::size_t m =
        do_slice_dim(old_desc, new_desc, keep_dim, current_dim, dim);
    std::size_t n =
        do_slice(old_desc, new_desc, keep_dim, ++current_dim, args...);
    return m + n;
}

}  // namespace slice_impl
}  // namespace tensor

