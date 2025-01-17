#include <cassert>
#include <cstddef>
#include <cstdio>
#include <slice.hpp>
#include <stdexcept>
#include <vector>

namespace tensor {

namespace slice_impl {

inline std::size_t do_slice(const std::vector<std::size_t>& os,
                            const std::vector<std::size_t>& ns,
                            const std::vector<std::size_t>& old_strides,
                            const std::vector<std::size_t>& new_strides,
                            std::vector<bool>& keep_dim,
                            const std::size_t current_dim) {
    return 0;
}

inline std::size_t do_slice_dim(const std::vector<std::size_t>& os,
                                std::vector<std::size_t>& ns,
                                const std::vector<std::size_t>& old_strides,
                                std::vector<std::size_t>& new_strides,
                                std::vector<bool>& keep_dim,
                                const std::size_t current_dim,
                                const std::size_t dim) {
    if (dim >= os[current_dim]) {
        throw std::out_of_range("Index out of range");
    }
    std::size_t offset = dim * old_strides[current_dim];
    keep_dim[current_dim] = false;
    return offset;
}

inline std::size_t do_slice_dim(const std::vector<std::size_t>& os,
                                std::vector<std::size_t>& ns,
                                const std::vector<std::size_t>& old_strides,
                                const std::vector<std::size_t>& new_strides,
                                std::vector<bool>& keep_dim,
                                const std::size_t current_dim,
                                const slice::slice& dim) {
    if (dim.start > os[current_dim] || dim.stop > os[current_dim]) {
        throw std::out_of_range("Slice bounds out of range");
    }
    std::size_t offset = dim.start * old_strides[current_dim];
    ns[current_dim] = dim.stop - dim.start;
    keep_dim[current_dim] = true;
    return offset;
}

template <typename T, typename... Args>
inline std::size_t do_slice(const std::vector<std::size_t>& os,
                            std::vector<std::size_t>& ns,
                            const std::vector<std::size_t>& old_strides,
                            std::vector<std::size_t>& new_strides,
                            std::vector<bool>& keep_dim,
                            std::size_t current_dim, const T& dim,
                            const Args&... args) {
    std::size_t m = do_slice_dim(os, ns, old_strides, new_strides, keep_dim,
                                 current_dim, dim);
    std::size_t n = do_slice(os, ns, old_strides, new_strides, keep_dim,
                             ++current_dim, args...);
    return m + n;
}

}  // namespace slice_impl
}  // namespace tensor

