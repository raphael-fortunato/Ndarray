
#include <cassert>
#include <cstddef>
#include <slice.hpp>
#include <stdexcept>
#include <vector>

namespace tensor {

namespace slice_impl {

inline std::size_t do_slice(const std::vector<std::size_t>& os,
                            const std::vector<std::size_t>& ns) {
    return 0;
}

template <std::size_t N>
inline std::size_t do_slice_dim(const std::vector<std::size_t>& os,
                                std::vector<std::size_t>& ns, std::size_t dim) {
    if (dim >= ns.front()) {
        throw std::out_of_range("Index out of range");
    }
    std::size_t offset = dim * *ns.begin();
    ns.erase(ns.begin());
    return offset;
}

template <std::size_t N>
inline std::size_t do_slice_dim(const std::vector<std::size_t>& os,
                                std::vector<std::size_t>& ns,
                                const slice::slice& dim) {
    assert(dim.start < *ns.begin() && dim.stop <= *ns.begin() &&
           "Slice bounds out of range");
    std::size_t offset = dim.start * *ns.begin();
    ns[0] = dim.stop - dim.start;
    return offset;
}

template <typename T, typename... Args>
inline std::size_t do_slice(const std::vector<std::size_t>& os,
                            std::vector<std::size_t>& ns, const T& dim,
                            const Args&... args) {
    std::size_t m = do_slice_dim<sizeof...(Args) - 1>(os, ns, dim);
    std::size_t n = do_slice(os, ns, args...);
    return m + n;
}

}  // namespace slice_impl
}  // namespace tensor

