#pragma once

#include <cstddef>
#include <vector>

namespace slice {
struct slice {
    slice() = default;

    slice(std::size_t start = -1, std::size_t end = -1)
        : start{start}, stop{end} {}

    std::size_t start{0};
    std::size_t stop{0};
    std::vector<std::size_t> shape;
    std::vector<std::size_t> strides;
};
}  // namespace slice
