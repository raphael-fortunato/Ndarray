#pragma once

#include <cstddef>

namespace slice {
struct slice {
    slice()
        : start{static_cast<std::size_t>(-1)},
          stop{static_cast<std::size_t>(-1)} {};

    explicit slice(std::size_t start)
        : start{start}, stop{static_cast<std::size_t>(-1)} {}

    explicit slice(std::size_t start, std::size_t stop)
        : start{start}, stop{stop} {}

    std::size_t start{static_cast<std::size_t>(-1)};
    std::size_t stop{static_cast<std::size_t>(-1)};
};
}  // namespace slice
