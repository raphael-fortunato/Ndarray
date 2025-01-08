**IN EARLY DEVELOPMENT**

# NDArray: An N-Dimensional Matrix Library for C++

NDArray is a lightweight and flexible C++ library for working with
N-dimensional matrices (tensors). It provides a dynamic, efficient, and
user-friendly way to manage and manipulate multi-dimensional arrays, inspired
by libraries like NumPy.

# Installation

1. Clone this repository.
2. `mkdir build && cd build`
3. `cmake -DCMAKE_BUILD_TYPE=Debug ..`
4. `make`

# Usage

```cpp
#include <tensor.hpp>

Tensor<int,1> a{10, 11, 12};
Tensor<int,2> b{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

```

# Roadmap

1. Implement slicing
2. Implement basic arithmetic operations
3. Implement random number generation
4. Implement Reference counting
