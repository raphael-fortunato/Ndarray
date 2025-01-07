#pragma once
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <numeric>

namespace tensor_impl {

template <typename dtype, std::size_t N>
struct tensor_init {
    using type =
        std::initializer_list<typename tensor_init<dtype, N - 1>::type>;
};

template <typename dtype>
struct tensor_init<dtype, 1> {
    using type = std::initializer_list<dtype>;
};

template <typename dtype>
struct tensor_init<dtype, 0>;

template <typename dtype, std::size_t N>
using tensor_initializer = typename tensor_init<dtype, N>::type;

template <bool B, class T = void>
struct enable_if {};

template <class T>
struct enable_if<true, T> {
    typedef T type;
};
template <bool B, class T = void>
using enable_if_t = typename enable_if<B, T>::type;

template <typename... Args>
concept AllConvertibleToSizeT = (std::convertible_to<Args, std::size_t> && ...);

template <std::size_t N, typename... Args>
concept ValidateTensorRefReturn =
    (sizeof...(Args) < N) && AllConvertibleToSizeT<Args...>;

template <std::size_t N, typename... Args>
concept ValidateElementReturn =
    (sizeof...(Args) == N) && AllConvertibleToSizeT<Args...>;

template <std::size_t N>
bool check_bounds(const std::vector<std::size_t>& s,
                  const std::vector<std::size_t>& shape) {
    return std::all_of(s.begin(), s.end(), [&, i = 0](std::size_t dim) mutable {
        return i < shape[i++];
    });
}

template <size_t N, typename List, typename l>
enable_if_t<(N == 1), void> add_extended_shape(const List& list, l& first) {
    static_assert(N == 1,
                  "This function should only be instantiated for N == 1.");
    *first++ = list.size();
}

template <typename T>
constexpr bool check_non_jagged_list(const std::initializer_list<T>& list) {
    auto i = list.begin();
    for (auto j = i + 1; j != list.end(); ++j) {
        if (i->size() != j->size()) {
            return false;
        }
    }
    return true;
}

template <size_t N, typename Nested_List, typename l>
enable_if_t<(N > 1), void> add_extended_shape(const Nested_List& nested_list,
                                              l& first) {
    static_assert(N > 1,
                  "This function should only be instantiated for N > 1.");
    assert(check_non_jagged_list(nested_list) && "Jagged list dectected");
    *first = nested_list.size();
    add_extended_shape<N - 1>(*nested_list.begin(), ++first);
}

template <size_t N, typename List>
std::vector<std::size_t> derive_shape(const List& init_list) {
    std::vector<std::size_t> a;
    a.resize(N);
    auto f = a.begin();
    add_extended_shape<N>(init_list, f);
    return a;
}

template <typename T, typename dtype>
void add_flattened_data(const T* begin, const T* end, dtype*& ptr) {
    std::copy(begin, end, ptr);
    ptr += end - begin;
}

template <typename T, typename dtype>
void add_flattened_data(const std::initializer_list<T>* begin,
                        const std::initializer_list<T>* end, dtype*& ptr) {
    for (; begin != end; begin++) {
        add_flattened_data(begin->begin(), begin->end(), ptr);
    }
}

template <typename T, typename dtype>
void insert_flat(std::initializer_list<T> data, dtype* ptr) {
    add_flattened_data(data.begin(), data.end(), ptr);
}

inline std::size_t m_compute_size(const std::vector<std::size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t(1),
                           std::multiplies<>());
}

inline std::vector<std::size_t> m_compute_strides(
    const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> strides(shape.size());
    if (shape.empty()) {
        return strides;
    }
    strides.back() = 1;
    for (int i = strides.size() - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

// TODO implement check on slicing
// constexpr bool all() { return true; }
//
// template <typename... Args>
// constexpr bool all(bool b, Args... args) {
//     return b && all(args...);
// }
//
// constexpr bool some() { return false; }
//
// template <typename... Args>
// constexpr bool some(bool b, Args... args) {
//     return b || some(args...);
// }
//
// template <typename... Args>
// constexpr bool requesting_element() {
//     return all(std::is_convertible<Args, std::size_t>()...);
// }
//
// template <typename... Args>
// constexpr bool requesting_slice() {
//     return all((std::is_convertible<Args, std::size_t>() ||
//                 std::is_same<Args, slice::slice>())...) &&
//            some(std::is_same<Args, slice::slice>()...);
// }

}  // namespace tensor_impl
