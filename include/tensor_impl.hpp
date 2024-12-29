
#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <queue>
#include <type_traits>

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

template <size_t N, typename List, typename l>
enable_if_t<(N == 1), void> add_extended_shape(const List& list, l& first) {
    static_assert(N == 1,
                  "This function should only be instantiated for N == 1.");
    *first++ = list.size();
}

template <size_t N, typename Nested_List, typename l>
enable_if_t<(N > 1), void> add_extended_shape(const Nested_List& nested_list,
                                              l& first) {
    static_assert(N > 1,
                  "This function should only be instantiated for N > 1.");
    *first = nested_list.size();
    add_extended_shape<N - 1>(*nested_list.begin(), ++first);
}

template <size_t N, typename List>
std::array<std::size_t, N> derive_shape(const List& init_list) {
    std::array<std::size_t, N> a;
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

}  // namespace tensor_impl
