#include <gtest/gtest.h>

#include "ndarray.hpp"

using namespace ndarray;

size_t get_expected_size(const std::vector<std::size_t>& shape) {
    size_t size = 1;
    for (const auto& dim : shape) {
        size *= dim;
    }
    return size;
}
TEST(NdarrayTest, TestShape) {
    // Test 1 dim array
    std::vector<std::size_t> shape = {2};
    Ndarray<double> test_array(shape);
    EXPECT_EQ(test_array.get_end_itr() - test_array.get_start_itr(),
              get_expected_size(shape));

    // Test 2 dim array
    shape = {2, 3};
    Ndarray<double> test_array2(shape);
    EXPECT_EQ(test_array2.get_end_itr() - test_array2.get_start_itr(),
              get_expected_size(shape));

    // Test 3 dim array
    shape = {2, 3, 4};
    Ndarray<double> test_array3(shape);
    EXPECT_EQ(test_array3.get_end_itr() - test_array3.get_start_itr(),
              get_expected_size(shape));
}

TEST(NdarrayTest, TestSize) {
    std::vector<std::size_t> shape = {2, 3, 4};
    Ndarray<double> test_array(shape);
    EXPECT_EQ(test_array.get_end_itr() - test_array.get_start_itr(),
              test_array.size());
    EXPECT_EQ(test_array.get_end_itr() - test_array.get_start_itr(),
              get_expected_size(shape));
}
