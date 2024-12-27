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
    EXPECT_EQ(test_array.cend() - test_array.cbegin(),
              get_expected_size(shape));

    // Test 2 dim array
    shape = {2, 3};
    Ndarray<double> test_array2(shape);
    EXPECT_EQ(test_array2.cend() - test_array2.cbegin(),
              get_expected_size(shape));

    // Test 3 dim array
    shape = {2, 3, 4};
    Ndarray<double> test_array3(shape);
    EXPECT_EQ(test_array3.cend() - test_array3.cbegin(),
              get_expected_size(shape));
}

TEST(NdarrayTest, TestSize) {
    std::vector<std::size_t> shape = {2, 3, 4};
    Ndarray<double> test_array(shape);
    EXPECT_EQ(test_array.cend() - test_array.cbegin(), test_array.size());
    EXPECT_EQ(test_array.cend() - test_array.cbegin(),
              get_expected_size(shape));
}

TEST(NdarrayTest, TestConstruction) {
    Ndarray<double> test_array({2, 3, 4});
    EXPECT_EQ(test_array.cend() - test_array.cbegin(), test_array.size());
}

TEST(NdarrayTest, TestCopyConstructor) {
    Ndarray<double> test_array1({2, 3, 4});
    Ndarray<double> test_array2 = test_array1;
    EXPECT_EQ(test_array1.size(), test_array2.size());
    EXPECT_EQ(test_array1.cend() - test_array1.cbegin(),
              test_array2.cend() - test_array2.cbegin());
}

// Test move constructor
TEST(NdarrayTest, TestMoveConstructor) {
    Ndarray<double> test_array1({2, 3, 4});
    Ndarray<double> test_array2 = std::move(test_array1);
    EXPECT_EQ(test_array1.begin(), nullptr);
    EXPECT_EQ(test_array1.end(), nullptr);
    EXPECT_EQ(test_array1.size(), 0);
    EXPECT_EQ(test_array2.size(), 24);
    EXPECT_EQ(test_array2.cend() - test_array2.cbegin(), test_array2.size());
}
