#include <gtest/gtest.h>

#include "gmock/gmock.h"
#include "tensorlibpp.hpp"

using namespace tensor;

TEST(TensorTest, TestScalarInplaceOperations) {
    Tensor<double, 1> test_array1{0, 0, 0, 0};
    test_array1 += 1;
    ASSERT_THAT(test_array1, ::testing::ElementsAre(1, 1, 1, 1));
    test_array1 *= 2;
    ASSERT_THAT(test_array1, ::testing::ElementsAre(2, 2, 2, 2));
    test_array1 -= 1;
    ASSERT_THAT(test_array1, ::testing::ElementsAre(1, 1, 1, 1));
    test_array1 /= 2;
    ASSERT_THAT(test_array1, ::testing::ElementsAre(0.5, 0.5, 0.5, 0.5));

    Tensor<int, 1> test_array2{3, 3, 3, 3};
    test_array2 %= 2;
    ASSERT_THAT(test_array2, ::testing::ElementsAre(1, 1, 1, 1));
}

TEST(TensorTest, TestScalarOperations) {
    Tensor<double, 1> test_array1{0, 0, 0, 0};
    auto test_array2 = test_array1 + 1.0;
    ASSERT_THAT(test_array2, ::testing::ElementsAre(1, 1, 1, 1));
    ASSERT_THAT(test_array1, ::testing::ElementsAre(0, 0, 0, 0));
    auto test_array3 = test_array2 * 2.0;
    ASSERT_THAT(test_array3, ::testing::ElementsAre(2, 2, 2, 2));
    auto test_array4 = test_array3 - 1.0;
    ASSERT_THAT(test_array4, ::testing::ElementsAre(1, 1, 1, 1));
    auto test_array5 = test_array4 / 2.0;
    ASSERT_THAT(test_array5, ::testing::ElementsAre(0.5, 0.5, 0.5, 0.5));

    Tensor<int, 1> test_array6{3, 3, 3, 3};
    auto test_array7 = test_array6 % 2;
    ASSERT_THAT(test_array7, ::testing::ElementsAre(1, 1, 1, 1));
}
TEST(TensorTest, TestMatrixInplaceOperations) {
    Tensor<int, 2> test_array1{{1, 2}, {3, 4}};
    Tensor<int, 2> test_array2{{1, 2}, {3, 4}};
    test_array1 += test_array2;
    ASSERT_THAT(test_array1, ::testing::ElementsAre(2, 4, 6, 8));
    test_array1 -= test_array2;
    ASSERT_THAT(test_array1, ::testing::ElementsAre(1, 2, 3, 4));
    test_array1 *= test_array2;
    ASSERT_THAT(test_array1, ::testing::ElementsAre(1, 4, 9, 16));
    test_array1 /= test_array2;
    ASSERT_THAT(test_array1, ::testing::ElementsAre(1, 2, 3, 4));
    test_array1 %= test_array2;
    ASSERT_THAT(test_array1, ::testing::ElementsAre(0, 0, 0, 0));
}
TEST(TensorTest, TestMatrixOperations) {
    Tensor<int, 2> test_array1{{1, 2}, {3, 4}};
    Tensor<int, 2> test_array2{{1, 2}, {3, 4}};
    auto test_array3 = test_array1 + test_array2;
    ASSERT_THAT(test_array3, ::testing::ElementsAre(2, 4, 6, 8));
    auto test_array4 = test_array3 - test_array2;
    ASSERT_THAT(test_array4, ::testing::ElementsAre(1, 2, 3, 4));
    auto test_array5 = test_array4 * test_array2;
    ASSERT_THAT(test_array5, ::testing::ElementsAre(1, 4, 9, 16));
    auto test_array6 = test_array5 / test_array2;
    ASSERT_THAT(test_array6, ::testing::ElementsAre(1, 2, 3, 4));
    auto test_array7 = test_array6 % test_array2;
    ASSERT_THAT(test_array7, ::testing::ElementsAre(0, 0, 0, 0));
}
TEST(TensorTest, TestOutOfRangeExceptions) {
    // Test out of rang exceptions for math operations
    Tensor<int, 1> test_array1{1, 2, 3, 4};
    Tensor<int, 1> test_array2{1, 2, 3};
    ASSERT_THROW(test_array1 += test_array2, std::out_of_range);
    ASSERT_THROW(test_array1 -= test_array2, std::out_of_range);
    ASSERT_THROW(test_array1 *= test_array2, std::out_of_range);
    ASSERT_THROW(test_array1 /= test_array2, std::out_of_range);
    ASSERT_THROW(test_array1 %= test_array2, std::out_of_range);

    ASSERT_THROW(test_array1 + test_array2, std::out_of_range);
    ASSERT_THROW(test_array1 - test_array2, std::out_of_range);
    ASSERT_THROW(test_array1 * test_array2, std::out_of_range);
    ASSERT_THROW(test_array1 / test_array2, std::out_of_range);
    ASSERT_THROW(test_array1 % test_array2, std::out_of_range);
}
