#include <gtest/gtest.h>

#include <vector>

#include "gmock/gmock.h"
#include "ndarray.hpp"

using namespace ndarray;

TEST(NdarrayTest, TestShape) {
    // Test 1 dim array
    Ndarray<double, 1> test_array = {12.5, 13.5};
    ASSERT_THAT(test_array, ::testing::ElementsAre(12.5, 13.5));
    ASSERT_THAT(test_array.shape(), ::testing::ElementsAre(2));

    Ndarray<int, 2> test_array2 = {{12, 13}, {14, 15}};
    ASSERT_THAT(test_array2.shape(), ::testing::ElementsAre(2, 2));
    ASSERT_THAT(test_array2, ::testing::ElementsAre(12, 13, 14, 15));

    // // Test 3 dim array
    Ndarray<double, 3> test_array3 = {{{12.5, 13.5}, {14.5, 15.5}},
                                      {{16.5, 17.5}, {18.5, 19.5}}};
    ASSERT_THAT(test_array3.shape(), ::testing::ElementsAre(2, 2, 2));
    ASSERT_THAT(test_array3, ::testing::ElementsAre(12.5, 13.5, 14.5, 15.5,
                                                    16.5, 17.5, 18.5, 19.5));

    // Test array with shap 2x3x4
    Ndarray<double, 3> test_array4 = {{{12.5, 13.5, 14.5, 15.5},
                                       {16.5, 17.5, 18.5, 19.5},
                                       {20.5, 21.5, 22.5, 23.5}},
                                      {{24.5, 25.5, 26.5, 27.5},
                                       {28.5, 29.5, 30.5, 31.5},
                                       {32.5, 33.5, 34.5, 35.5}}};
    ASSERT_THAT(test_array4.shape(), ::testing::ElementsAre(2, 3, 4));
    ASSERT_THAT(
        test_array4,
        ::testing::ElementsAre(12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5,
                               20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5,
                               28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5));
}

TEST(NdarrayTest, TestSize) {
    // Test the size of an array with shape 2x3x4
    Ndarray<double, 3> test_array = {{{12.5, 13.5, 14.5, 15.5},
                                      {16.5, 17.5, 18.5, 19.5},
                                      {20.5, 21.5, 22.5, 23.5}},
                                     {{24.5, 25.5, 26.5, 27.5},
                                      {28.5, 29.5, 30.5, 31.5},
                                      {32.5, 33.5, 34.5, 35.5}}};
    ASSERT_THAT(test_array.size(), ::testing::Eq(24));
    ASSERT_THAT(
        test_array,
        ::testing::ElementsAre(12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5,
                               20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5,
                               28.5, 29.5, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5));

    // Test the size of an array with shape 2x2x2x2
    Ndarray<double, 4> test_array2 = {
        {{{12.5, 13.5}, {14.5, 15.5}}, {{16.5, 17.5}, {18.5, 19.5}}},
        {{{20.5, 21.5}, {22.5, 23.5}}, {{24.5, 25.5}, {26.5, 27.5}}}};
    ASSERT_THAT(test_array2.size(), ::testing::Eq(16));
    ASSERT_THAT(
        test_array2,
        ::testing::ElementsAre(12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5,
                               20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5));
    // Test the size of an array with shape 1x1x3x3
    Ndarray<double, 4> test_array3 = {
        {{{12.5, 13.5, 14.5}, {15.5, 16.5, 17.5}, {18.5, 19.5, 20.5}}}};
    ASSERT_THAT(test_array3.size(), ::testing::Eq(9));
    ASSERT_THAT(test_array3,
                ::testing::ElementsAre(12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5,
                                       19.5, 20.5));
}
TEST(NdarrayTest, TestCopyConstructor) {
    Ndarray<double, 1> test_array1{2, 3, 4};
    auto test_array2 = test_array1;
    EXPECT_EQ(test_array1.size(), test_array2.size());
    EXPECT_EQ(test_array1.cend() - test_array1.cbegin(),
              test_array2.cend() - test_array2.cbegin());
}
TEST(NdarrayTest, TestMoveConstructor) {
    Ndarray<double, 1> test_array1{2, 3, 4};
    auto test_array2 = std::move(test_array1);
    EXPECT_EQ(test_array1.begin(), nullptr);
    EXPECT_EQ(test_array1.end(), nullptr);
    EXPECT_EQ(test_array1.size(), 0);
    EXPECT_EQ(test_array2.size(), 3);
    EXPECT_EQ(test_array2.cend() - test_array2.cbegin(), test_array2.size());
}
