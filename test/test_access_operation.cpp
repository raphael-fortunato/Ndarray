#include <gtest/gtest.h>

#include "gmock/gmock.h"
#include "tensor.hpp"

using namespace tensor;

TEST(TensorTest, TestTensorElementAccess) {
    Tensor<double, 1> test_array{2, 3, 4};
    EXPECT_EQ(test_array(0), 2);
    EXPECT_EQ(test_array(1), 3);
    EXPECT_EQ(test_array(2), 4);

    Tensor<double, 2> test_array2{{2, 3}, {4, 5}};
    EXPECT_EQ(test_array2(0, 0), 2);
    EXPECT_EQ(test_array2(0, 1), 3);
    EXPECT_EQ(test_array2(1, 0), 4);
    EXPECT_EQ(test_array2(1, 1), 5);

    EXPECT_THROW(test_array2(2, 0), std::out_of_range);

    Tensor<double, 3> test_array3{{{12.5, 13.5, 14.5, 15.5},
                                   {16.5, 17.5, 18.5, 19.5},
                                   {20.5, 21.5, 22.5, 23.5}},
                                  {{24.5, 25.5, 26.5, 27.5},
                                   {28.5, 29.5, 30.5, 31.5},
                                   {32.5, 33.5, 34.5, 35.5}}};
    EXPECT_EQ(test_array3(0, 0, 0), 12.5);
    EXPECT_EQ(test_array3(1, 1, 1), 29.5);
    EXPECT_EQ(test_array3(1, 2, 2), 34.5);
}

TEST(TensorTest, TestTensorRefAccess) {
    Tensor<double, 2> test_array{{2, 3}, {4, 5}};
    ASSERT_THAT(test_array(0).shape(), testing::ElementsAre(2));
    ASSERT_THAT(test_array(1).shape(), testing::ElementsAre(2));
    ASSERT_THAT(test_array(0), testing::ElementsAre(2, 3));
    ASSERT_THAT(test_array(1), testing::ElementsAre(4, 5));
    Tensor<double, 2> test_array2{{2, 3, 4}, {4, 5, 6}, {6, 7, 8}};
    ASSERT_THAT(test_array2(2).shape(), testing::ElementsAre(3));
    ASSERT_THAT(test_array2(0), testing::ElementsAre(2, 3, 4));
    ASSERT_THAT(test_array2(1), testing::ElementsAre(4, 5, 6));
    ASSERT_THAT(test_array2(2), testing::ElementsAre(6, 7, 8));
    Tensor<double, 3> test_array3{{{12.5, 13.5, 14.5, 15.5},
                                   {16.5, 17.5, 18.5, 19.5},
                                   {20.5, 21.5, 22.5, 23.5}},
                                  {{24.5, 25.5, 26.5, 27.5},
                                   {28.5, 29.5, 30.5, 31.5},
                                   {32.5, 33.5, 34.5, 35.5}}};

    ASSERT_THAT(test_array3(1).shape(), testing::ElementsAre(3, 4));
    ASSERT_THAT(test_array3(1),
                testing::ElementsAre(24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5,
                                     31.5, 32.5, 33.5, 34.5, 35.5));
    ASSERT_THAT(test_array3(1, 1).shape(), testing::ElementsAre(4));
    ASSERT_THAT(test_array3(1, 1),
                testing::ElementsAre(28.5, 29.5, 30.5, 31.5));
    Tensor<double, 4> test_array4{
        {{{12.5, 13.5}, {14.5, 15.5}}, {{16.5, 17.5}, {18.5, 19.5}}},
        {{{20.5, 21.5}, {22.5, 23.5}}, {{24.5, 25.5}, {26.5, 27.5}}}};
    ASSERT_THAT(test_array4(1).shape(), testing::ElementsAre(2, 2, 2));
    ASSERT_THAT(test_array4(1), testing::ElementsAre(20.5, 21.5, 22.5, 23.5,
                                                     24.5, 25.5, 26.5, 27.5));
    ASSERT_THAT(test_array4(1, 0).shape(), testing::ElementsAre(2, 2));
    ASSERT_THAT(test_array4(1, 0),
                testing::ElementsAre(20.5, 21.5, 22.5, 23.5));
    ASSERT_THAT(test_array4(1, 0, 1).shape(), testing::ElementsAre(2));
    ASSERT_THAT(test_array4(1, 0, 1), testing::ElementsAre(22.5, 23.5));
}
