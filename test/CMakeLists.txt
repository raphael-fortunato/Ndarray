file(GLOB TEST_FILES *.cpp)
# Add an executable for testing
add_executable(TestTensorLibpp ${TEST_FILES})

set(gtest_force_shared_crt on)
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.0
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)
# Link the main project's library to the tests
target_link_libraries(TestTensorLibpp PRIVATE gtest gtest_main gmock)

# Add test cases (if using a testing framework like Google Test, Catch2, etc.)
add_test(NAME TestTensorLibpp COMMAND TestTensorLibpp)
gtest_discover_tests(TestTensorLibpp)
