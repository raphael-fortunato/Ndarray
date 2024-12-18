file(GLOB TEST_FILES *.cpp)
# Add an executable for testing
add_executable(TestNDArray ${TEST_FILES})

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
target_link_libraries(TestNDArray PRIVATE gtest gtest_main)
target_link_libraries(TestNDArray PRIVATE NDArray)

# Add test cases (if using a testing framework like Google Test, Catch2, etc.)
enable_testing()
add_test(NAME TestNDArray COMMAND TestNDArray)
gtest_discover_tests(TestNDArray)
