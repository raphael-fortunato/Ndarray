file(GLOB TEST_FILES *.cpp)
# Add an executable for testing
add_executable(TestNDArray ${TEST_FILES})

# Link the main project's library to the tests
target_link_libraries(TestNDArray PRIVATE NDArrayLib)

# Add test cases (if using a testing framework like Google Test, Catch2, etc.)
enable_testing()
add_test(NAME TestNDArray COMMAND TestNDArray)
