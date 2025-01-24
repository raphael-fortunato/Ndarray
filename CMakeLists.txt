# Minimum required version of CMake
cmake_minimum_required(VERSION 3.14)
# Project name and version
project(TensorLibpp VERSION 1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -Wno-deprecated-register -O0 -g -fprofile-arcs -ftest-coverage ${CMAKE_CXX_FLAGS_DEBUG}")
option(CODE_COVERAGE "Enable coverage reporting" OFF)
option(CODE_TEST "Enable tests" OFF)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the include directories
include_directories(include)

if(CODE_TEST)
    message(STATUS "Test code enabled")
    add_subdirectory(test)
    enable_testing()
endif()

if(CODE_COVERAGE)
if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "Coverage enabled")
    include(CodeCoverage)
    append_coverage_compiler_flags()
    target_link_libraries(TestTensorLibpp PRIVATE gcov)
    set(COVERAGE_EXCLUDES
    "${PROJECT_SOURCE_DIR}/test/*"           # Exclude all test files
    "googlemock/*"  # Exclude gtest source files
    "googletest/*"  # Exclude gtest source files
    "/usr/include/*"  # Exclude gtest source files
    )
    setup_target_for_coverage_lcov(
        NAME coverage
        EXECUTABLE TestTensorLibpp
    )
endif()
endif()
