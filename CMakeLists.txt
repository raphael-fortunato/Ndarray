# Minimum required version of CMake
cmake_minimum_required(VERSION 3.14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
# Project name and version
project(TensorLibpp VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the include directories
include_directories(include)

file(GLOB SRC_FILES src/*.cpp)
add_library(TensorLibpp STATIC ${SRC_FILES})

# Add the tests directory (optional, if you have tests)
add_subdirectory(test)
