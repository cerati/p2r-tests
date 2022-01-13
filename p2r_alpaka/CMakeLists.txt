# Required CMake version.

cmake_minimum_required(VERSION 3.15)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.

set(_TARGET_NAME p2r_alpaka)

project(${_TARGET_NAME})

#-------------------------------------------------------------------------------
# Find alpaka.

find_package(alpaka REQUIRED)

#-------------------------------------------------------------------------------
# Add executable.

set(CMAKE_CUDA_ARCHITECTURES 70)
alpaka_add_executable(
    ${_TARGET_NAME}
    src/propagate-tor-test_alpaka.cpp)
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC alpaka::alpaka)
