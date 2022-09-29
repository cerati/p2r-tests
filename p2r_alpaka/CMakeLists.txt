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
# Set common var

set(alpaka_CXX_STANDARD 17)
#-------------------------------------------------------------------------------
# Add executable.
add_compile_definitions(EXCLUDE_H2D_TRANSFER=1)
add_compile_definitions(EXCLUDE_D2H_TRANSFER=1)
if(alpaka_ACC_GPU_CUDA_ENABLE)
    #set(CMAKE_CUDA_ARCHITECTURES 70)
    #add_compile_definitions(num_streams=1)
    #set(CMAKE_BUILD_TYPE Debug)
    #set(CMAKE_BUILD_TYPE RelWithDebInfo)
    add_compile_options(-std=c++17)
    add_compile_options(-maxrregcount=64)
    add_compile_options(--use_fast_math)
##HIP flags
elseif (alpaka_ACC_GPU_HIP_ENABLE)
    add_compile_options(-std=c++17)
    add_compile_options(-ffast-math)
    add_compile_options(-O3)
##CPU flags
else()
    add_compile_options(-ffast-math)
    add_compile_options(-march=native)
    add_compile_options(-O3)
    #add_compile_options(-fopt-info-vec-missed)
    add_compile_options(-fopenmp-simd)
endif()

alpaka_add_executable(
    ${_TARGET_NAME}
#    src/propagate-tor-test_alpaka_local.cpp)
    src/propagate-tor-test_alpaka.cpp)

target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC alpaka::alpaka
)

