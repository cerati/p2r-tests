# CUDA backend
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#       -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_CUDA_COMPILER=nvcc\
#       -Dalpaka_ACC_GPU_CUDA_ENABLE=on -DCMAKE_CUDA_ARCHITECTURES=80 ..

## HIP backend
# JLAB config
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#       -DCMAKE_PREFIX_PATH=/soft/compilers/rocm/rocm-5.2.0/bin -Dalpaka_ACC_GPU_HIP_ENABLE=ON -DGPU_TARGETS=gfx906 ..

# JLSE config
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#       -DCMAKE_PREFIX_PATH=/soft/compilers/rocm/rocm-5.1.3/hip/\
#       -DCMAKE_CXX_COMPILER=hipcc -Dalpaka_CXX_STANDARD=17\
#       -Dalpaka_ACC_GPU_HIP_ENABLE=ON -DGPU_TARGETS=gfx908 ..
## HIP with nvcc compiler -- DOES NOT WORK --
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#       -DCMAKE_PREFIX_PATH=/soft/compilers/rocm/rocm-5.1.3/hip/\
#       -DCMAKE_CXX_COMPILER=nvcc \
#       -Dalpaka_ACC_GPU_HIP_ENABLE=ON -DCMAKE_CUDA_ARCHITECTURES=80 ..

# SYCL backend
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#      -DCMAKE_CXX_COMPILER=/home/kkwok/sycl_workspace/llvm/build_intel/bin/clang++\
#      -Dalpaka_CXX_STANDARD=17   -Dalpaka_ACC_SYCL_ENABLE=ON -Dalpaka_SYCL_PLATFORM_ONEAPI=ON -Dalpaka_SYCL_ONEAPI_GPU=ON ..

# TBB backend 
cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
    -DCMAKE_CXX_COMPILER=g++  -DCMAKE_C_COMPILER=gcc -Dalpaka_ACC_CPU_B_TBB_T_SEQ_ENABLE=on\
#    -DTBB_DIR="~/PPS/pixeltrack-standalone/external/tbb/lib/cmake/TBB/" ..

# OMP5 host backend
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#    -DCMAKE_CXX_COMPILER=g++  -DOpenMP_CXX_VERSION=5 -Dalpaka_ACC_ANY_BT_OMP5_ENABLE=on\
#    -DCMAKE_CXX_FLAGS="-foffload=disable -fno-lto"\
#    -DALPAKA_CXX_STANDARD=17 ..\

#################Cori configs################################################
# C++ serial backend
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#    -DCMAKE_CXX_COMPILER=g++  -DCMAKE_C_COMPILER=gcc -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE=on -DALPAKA_CXX_STANDARD=17 ..

## C++ Threads backend
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#    -DCMAKE_CXX_COMPILER=g++  -DCMAKE_C_COMPILER=gcc -DALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE=on -DALPAKA_CXX_STANDARD=17 ..

## C++ BoostFiber backend
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#    -DCMAKE_CXX_COMPILER=g++  -DCMAKE_C_COMPILER=gcc -DALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE=on -DALPAKA_CXX_STANDARD=17 ..

## TBB backend
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#    -DCMAKE_CXX_COMPILER=g++  -DCMAKE_C_COMPILER=gcc -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE=on -DALPAKA_CXX_STANDARD=17\
#    -DTBB_DIR="~/PPS/pixeltrack-standalone/external/tbb/lib/cmake/TBB/" ..

## OMP2 blockthread backend
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#    -DCMAKE_CXX_COMPILER=g++  -DCMAKE_C_COMPILER=gcc -DALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE=on -DALPAKA_CXX_STANDARD=17 ..
## OMP2 grid block backend
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#    -DCMAKE_CXX_COMPILER=g++  -DCMAKE_C_COMPILER=gcc -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE=on -DALPAKA_CXX_STANDARD=17 ..

## OMP5 CUDA backend (GCC) (crashed on running)
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#    -DCMAKE_CXX_COMPILER=g++  -DCMAKE_C_COMPILER=gcc -DOpenMP_CXX_VERSION=5 -DALPAKA_ACC_ANY_BT_OMP5_ENABLE=on\
#    -DCMAKE_CXX_FLAGS="-foffload=nvptx-none -foffload=-lm -fno-lto"\
#    -DALPAKA_CXX_STANDARD=17 ..\

## OMP5 CUDA backend (clang) (failed to compile)
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#    -DCMAKE_CXX_COMPILER=clang  -DCMAKE_C_COMPILER=gcc -DOpenMP_CXX_VERSION=5 -DALPAKA_ACC_ANY_BT_OMP5_ENABLE=on\
#    -DCMAKE_CXX_FLAGS="-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -O2" \
#    -DCMAKE_EXE_LINKER_FLAGS="-fopenmp"\
#    -DALPAKA_CXX_STANDARD=17 ..\


## OMP5 host backend
#cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_ROOT=~/PPS/p2r-tests/alpaka/install/\
#    -DCMAKE_CXX_COMPILER=g++  -DCMAKE_C_COMPILER=gcc -DOpenMP_CXX_VERSION=5 -DALPAKA_ACC_ANY_BT_OMP5_ENABLE=on\
#    -DCMAKE_CXX_FLAGS="-foffload=disable -fno-lto"\
#    -DALPAKA_CXX_STANDARD=17 ..\

