Kokkos_source="/home/kkwok/PPS/p2r-tests/kokkos"
## Kokkos ROOT
#cmake ../  -DKokkos_ENABLE_SERIAL=On -DKokkos_CXX_STANDARD=17  -DKokkos_ROOT=$Kokkos_serial/lib64/cmake/Kokkos

## CUDA backend
#cmake ../ -DCMAKE_CXX_COMPILER=$Kokkos_source/bin/nvcc_wrapper -DCMAKE_CXX_FLAGS="-lineinfo"\
#     -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_CONSTEXPR=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_CXX_STANDARD=17 -DKokkos_ARCH_VOLTA70=On -DCMAKE_BUIL_TYPE=Debug
     #-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_CONSTEXPR=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_CXX_STANDARD=17 -DKokkos_ARCH_AMPERE80=On

## HIP backend
#cmake ../ -DCMAKE_CXX_COMPILER=/soft/compilers/rocm/rocm-5.2.0/hip/bin/hipcc\
#          -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_VEGA908=On -DCMAKE_CXX_STANDARD=17

## SYCL backend
#cmake ../ -DCMAKE_CXX_COMPILER=dpcpp  -DKokkos_ENABLE_SYCL=ON -DCMAKE_CXX_STANDARD=17
#export KOKKOS_HOME=/soft/restricted/CNDA/kokkos/sycl_intel/2022.06.30.002/eng-compiler/jit
#cmake ../ -DCMAKE_CXX_COMPILER=dpcpp  -DKokkos_ENABLE_SYCL=ON -DKokkos_ARCH_INTEL_GEN=ON  -DCMAKE_CXX_STANDARD=17
# Serial backend
#cmake ../  -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_SERIAL=On -DKokkos_CXX_STANDARD=17 
#
## Threads backend
#cmake ../  -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_THREADS=On -DKokkos_CXX_STANDARD=17 

## Pthread backend (require hwloc...)
#cmake ../  -DCMAKE_CXX_COMPILER=gcc -DKokkos_ENABLE_PTHREAD=On -DKokkos_CXX_STANDARD=17 -DKokkos_HWLOC_DIR=???

## OpenMP backend
#cmake ../  -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_OPENMP=On -DKokkos_CXX_STANDARD=17 --kokkos-threads=4
#cmake ../ -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_OPENMP=On -DKokkos_CXX_STANDARD=17 --Dkokkos-threads=16
#cmake ../ -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_OPENMP=On -DKokkos_CXX_STANDARD=17 -Dkokkos-threads=32
cmake ../ -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_OPENMP=On -DKokkos_CXX_STANDARD=17 
#cmake ../ -DCMAKE_CXX_COMPILER=icc -DKokkos_ENABLE_OPENMP=On -DKokkos_CXX_STANDARD=17 
