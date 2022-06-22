Kokkos_source="/global/homes/k/kkwok/PPS/p2r-tests/kokkos/"
Kokkos_serial="/global/homes/k/kkwok/PPS/kokkos_serial/"
## Kokkos ROOT
#cmake ../  -DKokkos_ENABLE_SERIAL=On -DKokkos_CXX_STANDARD=17  -DKokkos_ROOT=$Kokkos_serial/lib64/cmake/Kokkos

## CUDA backend
#cmake ../ -DCMAKE_CXX_COMPILER=$Kokkos_source/bin/nvcc_wrapper\
#     -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_CONSTEXPR=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_CXX_STANDARD=17

## Serial backend
#cmake ../  -DCMAKE_CXX_COMPILER=gcc -DKokkos_ENABLE_SERIAL=On -DKokkos_CXX_STANDARD=17 
#
## Pthread backend (require hwloc...)
#cmake ../  -DCMAKE_CXX_COMPILER=gcc -DKokkos_ENABLE_PTHREAD=On -DKokkos_CXX_STANDARD=17 -DKokkos_HWLOC_DIR=???

## OpenMP backend
#cmake ../  -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_OPENMP=On -DKokkos_CXX_STANDARD=17 --kokkos-threads=4
#cmake ../ -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_OPENMP=On -DKokkos_CXX_STANDARD=17 --Dkokkos-threads=16
#cmake ../ -DCMAKE_CXX_COMPILER=g++ -DKokkos_ENABLE_OPENMP=On -DKokkos_CXX_STANDARD=17 -Dkokkos-threads=16
cmake ../ -DCMAKE_CXX_COMPILER=icc -DKokkos_ENABLE_OPENMP=On -DKokkos_CXX_STANDARD=17 
