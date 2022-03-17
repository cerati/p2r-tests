Kokkos_source="/global/homes/k/kkwok/PPS/p2r-tests/kokkos"

## CUDA backend
#cmake ./bin -DCMAKE_CXX_COMPILER=$Kokkos_source/bin/nvcc_wrapper \
#     -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_CONSTEXPR=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_ARCH_VOLTA70=ON -DKokkos_CXX_STANDARD=17
## CPU SERIAL
cmake ./bin -DCMAKE_CXX_COMPILER=g++       -DKokkos_ENABLE_SERIAL=ON 
 
## PThread
#cmake ./bin -DCMAKE_CXX_COMPILER=g++       -DKokkos_ENABLE_PTHREAD=ON 
#export OMP_PROC_BIND=spread
#export OMP_PLACES=threads
#cmake -B./bin -S./  -DKokkos_ENABLE_OPENMP=ON

## OMP TARGET 
#cmake ./bin   -DKokkos_ENABLE_OPENMPTARGET=ON -DKokkos_ARCH_VOLTA70=ON -DKokkos_CXX_STANDARD=c++17 -DCMAKE_CXX_COMPILER=$Kokkos_source/bin/nvcc_wrapper

cd bin
cmake --build ./
cd ../
