Kokkos_source="/global/homes/k/kkwok/PPS/p2r-tests/kokkos"


cmake ./bin -DCMAKE_CXX_COMPILER=$Kokkos_source/bin/nvcc_wrapper \
     -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_CONSTEXPR=On -DKokkos_ENABLE_CUDA_LAMBDA=On

cd bin
cmake --build ./
cd ../
