nvcc -arch=sm_86 -O3 --extended-lambda --expt-relaxed-constexpr --default-stream per-thread -std=c++17 ./propagate-tor-test_cuda_upd13.cu -L -lcudart   -o ./"propagate_nvcc_cuda"

