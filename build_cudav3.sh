
nvcc -arch=sm_75 -Iinclude -std=c++17 -maxrregcount=64 -g -lineinfo -o propagate-tor-test_cuda_v3_nvcc src/propagate-tor-test_cuda_v3.cu -Dntrks=9600 -Dnevts=100 -DNITER=5 -Dbsize=128 -Dnlayer=20 -Dnthreads=1 -Dnum_streams=1 -Dthreadsperblock=1000 -Dthreadsperblockx=16 -Dthreadsperblocky=2 -Dblockspergrid=40

