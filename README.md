# p2r-tests

## Peer-reviewed publication

Our results are published as "Exploring code portability solutions for HEP with a particle tracking test code", Front. Big Data, 22 October 2024, Sec. Big Data and AI in High Energy Physics, Volume 7 - 2024 (https://doi.org/10.3389/fdata.2024.1485344).

Versions from this repository used in the paper figures are:
* Figure 2,6,7:
	* CUDA: [src/propagate-tor-test_cuda_uvm.cu](https://github.com/cerati/p2r-tests/blob/v1.0/src/propagate-tor-test_cuda_uvm.cu)
	* Alpaka: [p2r_alpaka/src/propagate-tor-test_alpaka_local.cpp](https://github.com/cerati/p2r-tests/blob/v1.0/p2r_alpaka/src/propagate-tor-test_alpaka_local.cpp)
	* Kokkos: [src/propagate-tor-test_kokkos_local_tvr.cpp](https://github.com/cerati/p2r-tests/blob/v1.0/src/propagate-tor-test_kokkos_local_tvr.cpp)
	* SYCL: [src/propagate-tor-test_sycl.cpp](https://github.com/cerati/p2r-tests/blob/v1.0/src/propagate-tor-test_sycl.cpp)
	* stdpar: [src/propagate-tor-test_pstl.cpp](https://github.com/cerati/p2r-tests/blob/v1.0/src/propagate-tor-test_pstl.cpp)
	* TBB: [src/propagate-tor-test_tbb.cpp](https://github.com/cerati/p2r-tests/blob/v1.0/src/propagate-tor-test_tbb.cpp)
	* OpenACC: [src/propagate-tor-test_acc.v3.cpp](https://github.com/cerati/p2r-tests/blob/v1.0/src/propagate-tor-test_acc.v3.cpp)
	* HIP: [src/propagate-tor-test_hip_local.cu](https://github.com/cerati/p2r-tests/blob/v1.0/src/propagate-tor-test_hip_local.cu)

Please find them as used in the paper at the tagged version of this repository: https://github.com/cerati/p2r-tests/releases/tag/v1.0. 

## Build instructions on JLSE

### Kokkos versions with CUDA backend:
 - Clone Kokkos as a sub-module inside the main directory of the `p2r-tests` repo:
 
   `git clone git@github.com:kokkos/kokkos.git`
   
   Set the environment var `Kokkos_source` to the kokkos directory
   
 - Load CUDA modules:
  
   `module loda cuda/11.6.2`

 - Configure CMake with: 
 ```
 cd ./p2r-tests/bin
 cmake ../ -DCMAKE_CXX_COMPILER=$Kokkos_source/bin/nvcc_wrapper\
     -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_CONSTEXPR=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_CXX_STANDARD=17 -DKokkos_ARCH_AMPERE80=On
 ```
 for A-100 GPU architecture
     
 - Build with `make`
 - Run with `./p2r_kokkos`
 - For other backends, see `bin/config.sh` for examples
 
## instructions to compile and run p2r on apollo@cs.uoregon.edu

```
module load intel

export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/packages/intel/20/compilers_and_libraries_2020.1.217/linux/tbb/lib/intel64_lin/gcc4.8/

icc -Wall -I. -O3 -fopenmp -march=native -xHost -qopt-zmm-usage=high propagate-tor-test_tbb.cpp -I/packages/intel/20/compilers_and_libraries/linux/tbb/include/ -L/packages/intel/20/compilers_and_libraries_2020.1.217/linux/tbb/lib/intel64_lin/gcc4.8/ -Wl,-rpath,/lib -ltbb -o propagate-tor-test.exe
```

add the following to create detailed optimization report: `-qopt-report=5`

```./propagate-tor-test.exe```

## instructions to compile and run p2r on lnx7188.classe.cornell.edu

```
source /opt/intel/oneapi/setvars.sh
source /cvmfs/cms.cern.ch/slc7_amd64_gcc820/lcg/root/6.18.04-bcolbf/etc/profile.d/init.sh
export TBB_GCC=/cvmfs/cms.cern.ch/slc7_amd64_gcc820/external/tbb/2019_U9
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBJPEG_TURBO_ROOT/lib64

icc -Wall -I. -O3 -fopenmp -march=native -xHost -qopt-zmm-usage=high src/propagate-tor-test_tbb.cpp -I$TBBROOT/include/ -L$TBBROOT/lib/intel64/gcc4.8/ -Wl,-rpath,/lib -ltbb -o propagate-tor-test.exe

./propagate-tor-test.exe
```

## instructions on cori
```
module load intel
module load tbb
#Build and run once with icc as compiler
python build.py -t tbb -c icc -v
```
Example commands:
```
#print out compile command
python build.py -t tbb -c icc -v --dryRun
#build and scan with multiple threads
python build.py -t tbb -c icc -v --nthreads 1,2,3,4,5
#Scan for two compilers with multiple threads
python build.py -t tbb -c icc,gcc -v --nthreads 1,2,3,4,5
#Append results to the same result json (Default is to skip existing scan points)
python build.py -t tbb -c icc -v --nthreads 1,2,3,4,5 --append
```
To run the `CUDA` version on cori:
```
#load the module once
module load cgpu
module load cuda
#Connect to a GPU node:
alloc -A m2845 -C gpu -N 1 --gres=gpu:1 -t 2:00:00 --exclusive
#Example command:
python build.py -t cuda --num_streams 1 --bsize 1 -v
```

### CUDA versions

There are 3 different versions of CUDA implementations, with different indexing scheme and kernel launch patterns.
For details of how the 3 implementaion differs, see slides [here](https://github.com/kakwok/p2r-tests/blob/main/slides/p2z-slides_mar30.pdf)

`cuda`: Always run with `bsize=1`. Kernels are launched in 1D blocks with a constant threads per block inside. 
```
Blocks per grid = (nevts * nTrks) / threads_per_block 
Threads_per_block = const
```
Example command:
```
python build.py -t cuda --num_streams 1 --bsize 1 -v --nevts 1 --nlayer 2 --ntrks 32,64,128,256
```

`cuda_v2`: `bsize` is set to `ntrks` in the implementation. Kernels are launched in 1D blocks with: 
```
Blocks per grid = nevts  
Threads_per_block = ntrks 
```
On a V100 GPU, `ntrks` cannot exceed `300`. Example command:
```
python build.py -t cuda_v2 --num_streams 1 -v --nevts 1 --nlayer 20 --ntrks 32,64,128,256
```

`cuda_v3`: Follows `p2z CUDA V2` conventions, default with `bsize=128`. Kernels are launched in 2D blocks.

Example command:
```
python build.py -t cuda_v3 --num_streams 1  -v --ntrks 9600 --nevts 100 --nlayer 20 --threadsperblockx 16 --threadsperblocky 2
python build.py -t cuda_v4 --num_streams 1  -v --noH2D --noD2H --dryRun
```
### PSTL version

```
module load nvhpc/21.7
module load gcc/9.3.0 # needed for gcc  version
module load tbb       # needed for gcc  version
``` 

Example command:
```
python build.py -t pstl -v -c nvc++,nvc++_x86,gcc --dryRun
```
