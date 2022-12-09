/*
nvc++ -O2 -std=c++23 --gcc-toolchain=path-to-gnu-compiler -stdpar=gpu -gpu=cc86 -gpu=managed -gpu=fma -gpu=fastmath -gpu=autocollapse -gpu=loadcache:L1 -gpu=unroll ./src/propagate-tor-test_cuda_hybrid_native.cpp  -o ./propagate_nvcpp_cuda -Dntrks=8192 -Dnevts=100 -DNITER=5 -Dbsize=1 -Dnlayer=20

nvc++ -O2 -std=c++23 --gcc-toolchain=path-to-gnu-compiler -stdpar=multicore -gpu=cc86 -gpu=managed -gpu=fma -gpu=fastmath -gpu=autocollapse -gpu=loadcache:L1 -gpu=unroll ./src/propagate-tor-test_cuda_hybrid_cpp23.cpp  -o ./propagate_nvcpp_cuda -Dntrks=8192 -Dnevts=100 -DNITER=5 -Dbsize=1 -Dnlayer=20 -Dstdpar_launcher

nvc++ -O2 -std=c++20 --gcc-toolchain=path-to-gnu-compiler -stdpar=multicore ./src/propagate-tor-test_cuda_hybrid_native.cpp  -o ./propagate_nvcpp_x86 -Dntrks=8192 -Dnevts=100 -DNITER=5 -Dbsize=32 -Dnlayer=20

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>

#include <concepts> 
#include <ranges>
#include <type_traits>

#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>
#include <execution>
#include <random>

#include <experimental/mdspan>
//
#include <experimental/stdexec/execution.hpp>
#include <experimental/exec/on.hpp>

#include <experimental/nvexec/detail/throw_on_cuda_error.cuh>
#include <experimental/nvexec/stream_context.cuh>
#include <experimental/nvexec/multi_gpu_context.cuh>

namespace stdex = std::experimental;

#ifndef bsize
#define bsize 32
#endif

#ifndef ntrks
#define ntrks 8192
#endif

#define nb    (ntrks/bsize)

#ifndef nevts
#define nevts 100
#endif
#define smear 0.00001

#ifndef NITER
#define NITER 5
#endif

#ifndef nlayer
#define nlayer 20
#endif

#ifndef num_streams
#define num_streams 1
#endif

#ifndef threadsperblock
#define threadsperblock 32
#endif

#ifdef __NVCOMPILER_CUDA__

constexpr bool enable_cuda             = true;

#ifdef cuda_launcher

//#include <nv/target>
#define __cuda_kernel__ __global__

constexpr bool enable_cuda_launcher    = true;
constexpr bool enable_stdexec_launcher = false;
static int threads_per_block           = threadsperblock;

#elif defined (stdexec_launcher)

#define __cuda_kernel__
constexpr bool enable_cuda_launcher    = false;
constexpr bool enable_stdexec_launcher = true;

#else //running cuda backend with stdpar launcher

#define __cuda_kernel__
constexpr bool enable_cuda_launcher    = false;
constexpr bool enable_stdexec_launcher = false;

#endif

#else // non-cuda targets

#define __cuda_kernel__
constexpr bool enable_cuda             = false;
constexpr bool enable_cuda_launcher    = false;

#endif

constexpr int host_id = -1; /*cudaCpuDeviceId*/

#ifdef include_data
constexpr bool include_data_transfer = true;
#else
constexpr bool include_data_transfer = false;
#endif

static int nstreams           = num_streams;//we have only one stream, though

using indx_type = int;

template<typename data_tp, int N, int B> using Right2DView        = stdex::mdspan<data_tp, stdex::extents<indx_type, N, B>, stdex::layout_right, stdex::default_accessor<data_tp>>;
template<typename data_tp, int M, int N, int B> using Right3DView = stdex::mdspan<data_tp, stdex::extents<indx_type, M, N, B>, stdex::layout_right, stdex::default_accessor<data_tp>>;
template<typename data_tp, int N, int B> using Right2DCView       = stdex::mdspan<const data_tp, stdex::extents<indx_type, N, B>, stdex::layout_right, stdex::default_accessor<const data_tp>>;


template <bool is_cuda_target>
concept CudaCompute = is_cuda_target == true;

//Collection of helper methods
//General helper routines:
template <bool is_cuda_target>
requires CudaCompute<is_cuda_target>
void p2r_check_error(){
  //	
  auto error = cudaGetLastError();
  if(error != cudaSuccess) std::cout << "Error detected, error " << error << std::endl;
  //
  return;
}

template <bool is_cuda_target>
void p2r_check_error(){
  return;
}

template <bool is_cuda_target>
requires CudaCompute<is_cuda_target>
int p2r_get_compute_device_id(){
  int dev = -1;
  cudaGetDevice(&dev);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  return dev;
}

//default version:
template <bool is_cuda_target>
int p2r_get_compute_device_id(){
  return 0;
}

template <bool is_cuda_target>
requires CudaCompute<is_cuda_target>
void p2r_set_compute_device(const int dev){
  cudaSetDevice(dev);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  return;
}

//default version:
template <bool is_cuda_target>
void p2r_set_compute_device(const int dev){
  return;
}

template <bool is_cuda_target>
requires CudaCompute<is_cuda_target>
decltype(auto) p2r_get_streams(const int n){
  std::vector<cudaStream_t> streams;
  streams.reserve(n);
  for (int i = 0; i < n; i++) {  
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams.push_back(stream);
  }
  return streams;
}

template <bool is_cuda_target>
decltype(auto) p2r_get_streams(const int n){
  if(n > 1) std::cout << "Number of compute streams is not supported : " << n << std::endl; 
  std::vector<int> streams = {0};
  return streams;
}

//CUDA specialized version:
template <typename data_tp, bool is_cuda_target, typename stream_t, bool is_sync = false>
requires CudaCompute<is_cuda_target>
void p2r_prefetch(std::vector<data_tp> &v, int devId, stream_t stream) {
  cudaMemPrefetchAsync(v.data(), v.size() * sizeof(data_tp), devId, stream);
  //
  if constexpr (is_sync) {cudaStreamSynchronize(stream);}

  return;
}

//Default implementation
template <typename data_tp, bool is_cuda_target, typename stream_t, bool is_sync = false>
void p2r_prefetch(std::vector<data_tp> &v, int dev_id, stream_t stream) {
  return;
}


//CUDA specialized version:
template <bool is_cuda_target>
requires CudaCompute<is_cuda_target>
void p2r_wait() { 
  cudaDeviceSynchronize(); 
  return; 
}

template <bool is_cuda_target>
void p2r_wait() { 
  return; 
}

//used only in cuda 
template <bool is_cuda_target, bool is_verbose = true>
requires CudaCompute<is_cuda_target>
void info(int device) {
  cudaDeviceProp deviceProp;

  int driver_version;
  cudaDriverGetVersion(&driver_version);
  if constexpr (is_verbose) { std::cout << "CUDA Driver version = " << driver_version << std::endl;}

  int runtime_version;
  cudaRuntimeGetVersion(&runtime_version);
  if constexpr (is_verbose) { std::cout << "CUDA Runtime version = " << runtime_version << std::endl;}

  cudaGetDeviceProperties(&deviceProp, device);

  if constexpr (is_verbose) {
    printf("%d - name:                    %s\n", device, deviceProp.name);
    printf("%d - totalGlobalMem:          %lu bytes ( %.2f Gbytes)\n", device, deviceProp.totalGlobalMem,
                   deviceProp.totalGlobalMem / (float)(1024 * 1024 * 1024));
    printf("%d - sharedMemPerBlock:       %lu bytes ( %.2f Kbytes)\n", device, deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerBlock / (float)1024);
    printf("%d - regsPerBlock:            %d\n", device, deviceProp.regsPerBlock);
    printf("%d - warpSize:                %d\n", device, deviceProp.warpSize);
    printf("%d - memPitch:                %lu\n", device, deviceProp.memPitch);
    printf("%d - maxThreadsPerBlock:      %d\n", device, deviceProp.maxThreadsPerBlock);
    printf("%d - maxThreadsDim[0]:        %d\n", device, deviceProp.maxThreadsDim[0]);
    printf("%d - maxThreadsDim[1]:        %d\n", device, deviceProp.maxThreadsDim[1]);
    printf("%d - maxThreadsDim[2]:        %d\n", device, deviceProp.maxThreadsDim[2]);
    printf("%d - maxGridSize[0]:          %d\n", device, deviceProp.maxGridSize[0]);
    printf("%d - maxGridSize[1]:          %d\n", device, deviceProp.maxGridSize[1]);
    printf("%d - maxGridSize[2]:          %d\n", device, deviceProp.maxGridSize[2]);
    printf("%d - totalConstMem:           %lu bytes ( %.2f Kbytes)\n", device, deviceProp.totalConstMem,
                   deviceProp.totalConstMem / (float)1024);
    printf("%d - compute capability:      %d.%d\n", device, deviceProp.major, deviceProp.minor);
    printf("%d - deviceOverlap            %s\n", device, (deviceProp.deviceOverlap ? "true" : "false"));
    printf("%d - multiProcessorCount      %d\n", device, deviceProp.multiProcessorCount);
    printf("%d - kernelExecTimeoutEnabled %s\n", device,
                   (deviceProp.kernelExecTimeoutEnabled ? "true" : "false"));
    printf("%d - integrated               %s\n", device, (deviceProp.integrated ? "true" : "false"));
    printf("%d - canMapHostMemory         %s\n", device, (deviceProp.canMapHostMemory ? "true" : "false"));
    switch (deviceProp.computeMode) {
      case 0: printf("%d - computeMode              0: cudaComputeModeDefault\n", device); break;
      case 1: printf("%d - computeMode              1: cudaComputeModeExclusive\n", device); break;
      case 2: printf("%d - computeMode              2: cudaComputeModeProhibited\n", device); break;
      case 3: printf("%d - computeMode              3: cudaComputeModeExclusiveProcess\n", device); break;
      default: printf("Error: unknown deviceProp.computeMode."), exit(-1);
    }
    printf("%d - surfaceAlignment         %lu\n", device, deviceProp.surfaceAlignment);
    printf("%d - concurrentKernels        %s\n", device, (deviceProp.concurrentKernels ? "true" : "false"));
    printf("%d - ECCEnabled               %s\n", device, (deviceProp.ECCEnabled ? "true" : "false"));
    printf("%d - pciBusID                 %d\n", device, deviceProp.pciBusID);
    printf("%d - pciDeviceID              %d\n", device, deviceProp.pciDeviceID);
    printf("%d - pciDomainID              %d\n", device, deviceProp.pciDomainID);
    printf("%d - tccDriver                %s\n", device, (deviceProp.tccDriver ? "true" : "false"));

    switch (deviceProp.asyncEngineCount) {
      case 0: printf("%d - asyncEngineCount         1: host -> device only\n", device); break;
      case 1: printf("%d - asyncEngineCount         2: host <-> device\n", device); break;
      case 2: printf("%d - asyncEngineCount         0: not supported\n", device); break;
      default: printf("Error: unknown deviceProp.asyncEngineCount."), exit(-1);
    }
    printf("%d - unifiedAddressing        %s\n", device, (deviceProp.unifiedAddressing ? "true" : "false"));
    printf("%d - memoryClockRate          %d kilohertz\n", device, deviceProp.memoryClockRate);
    printf("%d - memoryBusWidth           %d bits\n", device, deviceProp.memoryBusWidth);
    printf("%d - l2CacheSize              %d bytes\n", device, deviceProp.l2CacheSize);
    printf("%d - maxThreadsPerMultiProcessor          %d\n\n", device, deviceProp.maxThreadsPerMultiProcessor);


  }

  p2r_check_error<is_cuda_target>();

  return;	  
}

template <bool is_cuda_target, bool is_verbose = true>
void info(int device) {
  return;
}


const std::array<size_t, 36> SymOffsets66{0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20};

struct ATRK {
  std::array<float,6> par;
  std::array<float,21> cov;
  int q;
};

struct AHIT {
  std::array<float,3> pos;
  std::array<float,6> cov;
};

constexpr int iparX     = 0;
constexpr int iparY     = 1;
constexpr int iparZ     = 2;
constexpr int iparIpt   = 3;
constexpr int iparPhi   = 4;
constexpr int iparTheta = 5;

template <typename T, int N, int bSize = 1>
struct MPNX {
   std::array<T,N*bSize> data;

   MPNX() = default;
   MPNX(const MPNX<T, N, bSize> &) = default;
   MPNX(MPNX<T, N, bSize> &&)      = default;
   
   //basic accessors
   constexpr T &operator[](const int i) { return data[i]; }
   constexpr const T &operator[](const int i) const { return data[i]; }

   constexpr int size() const { return N*bSize; }   
   //
   inline void load(MPNX<T, N, 1>& dst, const int b) const {
     Right2DCView<T, N, bSize> data_view{data.data()};
#pragma unroll
     for (int ip = 0;ip < data_view.extent(1);++ip) {   	
       dst.data[ip] = data_view(ip,b); 
     }
     
     return;
   }

   inline void save(const MPNX<T, N, 1>& src, const int b) {	   
     Right2DView<T, N, bSize> data_view{data.data()};	   
#pragma unroll
     for (int ip = 0;ip < data_view.extent(1);++ip) {    	
       data_view(ip, b) = src.data[ip]; 
     }

     return;
   }  

   auto operator=(const MPNX&) -> MPNX& = default;
   auto operator=(MPNX&&     ) -> MPNX& = default;
};


// internal data formats (coinside with external ones for x86):
template<int bSize = 1> using MP1I_    = MPNX<int,   1 , bSize>;
template<int bSize = 1> using MP1F_    = MPNX<float, 1 , bSize>;
template<int bSize = 1> using MP2F_    = MPNX<float, 2 , bSize>;
template<int bSize = 1> using MP3F_    = MPNX<float, 3 , bSize>;
template<int bSize = 1> using MP6F_    = MPNX<float, 6 , bSize>;
template<int bSize = 1> using MP2x2SF_ = MPNX<float, 3 , bSize>;
template<int bSize = 1> using MP3x3SF_ = MPNX<float, 6 , bSize>;
template<int bSize = 1> using MP6x6SF_ = MPNX<float, 21, bSize>;
template<int bSize = 1> using MP6x6F_  = MPNX<float, 36, bSize>;
template<int bSize = 1> using MP3x3_   = MPNX<float, 9 , bSize>;
template<int bSize = 1> using MP3x6_   = MPNX<float, 18, bSize>;

// external data formats:
using MP1I    = MPNX<int,   1 , bsize>;
using MP1F    = MPNX<float, 1 , bsize>;
using MP2F    = MPNX<float, 2 , bsize>;
using MP3F    = MPNX<float, 3 , bsize>;
using MP6F    = MPNX<float, 6 , bsize>;
using MP2x2SF = MPNX<float, 3 , bsize>;
using MP3x3SF = MPNX<float, 6 , bsize>;
using MP6x6SF = MPNX<float, 21, bsize>;
using MP6x6F  = MPNX<float, 36, bsize>;
using MP3x3   = MPNX<float, 9 , bsize>;
using MP3x6   = MPNX<float, 18, bsize>;

template <int N = 1>
struct MPTRK_ {
  MP6F_<N>    par;
  MP6x6SF_<N> cov;
  MP1I_<N>    q;
};

template <int N = 1>
struct MPHIT_ {
  MP3F_<N>    pos;
  MP3x3SF_<N> cov;
};

struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;

  MPTRK() = default;

  template<int S>
  inline decltype(auto) load(const int batch_id = 0) const{
  
    MPTRK_<S> dst;

    if constexpr (std::is_same<MP6F, MP6F_<S>>::value        
                  and std::is_same<MP6x6SF, MP6x6SF_<S>>::value
                  and std::is_same<MP1I, MP1I_<S>>::value)  { //just do a copy of the whole objects
      dst.par = this->par;
      dst.cov = this->cov;
      dst.q   = this->q;
      
    } else { //ok, do manual load of the batch component instead
      this->par.load(dst.par, batch_id);
      this->cov.load(dst.cov, batch_id);
      this->q.load(dst.q, batch_id);
    }//done
    
    return dst;  
  }
  
  template<int S>
  inline void save(MPTRK_<S> &src, const int batch_id = 0) {
  
    if constexpr (std::is_same<MP6F, MP6F_<S>>::value        
                  and std::is_same<MP6x6SF, MP6x6SF_<S>>::value
                  and std::is_same<MP1I, MP1I_<S>>::value) { //just do a copy of the whole objects
      this->par = src.par;
      this->cov = src.cov;
      this->q   = src.q;

    } else { //ok, do manual load of the batch component instead
      this->par.save(src.par, batch_id);
      this->cov.save(src.cov, batch_id);
      this->q.save(src.q, batch_id);
    }//done
    
    return;
  }
};

struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;
  //
  MPHIT() = default;

  template<int S>
  inline decltype(auto) load(const int batch_id = 0) const {
    MPHIT_<S> dst;
    
    if constexpr (std::is_same<MP3F, MP3F_<S>>::value        
                  and std::is_same<MP3x3SF, MP3x3SF_<S>>::value) { //just do a copy of the whole object
      dst.pos = this->pos;
      dst.cov = this->cov;
    } else { //ok, do manual load of the batch component instead
      this->pos.load(dst.pos, batch_id);
      this->cov.load(dst.cov, batch_id);
    }//done    
    
    return dst;
  }
};

///////////////////////////////////////
//Gen. utils

float randn(float mu, float sigma) {
  float U1, U2, W, mult;
  static float X1, X2;
  static int call = 0;
  if (call == 1) {
    call = !call;
    return (mu + sigma * (float) X2);
  } do {
    U1 = -1 + ((float) rand () / RAND_MAX) * 2;
    U2 = -1 + ((float) rand () / RAND_MAX) * 2;
    W = pow (U1, 2) + pow (U2, 2);
  }
  while (W >= 1 || W == 0); 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult; 
  call = !call; 
  return (mu + sigma * (float) X1);
}

void prepareTracks(std::vector<MPTRK> &trcks, ATRK &inputtrk) {
  //
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
	      //par
	      for (size_t ip=0;ip<6;++ip) {
	        trcks[ib + nb*ie].par.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.par[ip];
	      }
	      //cov, scale by factor 100
	      for (size_t ip=0;ip<21;++ip) {
	        trcks[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.cov[ip]*100;
	      }
	      //q
	      trcks[ib + nb*ie].q.data[it] = inputtrk.q;//can't really smear this or fit will be wrong
      }
    }
  }
  //
  return;
}

void prepareHits(std::vector<MPHIT> &hits, std::vector<AHIT>& inputhits) {
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t lay=0;lay<nlayer;++lay) {

    size_t mylay = lay;
    if (lay>=inputhits.size()) {
      // int wraplay = inputhits.size()/lay;
      exit(1);
    }
    AHIT& inputhit = inputhits[mylay];

    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        for (size_t it=0;it<bsize;++it) {
        	//pos
        	for (size_t ip=0;ip<3;++ip) {
        	  hits[lay+nlayer*(ib + nb*ie)].pos.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
        	  hits[lay+nlayer*(ib + nb*ie)].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.cov[ip];
        	}
        }
      }
    }
  }
  return;
}

//////////////////////////////////////////////////////////////////////////////////////
// Aux utils 
MPTRK* bTk(MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

const MPTRK* bTk(const MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

float q(const MP1I* bq, size_t it){
  return (*bq).data[it];
}
//
float par(const MP6F* bpars, size_t it, size_t ipar){
  return (*bpars).data[it + ipar*bsize];
}
float x    (const MP6F* bpars, size_t it){ return par(bpars, it, 0); }
float y    (const MP6F* bpars, size_t it){ return par(bpars, it, 1); }
float z    (const MP6F* bpars, size_t it){ return par(bpars, it, 2); }
float ipt  (const MP6F* bpars, size_t it){ return par(bpars, it, 3); }
float phi  (const MP6F* bpars, size_t it){ return par(bpars, it, 4); }
float theta(const MP6F* bpars, size_t it){ return par(bpars, it, 5); }
//
float par(const MPTRK* btracks, size_t it, size_t ipar){
  return par(&(*btracks).par,it,ipar);
}
float x    (const MPTRK* btracks, size_t it){ return par(btracks, it, 0); }
float y    (const MPTRK* btracks, size_t it){ return par(btracks, it, 1); }
float z    (const MPTRK* btracks, size_t it){ return par(btracks, it, 2); }
float ipt  (const MPTRK* btracks, size_t it){ return par(btracks, it, 3); }
float phi  (const MPTRK* btracks, size_t it){ return par(btracks, it, 4); }
float theta(const MPTRK* btracks, size_t it){ return par(btracks, it, 5); }
//
float par(const MPTRK* tracks, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPTRK* btracks = bTk(tracks, ev, ib);
  size_t it = tk % bsize;
  return par(btracks, it, ipar);
}
float x    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 0); }
float y    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 1); }
float z    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 2); }
float ipt  (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 3); }
float phi  (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 4); }
float theta(const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 5); }
//

const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib,size_t lay) {
return &(hits[lay + (ib*nlayer) +(ev*nlayer*nb)]);
}
//
float Pos(const MP3F* hpos, size_t it, size_t ipar){
  return (*hpos).data[it + ipar*bsize];
}
float x(const MP3F* hpos, size_t it)    { return Pos(hpos, it, 0); }
float y(const MP3F* hpos, size_t it)    { return Pos(hpos, it, 1); }
float z(const MP3F* hpos, size_t it)    { return Pos(hpos, it, 2); }
//
float Pos(const MPHIT* hits, size_t it, size_t ipar){
  return Pos(&(*hits).pos,it,ipar);
}
float x(const MPHIT* hits, size_t it)    { return Pos(hits, it, 0); }
float y(const MPHIT* hits, size_t it)    { return Pos(hits, it, 1); }
float z(const MPHIT* hits, size_t it)    { return Pos(hits, it, 2); }
//
float Pos(const MPHIT* hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPHIT* bhits = bHit(hits, ev, ib);
  size_t it = tk % bsize;
  return Pos(bhits,it,ipar);
}
float x(const MPHIT* hits, size_t ev, size_t tk)    { return Pos(hits, ev, tk, 0); }
float y(const MPHIT* hits, size_t ev, size_t tk)    { return Pos(hits, ev, tk, 1); }
float z(const MPHIT* hits, size_t ev, size_t tk)    { return Pos(hits, ev, tk, 2); }


////////////////////////////////////////////////////////////////////////
///MAIN compute kernels

template<size_t N = 1>
inline void MultHelixProp(const MP6x6F_<N> &a_, const MP6x6SF_<N> &b_, MP6x6F_<N> &c_) {//ok
  Right2DCView<float, 36, N> a{a_.data.data()};
  Right2DCView<float, 21, N> b{b_.data.data()};
  Right2DView<float, 36, N>  c{c_.data.data()};
#pragma unroll
  for (int it = 0;it < N; it++) {
    c( 0, it) = a( 0, it)*b( 0, it) + a( 1, it)*b( 1, it) + a( 3, it)*b( 6, it) + a( 4, it)*b(10, it);
    c( 1, it) = a( 0, it)*b( 1, it) + a( 1, it)*b( 2, it) + a( 3, it)*b( 7, it) + a( 4, it)*b(11, it);
    c( 2, it) = a( 0, it)*b( 3, it) + a( 1, it)*b( 4, it) + a( 3, it)*b( 8, it) + a( 4, it)*b(12, it);
    c( 3, it) = a( 0, it)*b( 6, it) + a( 1, it)*b( 7, it) + a( 3, it)*b( 9, it) + a( 4, it)*b(13, it);
    c( 4, it) = a( 0, it)*b(10, it) + a( 1, it)*b(11, it) + a( 3, it)*b(13, it) + a( 4, it)*b(14, it);
    c( 5, it) = a( 0, it)*b(15, it) + a( 1, it)*b(16, it) + a( 3, it)*b(18, it) + a( 4, it)*b(19, it);
    c( 6, it) = a( 6, it)*b( 0, it) + a( 7, it)*b( 1, it) + a( 9, it)*b( 6, it) + a(10, it)*b(10, it);
    c( 7, it) = a( 6, it)*b( 1, it) + a( 7, it)*b( 2, it) + a( 9, it)*b( 7, it) + a(10, it)*b(11, it);
    c( 8, it) = a( 6, it)*b( 3, it) + a( 7, it)*b( 4, it) + a( 9, it)*b( 8, it) + a(10, it)*b(12, it);
    c( 9, it) = a( 6, it)*b( 6, it) + a( 7, it)*b( 7, it) + a( 9, it)*b( 9, it) + a(10, it)*b(13, it);
    c(10, it) = a( 6, it)*b(10, it) + a( 7, it)*b(11, it) + a( 9, it)*b(13, it) + a(10, it)*b(14, it);
    c(11, it) = a( 6, it)*b(15, it) + a( 7, it)*b(16, it) + a( 9, it)*b(18, it) + a(10, it)*b(19, it);
    
    c(12, it) = a(12, it)*b( 0, it) + a(13, it)*b( 1, it) + b( 3, it) + a(15, it)*b( 6, it) + a(16, it)*b(10, it) + a(17, it)*b(15, it);
    c(13, it) = a(12, it)*b( 1, it) + a(13, it)*b( 2, it) + b( 4, it) + a(15, it)*b( 7, it) + a(16, it)*b(11, it) + a(17, it)*b(16, it);
    c(14, it) = a(12, it)*b( 3, it) + a(13, it)*b( 4, it) + b( 5, it) + a(15, it)*b( 8, it) + a(16, it)*b(12, it) + a(17, it)*b(17, it);
    c(15, it) = a(12, it)*b( 6, it) + a(13, it)*b( 7, it) + b( 8, it) + a(15, it)*b( 9, it) + a(16, it)*b(13, it) + a(17, it)*b(18, it);
    c(16, it) = a(12, it)*b(10, it) + a(13, it)*b(11, it) + b(12, it) + a(15, it)*b(13, it) + a(16, it)*b(14, it) + a(17, it)*b(19, it);
    c(17, it) = a(12, it)*b(15, it) + a(13, it)*b(16, it) + b(17, it) + a(15, it)*b(18, it) + a(16, it)*b(19, it) + a(17, it)*b(20, it);
    
    c(18, it) = a(18, it)*b( 0, it) + a(19, it)*b( 1, it) + a(21, it)*b( 6, it) + a(22, it)*b(10, it);
    c(19, it) = a(18, it)*b( 1, it) + a(19, it)*b( 2, it) + a(21, it)*b( 7, it) + a(22, it)*b(11, it);
    c(20, it) = a(18, it)*b( 3, it) + a(19, it)*b( 4, it) + a(21, it)*b( 8, it) + a(22, it)*b(12, it);
    c(21, it) = a(18, it)*b( 6, it) + a(19, it)*b( 7, it) + a(21, it)*b( 9, it) + a(22, it)*b(13, it);
    c(22, it) = a(18, it)*b(10, it) + a(19, it)*b(11, it) + a(21, it)*b(13, it) + a(22, it)*b(14, it);
    c(23, it) = a(18, it)*b(15, it) + a(19, it)*b(16, it) + a(21, it)*b(18, it) + a(22, it)*b(19, it);
    c(24, it) = a(24, it)*b( 0, it) + a(25, it)*b( 1, it) + a(27, it)*b( 6, it) + a(28, it)*b(10, it);
    c(25, it) = a(24, it)*b( 1, it) + a(25, it)*b( 2, it) + a(27, it)*b( 7, it) + a(28, it)*b(11, it);
    c(26, it) = a(24, it)*b( 3, it) + a(25, it)*b( 4, it) + a(27, it)*b( 8, it) + a(28, it)*b(12, it);
    c(27, it) = a(24, it)*b( 6, it) + a(25, it)*b( 7, it) + a(27, it)*b( 9, it) + a(28, it)*b(13, it);
    c(28, it) = a(24, it)*b(10, it) + a(25, it)*b(11, it) + a(27, it)*b(13, it) + a(28, it)*b(14, it);
    c(29, it) = a(24, it)*b(15, it) + a(25, it)*b(16, it) + a(27, it)*b(18, it) + a(28, it)*b(19, it);
    c(30, it) = b(15, it);
    c(31, it) = b(16, it);
    c(32, it) = b(17, it);
    c(33, it) = b(18, it);
    c(34, it) = b(19, it);
    c(35, it) = b(20, it);  
  }
  return;
}

template<size_t N = 1>
inline void MultHelixPropTransp(const MP6x6F_<N> &a_, const MP6x6F_<N> &b_, MP6x6SF_<N> &c_) {//
  Right2DCView<float, 36, N> a{a_.data.data()};
  Right2DCView<float, 36, N> b{b_.data.data()};
  Right2DView<float, 21, N>  c{c_.data.data()};
#pragma unroll
  for (int it = 0;it < N; it++) {
    c( 0, it) = b( 0, it)*a( 0, it) + b( 1, it)*a( 1, it) + b( 3, it)*a( 3, it) + b( 4, it)*a( 4, it);
    c( 1, it) = b( 6, it)*a( 0, it) + b( 7, it)*a( 1, it) + b( 9, it)*a( 3, it) + b(10, it)*a( 4, it);
    c( 2, it) = b( 6, it)*a( 6, it) + b( 7, it)*a( 7, it) + b( 9, it)*a( 9, it) + b(10, it)*a(10, it);
    c( 3, it) = b(12, it)*a( 0, it) + b(13, it)*a( 1, it) + b(15, it)*a( 3, it) + b(16, it)*a( 4, it);
    c( 4, it) = b(12, it)*a( 6, it) + b(13, it)*a( 7, it) + b(15, it)*a( 9, it) + b(16, it)*a(10, it);
    c( 5, it) = b(12, it)*a(12, it) + b(13, it)*a(13, it) + b(14, it) + b(15, it)*a(15, it) + b(16, it)*a(16, it) + b(17, it)*a(17, it);
    c( 6, it) = b(18, it)*a( 0, it) + b(19, it)*a( 1, it) + b(21, it)*a( 3, it) + b(22, it)*a( 4, it);
    c( 7, it) = b(18, it)*a( 6, it) + b(19, it)*a( 7, it) + b(21, it)*a( 9, it) + b(22, it)*a(10, it);
    c( 8, it) = b(18, it)*a(12, it) + b(19, it)*a(13, it) + b(20, it) + b(21, it)*a(15, it) + b(22, it)*a(16, it) + b(23, it)*a(17, it);
    c( 9, it) = b(18, it)*a(18, it) + b(19, it)*a(19, it) + b(21, it)*a(21, it) + b(22, it)*a(22, it);
    c(10, it) = b(24, it)*a( 0, it) + b(25, it)*a( 1, it) + b(27, it)*a( 3, it) + b(28, it)*a( 4, it);
    c(11, it) = b(24, it)*a( 6, it) + b(25, it)*a( 7, it) + b(27, it)*a( 9, it) + b(28, it)*a(10, it);
    c(12, it) = b(24, it)*a(12, it) + b(25, it)*a(13, it) + b(26, it) + b(27, it)*a(15, it) + b(28, it)*a(16, it) + b(29, it)*a(17, it);
    c(13, it) = b(24, it)*a(18, it) + b(25, it)*a(19, it) + b(27, it)*a(21, it) + b(28, it)*a(22, it);
    c(14, it) = b(24, it)*a(24, it) + b(25, it)*a(25, it) + b(27, it)*a(27, it) + b(28, it)*a(28, it);
    c(15, it) = b(30, it)*a( 0, it) + b(31, it)*a( 1, it) + b(33, it)*a( 3, it) + b(34, it)*a( 4, it);
    c(16, it) = b(30, it)*a( 6, it) + b(31, it)*a( 7, it) + b(33, it)*a( 9, it) + b(34, it)*a(10, it);
    c(17, it) = b(30, it)*a(12, it) + b(31, it)*a(13, it) + b(32, it) + b(33, it)*a(15, it) + b(34, it)*a(16, it) + b(35, it)*a(17, it);
    c(18, it) = b(30, it)*a(18, it) + b(31, it)*a(19, it) + b(33, it)*a(21, it) + b(34, it)*a(22, it);
    c(19, it) = b(30, it)*a(24, it) + b(31, it)*a(25, it) + b(33, it)*a(27, it) + b(34, it)*a(28, it);
    c(20, it) = b(35, it);    
  }
  return;  
}

auto hipo = [](const float x, const float y) {return std::sqrt(x*x + y*y);};

template <size_t N = 1>
void KalmanUpdate(MP6x6SF_<N> &trkErr_, MP6F_<N> &inPar_, const MP3x3SF_<N> &hitErr_, const MP3F_<N> &msP_){
  //	  
  Right2DView<float,21, N>   trkErr{trkErr_.data.data()};
  Right2DView<float, 6, N>   inPar{inPar_.data.data()};
  //
  Right2DCView<float, 6, N>   hitErr{hitErr_.data.data()};
  Right2DCView<float, 3, N>   msP{msP_.data.data()};
  
  MP1F_<N>    rotT00;
  MP1F_<N>    rotT01;
  MP2x2SF_<N> resErr_loc_;
  //MP3x3SF resErr_glo;
  Right2DView<float, 3, N> resErr_loc{resErr_loc_.data.data()};
 
  for (size_t it = 0;it < N; ++it) {   
    const auto msPX = msP(iparX, it);
    const auto msPY = msP(iparY, it);
    const auto inParX = inPar(iparX, it);
    const auto inParY = inPar(iparY, it);          
  
    const auto r = hipo(msPX, msPY);
    rotT00[it] = -(msPY + inParY) / (2*r);
    rotT01[it] =  (msPX + inParX) / (2*r);    
    
    resErr_loc( 0, it) = (rotT00[it]*(trkErr(0, it) + hitErr(0, it)) +
                                    rotT01[it]*(trkErr(1, it) + hitErr(1, it)))*rotT00[it] +
                                   (rotT00[it]*(trkErr(1, it) + hitErr(1, it)) +
                                    rotT01[it]*(trkErr(2, it) + hitErr(2, it)))*rotT01[it];
    resErr_loc( 1, it) = (trkErr(3, it) + hitErr(3, it))*rotT00[it] +
                                   (trkErr(4, it) + hitErr(4, it))*rotT01[it];
    resErr_loc( 2, it) = (trkErr(5, it) + hitErr(5, it));
  }
  
  for (size_t it=0;it<N;++it) {
  
    const double det = (double)resErr_loc(0, it) * resErr_loc(2, it) -
                       (double)resErr_loc(1, it) * resErr_loc(1, it);
    const float s   = 1.f / det;
    const float tmp = s * resErr_loc(2, it);
    resErr_loc(1, it) *= -s;
    resErr_loc(2, it)  = s * resErr_loc(0, it);
    resErr_loc(0, it)  = tmp; 
  }     
  
  MP3x6_<N> kGain_;
  Right2DView<float, 18, N> kGain{kGain_.data.data()};
  
#pragma omp simd
  for (size_t it=0; it<N; ++it) {
    const auto t00 = rotT00[it]*resErr_loc( 0, it);
    const auto t01 = rotT01[it]*resErr_loc( 0, it);
    const auto t10 = rotT00[it]*resErr_loc( 1, it);
    const auto t11 = rotT01[it]*resErr_loc( 1, it);

    kGain( 0, it) = trkErr( 0, it)*t00 + trkErr( 1, it)*t01 +
	                        trkErr( 3, it)*resErr_loc( 1, it);
    kGain( 1, it) = trkErr( 0, it)*t10 + trkErr( 1, it)*t11 +
	                        trkErr( 3, it)*resErr_loc( 2, it);
    kGain( 2, it) = 0.f;
    kGain( 3, it) = trkErr( 1, it)*t00 + trkErr( 2, it)*t01 +
	                        trkErr( 4, it)*resErr_loc( 1, it);
    kGain( 4, it) = trkErr( 1, it)*t10 + trkErr( 2, it)*t11 +
	                        trkErr( 4, it)*resErr_loc( 2, it);
    kGain( 5, it) = 0.f;
    kGain( 6, it) = trkErr( 3, it)*t00 + trkErr( 4, it)*t01 +
	                        trkErr( 5, it)*resErr_loc( 1, it);
    kGain( 7, it) = trkErr( 3, it)*t10 + trkErr( 4, it)*t11 +
	                        trkErr( 5, it)*resErr_loc( 2, it);
    kGain( 8, it) = 0.f;
    kGain( 9, it) = trkErr( 6, it)*t00 + trkErr( 7, it)*t01 +
	                        trkErr( 8, it)*resErr_loc( 1, it);
    kGain(10, it) = trkErr( 6, it)*t10 + trkErr( 7, it)*t11 +
	                        trkErr( 8, it)*resErr_loc( 2, it);
    kGain(11, it) = 0.f;
    kGain(12, it) = trkErr(10, it)*t00 + trkErr(11, it)*t01 +
	                        trkErr(12, it)*resErr_loc( 1, it);
    kGain(13, it) = trkErr(10, it)*t10 + trkErr(11, it)*t11 +
	                        trkErr(12, it)*resErr_loc( 2, it);
    kGain(14, it) = 0.f;
    kGain(15, it) = trkErr(15, it)*t00 + trkErr(16, it)*t01 +
	                        trkErr(17, it)*resErr_loc( 1, it);
    kGain(16, it) = trkErr(15, it)*t10 + trkErr(16, it)*t11 +
	                        trkErr(17, it)*resErr_loc( 2, it);
    kGain(17, it) = 0.f;  
  }  
     
  MP2F_<N> res_loc_; 
  Right2DView<float, 2, N> res_loc{res_loc_.data.data()};
  
  for (size_t it = 0; it < N; ++it) {
    const auto msPX = msP(iparX, it);
    const auto msPY = msP(iparY, it);
    const auto msPZ = msP(iparZ, it);    
    const auto inParX = inPar(iparX, it);
    const auto inParY = inPar(iparY, it);     
    const auto inParZ = inPar(iparZ, it); 
    
    const auto inParIpt   = inPar(iparIpt, it);
    const auto inParPhi   = inPar(iparPhi, it);
    const auto inParTheta = inPar(iparTheta, it);            
    
    res_loc(0, it) =  rotT00[it]*(msPX - inParX) + rotT01[it]*(msPY - inParY);
    res_loc(1, it) =  msPZ - inParZ;

    inPar(iparX,it)     = inParX + kGain( 0, it) * res_loc( 0, it) + kGain( 1, it) * res_loc( 1, it);
    inPar(iparY,it)     = inParY + kGain( 3, it) * res_loc( 0, it) + kGain( 4, it) * res_loc( 1, it);
    inPar(iparZ,it)     = inParZ + kGain( 6, it) * res_loc( 0, it) + kGain( 7, it) * res_loc( 1, it);
    inPar(iparIpt,it)   = inParIpt + kGain( 9, it) * res_loc( 0, it) + kGain(10, it) * res_loc( 1, it);
    inPar(iparPhi,it)   = inParPhi + kGain(12, it) * res_loc( 0, it) + kGain(13, it) * res_loc( 1, it);
    inPar(iparTheta,it) = inParTheta + kGain(15, it) * res_loc( 0, it) + kGain(16, it) * res_loc( 1, it);          
  }

   MP6x6SF_<N> newErr_;
   Right2DView<float,21, N>   newErr{newErr_.data.data()};
   
   for (size_t it=0;it<N;++it)   {
     const auto t0 = rotT00[it]*trkErr( 0, it) + rotT01[it]*trkErr( 1, it);
     const auto t1 = rotT00[it]*trkErr( 1, it) + rotT01[it]*trkErr( 2, it);
     const auto t2 = rotT00[it]*trkErr( 3, it) + rotT01[it]*trkErr( 4, it);
     const auto t3 = rotT00[it]*trkErr( 6, it) + rotT01[it]*trkErr( 7, it);
     const auto t4 = rotT00[it]*trkErr(10, it) + rotT01[it]*trkErr(11, it);

     newErr( 0, it) = kGain( 0, it)*t0 + kGain( 1, it)*trkErr( 3, it);
     newErr( 1, it) = kGain( 3, it)*t0 + kGain( 4, it)*trkErr( 3, it);
     newErr( 2, it) = kGain( 3, it)*t1 + kGain( 4, it)*trkErr( 4, it);
     newErr( 3, it) = kGain( 6, it)*t0 + kGain( 7, it)*trkErr( 3, it);
     newErr( 4, it) = kGain( 6, it)*t1 + kGain( 7, it)*trkErr( 4, it);
     newErr( 5, it) = kGain( 6, it)*t2 + kGain( 7, it)*trkErr( 5, it);
     newErr( 6, it) = kGain( 9, it)*t0 + kGain(10, it)*trkErr( 3, it);
     newErr( 7, it) = kGain( 9, it)*t1 + kGain(10, it)*trkErr( 4, it);
     newErr( 8, it) = kGain( 9, it)*t2 + kGain(10, it)*trkErr( 5, it);
     newErr( 9, it) = kGain( 9, it)*t3 + kGain(10, it)*trkErr( 8, it);
     newErr(10, it) = kGain(12, it)*t0 + kGain(13, it)*trkErr( 3, it);
     newErr(11, it) = kGain(12, it)*t1 + kGain(13, it)*trkErr( 4, it);
     newErr(12, it) = kGain(12, it)*t2 + kGain(13, it)*trkErr( 5, it);
     newErr(13, it) = kGain(12, it)*t3 + kGain(13, it)*trkErr( 8, it);
     newErr(14, it) = kGain(12, it)*t4 + kGain(13, it)*trkErr(12, it);
     newErr(15, it) = kGain(15, it)*t0 + kGain(16, it)*trkErr( 3, it);
     newErr(16, it) = kGain(15, it)*t1 + kGain(16, it)*trkErr( 4, it);
     newErr(17, it) = kGain(15, it)*t2 + kGain(16, it)*trkErr( 5, it);
     newErr(18, it) = kGain(15, it)*t3 + kGain(16, it)*trkErr( 8, it);
     newErr(19, it) = kGain(15, it)*t4 + kGain(16, it)*trkErr(12, it);
     newErr(20, it) = kGain(15, it)*(rotT00[it]*trkErr(15, it) + rotT01[it]*trkErr(16, it)) +
                         kGain(16, it)*trkErr(17, it);
 #pragma unroll
     for (int i = 0; i < 21; i++){
       trkErr( i, it) = trkErr( i, it) - newErr( i, it);
     }
   }
   //
   return;                 
}
                  
constexpr float kfact= 100/(-0.299792458*3.8112);
constexpr int Niter=5;

template <size_t N = 1>
void propagateToR(const MP6x6SF_<N> &inErr, const MP6F_<N> &inPar, const MP1I_<N> &inChg, 
                  const MP3F_<N> &msP, MP6x6SF_<N> &outErr, MP6F_<N> &outPar) {
  //aux objects  
  MP6x6F_<N> errorProp;
  MP6x6F_<N> temp;
  
  Right3DView<float, 6, 6, N> errorProp_{errorProp.data.data()};
  //
  Right2DCView<float, 6, N>   inPar_{inPar.data.data()};
  Right2DCView<float, 3, N>   msP_{msP.data.data()};
  Right2DView<float, 6, N>    outPar_{outPar.data.data()};
  
  auto sincos4 = [] (const float x, float& sin, float& cos) {
    const float x2 = x*x;
    cos  = 1.f - 0.5f*x2 + 0.04166667f*x2*x2;
    sin  = x - 0.16666667f*x*x2;
  };

  
  for (size_t it = 0; it < N; ++it) {
    //initialize erroProp to identity matrix
    //for (size_t i=0;i<6;++i) errorProp.data[bsize*PosInMtrx(i,i,6) + it] = 1.f; 
    errorProp_(0,0,it) = 1.0f;
    errorProp_(1,1,it) = 1.0f;
    errorProp_(2,2,it) = 1.0f;
    errorProp_(3,3,it) = 1.0f;
    errorProp_(4,4,it) = 1.0f;
    errorProp_(5,5,it) = 1.0f;
    //
    const auto xin = inPar_(iparX, it);
    const auto yin = inPar_(iparY, it);     
    const auto zin = inPar_(iparZ, it); 
    
    const auto iptin   = inPar_(iparIpt,   it);
    const auto phiin   = inPar_(iparPhi,   it);
    const auto thetain = inPar_(iparTheta, it); 
    //
    auto r0 = hipo(xin, yin);
    const auto k = inChg[it]*kfact;//?
    
    const auto xmsP = msP_(iparX, it);//?
    const auto ymsP = msP_(iparY, it);//?
    
    const auto r = hipo(xmsP, ymsP);    
    
    outPar_(iparX,it) = xin;
    outPar_(iparY,it) = yin;
    outPar_(iparZ,it) = zin;

    outPar_(iparIpt,it)   = iptin;
    outPar_(iparPhi,it)   = phiin;
    outPar_(iparTheta,it) = thetain;
 
    const auto kinv  = 1.f/k;
    const auto pt = 1.f/iptin;

    auto D = 0.f, cosa = 0.f, sina = 0.f, id = 0.f;
    //no trig approx here, phi can be large
    auto cosPorT = std::cos(phiin), sinPorT = std::sin(phiin);
    auto pxin = cosPorT*pt;
    auto pyin = sinPorT*pt;

    //derivatives initialized to value for first iteration, i.e. distance = r-r0in
    auto dDdx = r0 > 0.f ? -xin/r0 : 0.f;
    auto dDdy = r0 > 0.f ? -yin/r0 : 0.f;
    auto dDdipt = 0.;
    auto dDdphi = 0.;  
#pragma unroll    
    for (int i = 0; i < Niter; ++i)
    {
     //compute distance and path for the current iteration
      const auto xout = outPar_(iparX, it);
      const auto yout = outPar_(iparY, it);     
      
      r0 = hipo(xout, yout);
      id = (r-r0);
      D+=id;
      sincos4(id*iptin*kinv, sina, cosa);

      //update derivatives on total distance
      if (i+1 != Niter) {

	const auto oor0 = (r0>0.f && std::abs(r-r0)<0.0001f) ? 1.f/r0 : 0.f;

	const auto dadipt = id*kinv;

	const auto dadx = -xout*iptin*kinv*oor0;
	const auto dady = -yout*iptin*kinv*oor0;

	const auto pxca = pxin*cosa;
	const auto pxsa = pxin*sina;
	const auto pyca = pyin*cosa;
	const auto pysa = pyin*sina;

	auto tmp = k*dadx;
	dDdx   -= ( xout*(1.f + tmp*(pxca - pysa)) + yout*tmp*(pyca + pxsa) )*oor0;
	tmp = k*dady;
	dDdy   -= ( xout*tmp*(pxca - pysa) + yout*(1.f + tmp*(pyca + pxsa)) )*oor0;
	//now r0 depends on ipt and phi as well
	tmp = dadipt*iptin;
	dDdipt -= k*( xout*(pxca*tmp - pysa*tmp - pyca - pxsa + pyin) +
		      yout*(pyca*tmp + pxsa*tmp - pysa + pxca - pxin))*pt*oor0;
	dDdphi += k*( xout*(pysa - pxin + pxca) - yout*(pxsa - pyin + pyca))*oor0;
      } 
      
      //update parameters
      outPar_(iparX,it) = xout + k*(pxin*sina - pyin*(1.f-cosa));
      outPar_(iparY,it) = yout + k*(pyin*sina + pxin*(1.f-cosa));
      const float pxinold = pxin;//copy before overwriting
      pxin = pxin*cosa - pyin*sina;
      pyin = pyin*cosa + pxinold*sina;
  
    }
    //
    const auto alpha  = D*iptin*kinv;
    const auto dadx   = dDdx*iptin*kinv;
    const auto dady   = dDdy*iptin*kinv;
    const auto dadipt = (iptin*dDdipt + D)*kinv;
    const auto dadphi = dDdphi*iptin*kinv;

    sincos4(alpha, sina, cosa);
 
    const auto kcosPorTpt = k*cosPorT*pt;
    const auto ksinPorTpt = k*sinPorT*pt;
    
    const auto t1 = (kcosPorTpt*cosa-ksinPorTpt*sina);
    const auto t2 = (ksinPorTpt*cosa+kcosPorTpt*sina);
    //
    const auto t3 = (iptin*dadipt*cosa-sina)*pt;
    const auto t4 = (iptin*dadipt*sina-(1.f-cosa))*pt;
 
    errorProp_(0,0,it) = 1.f+dadx*t1;
    errorProp_(0,1,it) =     dady*t1;
    errorProp_(0,2,it) = 0.f;
    errorProp_(0,3,it) = (kcosPorTpt*t3-ksinPorTpt*t4);
    errorProp_(0,4,it) = (kcosPorTpt*dadphi*cosa - ksinPorTpt*dadphi*sina - ksinPorTpt*sina + kcosPorTpt*cosa - kcosPorTpt);
    errorProp_(0,5,it) = 0.f;

    errorProp_(1,0,it) =     dadx*t2;
    errorProp_(1,1,it) = 1.f+dady*t2;
    errorProp_(1,2,it) = 0.f;
    errorProp_(1,3,it) = (ksinPorTpt*t3+kcosPorTpt*t4);
    errorProp_(1,4,it) = (ksinPorTpt*dadphi*cosa + kcosPorTpt*dadphi*sina + ksinPorTpt*cosa + kcosPorTpt*sina - ksinPorTpt);
    errorProp_(1,5,it) = 0.f;

    //no trig approx here, theta can be large
    cosPorT=std::cos(thetain);
    sinPorT=std::sin(thetain);
    //redefine sinPorT as 1./sinPorT to reduce the number of temporaries
    sinPorT = 1.f/sinPorT;

    const auto t5 = k*cosPorT*pt*sinPorT;

    outPar_(iparZ,it) = zin + alpha*t5;

    errorProp_(2,0, it) = dadx*t5;
    errorProp_(2,1, it) = dady*t5;
    errorProp_(2,2, it) = 1.f;
    errorProp_(2,3, it) = t5*(iptin*dadipt-alpha)*pt;
    errorProp_(2,4, it) = dadphi*t5;
    errorProp_(2,5, it) =-k*alpha*pt*sinPorT*sinPorT;   
    //
    outPar_(iparIpt,it) = iptin;
 
    errorProp_(3,0, it) = 0.f;
    errorProp_(3,1, it) = 0.f;
    errorProp_(3,2, it) = 0.f;
    errorProp_(3,3, it) = 1.f;
    errorProp_(3,4, it) = 0.f;
    errorProp_(3,5, it) = 0.f; 
    
    outPar_(iparPhi,it) = phiin+alpha;
   
    errorProp_(4,0, it) = dadx;
    errorProp_(4,1, it) = dady;
    errorProp_(4,2, it) = 0.f;
    errorProp_(4,3, it) = dadipt;
    errorProp_(4,4, it) = 1.f+dadphi;
    errorProp_(4,5, it) = 0.f; 
  
    outPar_(iparTheta,it) = thetain;        

    errorProp_(5,0, it) = 0.f;
    errorProp_(5,1, it) = 0.f;
    errorProp_(5,2, it) = 0.f;
    errorProp_(5,3, it) = 0.f;
    errorProp_(5,4, it) = 0.f;
    errorProp_(5,5, it) = 1.f; 
                                 
  }
  
  MultHelixProp<N>(errorProp, inErr, temp);
  MultHelixPropTransp<N>(errorProp, temp, outErr);  
  
  return;
}

template <bool is_cuda_target, typename lambda_tp, bool grid_stride = false>
requires (CudaCompute<is_cuda_target> and (enable_cuda_launcher == true))
__cuda_kernel__ void launch_p2r_cuda_kernel(const lambda_tp p2r_kernel, const int length){

  auto i = threadIdx.x + blockIdx.x * blockDim.x;
  const auto stride = (gridDim.x * blockDim.x);
  
  while (i < length) {   

    p2r_kernel(i);

    if constexpr (grid_stride) { i += stride;}
    else  break;
  }


  return;
}

//CUDA specialized version:
template <int bSize, typename stream_tp, bool is_cuda_target>
requires (CudaCompute<is_cuda_target> and (enable_cuda_launcher == true))
void dispatch_p2r_kernels(auto&& p2r_kernel, auto&& pref_to_device, auto&& pref_to_host, stream_tp stream, const int nb_, const int nevnts_){

  pref_to_device();	

  const int outer_loop_range = nevnts_*nb_*bSize;//re-scale exec domain for the cuda backend

  dim3 blocks(threads_per_block, 1, 1);
  dim3 grid(((outer_loop_range + threads_per_block - 1)/ threads_per_block), 1, 1);
  //
  launch_p2r_cuda_kernel<is_cuda_target><<<grid, blocks, 0, stream>>>(p2r_kernel, outer_loop_range);
  //
  p2r_check_error<is_cuda_target>();
  //
  pref_to_host();	
  
  p2r_wait<enable_cuda>();
}

namespace ex = stdexec;

//c++2X implementation for nvidia accelerators:
template <int bSize, typename stream_tp, bool is_cuda_target>
void dispatch_p2r_kernels(auto&& p2r_kernel, auto&& pref_to_device, auto&& pref_to_host, stream_tp, const int nb_, const int nevnts_){
  if constexpr (enable_stdexec_launcher) {
    const auto exe_range =  nb_*nevnts_*(is_cuda_target ? bSize : 1);

    nvexec::stream_context stream_cxt{};
    //
    nvexec::stream_scheduler gpu = stream_cxt.get_scheduler(nvexec::stream_priority::low);
    //nvexec::stream_scheduler gpu = stream_cxt.get_scheduler(nvexec::stream_priority::high);

    auto compute_p2r = ex::just()
	               | ex::then(pref_to_device)
                       | exec::on(gpu, ex::bulk(exe_range, p2r_kernel))
                       | ex::then(pref_to_host);
 
    stdexec::this_thread::sync_wait(std::move(compute_p2r));

  } else { //stdpar launcher
    //
    pref_to_device();

    auto policy = std::execution::par_unseq;
    //
    auto outer_loop_range = std::ranges::views::iota(0, nb_*nevnts_*(is_cuda_target ? bSize : 1));
    //
    std::for_each(policy,
                  std::ranges::begin(outer_loop_range),
                  std::ranges::end(outer_loop_range),
                  p2r_kernel);

    pref_to_host();
  }
}



int main (int argc, char* argv[]) {
#ifdef __NVCOMPILER_CUDA__
#ifdef cuda_launcher
   std::cout << "Running CUDA backend with CUDA launcher.." << std::endl;
#elif defined (stdexec_launcher)
   std::cout << "Running CUDA backend with stdexec launcher.." << std::endl;
#else
   std::cout << "Running CUDA backend with stdpar launcher.." << std::endl;	
#endif
#endif
   #include "input_track.h"

   std::vector<AHIT> inputhits{inputhit21,inputhit20,inputhit19,inputhit18,inputhit17,inputhit16,inputhit15,inputhit14,
                               inputhit13,inputhit12,inputhit11,inputhit10,inputhit09,inputhit08,inputhit07,inputhit06,
                               inputhit05,inputhit04,inputhit03,inputhit02,inputhit01,inputhit00};

   printf("track in pos: x=%f, y=%f, z=%f, r=%f, pt=%f, phi=%f, theta=%f \n", inputtrk.par[0], inputtrk.par[1], inputtrk.par[2],
	  sqrtf(inputtrk.par[0]*inputtrk.par[0] + inputtrk.par[1]*inputtrk.par[1]),
	  1./inputtrk.par[3], inputtrk.par[4], inputtrk.par[5]);
   printf("track in cov: xx=%.2e, yy=%.2e, zz=%.2e \n", inputtrk.cov[SymOffsets66[0]],
	                                       inputtrk.cov[SymOffsets66[(1*6+1)]],
	                                       inputtrk.cov[SymOffsets66[(2*6+2)]]);
   for (size_t lay=0; lay<nlayer; lay++){
     printf("hit in layer=%lu, pos: x=%f, y=%f, z=%f, r=%f \n", lay, inputhits[lay].pos[0], inputhits[lay].pos[1], inputhits[lay].pos[2], sqrtf(inputhits[lay].pos[0]*inputhits[lay].pos[0] + inputhits[lay].pos[1]*inputhits[lay].pos[1]));
   }
   
   printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
   printf("NITER=%d\n", NITER);

   long setup_start, setup_stop;
   struct timeval timecheck;

   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
   
   auto dev_id = p2r_get_compute_device_id<enable_cuda>();
   auto streams= p2r_get_streams<enable_cuda>(nstreams);

   auto stream = streams[0];//with UVM, we use only one compute stream 

   std::vector<MPTRK> outtrcks(nevts*nb);
   // migrate output object to dev memory:
   p2r_prefetch<MPTRK, enable_cuda>(outtrcks, dev_id, stream);

   std::vector<MPTRK> trcks(nevts*nb); 
   prepareTracks(trcks, inputtrk);
   //
   std::vector<MPHIT> hits(nlayer*nevts*nb);
   prepareHits(hits, inputhits);

   // migrate the remaining objects if we don't measure transfers
   if constexpr (include_data_transfer == false) {
     p2r_prefetch<MPTRK, enable_cuda>(trcks, dev_id, stream);
     p2r_prefetch<MPHIT, enable_cuda>(hits,  dev_id, stream);
   }
   
   // synchronize to ensure that all needed data is on the device:
   p2r_wait<enable_cuda>();   

   // create compute kernel
   auto p2r_kernels = [=,btracksPtr    = trcks.data(),
                         outtracksPtr  = outtrcks.data(),
                         bhitsPtr      = hits.data()] (const auto i) {
                         //  
                         constexpr int N      = enable_cuda ? 1 : bsize;
                         //
                         const auto tid      = (enable_cuda ? i / bsize : i);
                         const auto batch_id = (enable_cuda ? i % bsize : 0);
                         //
                         MPTRK_<N> obtracks;
                         //
                         const auto& btracks = btracksPtr[tid].load<N>(batch_id);
                         //
                         constexpr int layers = nlayer;
                         //
                         for(int layer = 0; layer < layers; ++layer) {
                           //
                           const auto& bhits = bhitsPtr[layer+layers*tid].load<N>(batch_id);
                           //
                           propagateToR<N>(btracks.cov, btracks.par, btracks.q, bhits.pos, obtracks.cov, obtracks.par);
                           KalmanUpdate<N>(obtracks.cov, obtracks.par, bhits.cov, bhits.pos);
                           //
                         }
                         //
                         outtracksPtr[tid].save<N>(obtracks, batch_id);
                       };

   // create prefetchers:
   auto prefetch_to_device = [&] {
     if constexpr (include_data_transfer) {
       p2r_prefetch<MPTRK, enable_cuda>(trcks, dev_id, stream);
       p2r_prefetch<MPHIT, enable_cuda>(hits,  dev_id, stream);
     }
   };

   auto prefetch_to_host   = [&] {
     if constexpr (include_data_transfer) {
       p2r_prefetch<MPTRK, enable_cuda>(outtrcks, host_id, stream);
     }
   };   

   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");

   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nevts*nb*sizeof(MPHIT));

   //info<enable_cuda>(dev_id);
   dispatch_p2r_kernels<bsize, decltype(stream), enable_cuda>(p2r_kernels, prefetch_to_device, prefetch_to_host, stream, nb, nevts);

   double wall_time = 0.0;

   for(int itr=0; itr<NITER; itr++) {
     //	   
     auto wall_start = std::chrono::high_resolution_clock::now();
     //
     dispatch_p2r_kernels<bsize, decltype(stream), enable_cuda>(p2r_kernels, prefetch_to_device, prefetch_to_host, stream, nb, nevts);
     //
     auto wall_stop = std::chrono::high_resolution_clock::now();
     //
     auto wall_diff = wall_stop - wall_start;
     //
     wall_time += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
     // reset initial states (don't need if we won't measure data migrations):
     if constexpr (include_data_transfer) {
       p2r_prefetch<MPTRK, enable_cuda>(trcks, host_id, stream);
       p2r_prefetch<MPHIT, enable_cuda>(hits,  host_id, stream);
       //
       p2r_prefetch<MPTRK, enable_cuda, decltype(stream), true>(outtrcks, dev_id, stream);
     }
   } //end of itr loop

   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, (setup_stop-setup_start)*0.001, -1);

   auto outtrk = outtrcks.data();

   int nnans = 0, nfail = 0;
   float avgx = 0, avgy = 0, avgz = 0, avgr = 0;
   float avgpt = 0, avgphi = 0, avgtheta = 0;
   float avgdx = 0, avgdy = 0, avgdz = 0, avgdr = 0;

   for (size_t ie=0;ie<nevts;++ie) {
     for (size_t it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
       float r_ = sqrtf(x_*x_ + y_*y_);
       float pt_ = std::abs(1./ipt(outtrk,ie,it));
       float phi_ = phi(outtrk,ie,it);
       float theta_ = theta(outtrk,ie,it);
       float hx_ = inputhits[nlayer-1].pos[0];
       float hy_ = inputhits[nlayer-1].pos[1];
       float hz_ = inputhits[nlayer-1].pos[2];
       float hr_ = sqrtf(hx_*hx_ + hy_*hy_);
       if (std::isfinite(x_)==false ||
          std::isfinite(y_)==false ||
          std::isfinite(z_)==false ||
          std::isfinite(pt_)==false ||
          std::isfinite(phi_)==false ||
          std::isfinite(theta_)==false
          ) {
        nnans++;
        continue;
       }
       if (fabs( (x_-hx_)/hx_ )>1. ||
	   fabs( (y_-hy_)/hy_ )>1. ||
	   fabs( (z_-hz_)/hz_ )>1.) {
	 nfail++;
	 continue;
       }
       avgpt += pt_;
       avgphi += phi_;
       avgtheta += theta_;
       avgx += x_;
       avgy += y_;
       avgz += z_;
       avgr += r_;
       avgdx += (x_-hx_)/x_;
       avgdy += (y_-hy_)/y_;
       avgdz += (z_-hz_)/z_;
       avgdr += (r_-hr_)/r_;
       //if((it+ie*ntrks)%100000==0) printf("iTrk = %i,  track (x,y,z,r)=(%.6f,%.6f,%.6f,%.6f) \n", it+ie*ntrks, x_,y_,z_,r_);
     }
   }

   avgpt = avgpt/float(nevts*ntrks);
   avgphi = avgphi/float(nevts*ntrks);
   avgtheta = avgtheta/float(nevts*ntrks);
   avgx = avgx/float(nevts*ntrks);
   avgy = avgy/float(nevts*ntrks);
   avgz = avgz/float(nevts*ntrks);
   avgr = avgr/float(nevts*ntrks);
   avgdx = avgdx/float(nevts*ntrks);
   avgdy = avgdy/float(nevts*ntrks);
   avgdz = avgdz/float(nevts*ntrks);
   avgdr = avgdr/float(nevts*ntrks);

   float stdx = 0, stdy = 0, stdz = 0, stdr = 0;
   float stddx = 0, stddy = 0, stddz = 0, stddr = 0;
   for (size_t ie=0;ie<nevts;++ie) {
     for (size_t it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
       float r_ = sqrtf(x_*x_ + y_*y_);
       float hx_ = inputhits[nlayer-1].pos[0];
       float hy_ = inputhits[nlayer-1].pos[1];
       float hz_ = inputhits[nlayer-1].pos[2];
       float hr_ = sqrtf(hx_*hx_ + hy_*hy_);
       if (std::isfinite(x_)==false ||
          std::isfinite(y_)==false ||
          std::isfinite(z_)==false
          ) {
        continue;
       }
       if (fabs( (x_-hx_)/hx_ )>1. ||
	   fabs( (y_-hy_)/hy_ )>1. ||
	   fabs( (z_-hz_)/hz_ )>1.) {
	 continue;
       }
       stdx += (x_-avgx)*(x_-avgx);
       stdy += (y_-avgy)*(y_-avgy);
       stdz += (z_-avgz)*(z_-avgz);
       stdr += (r_-avgr)*(r_-avgr);
       stddx += ((x_-hx_)/x_-avgdx)*((x_-hx_)/x_-avgdx);
       stddy += ((y_-hy_)/y_-avgdy)*((y_-hy_)/y_-avgdy);
       stddz += ((z_-hz_)/z_-avgdz)*((z_-hz_)/z_-avgdz);
       stddr += ((r_-hr_)/r_-avgdr)*((r_-hr_)/r_-avgdr);
     }
   }

   stdx = sqrtf(stdx/float(nevts*ntrks));
   stdy = sqrtf(stdy/float(nevts*ntrks));
   stdz = sqrtf(stdz/float(nevts*ntrks));
   stdr = sqrtf(stdr/float(nevts*ntrks));
   stddx = sqrtf(stddx/float(nevts*ntrks));
   stddy = sqrtf(stddy/float(nevts*ntrks));
   stddz = sqrtf(stddz/float(nevts*ntrks));
   stddr = sqrtf(stddr/float(nevts*ntrks));

   printf("track x avg=%f std/avg=%f\n", avgx, fabs(stdx/avgx));
   printf("track y avg=%f std/avg=%f\n", avgy, fabs(stdy/avgy));
   printf("track z avg=%f std/avg=%f\n", avgz, fabs(stdz/avgz));
   printf("track r avg=%f std/avg=%f\n", avgr, fabs(stdr/avgz));
   printf("track dx/x avg=%f std=%f\n", avgdx, stddx);
   printf("track dy/y avg=%f std=%f\n", avgdy, stddy);
   printf("track dz/z avg=%f std=%f\n", avgdz, stddz);
   printf("track dr/r avg=%f std=%f\n", avgdr, stddr);
   printf("track pt avg=%f\n", avgpt);
   printf("track phi avg=%f\n", avgphi);
   printf("track theta avg=%f\n", avgtheta);
   printf("number of tracks with nans=%i\n", nnans);
   printf("number of tracks failed=%i\n", nfail);

   return 0;
}

