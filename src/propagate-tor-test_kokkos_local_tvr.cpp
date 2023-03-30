/*
nvcc -arch=sm_86 -O3 --extended-lambda --expt-relaxed-constexpr --default-stream per-thread -std=c++17 ./propagate-tor-test_cuda_native.cu -L -lcudart   -o ./"propagate_nvcc_cuda"
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>

#include <cassert>

#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>
#include <random>

#include <Kokkos_Core.hpp>

#ifndef ntrks
#define ntrks 8192
#endif

#ifndef bsize
#define bsize 32
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

#ifdef include_data
constexpr bool include_data_transfer = true;
#else
constexpr bool include_data_transfer = false;
#endif

#ifdef KOKKOS_ENABLE_CUDA
using MemSpace = Kokkos::CudaSpace;
constexpr bool use_gpu = true;
#else
#ifdef KOKKOS_ENABLE_HIP
using MemSpace = Kokkos::Experimental::HIP;
constexpr bool use_gpu = true;
#else
#ifdef KOKKOS_ENABLE_OPENMPTARGET
using MemSpace = Kokkos::OpenMPTargetSpace;
constexpr bool use_gpu = true;
#else
#ifdef KOKKOS_ENABLE_OPENMP
using MemSpace = Kokkos::OpenMP;
constexpr bool use_gpu = false;
#else
#ifdef KOKKOS_ENABLE_SYCL
using MemSpace = Kokkos::Experimental::SYCL;
constexpr bool use_gpu = true;
#else
#ifdef KOKKOS_ENABLE_THREADS
using MemSpace = Kokkos::HostSpace;
constexpr bool use_gpu = false;
#else
#ifdef KOKKOS_ENABLE_SERIAL 
using MemSpace =Kokkos::HostSpace;
constexpr bool use_gpu = false;
#endif
#endif
#endif
#endif
#endif
#endif
#endif
    

static int nstreams  = num_streams;//we have only one stream, though

constexpr int host_id = -1; /*cudaCpuDeviceId*/

const std::array<int, 36> SymOffsets66{0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20};

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
   T data[N*bSize];

   MPNX() = default;
   MPNX(const MPNX<T, N, bSize> &) = default;
   MPNX(MPNX<T, N, bSize> &&)      = default;

   //basic accessors   
   constexpr T &operator[](int i) { return data[i]; }
   constexpr const T &operator[](int i) const { return data[i]; }
   constexpr T& operator()(const int i, const int j) {return data[i*bSize+j];}
   constexpr const T& operator()(const int i, const int j) const {return data[i*bSize+j];}
   constexpr int size() const { return N*bSize; }   
   //
   
   KOKKOS_INLINE_FUNCTION  void load(MPNX<T, N, 1>& dst, const int b) const {
#pragma unroll
     for (int ip=0;ip<N;++ip) { //block load   	
    	dst.data[ip] = data[ip*bSize + b]; 
     }
     
     return;
   }

   KOKKOS_INLINE_FUNCTION  void save(const MPNX<T, N, 1>& src, const int b) {
#pragma unroll
     for (int ip=0;ip<N;++ip) {    	
    	 data[ip*bSize + b] = src.data[ip]; 
     }
     
     return;
   }  
   
   auto operator=(const MPNX&) -> MPNX& = default;
   auto operator=(MPNX&&     ) -> MPNX& = default; 
};

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

// internal data formats:
template<int bSize=1> using MP1I_    = MPNX<int,   1 ,bSize>;
template<int bSize=1> using MP1F_    = MPNX<float, 1 ,bSize>;
template<int bSize=1> using MP2F_    = MPNX<float, 2 ,bSize>;
template<int bSize=1> using MP3F_    = MPNX<float, 3 ,bSize>;
template<int bSize=1> using MP6F_    = MPNX<float, 6 ,bSize>;
template<int bSize=1> using MP2x2SF_ = MPNX<float, 3 ,bSize>;
template<int bSize=1> using MP3x3SF_ = MPNX<float, 6 ,bSize>;
template<int bSize=1> using MP6x6SF_ = MPNX<float, 21,bSize>;
template<int bSize=1> using MP6x6F_  = MPNX<float, 36,bSize>;
template<int bSize=1> using MP3x3_   = MPNX<float, 9 ,bSize>;
template<int bSize=1> using MP3x6_   = MPNX<float, 18,bSize>;

template<int N=1>
struct MPTRK_ {
  MP6F_<N>    par;
  MP6x6SF_<N> cov;
  MP1I_<N>    q;
};

template<int N=1>
struct MPHIT_ {
  MP3F_<N>    pos;
  MP3x3SF_<N> cov;
};


struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;

  template<int S>
  KOKKOS_INLINE_FUNCTION  const auto load_component (const int batch_id) const{//b is a batch idx
  
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
  KOKKOS_INLINE_FUNCTION   void save_component(MPTRK_<S> &src, const int batch_id) {

    if constexpr (std::is_same<MP6F, MP6F_<S>>::value
                  and std::is_same<MP6x6SF, MP6x6SF_<S>>::value
                  and std::is_same<MP1I, MP1I_<S>>::value) { //just do a copy of the whole objects

      this->par = src.par;
      this->cov = src.cov;
      this->q   = src.q;

    } else{
    this->par.save(src.par, batch_id);
    this->cov.save(src.cov, batch_id);
    this->q.save(src.q, batch_id);
    }
    
    return;
  }  
};

struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;
  //
  template<int S>
  KOKKOS_INLINE_FUNCTION   const auto load_component(const int batch_id) const {
    MPHIT_<S> dst;

    if constexpr (std::is_same<MP3F, MP3F_<S>>::value
                  and std::is_same<MP3x3SF, MP3x3SF_<S>>::value) { //just do a copy of the whole object
      dst.pos = this->pos;
      dst.cov = this->cov;
    } else { //ok, do manual load of the batch component instead
    this->pos.load(dst.pos, batch_id);
    this->cov.load(dst.cov, batch_id);
    }
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

Kokkos::View<MPTRK*, MemSpace>::HostMirror prepareTracks(ATRK inputtrk, Kokkos::View<MPTRK*, MemSpace> trk ) {

  auto result = Kokkos::create_mirror_view( trk );

  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
	      //par
	      for (size_t ip=0;ip<6;++ip) {
	        result(ib + nb*ie).par.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.par[ip];
	      }
	      //cov
	      for (size_t ip=0;ip<21;++ip) {
	        result(ib + nb*ie).cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.cov[ip]*100;
	      }
	      //q
	      result(ib + nb*ie).q.data[it] = inputtrk.q;//
      }
    }
  }
  return result;
}


Kokkos::View<MPHIT*, MemSpace>::HostMirror prepareHits(std::vector<AHIT> &inputhits, Kokkos::View<MPHIT*, MemSpace> hit) {
  
  auto result = Kokkos::create_mirror_view( hit );
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
        	  result(lay+nlayer*(ib + nb*ie)).pos.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
        	  result(lay+nlayer*(ib + nb*ie)).cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.cov[ip];
        	}
        }
      }
    }
  }
  return result;
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
template<size_t N=1,typename member_type>
KOKKOS_INLINE_FUNCTION  void MultHelixProp(const MP6x6F_<N> &a, const MP6x6SF_<N> &b, MP6x6F_<N> &c,const member_type& teamMember) {//ok

  //#pragma unroll
  #pragma omp simd
  for (int it =0;it<N;it++)
  //Kokkos::parallel_for( Kokkos::ThreadVectorRange(teamMember,N),[&] (const size_t it) 
  {
    //printf("league rank =  %i, team rank = %i, it =  %i, N= %d \n",int(teamMember.league_rank()),int(teamMember.team_rank()),it,N);
    c[ 0*N+it] = a[ 0*N+it]*b[ 0*N+it] + a[ 1*N+it]*b[ 1*N+it] + a[ 3*N+it]*b[ 6*N+it] + a[ 4*N+it]*b[10*N+it];
    c[ 1*N+it] = a[ 0*N+it]*b[ 1*N+it] + a[ 1*N+it]*b[ 2*N+it] + a[ 3*N+it]*b[ 7*N+it] + a[ 4*N+it]*b[11*N+it];
    c[ 2*N+it] = a[ 0*N+it]*b[ 3*N+it] + a[ 1*N+it]*b[ 4*N+it] + a[ 3*N+it]*b[ 8*N+it] + a[ 4*N+it]*b[12*N+it];
    c[ 3*N+it] = a[ 0*N+it]*b[ 6*N+it] + a[ 1*N+it]*b[ 7*N+it] + a[ 3*N+it]*b[ 9*N+it] + a[ 4*N+it]*b[13*N+it];
    c[ 4*N+it] = a[ 0*N+it]*b[10*N+it] + a[ 1*N+it]*b[11*N+it] + a[ 3*N+it]*b[13*N+it] + a[ 4*N+it]*b[14*N+it];
    c[ 5*N+it] = a[ 0*N+it]*b[15*N+it] + a[ 1*N+it]*b[16*N+it] + a[ 3*N+it]*b[18*N+it] + a[ 4*N+it]*b[19*N+it];
    c[ 6*N+it] = a[ 6*N+it]*b[ 0*N+it] + a[ 7*N+it]*b[ 1*N+it] + a[ 9*N+it]*b[ 6*N+it] + a[10*N+it]*b[10*N+it];
    c[ 7*N+it] = a[ 6*N+it]*b[ 1*N+it] + a[ 7*N+it]*b[ 2*N+it] + a[ 9*N+it]*b[ 7*N+it] + a[10*N+it]*b[11*N+it];
    c[ 8*N+it] = a[ 6*N+it]*b[ 3*N+it] + a[ 7*N+it]*b[ 4*N+it] + a[ 9*N+it]*b[ 8*N+it] + a[10*N+it]*b[12*N+it];
    c[ 9*N+it] = a[ 6*N+it]*b[ 6*N+it] + a[ 7*N+it]*b[ 7*N+it] + a[ 9*N+it]*b[ 9*N+it] + a[10*N+it]*b[13*N+it];
    c[10*N+it] = a[ 6*N+it]*b[10*N+it] + a[ 7*N+it]*b[11*N+it] + a[ 9*N+it]*b[13*N+it] + a[10*N+it]*b[14*N+it];
    c[11*N+it] = a[ 6*N+it]*b[15*N+it] + a[ 7*N+it]*b[16*N+it] + a[ 9*N+it]*b[18*N+it] + a[10*N+it]*b[19*N+it];
    
    c[12*N+it] = a[12*N+it]*b[ 0*N+it] + a[13*N+it]*b[ 1*N+it] + b[ 3*N+it] + a[15*N+it]*b[ 6*N+it] + a[16*N+it]*b[10*N+it] + a[17*N+it]*b[15*N+it];
    c[13*N+it] = a[12*N+it]*b[ 1*N+it] + a[13*N+it]*b[ 2*N+it] + b[ 4*N+it] + a[15*N+it]*b[ 7*N+it] + a[16*N+it]*b[11*N+it] + a[17*N+it]*b[16*N+it];
    c[14*N+it] = a[12*N+it]*b[ 3*N+it] + a[13*N+it]*b[ 4*N+it] + b[ 5*N+it] + a[15*N+it]*b[ 8*N+it] + a[16*N+it]*b[12*N+it] + a[17*N+it]*b[17*N+it];
    c[15*N+it] = a[12*N+it]*b[ 6*N+it] + a[13*N+it]*b[ 7*N+it] + b[ 8*N+it] + a[15*N+it]*b[ 9*N+it] + a[16*N+it]*b[13*N+it] + a[17*N+it]*b[18*N+it];
    c[16*N+it] = a[12*N+it]*b[10*N+it] + a[13*N+it]*b[11*N+it] + b[12*N+it] + a[15*N+it]*b[13*N+it] + a[16*N+it]*b[14*N+it] + a[17*N+it]*b[19*N+it];
    c[17*N+it] = a[12*N+it]*b[15*N+it] + a[13*N+it]*b[16*N+it] + b[17*N+it] + a[15*N+it]*b[18*N+it] + a[16*N+it]*b[19*N+it] + a[17*N+it]*b[20*N+it];
    
    c[18*N+it] = a[18*N+it]*b[ 0*N+it] + a[19*N+it]*b[ 1*N+it] + a[21*N+it]*b[ 6*N+it] + a[22*N+it]*b[10*N+it];
    c[19*N+it] = a[18*N+it]*b[ 1*N+it] + a[19*N+it]*b[ 2*N+it] + a[21*N+it]*b[ 7*N+it] + a[22*N+it]*b[11*N+it];
    c[20*N+it] = a[18*N+it]*b[ 3*N+it] + a[19*N+it]*b[ 4*N+it] + a[21*N+it]*b[ 8*N+it] + a[22*N+it]*b[12*N+it];
    c[21*N+it] = a[18*N+it]*b[ 6*N+it] + a[19*N+it]*b[ 7*N+it] + a[21*N+it]*b[ 9*N+it] + a[22*N+it]*b[13*N+it];
    c[22*N+it] = a[18*N+it]*b[10*N+it] + a[19*N+it]*b[11*N+it] + a[21*N+it]*b[13*N+it] + a[22*N+it]*b[14*N+it];
    c[23*N+it] = a[18*N+it]*b[15*N+it] + a[19*N+it]*b[16*N+it] + a[21*N+it]*b[18*N+it] + a[22*N+it]*b[19*N+it];
    c[24*N+it] = a[24*N+it]*b[ 0*N+it] + a[25*N+it]*b[ 1*N+it] + a[27*N+it]*b[ 6*N+it] + a[28*N+it]*b[10*N+it];
    c[25*N+it] = a[24*N+it]*b[ 1*N+it] + a[25*N+it]*b[ 2*N+it] + a[27*N+it]*b[ 7*N+it] + a[28*N+it]*b[11*N+it];
    c[26*N+it] = a[24*N+it]*b[ 3*N+it] + a[25*N+it]*b[ 4*N+it] + a[27*N+it]*b[ 8*N+it] + a[28*N+it]*b[12*N+it];
    c[27*N+it] = a[24*N+it]*b[ 6*N+it] + a[25*N+it]*b[ 7*N+it] + a[27*N+it]*b[ 9*N+it] + a[28*N+it]*b[13*N+it];
    c[28*N+it] = a[24*N+it]*b[10*N+it] + a[25*N+it]*b[11*N+it] + a[27*N+it]*b[13*N+it] + a[28*N+it]*b[14*N+it];
    c[29*N+it] = a[24*N+it]*b[15*N+it] + a[25*N+it]*b[16*N+it] + a[27*N+it]*b[18*N+it] + a[28*N+it]*b[19*N+it];
    c[30*N+it] = b[15*N+it];
    c[31*N+it] = b[16*N+it];
    c[32*N+it] = b[17*N+it];
    c[33*N+it] = b[18*N+it];
    c[34*N+it] = b[19*N+it];
    c[35*N+it] = b[20*N+it];    
  //});
  };
  return;
}

template<size_t N=1,typename member_type>
KOKKOS_INLINE_FUNCTION  void MultHelixPropTransp(const MP6x6F_<N> &a, const MP6x6F_<N> &b, MP6x6SF_<N> &c, const member_type& teamMember) {//

  //#pragma unroll
  #pragma omp simd
  for (int it=0;it<N;it++)
  //Kokkos::parallel_for( Kokkos::ThreadVectorRange(teamMember,N),[&] (const size_t it) 
  {
    
    c[ 0*N+it] = b[ 0*N+it]*a[ 0*N+it] + b[ 1*N+it]*a[ 1*N+it] + b[ 3*N+it]*a[ 3*N+it] + b[ 4*N+it]*a[ 4*N+it];
    c[ 1*N+it] = b[ 6*N+it]*a[ 0*N+it] + b[ 7*N+it]*a[ 1*N+it] + b[ 9*N+it]*a[ 3*N+it] + b[10*N+it]*a[ 4*N+it];
    c[ 2*N+it] = b[ 6*N+it]*a[ 6*N+it] + b[ 7*N+it]*a[ 7*N+it] + b[ 9*N+it]*a[ 9*N+it] + b[10*N+it]*a[10*N+it];
    c[ 3*N+it] = b[12*N+it]*a[ 0*N+it] + b[13*N+it]*a[ 1*N+it] + b[15*N+it]*a[ 3*N+it] + b[16*N+it]*a[ 4*N+it];
    c[ 4*N+it] = b[12*N+it]*a[ 6*N+it] + b[13*N+it]*a[ 7*N+it] + b[15*N+it]*a[ 9*N+it] + b[16*N+it]*a[10*N+it];
    c[ 5*N+it] = b[12*N+it]*a[12*N+it] + b[13*N+it]*a[13*N+it] + b[14*N+it] + b[15*N+it]*a[15*N+it] + b[16*N+it]*a[16*N+it] + b[17*N+it]*a[17*N+it];
    c[ 6*N+it] = b[18*N+it]*a[ 0*N+it] + b[19*N+it]*a[ 1*N+it] + b[21*N+it]*a[ 3*N+it] + b[22*N+it]*a[ 4*N+it];
    c[ 7*N+it] = b[18*N+it]*a[ 6*N+it] + b[19*N+it]*a[ 7*N+it] + b[21*N+it]*a[ 9*N+it] + b[22*N+it]*a[10*N+it];
    c[ 8*N+it] = b[18*N+it]*a[12*N+it] + b[19*N+it]*a[13*N+it] + b[20*N+it] + b[21*N+it]*a[15*N+it] + b[22*N+it]*a[16*N+it] + b[23*N+it]*a[17*N+it];
    c[ 9*N+it] = b[18*N+it]*a[18*N+it] + b[19*N+it]*a[19*N+it] + b[21*N+it]*a[21*N+it] + b[22*N+it]*a[22*N+it];
    c[10*N+it] = b[24*N+it]*a[ 0*N+it] + b[25*N+it]*a[ 1*N+it] + b[27*N+it]*a[ 3*N+it] + b[28*N+it]*a[ 4*N+it];
    c[11*N+it] = b[24*N+it]*a[ 6*N+it] + b[25*N+it]*a[ 7*N+it] + b[27*N+it]*a[ 9*N+it] + b[28*N+it]*a[10*N+it];
    c[12*N+it] = b[24*N+it]*a[12*N+it] + b[25*N+it]*a[13*N+it] + b[26*N+it] + b[27*N+it]*a[15*N+it] + b[28*N+it]*a[16*N+it] + b[29*N+it]*a[17*N+it];
    c[13*N+it] = b[24*N+it]*a[18*N+it] + b[25*N+it]*a[19*N+it] + b[27*N+it]*a[21*N+it] + b[28*N+it]*a[22*N+it];
    c[14*N+it] = b[24*N+it]*a[24*N+it] + b[25*N+it]*a[25*N+it] + b[27*N+it]*a[27*N+it] + b[28*N+it]*a[28*N+it];
    c[15*N+it] = b[30*N+it]*a[ 0*N+it] + b[31*N+it]*a[ 1*N+it] + b[33*N+it]*a[ 3*N+it] + b[34*N+it]*a[ 4*N+it];
    c[16*N+it] = b[30*N+it]*a[ 6*N+it] + b[31*N+it]*a[ 7*N+it] + b[33*N+it]*a[ 9*N+it] + b[34*N+it]*a[10*N+it];
    c[17*N+it] = b[30*N+it]*a[12*N+it] + b[31*N+it]*a[13*N+it] + b[32*N+it] + b[33*N+it]*a[15*N+it] + b[34*N+it]*a[16*N+it] + b[35*N+it]*a[17*N+it];
    c[18*N+it] = b[30*N+it]*a[18*N+it] + b[31*N+it]*a[19*N+it] + b[33*N+it]*a[21*N+it] + b[34*N+it]*a[22*N+it];
    c[19*N+it] = b[30*N+it]*a[24*N+it] + b[31*N+it]*a[25*N+it] + b[33*N+it]*a[27*N+it] + b[34*N+it]*a[28*N+it];
    c[20*N+it] = b[35*N+it];
  };
  return;  
}

KOKKOS_INLINE_FUNCTION  float hipo(const float x, const float y) {return std::sqrt(x*x + y*y);}

template<size_t N=1,typename member_type>
KOKKOS_INLINE_FUNCTION  void KalmanUpdate(MP6x6SF_<N> &trkErr_, MP6F_<N> &inPar_, const MP3x3SF_<N> &hitErr_, const MP3F_<N> &msP_, const member_type& teamMember){	  
  
  MP1F_<N>    rotT00;
  MP1F_<N>    rotT01;
  MP2x2SF_<N> resErr_loc;
  //MP3x3SF resErr_glo;
  #pragma omp simd
  for (size_t it = 0;it < N; ++it) {    
  //Kokkos::parallel_for( Kokkos::ThreadVectorRange(teamMember,N),[&] (const size_t it){ 
    const float msPX = msP_(iparX,it);
    const float msPY = msP_(iparY,it);
    const float inParX = inPar_(iparX,it);
    const float inParY = inPar_(iparY,it);          
  
    const float r = hipo(msPX, msPY);
    rotT00[it] = -(msPY + inParY) / (2*r);
    rotT01[it] =  (msPX + inParX) / (2*r);    
    
    resErr_loc[ 0*N+it] = (rotT00[it]*(trkErr_[0*N+it] + hitErr_[0*N+it]) +
                           rotT01[it]*(trkErr_[1*N+it] + hitErr_[1*N+it]))*rotT00[it] +
                          (rotT00[it]*(trkErr_[1*N+it] + hitErr_[1*N+it]) +
                           rotT01[it]*(trkErr_[2*N+it] + hitErr_[2*N+it]))*rotT01[it];
    resErr_loc[ 1*N+it] = (trkErr_[3*N+it] + hitErr_[3*N+it])*rotT00[it] +
                          (trkErr_[4*N+it] + hitErr_[4*N+it])*rotT01[it];
    resErr_loc[ 2*N+it] = (trkErr_[5*N+it] + hitErr_[5*N+it]);
  
  
    const double det = (double)resErr_loc[0*N+it] * resErr_loc[2*N+it] -
                       (double)resErr_loc[1*N+it] * resErr_loc[1*N+it];
    const float s   = 1.f / det;
    const float tmp = s * resErr_loc[2*N+it];
    resErr_loc[1*N+it] *= -s;
    resErr_loc[2*N+it]  = s * resErr_loc[0*N+it];
    resErr_loc[0*N+it]  = tmp;  
  };     
  
  MP3x6_<N> kGain;

  #pragma omp simd
  for (size_t it=0; it<N; ++it){
  //Kokkos::parallel_for( Kokkos::ThreadVectorRange(teamMember,N),[&] (const size_t it){ 
    kGain[ 0*N+it] = trkErr_[ 0*N+it]*(rotT00[0*N+it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 1*N+it]*(rotT01[0*N+it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 3*N+it]*resErr_loc[ 1*N+it];
    kGain[ 1*N+it] = trkErr_[ 0*N+it]*(rotT00[0*N+it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 1*N+it]*(rotT01[0*N+it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 3*N+it]*resErr_loc[ 2*N+it];
    kGain[ 2*N+it] = 0;
    kGain[ 3*N+it] = trkErr_[ 1*N+it]*(rotT00[0*N+it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 2*N+it]*(rotT01[0*N+it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 4*N+it]*resErr_loc[ 1*N+it];
    kGain[ 4*N+it] = trkErr_[ 1*N+it]*(rotT00[0*N+it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 2*N+it]*(rotT01[0*N+it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 4*N+it]*resErr_loc[ 2*N+it];
    kGain[ 5*N+it] = 0;
    kGain[ 6*N+it] = trkErr_[ 3*N+it]*(rotT00[0*N+it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 4*N+it]*(rotT01[0*N+it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 5*N+it]*resErr_loc[ 1*N+it];
    kGain[ 7*N+it] = trkErr_[ 3*N+it]*(rotT00[0*N+it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 4*N+it]*(rotT01[0*N+it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 5*N+it]*resErr_loc[ 2*N+it];
    kGain[ 8*N+it] = 0;
    kGain[ 9*N+it] = trkErr_[ 6*N+it]*(rotT00[0*N+it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 7*N+it]*(rotT01[0*N+it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 8*N+it]*resErr_loc[ 1*N+it];
    kGain[10*N+it] = trkErr_[ 6*N+it]*(rotT00[0*N+it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 7*N+it]*(rotT01[0*N+it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 8*N+it]*resErr_loc[ 2*N+it];
    kGain[11*N+it] = 0;
    kGain[12*N+it] = trkErr_[10*N+it]*(rotT00[0*N+it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[11*N+it]*(rotT01[0*N+it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[12*N+it]*resErr_loc[ 1*N+it];
    kGain[13*N+it] = trkErr_[10*N+it]*(rotT00[0*N+it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[11*N+it]*(rotT01[0*N+it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[12*N+it]*resErr_loc[ 2*N+it];
    kGain[14*N+it] = 0;
    kGain[15*N+it] = trkErr_[15*N+it]*(rotT00[0*N+it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[16*N+it]*(rotT01[0*N+it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[17*N+it]*resErr_loc[ 1*N+it];
    kGain[16*N+it] = trkErr_[15*N+it]*(rotT00[0*N+it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[16*N+it]*(rotT01[0*N+it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[17*N+it]*resErr_loc[ 2*N+it];
    kGain[17*N+it] = 0;  
  }; 
     
  MP2F_<N> res_loc;   
  #pragma omp simd
  for (size_t it=0 ;it<N ; ++it){
  //Kokkos::parallel_for( Kokkos::ThreadVectorRange(teamMember,N),[&] (const size_t it){ 
    const float msPX = msP_(iparX,it);
    const float msPY = msP_(iparY,it);
    const float msPZ = msP_(iparZ,it);    
    const float inParX = inPar_(iparX,it);
    const float inParY = inPar_(iparY,it);     
    const float inParZ = inPar_(iparZ,it); 
    
    const float inParIpt   = inPar_(iparIpt,it);
    const float inParPhi   = inPar_(iparPhi,it);
    const float inParTheta = inPar_(iparTheta,it);            
    
    res_loc[0*N+it] =  rotT00[0*N+it]*(msPX - inParX) + rotT01[0*N+it]*(msPY - inParY);
    res_loc[1*N+it] =  msPZ - inParZ;

    inPar_(iparX,it)     = inParX + kGain[ 0*N+it] * res_loc[ 0*N+it] + kGain[ 1*N+it] * res_loc[ 1*N+it];
    inPar_(iparY,it)     = inParY + kGain[ 3*N+it] * res_loc[ 0*N+it] + kGain[ 4*N+it] * res_loc[ 1*N+it];
    inPar_(iparZ,it)     = inParZ + kGain[ 6*N+it] * res_loc[ 0*N+it] + kGain[ 7*N+it] * res_loc[ 1*N+it];
    inPar_(iparIpt,it)   = inParIpt + kGain[ 9*N+it] * res_loc[ 0*N+it] + kGain[10*N+it] * res_loc[ 1*N+it];
    inPar_(iparPhi,it)   = inParPhi + kGain[12*N+it] * res_loc[ 0*N+it] + kGain[13*N+it] * res_loc[ 1*N+it];
    inPar_(iparTheta,it) = inParTheta + kGain[15*N+it] * res_loc[ 0*N+it] + kGain[16*N+it] * res_loc[ 1*N+it];     
  };

  MP6x6SF_<N> newErr;
  #pragma omp simd
  for (size_t it=0 ;it<N ; ++it){
  //Kokkos::parallel_for( Kokkos::ThreadVectorRange(teamMember,N),[&] (const size_t it){ 
     newErr[ 0*N+it] = kGain[ 0*N+it]*rotT00[0*N+it]*trkErr_[ 0*N+it] +
                         kGain[ 0*N+it]*rotT01[0*N+it]*trkErr_[ 1*N+it] +
                         kGain[ 1*N+it]*trkErr_[ 3*N+it];
     newErr[ 1*N+it] = kGain[ 3*N+it]*rotT00[0*N+it]*trkErr_[ 0*N+it] +
                         kGain[ 3*N+it]*rotT01[0*N+it]*trkErr_[ 1*N+it] +
                         kGain[ 4*N+it]*trkErr_[ 3*N+it];
     newErr[ 2*N+it] = kGain[ 3*N+it]*rotT00[0*N+it]*trkErr_[ 1*N+it] +
                         kGain[ 3*N+it]*rotT01[0*N+it]*trkErr_[ 2*N+it] +
                         kGain[ 4*N+it]*trkErr_[ 4*N+it];
     newErr[ 3*N+it] = kGain[ 6*N+it]*rotT00[0*N+it]*trkErr_[ 0*N+it] +
                         kGain[ 6*N+it]*rotT01[0*N+it]*trkErr_[ 1*N+it] +
                         kGain[ 7*N+it]*trkErr_[ 3*N+it];
     newErr[ 4*N+it] = kGain[ 6*N+it]*rotT00[0*N+it]*trkErr_[ 1*N+it] +
                         kGain[ 6*N+it]*rotT01[0*N+it]*trkErr_[ 2*N+it] +
                         kGain[ 7*N+it]*trkErr_[ 4*N+it];
     newErr[ 5*N+it] = kGain[ 6*N+it]*rotT00[0*N+it]*trkErr_[ 3*N+it] +
                         kGain[ 6*N+it]*rotT01[0*N+it]*trkErr_[ 4*N+it] +
                         kGain[ 7*N+it]*trkErr_[ 5*N+it];
     newErr[ 6*N+it] = kGain[ 9*N+it]*rotT00[0*N+it]*trkErr_[ 0*N+it] +
                         kGain[ 9*N+it]*rotT01[0*N+it]*trkErr_[ 1*N+it] +
                         kGain[10*N+it]*trkErr_[ 3*N+it];
     newErr[ 7*N+it] = kGain[ 9*N+it]*rotT00[0*N+it]*trkErr_[ 1*N+it] +
                         kGain[ 9*N+it]*rotT01[0*N+it]*trkErr_[ 2*N+it] +
                         kGain[10*N+it]*trkErr_[ 4*N+it];
     newErr[ 8*N+it] = kGain[ 9*N+it]*rotT00[0*N+it]*trkErr_[ 3*N+it] +
                         kGain[ 9*N+it]*rotT01[0*N+it]*trkErr_[ 4*N+it] +
                         kGain[10*N+it]*trkErr_[ 5*N+it];
     newErr[ 9*N+it] = kGain[ 9*N+it]*rotT00[0*N+it]*trkErr_[ 6*N+it] +
                         kGain[ 9*N+it]*rotT01[0*N+it]*trkErr_[ 7*N+it] +
                         kGain[10*N+it]*trkErr_[ 8*N+it];
     newErr[10*N+it] = kGain[12*N+it]*rotT00[0*N+it]*trkErr_[ 0*N+it] +
                         kGain[12*N+it]*rotT01[0*N+it]*trkErr_[ 1*N+it] +
                         kGain[13*N+it]*trkErr_[ 3*N+it];
     newErr[11*N+it] = kGain[12*N+it]*rotT00[0*N+it]*trkErr_[ 1*N+it] +
                         kGain[12*N+it]*rotT01[0*N+it]*trkErr_[ 2*N+it] +
                         kGain[13*N+it]*trkErr_[ 4*N+it];
     newErr[12*N+it] = kGain[12*N+it]*rotT00[0*N+it]*trkErr_[ 3*N+it] +
                         kGain[12*N+it]*rotT01[0*N+it]*trkErr_[ 4*N+it] +
                         kGain[13*N+it]*trkErr_[ 5*N+it];
     newErr[13*N+it] = kGain[12*N+it]*rotT00[0*N+it]*trkErr_[ 6*N+it] +
                         kGain[12*N+it]*rotT01[0*N+it]*trkErr_[ 7*N+it] +
                         kGain[13*N+it]*trkErr_[ 8*N+it];
     newErr[14*N+it] = kGain[12*N+it]*rotT00[0*N+it]*trkErr_[10*N+it] +
                         kGain[12*N+it]*rotT01[0*N+it]*trkErr_[11*N+it] +
                         kGain[13*N+it]*trkErr_[12*N+it];
     newErr[15*N+it] = kGain[15*N+it]*rotT00[0*N+it]*trkErr_[ 0*N+it] +
                         kGain[15*N+it]*rotT01[0*N+it]*trkErr_[ 1*N+it] +
                         kGain[16*N+it]*trkErr_[ 3*N+it];
     newErr[16*N+it] = kGain[15*N+it]*rotT00[0*N+it]*trkErr_[ 1*N+it] +
                         kGain[15*N+it]*rotT01[0*N+it]*trkErr_[ 2*N+it] +
                         kGain[16*N+it]*trkErr_[ 4*N+it];
     newErr[17*N+it] = kGain[15*N+it]*rotT00[0*N+it]*trkErr_[ 3*N+it] +
                         kGain[15*N+it]*rotT01[0*N+it]*trkErr_[ 4*N+it] +
                         kGain[16*N+it]*trkErr_[ 5*N+it];
     newErr[18*N+it] = kGain[15*N+it]*rotT00[0*N+it]*trkErr_[ 6*N+it] +
                         kGain[15*N+it]*rotT01[0*N+it]*trkErr_[ 7*N+it] +
                         kGain[16*N+it]*trkErr_[ 8*N+it];
     newErr[19*N+it] = kGain[15*N+it]*rotT00[0*N+it]*trkErr_[10*N+it] +
                         kGain[15*N+it]*rotT01[0*N+it]*trkErr_[11*N+it] +
                         kGain[16*N+it]*trkErr_[12*N+it];
     newErr[20*N+it] = kGain[15*N+it]*rotT00[0*N+it]*trkErr_[15*N+it] +
                         kGain[15*N+it]*rotT01[0*N+it]*trkErr_[16*N+it] +
                         kGain[16*N+it]*trkErr_[17*N+it];     
 #pragma unroll
     for (int i = 0; i < 21; i++){
       trkErr_[ i*N+it] = trkErr_[ i*N+it] - newErr[ i*N+it];
     }
   };
   //
   return;                 
}
                  

constexpr float kfact= 100/(-0.299792458*3.8112);
constexpr int Niter=5;

template <int N = 1,typename member_type>
KOKKOS_INLINE_FUNCTION  void propagateToR(const MP6x6SF_<N> &inErr_, const MP6F_<N> &inPar_, const MP1I_<N> &inChg_, 
                  const MP3F_<N> &msP_, MP6x6SF_<N> &outErr_, MP6F_<N> &outPar_,const member_type& teamMember) {
  //aux objects  
  MP6x6F_<N> errorProp;
  MP6x6F_<N> temp;
  
  //auto PosInMtrx = [=] (int i, int j, int D) constexpr {return (i*D+j);};
  auto PosInMtrx = [=](const size_t &&i, const size_t &&j, const size_t &&D, const size_t block_size = 1) constexpr {return block_size*(i*D+j);};
  
  auto sincos4 = [] (const float x, float& sin, float& cos) {
    const float x2 = x*x;
    cos  = 1.f - 0.5f*x2 + 0.04166667f*x2*x2;
    sin  = x - 0.16666667f*x*x2;
  };
 
  #pragma omp simd
  for (size_t it = 0; it < N; ++it){ 
  ///Thread vector range here : Loop over vector elements (1 for GPU, teamSize for CPU) 
  //Kokkos::parallel_for( Kokkos::ThreadVectorRange(teamMember,N),[&] (const int it){ 
    //printf("league rank =  %i, team rank = %i, it =  %i, N= %d \n",int(teamMember.league_rank()),int(teamMember.team_rank()),it,N);
    //initialize erroProp to identity matrix
    errorProp[PosInMtrx(0,0,6,N) + it] = 1.0f;
    errorProp[PosInMtrx(1,1,6,N) + it] = 1.0f;
    errorProp[PosInMtrx(2,2,6,N) + it] = 1.0f;
    errorProp[PosInMtrx(3,3,6,N) + it] = 1.0f;
    errorProp[PosInMtrx(4,4,6,N) + it] = 1.0f;
    errorProp[PosInMtrx(5,5,6,N) + it] = 1.0f;
    //
    const float xin = inPar_(iparX,it);
    const float yin = inPar_(iparY,it);     
    const float zin = inPar_(iparZ,it); 
    
    const float iptin   = inPar_(iparIpt,it);
    const float phiin   = inPar_(iparPhi,it);
    const float thetain = inPar_(iparTheta,it); 
    //
    float r0 = hipo(xin, yin);
    const float k = inChg_[it]*kfact;//?
    
    const float xmsP = msP_(iparX,it);//?
    const float ymsP = msP_(iparY,it);//?
    
    const float r = hipo(xmsP, ymsP);    
    
    outPar_(iparX,it) = xin;
    outPar_(iparY,it) = yin;
    outPar_(iparZ,it) = zin;

    outPar_(iparIpt,it)   = iptin;
    outPar_(iparPhi,it)   = phiin;
    outPar_(iparTheta,it) = thetain;
 
    const float kinv  = 1.f/k;
    const float pt = 1.f/iptin;

    float D = 0.f, cosa = 0.f, sina = 0.f, id = 0.f;
    //no trig approx here, phi can be large
    float cosPorT = std::cos(phiin), sinPorT = std::sin(phiin);
    float pxin = cosPorT*pt;
    float pyin = sinPorT*pt;

    //derivatives initialized to value for first iteration, i.e. distance = r-r0in
    float dDdx = r0 > 0.f ? -xin/r0 : 0.f;
    float dDdy = r0 > 0.f ? -yin/r0 : 0.f;
    float dDdipt = 0.;
    float dDdphi = 0.;  
    #pragma unroll    
    for (int i = 0; i < Niter; ++i)
    {
     //compute distance and path for the current iteration
      const float xout = outPar_(iparX,it);
      const float yout = outPar_(iparY,it);     
      
      r0 = hipo(xout, yout);
      id = (r-r0);
      D+=id;
      sincos4(id*iptin*kinv, sina, cosa);

      //update derivatives on total distance
      if (i+1 != Niter) {

	const float oor0 = (r0>0.f && std::abs(r-r0)<0.0001f) ? 1.f/r0 : 0.f;

	const float dadipt = id*kinv;

	const float dadx = -xout*iptin*kinv*oor0;
	const float dady = -yout*iptin*kinv*oor0;

	const float pxca = pxin*cosa;
	const float pxsa = pxin*sina;
	const float pyca = pyin*cosa;
	const float pysa = pyin*sina;

	float tmp = k*dadx;
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
    const float alpha  = D*iptin*kinv;
    const float dadx   = dDdx*iptin*kinv;
    const float dady   = dDdy*iptin*kinv;
    const float dadipt = (iptin*dDdipt + D)*kinv;
    const float dadphi = dDdphi*iptin*kinv;

    sincos4(alpha, sina, cosa);
 
    errorProp[PosInMtrx(0,0,6, N)+ it] = 1.f+k*dadx*(cosPorT*cosa-sinPorT*sina)*pt;
    errorProp[PosInMtrx(0,1,6, N)+ it] =     k*dady*(cosPorT*cosa-sinPorT*sina)*pt;
    errorProp[PosInMtrx(0,2,6, N)+ it] = 0.f;
    errorProp[PosInMtrx(0,3,6, N)+ it] = k*(cosPorT*(iptin*dadipt*cosa-sina)+sinPorT*((1.f-cosa)-iptin*dadipt*sina))*pt*pt;
    errorProp[PosInMtrx(0,4,6, N)+ it] = k*(cosPorT*dadphi*cosa - sinPorT*dadphi*sina - sinPorT*sina + cosPorT*cosa - cosPorT)*pt;
    errorProp[PosInMtrx(0,5,6, N)+ it] = 0.f;

    errorProp[PosInMtrx(1,0,6, N)+ it] =     k*dadx*(sinPorT*cosa+cosPorT*sina)*pt;
    errorProp[PosInMtrx(1,1,6, N)+ it] = 1.f+k*dady*(sinPorT*cosa+cosPorT*sina)*pt;
    errorProp[PosInMtrx(1,2,6, N)+ it] = 0.f;
    errorProp[PosInMtrx(1,3,6, N)+ it] = k*(sinPorT*(iptin*dadipt*cosa-sina)+cosPorT*(iptin*dadipt*sina-(1.f-cosa)))*pt*pt;
    errorProp[PosInMtrx(1,4,6, N)+ it] = k*(sinPorT*dadphi*cosa + cosPorT*dadphi*sina + sinPorT*cosa + cosPorT*sina - sinPorT)*pt;
    errorProp[PosInMtrx(1,5,6, N)+ it] = 0.f;

    //no trig approx here, theta can be large
    cosPorT=std::cos(thetain);
    sinPorT=std::sin(thetain);
    //redefine sinPorT as 1./sinPorT to reduce the number of temporaries
    sinPorT = 1.f/sinPorT;

    outPar_(iparZ,it) = zin + k*alpha*cosPorT*pt*sinPorT;    

    errorProp[PosInMtrx(2,0,6, N) + it] = k*cosPorT*dadx*pt*sinPorT;
    errorProp[PosInMtrx(2,1,6, N) + it] = k*cosPorT*dady*pt*sinPorT;
    errorProp[PosInMtrx(2,2,6, N) + it] = 1.f;
    errorProp[PosInMtrx(2,3,6, N) + it] = k*cosPorT*(iptin*dadipt-alpha)*pt*pt*sinPorT;
    errorProp[PosInMtrx(2,4,6, N) + it] = k*dadphi*cosPorT*pt*sinPorT;
    errorProp[PosInMtrx(2,5,6, N) + it] =-k*alpha*pt*sinPorT*sinPorT;   
    //
    outPar_(iparIpt,it) = iptin;
 
    errorProp[PosInMtrx(3,0,6, N) + it] = 0.f;
    errorProp[PosInMtrx(3,1,6, N) + it] = 0.f;
    errorProp[PosInMtrx(3,2,6, N) + it] = 0.f;
    errorProp[PosInMtrx(3,3,6, N) + it] = 1.f;
    errorProp[PosInMtrx(3,4,6, N) + it] = 0.f;
    errorProp[PosInMtrx(3,5,6, N) + it] = 0.f; 
    
    outPar_(iparPhi,it) = phiin+alpha;
   
    errorProp[PosInMtrx(4,0,6, N) + it] = dadx;
    errorProp[PosInMtrx(4,1,6, N) + it] = dady;
    errorProp[PosInMtrx(4,2,6, N) + it] = 0.f;
    errorProp[PosInMtrx(4,3,6, N) + it] = dadipt;
    errorProp[PosInMtrx(4,4,6, N) + it] = 1.f+dadphi;
    errorProp[PosInMtrx(4,5,6, N) + it] = 0.f; 
  
    outPar_(iparTheta,it) = thetain;        

    errorProp[PosInMtrx(5,0,6, N) + it] = 0.f;
    errorProp[PosInMtrx(5,1,6, N) + it] = 0.f;
    errorProp[PosInMtrx(5,2,6, N) + it] = 0.f;
    errorProp[PosInMtrx(5,3,6, N) + it] = 0.f;
    errorProp[PosInMtrx(5,4,6, N) + it] = 0.f;
    errorProp[PosInMtrx(5,5,6, N) + it] = 1.f; 
                                 
  };
  
  MultHelixProp<N>(errorProp, inErr_, temp, teamMember);
  MultHelixPropTransp<N>(errorProp, temp, outErr_, teamMember);  
  
  return;
}

template <int bSize, int layers, typename member_type, bool grid_stride = true>
KOKKOS_FUNCTION void launch_p2r_kernel(MPTRK *obtracks_, MPTRK *btracks_, MPHIT *bhits_, const member_type& teamMember){

     Kokkos::parallel_for(  Kokkos::TeamThreadRange(teamMember, teamMember.team_size()),[&] (const int& i_local){
     int i = teamMember.league_rank () * teamMember.team_size () + i_local;
     constexpr int  N             = use_gpu ? 1 : bSize;

     const int tid        = use_gpu ? i / bSize : i;
     const int batch_id   = use_gpu ? i % bSize : 0;

     MPTRK_<N> obtracks;
     //printf("league rank =  %i, team rank = %i, N=%i,  i= %i \n",int(teamMember.league_rank()),int(teamMember.team_rank()),N,i);
     //
     //
     const auto& btracks = btracks_[tid].load_component<N>(batch_id);
     #pragma unroll //improved performance by 40-60 %   
     for(int layer = 0; layer < layers; ++layer) {  
       //
       const auto& bhits = bhits_[layer+layers*tid].load_component<N>(batch_id);
       //
       propagateToR<N>(btracks.cov, btracks.par, btracks.q, bhits.pos, obtracks.cov, obtracks.par,teamMember);
       KalmanUpdate<N>(obtracks.cov, obtracks.par, bhits.cov, bhits.pos,teamMember);
       //
     }
        //
        obtracks_[tid].save_component<N>(obtracks, batch_id);
     });
  return;
}

int main (int argc, char* argv[]) {

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
   for (int lay=0; lay<nlayer; lay++){
     printf("hit in layer=%lu, pos: x=%f, y=%f, z=%f, r=%f \n", lay, inputhits[lay].pos[0], inputhits[lay].pos[1], inputhits[lay].pos[2], sqrtf(inputhits[lay].pos[0]*inputhits[lay].pos[0] + inputhits[lay].pos[1]*inputhits[lay].pos[1]));
   }
   
   printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
   printf("NITER=%d\n", NITER);

   long setup_start, setup_stop;
   struct timeval timecheck;
   
   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
   
   Kokkos::initialize(argc, argv);
   {


   printf("After kokkos::init\n");
   using ExecSpace = MemSpace::execution_space;
   ExecSpace e;
   e.print_configuration(std::cout, true);

   //auto dev_id = p2r_get_compute_device_id();
   //auto streams= p2r_get_streams(nstreams);

   //auto stream = streams[0];//with UVM, we use only one compute stream 

   Kokkos::View<MPTRK*, MemSpace>             outtrcks("outtrk",nevts*nb); // device pointer
   Kokkos::View<MPTRK*, MemSpace>::HostMirror h_outtrk = Kokkos::create_mirror_view( outtrcks);

   Kokkos::View<MPTRK*, MemSpace>            trcks("trk",nevts*nb); // device pointer
   Kokkos::View<MPTRK*, MemSpace>::HostMirror h_trk = prepareTracks(inputtrk,trcks);  // host pointer
   Kokkos::deep_copy( trcks , h_trk);

   Kokkos::View<MPHIT*, MemSpace>             hits("hit",nevts*nb*nlayer);
   Kokkos::View<MPHIT*, MemSpace>::HostMirror  h_hit = prepareHits(inputhits,hits);
   Kokkos::deep_copy( hits, h_hit );
   
   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");

   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nevts*nb*sizeof(MPHIT));

   const int phys_length      = nevts*nb;
   const int outer_loop_range = phys_length*(use_gpu?bsize:1); // re-scale the exe domain for gpu backends

   typedef Kokkos::TeamPolicy<>               team_policy;
   typedef Kokkos::TeamPolicy<>::member_type  member_type;
   int team_policy_range = nevts*nb;  // number of teams
   int team_size         = use_gpu?bsize:1;              // team size
   int vector_size       = use_gpu?1:bsize;  // thread size
   //
   //dim3 blocks(threadsperblock, 1, 1);
   //dim3 grid(((outer_loop_range + threadsperblock - 1)/ threadsperblock),1,1);
   printf("team range =  %i, team size= %i, vector_size =  %i\n",team_policy_range,team_size,vector_size);

   double wall_time = 0.0;
   
   // sync explicitly 
   Kokkos::fence();

   for(int itr=0; itr<NITER; itr++) {
     auto wall_start = std::chrono::high_resolution_clock::now();

     Kokkos::parallel_for("Kernel",
                          //Kokkos::RangePolicy<ExecSpace>(0,outer_loop_range), 
                          //KOKKOS_LAMBDA(const int i){
                         team_policy(team_policy_range,team_size,vector_size),
                         KOKKOS_LAMBDA( const member_type &teamMember){
                             //printf("league rank =  %i, team rank = %i, i =  %i\n",int(teamMember.league_rank()),int(teamMember.team_rank()),i);
                             launch_p2r_kernel<bsize, nlayer>(outtrcks.data(), trcks.data(), hits.data(),  teamMember); // kernel for 1 track
                         });
     //
     Kokkos::fence();
     auto wall_stop = std::chrono::high_resolution_clock::now();
     //
     auto wall_diff = wall_stop - wall_start;
     //
     wall_time += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
   } //end of itr loop
   Kokkos::fence();

   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, ntrks, wall_time, (setup_stop-setup_start)*0.001, -1);

   Kokkos::deep_copy( h_outtrk, outtrcks );

   int nnans = 0, nfail = 0;
   float avgx = 0, avgy = 0, avgz = 0, avgr = 0;
   float avgpt = 0, avgphi = 0, avgtheta = 0;
   float avgdx = 0, avgdy = 0, avgdz = 0, avgdr = 0;

   for (int ie=0;ie<nevts;++ie) {
     for (int it=0;it<ntrks;++it) {
       float x_ = x(h_outtrk.data(),ie,it);
       float y_ = y(h_outtrk.data(),ie,it);
       float z_ = z(h_outtrk.data(),ie,it);
       float r_ = sqrtf(x_*x_ + y_*y_);
       float pt_ = std::abs(1./ipt(h_outtrk.data(),ie,it));
       float phi_ = phi(h_outtrk.data(),ie,it);
       float theta_ = theta(h_outtrk.data(),ie,it);
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
       //if((it+ie*ntrks) < 64) printf("iTrk = %i,  track (x,y,z,r)=(%.6f,%.6f,%.6f,%.6f) \n", it+ie*ntrks, x_,y_,z_,r_);
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
   for (int ie=0;ie<nevts;++ie) {
     for (int it=0;it<ntrks;++it) {
       float x_ = x(h_outtrk.data(),ie,it);
       float y_ = y(h_outtrk.data(),ie,it);
       float z_ = z(h_outtrk.data(),ie,it);
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
   };
   Kokkos::finalize();
   return 0;
}
