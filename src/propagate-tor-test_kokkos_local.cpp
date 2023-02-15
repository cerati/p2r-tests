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
#define NITER 200
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
#endif
#ifdef KOKKOS_ENABLE_HIP
using MemSpace = Kokkos::Experimental::HIP;
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
using MemSpace = Kokkos::OpenMPTargetSpace;
#endif
#ifdef KOKKOS_ENABLE_OPENMP
using MemSpace = Kokkos::OpenMP;
#endif
#ifdef KOKKOS_ENABLE_SYCL
using MemSpace = Kokkos::Experimental::SYCL;
#endif
//#ifdef KOKKOS_ENABLE_SERIAL 
//using MemSpace =Kokkos::HostSpace;
//#endif
    

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
using MP1I_    = MPNX<int,   1 >;
using MP1F_    = MPNX<float, 1 >;
using MP2F_    = MPNX<float, 2 >;
using MP3F_    = MPNX<float, 3 >;
using MP6F_    = MPNX<float, 6 >;
using MP2x2SF_ = MPNX<float, 3 >;
using MP3x3SF_ = MPNX<float, 6 >;
using MP6x6SF_ = MPNX<float, 21>;
using MP6x6F_  = MPNX<float, 36>;
using MP3x3_   = MPNX<float, 9 >;
using MP3x6_   = MPNX<float, 18>;

struct MPTRK_ {
  MP6F_    par;
  MP6x6SF_ cov;
  MP1I_    q;
};

struct MPHIT_ {
  MP3F_    pos;
  MP3x3SF_ cov;
};


struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;

  KOKKOS_INLINE_FUNCTION  const auto load_component (const int batch_id) const{//b is a batch idx
  
    MPTRK_ dst;

    this->par.load(dst.par, batch_id);
    this->cov.load(dst.cov, batch_id);
    this->q.load(dst.q, batch_id);
    
    return dst;  
  }
  
  KOKKOS_INLINE_FUNCTION   void save_component(MPTRK_ &src, const int batch_id) {
    this->par.save(src.par, batch_id);
    this->cov.save(src.cov, batch_id);
    this->q.save(src.q, batch_id);
    
    return;
  }  
};

struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;
  //
  KOKKOS_INLINE_FUNCTION   const auto load_component(const int batch_id) const {
    MPHIT_ dst;

    this->pos.load(dst.pos, batch_id);
    this->cov.load(dst.cov, batch_id);
    
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

KOKKOS_INLINE_FUNCTION  void MultHelixProp(const MP6x6F_ &a, const MP6x6SF_ &b, MP6x6F_ &c) {//ok

  {
    c[ 0] = a[ 0]*b[ 0] + a[ 1]*b[ 1] + a[ 3]*b[ 6] + a[ 4]*b[10];
    c[ 1] = a[ 0]*b[ 1] + a[ 1]*b[ 2] + a[ 3]*b[ 7] + a[ 4]*b[11];
    c[ 2] = a[ 0]*b[ 3] + a[ 1]*b[ 4] + a[ 3]*b[ 8] + a[ 4]*b[12];
    c[ 3] = a[ 0]*b[ 6] + a[ 1]*b[ 7] + a[ 3]*b[ 9] + a[ 4]*b[13];
    c[ 4] = a[ 0]*b[10] + a[ 1]*b[11] + a[ 3]*b[13] + a[ 4]*b[14];
    c[ 5] = a[ 0]*b[15] + a[ 1]*b[16] + a[ 3]*b[18] + a[ 4]*b[19];
    c[ 6] = a[ 6]*b[ 0] + a[ 7]*b[ 1] + a[ 9]*b[ 6] + a[10]*b[10];
    c[ 7] = a[ 6]*b[ 1] + a[ 7]*b[ 2] + a[ 9]*b[ 7] + a[10]*b[11];
    c[ 8] = a[ 6]*b[ 3] + a[ 7]*b[ 4] + a[ 9]*b[ 8] + a[10]*b[12];
    c[ 9] = a[ 6]*b[ 6] + a[ 7]*b[ 7] + a[ 9]*b[ 9] + a[10]*b[13];
    c[10] = a[ 6]*b[10] + a[ 7]*b[11] + a[ 9]*b[13] + a[10]*b[14];
    c[11] = a[ 6]*b[15] + a[ 7]*b[16] + a[ 9]*b[18] + a[10]*b[19];
    
    c[12] = a[12]*b[ 0] + a[13]*b[ 1] + b[ 3] + a[15]*b[ 6] + a[16]*b[10] + a[17]*b[15];
    c[13] = a[12]*b[ 1] + a[13]*b[ 2] + b[ 4] + a[15]*b[ 7] + a[16]*b[11] + a[17]*b[16];
    c[14] = a[12]*b[ 3] + a[13]*b[ 4] + b[ 5] + a[15]*b[ 8] + a[16]*b[12] + a[17]*b[17];
    c[15] = a[12]*b[ 6] + a[13]*b[ 7] + b[ 8] + a[15]*b[ 9] + a[16]*b[13] + a[17]*b[18];
    c[16] = a[12]*b[10] + a[13]*b[11] + b[12] + a[15]*b[13] + a[16]*b[14] + a[17]*b[19];
    c[17] = a[12]*b[15] + a[13]*b[16] + b[17] + a[15]*b[18] + a[16]*b[19] + a[17]*b[20];
    
    c[18] = a[18]*b[ 0] + a[19]*b[ 1] + a[21]*b[ 6] + a[22]*b[10];
    c[19] = a[18]*b[ 1] + a[19]*b[ 2] + a[21]*b[ 7] + a[22]*b[11];
    c[20] = a[18]*b[ 3] + a[19]*b[ 4] + a[21]*b[ 8] + a[22]*b[12];
    c[21] = a[18]*b[ 6] + a[19]*b[ 7] + a[21]*b[ 9] + a[22]*b[13];
    c[22] = a[18]*b[10] + a[19]*b[11] + a[21]*b[13] + a[22]*b[14];
    c[23] = a[18]*b[15] + a[19]*b[16] + a[21]*b[18] + a[22]*b[19];
    c[24] = a[24]*b[ 0] + a[25]*b[ 1] + a[27]*b[ 6] + a[28]*b[10];
    c[25] = a[24]*b[ 1] + a[25]*b[ 2] + a[27]*b[ 7] + a[28]*b[11];
    c[26] = a[24]*b[ 3] + a[25]*b[ 4] + a[27]*b[ 8] + a[28]*b[12];
    c[27] = a[24]*b[ 6] + a[25]*b[ 7] + a[27]*b[ 9] + a[28]*b[13];
    c[28] = a[24]*b[10] + a[25]*b[11] + a[27]*b[13] + a[28]*b[14];
    c[29] = a[24]*b[15] + a[25]*b[16] + a[27]*b[18] + a[28]*b[19];
    c[30] = b[15];
    c[31] = b[16];
    c[32] = b[17];
    c[33] = b[18];
    c[34] = b[19];
    c[35] = b[20];    
  }
  return;
}

KOKKOS_INLINE_FUNCTION  void MultHelixPropTransp(const MP6x6F_ &a, const MP6x6F_ &b, MP6x6SF_ &c) {//

  {
    
    c[ 0] = b[ 0]*a[ 0] + b[ 1]*a[ 1] + b[ 3]*a[ 3] + b[ 4]*a[ 4];
    c[ 1] = b[ 6]*a[ 0] + b[ 7]*a[ 1] + b[ 9]*a[ 3] + b[10]*a[ 4];
    c[ 2] = b[ 6]*a[ 6] + b[ 7]*a[ 7] + b[ 9]*a[ 9] + b[10]*a[10];
    c[ 3] = b[12]*a[ 0] + b[13]*a[ 1] + b[15]*a[ 3] + b[16]*a[ 4];
    c[ 4] = b[12]*a[ 6] + b[13]*a[ 7] + b[15]*a[ 9] + b[16]*a[10];
    c[ 5] = b[12]*a[12] + b[13]*a[13] + b[14] + b[15]*a[15] + b[16]*a[16] + b[17]*a[17];
    c[ 6] = b[18]*a[ 0] + b[19]*a[ 1] + b[21]*a[ 3] + b[22]*a[ 4];
    c[ 7] = b[18]*a[ 6] + b[19]*a[ 7] + b[21]*a[ 9] + b[22]*a[10];
    c[ 8] = b[18]*a[12] + b[19]*a[13] + b[20] + b[21]*a[15] + b[22]*a[16] + b[23]*a[17];
    c[ 9] = b[18]*a[18] + b[19]*a[19] + b[21]*a[21] + b[22]*a[22];
    c[10] = b[24]*a[ 0] + b[25]*a[ 1] + b[27]*a[ 3] + b[28]*a[ 4];
    c[11] = b[24]*a[ 6] + b[25]*a[ 7] + b[27]*a[ 9] + b[28]*a[10];
    c[12] = b[24]*a[12] + b[25]*a[13] + b[26] + b[27]*a[15] + b[28]*a[16] + b[29]*a[17];
    c[13] = b[24]*a[18] + b[25]*a[19] + b[27]*a[21] + b[28]*a[22];
    c[14] = b[24]*a[24] + b[25]*a[25] + b[27]*a[27] + b[28]*a[28];
    c[15] = b[30]*a[ 0] + b[31]*a[ 1] + b[33]*a[ 3] + b[34]*a[ 4];
    c[16] = b[30]*a[ 6] + b[31]*a[ 7] + b[33]*a[ 9] + b[34]*a[10];
    c[17] = b[30]*a[12] + b[31]*a[13] + b[32] + b[33]*a[15] + b[34]*a[16] + b[35]*a[17];
    c[18] = b[30]*a[18] + b[31]*a[19] + b[33]*a[21] + b[34]*a[22];
    c[19] = b[30]*a[24] + b[31]*a[25] + b[33]*a[27] + b[34]*a[28];
    c[20] = b[35];
  }
  return;  
}

KOKKOS_INLINE_FUNCTION  float hipo(const float x, const float y) {return std::sqrt(x*x + y*y);}

KOKKOS_INLINE_FUNCTION  void KalmanUpdate(MP6x6SF_ &trkErr_, MP6F_ &inPar_, const MP3x3SF_ &hitErr_, const MP3F_ &msP_){	  
  
  MP1F_    rotT00;
  MP1F_    rotT01;
  MP2x2SF_ resErr_loc;
  //MP3x3SF resErr_glo;
  {   
    const auto msPX = msP_[iparX];
    const auto msPY = msP_[iparY];
    const auto inParX = inPar_[iparX];
    const auto inParY = inPar_[iparY];          
  
    const auto r = hipo(msPX, msPY);
    rotT00[0] = -(msPY + inParY) / (2*r);
    rotT01[0] =  (msPX + inParX) / (2*r);    
    
    resErr_loc[ 0] = (rotT00[0]*(trkErr_[0] + hitErr_[0]) +
                                    rotT01[0]*(trkErr_[1] + hitErr_[1]))*rotT00[0] +
                                   (rotT00[0]*(trkErr_[1] + hitErr_[1]) +
                                    rotT01[0]*(trkErr_[2] + hitErr_[2]))*rotT01[0];
    resErr_loc[ 1] = (trkErr_[3] + hitErr_[3])*rotT00[0] +
                                   (trkErr_[4] + hitErr_[4])*rotT01[0];
    resErr_loc[ 2] = (trkErr_[5] + hitErr_[5]);
  } 
  
  {
  
    const double det = (double)resErr_loc[0] * resErr_loc[2] -
                       (double)resErr_loc[1] * resErr_loc[1];
    const float s   = 1.f / det;
    const float tmp = s * resErr_loc[2];
    resErr_loc[1] *= -s;
    resErr_loc[2]  = s * resErr_loc[0];
    resErr_loc[0]  = tmp;  
  }     
  
  MP3x6_ kGain;
  
  {
    kGain[ 0] = trkErr_[ 0]*(rotT00[0]*resErr_loc[ 0]) +
	                        trkErr_[ 1]*(rotT01[0]*resErr_loc[ 0]) +
	                        trkErr_[ 3]*resErr_loc[ 1];
    kGain[ 1] = trkErr_[ 0]*(rotT00[0]*resErr_loc[ 1]) +
	                        trkErr_[ 1]*(rotT01[0]*resErr_loc[ 1]) +
	                        trkErr_[ 3]*resErr_loc[ 2];
    kGain[ 2] = 0;
    kGain[ 3] = trkErr_[ 1]*(rotT00[0]*resErr_loc[ 0]) +
	                        trkErr_[ 2]*(rotT01[0]*resErr_loc[ 0]) +
	                        trkErr_[ 4]*resErr_loc[ 1];
    kGain[ 4] = trkErr_[ 1]*(rotT00[0]*resErr_loc[ 1]) +
	                        trkErr_[ 2]*(rotT01[0]*resErr_loc[ 1]) +
	                        trkErr_[ 4]*resErr_loc[ 2];
    kGain[ 5] = 0;
    kGain[ 6] = trkErr_[ 3]*(rotT00[0]*resErr_loc[ 0]) +
	                        trkErr_[ 4]*(rotT01[0]*resErr_loc[ 0]) +
	                        trkErr_[ 5]*resErr_loc[ 1];
    kGain[ 7] = trkErr_[ 3]*(rotT00[0]*resErr_loc[ 1]) +
	                        trkErr_[ 4]*(rotT01[0]*resErr_loc[ 1]) +
	                        trkErr_[ 5]*resErr_loc[ 2];
    kGain[ 8] = 0;
    kGain[ 9] = trkErr_[ 6]*(rotT00[0]*resErr_loc[ 0]) +
	                        trkErr_[ 7]*(rotT01[0]*resErr_loc[ 0]) +
	                        trkErr_[ 8]*resErr_loc[ 1];
    kGain[10] = trkErr_[ 6]*(rotT00[0]*resErr_loc[ 1]) +
	                        trkErr_[ 7]*(rotT01[0]*resErr_loc[ 1]) +
	                        trkErr_[ 8]*resErr_loc[ 2];
    kGain[11] = 0;
    kGain[12] = trkErr_[10]*(rotT00[0]*resErr_loc[ 0]) +
	                        trkErr_[11]*(rotT01[0]*resErr_loc[ 0]) +
	                        trkErr_[12]*resErr_loc[ 1];
    kGain[13] = trkErr_[10]*(rotT00[0]*resErr_loc[ 1]) +
	                        trkErr_[11]*(rotT01[0]*resErr_loc[ 1]) +
	                        trkErr_[12]*resErr_loc[ 2];
    kGain[14] = 0;
    kGain[15] = trkErr_[15]*(rotT00[0]*resErr_loc[ 0]) +
	                        trkErr_[16]*(rotT01[0]*resErr_loc[ 0]) +
	                        trkErr_[17]*resErr_loc[ 1];
    kGain[16] = trkErr_[15]*(rotT00[0]*resErr_loc[ 1]) +
	                        trkErr_[16]*(rotT01[0]*resErr_loc[ 1]) +
	                        trkErr_[17]*resErr_loc[ 2];
    kGain[17] = 0;  
  }  
     
  MP2F_ res_loc;   
  {
    const auto msPX = msP_[iparX];
    const auto msPY = msP_[iparY];
    const auto msPZ = msP_[iparZ];    
    const auto inParX = inPar_[iparX];
    const auto inParY = inPar_[iparY];     
    const auto inParZ = inPar_[iparZ]; 
    
    const auto inParIpt   = inPar_[iparIpt];
    const auto inParPhi   = inPar_[iparPhi];
    const auto inParTheta = inPar_[iparTheta];            
    
    res_loc[0] =  rotT00[0]*(msPX - inParX) + rotT01[0]*(msPY - inParY);
    res_loc[1] =  msPZ - inParZ;

    inPar_[iparX]     = inParX + kGain[ 0] * res_loc[ 0] + kGain[ 1] * res_loc[ 1];
    inPar_[iparY]     = inParY + kGain[ 3] * res_loc[ 0] + kGain[ 4] * res_loc[ 1];
    inPar_[iparZ]     = inParZ + kGain[ 6] * res_loc[ 0] + kGain[ 7] * res_loc[ 1];
    inPar_[iparIpt]   = inParIpt + kGain[ 9] * res_loc[ 0] + kGain[10] * res_loc[ 1];
    inPar_[iparPhi]   = inParPhi + kGain[12] * res_loc[ 0] + kGain[13] * res_loc[ 1];
    inPar_[iparTheta] = inParTheta + kGain[15] * res_loc[ 0] + kGain[16] * res_loc[ 1];     
  }

  MP6x6SF_ newErr;
  {

     newErr[ 0] = kGain[ 0]*rotT00[0]*trkErr_[ 0] +
                         kGain[ 0]*rotT01[0]*trkErr_[ 1] +
                         kGain[ 1]*trkErr_[ 3];
     newErr[ 1] = kGain[ 3]*rotT00[0]*trkErr_[ 0] +
                         kGain[ 3]*rotT01[0]*trkErr_[ 1] +
                         kGain[ 4]*trkErr_[ 3];
     newErr[ 2] = kGain[ 3]*rotT00[0]*trkErr_[ 1] +
                         kGain[ 3]*rotT01[0]*trkErr_[ 2] +
                         kGain[ 4]*trkErr_[ 4];
     newErr[ 3] = kGain[ 6]*rotT00[0]*trkErr_[ 0] +
                         kGain[ 6]*rotT01[0]*trkErr_[ 1] +
                         kGain[ 7]*trkErr_[ 3];
     newErr[ 4] = kGain[ 6]*rotT00[0]*trkErr_[ 1] +
                         kGain[ 6]*rotT01[0]*trkErr_[ 2] +
                         kGain[ 7]*trkErr_[ 4];
     newErr[ 5] = kGain[ 6]*rotT00[0]*trkErr_[ 3] +
                         kGain[ 6]*rotT01[0]*trkErr_[ 4] +
                         kGain[ 7]*trkErr_[ 5];
     newErr[ 6] = kGain[ 9]*rotT00[0]*trkErr_[ 0] +
                         kGain[ 9]*rotT01[0]*trkErr_[ 1] +
                         kGain[10]*trkErr_[ 3];
     newErr[ 7] = kGain[ 9]*rotT00[0]*trkErr_[ 1] +
                         kGain[ 9]*rotT01[0]*trkErr_[ 2] +
                         kGain[10]*trkErr_[ 4];
     newErr[ 8] = kGain[ 9]*rotT00[0]*trkErr_[ 3] +
                         kGain[ 9]*rotT01[0]*trkErr_[ 4] +
                         kGain[10]*trkErr_[ 5];
     newErr[ 9] = kGain[ 9]*rotT00[0]*trkErr_[ 6] +
                         kGain[ 9]*rotT01[0]*trkErr_[ 7] +
                         kGain[10]*trkErr_[ 8];
     newErr[10] = kGain[12]*rotT00[0]*trkErr_[ 0] +
                         kGain[12]*rotT01[0]*trkErr_[ 1] +
                         kGain[13]*trkErr_[ 3];
     newErr[11] = kGain[12]*rotT00[0]*trkErr_[ 1] +
                         kGain[12]*rotT01[0]*trkErr_[ 2] +
                         kGain[13]*trkErr_[ 4];
     newErr[12] = kGain[12]*rotT00[0]*trkErr_[ 3] +
                         kGain[12]*rotT01[0]*trkErr_[ 4] +
                         kGain[13]*trkErr_[ 5];
     newErr[13] = kGain[12]*rotT00[0]*trkErr_[ 6] +
                         kGain[12]*rotT01[0]*trkErr_[ 7] +
                         kGain[13]*trkErr_[ 8];
     newErr[14] = kGain[12]*rotT00[0]*trkErr_[10] +
                         kGain[12]*rotT01[0]*trkErr_[11] +
                         kGain[13]*trkErr_[12];
     newErr[15] = kGain[15]*rotT00[0]*trkErr_[ 0] +
                         kGain[15]*rotT01[0]*trkErr_[ 1] +
                         kGain[16]*trkErr_[ 3];
     newErr[16] = kGain[15]*rotT00[0]*trkErr_[ 1] +
                         kGain[15]*rotT01[0]*trkErr_[ 2] +
                         kGain[16]*trkErr_[ 4];
     newErr[17] = kGain[15]*rotT00[0]*trkErr_[ 3] +
                         kGain[15]*rotT01[0]*trkErr_[ 4] +
                         kGain[16]*trkErr_[ 5];
     newErr[18] = kGain[15]*rotT00[0]*trkErr_[ 6] +
                         kGain[15]*rotT01[0]*trkErr_[ 7] +
                         kGain[16]*trkErr_[ 8];
     newErr[19] = kGain[15]*rotT00[0]*trkErr_[10] +
                         kGain[15]*rotT01[0]*trkErr_[11] +
                         kGain[16]*trkErr_[12];
     newErr[20] = kGain[15]*rotT00[0]*trkErr_[15] +
                         kGain[15]*rotT01[0]*trkErr_[16] +
                         kGain[16]*trkErr_[17];     
 #pragma unroll
     for (int i = 0; i < 21; i++){
       trkErr_[ i] = trkErr_[ i] - newErr[ i];
     }
   }
   //
   return;                 
}
                  

constexpr float kfact= 100/(-0.299792458*3.8112);
constexpr int Niter=5;

KOKKOS_INLINE_FUNCTION  void propagateToR(const MP6x6SF_ &inErr_, const MP6F_ &inPar_, const MP1I_ &inChg_, 
                  const MP3F_ &msP_, MP6x6SF_ &outErr_, MP6F_ &outPar_) {
  //aux objects  
  MP6x6F_ errorProp;
  MP6x6F_ temp;
  
  auto PosInMtrx = [=] (int i, int j, int D) constexpr {return (i*D+j);};
  
  auto sincos4 = [] (const float x, float& sin, float& cos) {
    const float x2 = x*x;
    cos  = 1.f - 0.5f*x2 + 0.04166667f*x2*x2;
    sin  = x - 0.16666667f*x*x2;
  };
  
  {
    //initialize erroProp to identity matrix
    //for (int i=0;i<6;++i) errorProp.data[bsize*PosInMtrx(i,i,6) + it] = 1.f; 
    errorProp[PosInMtrx(0,0,6)] = 1.0f;
    errorProp[PosInMtrx(1,1,6)] = 1.0f;
    errorProp[PosInMtrx(2,2,6)] = 1.0f;
    errorProp[PosInMtrx(3,3,6)] = 1.0f;
    errorProp[PosInMtrx(4,4,6)] = 1.0f;
    errorProp[PosInMtrx(5,5,6)] = 1.0f;
    //
    const auto xin = inPar_[iparX];
    const auto yin = inPar_[iparY];     
    const auto zin = inPar_[iparZ]; 
    
    const auto iptin   = inPar_[iparIpt];
    const auto phiin   = inPar_[iparPhi];
    const auto thetain = inPar_[iparTheta]; 
    //
    auto r0 = hipo(xin, yin);
    const auto k = inChg_[0]*kfact;//?
    
    const auto xmsP = msP_[iparX];//?
    const auto ymsP = msP_[iparY];//?
    
    const auto r = hipo(xmsP, ymsP);    
    
    outPar_[iparX] = xin;
    outPar_[iparY] = yin;
    outPar_[iparZ] = zin;

    outPar_[iparIpt]   = iptin;
    outPar_[iparPhi]   = phiin;
    outPar_[iparTheta] = thetain;
 
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
      const auto xout = outPar_[iparX];
      const auto yout = outPar_[iparY];     
      
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
      outPar_[iparX] = xout + k*(pxin*sina - pyin*(1.f-cosa));
      outPar_[iparY] = yout + k*(pyin*sina + pxin*(1.f-cosa));
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
 
    errorProp[PosInMtrx(0,0,6)] = 1.f+k*dadx*(cosPorT*cosa-sinPorT*sina)*pt;
    errorProp[PosInMtrx(0,1,6)] =     k*dady*(cosPorT*cosa-sinPorT*sina)*pt;
    errorProp[PosInMtrx(0,2,6)] = 0.f;
    errorProp[PosInMtrx(0,3,6)] = k*(cosPorT*(iptin*dadipt*cosa-sina)+sinPorT*((1.f-cosa)-iptin*dadipt*sina))*pt*pt;
    errorProp[PosInMtrx(0,4,6)] = k*(cosPorT*dadphi*cosa - sinPorT*dadphi*sina - sinPorT*sina + cosPorT*cosa - cosPorT)*pt;
    errorProp[PosInMtrx(0,5,6)] = 0.f;

    errorProp[PosInMtrx(1,0,6)] =     k*dadx*(sinPorT*cosa+cosPorT*sina)*pt;
    errorProp[PosInMtrx(1,1,6)] = 1.f+k*dady*(sinPorT*cosa+cosPorT*sina)*pt;
    errorProp[PosInMtrx(1,2,6)] = 0.f;
    errorProp[PosInMtrx(1,3,6)] = k*(sinPorT*(iptin*dadipt*cosa-sina)+cosPorT*(iptin*dadipt*sina-(1.f-cosa)))*pt*pt;
    errorProp[PosInMtrx(1,4,6)] = k*(sinPorT*dadphi*cosa + cosPorT*dadphi*sina + sinPorT*cosa + cosPorT*sina - sinPorT)*pt;
    errorProp[PosInMtrx(1,5,6)] = 0.f;

    //no trig approx here, theta can be large
    cosPorT=std::cos(thetain);
    sinPorT=std::sin(thetain);
    //redefine sinPorT as 1./sinPorT to reduce the number of temporaries
    sinPorT = 1.f/sinPorT;

    outPar_[iparZ] = zin + k*alpha*cosPorT*pt*sinPorT;    

    errorProp[PosInMtrx(2,0,6)] = k*cosPorT*dadx*pt*sinPorT;
    errorProp[PosInMtrx(2,1,6)] = k*cosPorT*dady*pt*sinPorT;
    errorProp[PosInMtrx(2,2,6)] = 1.f;
    errorProp[PosInMtrx(2,3,6)] = k*cosPorT*(iptin*dadipt-alpha)*pt*pt*sinPorT;
    errorProp[PosInMtrx(2,4,6)] = k*dadphi*cosPorT*pt*sinPorT;
    errorProp[PosInMtrx(2,5,6)] =-k*alpha*pt*sinPorT*sinPorT;   
    //
    outPar_[iparIpt] = iptin;
 
    errorProp[PosInMtrx(3,0,6)] = 0.f;
    errorProp[PosInMtrx(3,1,6)] = 0.f;
    errorProp[PosInMtrx(3,2,6)] = 0.f;
    errorProp[PosInMtrx(3,3,6)] = 1.f;
    errorProp[PosInMtrx(3,4,6)] = 0.f;
    errorProp[PosInMtrx(3,5,6)] = 0.f; 
    
    outPar_[iparPhi] = phiin+alpha;
   
    errorProp[PosInMtrx(4,0,6)] = dadx;
    errorProp[PosInMtrx(4,1,6)] = dady;
    errorProp[PosInMtrx(4,2,6)] = 0.f;
    errorProp[PosInMtrx(4,3,6)] = dadipt;
    errorProp[PosInMtrx(4,4,6)] = 1.f+dadphi;
    errorProp[PosInMtrx(4,5,6)] = 0.f; 
  
    outPar_[iparTheta] = thetain;        

    errorProp[PosInMtrx(5,0,6)] = 0.f;
    errorProp[PosInMtrx(5,1,6)] = 0.f;
    errorProp[PosInMtrx(5,2,6)] = 0.f;
    errorProp[PosInMtrx(5,3,6)] = 0.f;
    errorProp[PosInMtrx(5,4,6)] = 0.f;
    errorProp[PosInMtrx(5,5,6)] = 1.f; 
                                 
  }
  
  MultHelixProp(errorProp, inErr_, temp);
  MultHelixPropTransp(errorProp, temp, outErr_);  
  
  return;
}

template <int bSize, int layers, bool grid_stride = true>
KOKKOS_FUNCTION void launch_p2r_kernel(MPTRK *obtracks_, MPTRK *btracks_, MPHIT *bhits_, int i){

     MPTRK_ obtracks;
     //
     const auto tid        = i / bSize;
     const auto batch_id   = i % bSize;
     //
     const auto& btracks = btracks_[tid].load_component(batch_id);
#pragma unroll //improved performance by 40-60 %   
     for(int layer = 0; layer < layers; ++layer) {  
       //
       const auto& bhits = bhits_[layer+layers*tid].load_component(batch_id);
       //
       propagateToR(btracks.cov, btracks.par, btracks.q, bhits.pos, obtracks.cov, obtracks.par);
       KalmanUpdate(obtracks.cov, obtracks.par, bhits.cov, bhits.pos);
       //
     }
     //
     obtracks_[tid].save_component(obtracks, batch_id);
     
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
   const int outer_loop_range = phys_length*bsize;

   typedef Kokkos::TeamPolicy<>               team_policy;
   typedef Kokkos::TeamPolicy<>::member_type  member_type;
   int team_policy_range = nevts*nb;  // number of teams
   int team_size = bsize;  // team size
   //
   //dim3 blocks(threadsperblock, 1, 1);
   //dim3 grid(((outer_loop_range + threadsperblock - 1)/ threadsperblock),1,1);

   double wall_time = 0.0;
   
   // sync explicitly 
   Kokkos::fence();

   for(int itr=0; itr<NITER; itr++) {
     auto wall_start = std::chrono::high_resolution_clock::now();

     Kokkos::parallel_for("Kernel",
                          //Kokkos::RangePolicy<ExecSpace>(0,outer_loop_range), 
                          //KOKKOS_LAMBDA(const int i){
                         team_policy(team_policy_range,team_size),
                         KOKKOS_LAMBDA( const member_type &teamMember){
                             int i = teamMember.league_rank () * teamMember.team_size () +teamMember.team_rank ();
                             launch_p2r_kernel<bsize, nlayer>(outtrcks.data(), trcks.data(), hits.data(), i); // kernel for 1 track
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
