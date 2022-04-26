/*
export PSTL_USAGE_WARNINGS=1
export ONEDPL_USE_DPCPP_BACKEND=1

dpcpp -std=c++17 -O2 src/propagate-tor-test_dpl.cpp -o test-dpl.exe -Dntrks=8192 -Dnevts=100 -DNITER=5 -Dbsize=1 -Dnlayer=20

*/

//NOTE: For better coexistence with the C++ standard library, include oneAPI DPC++ Library (oneDPL) 
//      header files BEFORE the standard C++ header files.

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/random>
#include <oneapi/dpl/async>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>

#include <vector>
#include <memory>
#include <numeric>

#ifndef USE_GPU
#define USE_CPU
#endif

#ifndef bsize
#define bsize 1
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

#include <CL/sycl.hpp>
using oneapi::dpl::counting_iterator;

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

template <typename T, int N, int bSize>
struct MPNX {
   std::array<T,N*bSize> data;
   //basic accessors
   const T& operator[](const int idx) const {return data[idx];}
   T& operator[](const int idx) {return data[idx];}
   const T& operator()(const int m, const int b) const {return data[m*bSize+b];}
   T& operator()(const int m, const int b) {return data[m*bSize+b];}
   //
   void load(MPNX& dst) const{
     for (int it=0;it<bSize;++it) {
     //const int l = it+ib*bsize+ie*nb*bsize;
       for (int ip=0;ip<N;++ip) {    	
    	 dst.data[it + ip*bSize] = this->operator()(ip, it);  
       }
     }//
     
     return;
   }

   void save(const MPNX& src) {
     for (int it=0;it<bSize;++it) {
     //const int l = it+ib*bsize+ie*nb*bsize;
       for (int ip=0;ip<N;++ip) {    	
    	 this->operator()(ip, it) = src.data[it + ip*bSize];  
       }
     }//
     
     return;
   }
};

using MP1I    = MPNX<int,   1 , bsize>;
using MP1F    = MPNX<float, 1 , bsize>;
using MP2F    = MPNX<float, 3 , bsize>;
using MP3F    = MPNX<float, 3 , bsize>;
using MP6F    = MPNX<float, 6 , bsize>;
using MP2x2SF = MPNX<float, 3 , bsize>;
using MP3x3SF = MPNX<float, 6 , bsize>;
using MP6x6SF = MPNX<float, 21, bsize>;
using MP6x6F  = MPNX<float, 36, bsize>;
using MP3x3   = MPNX<float, 9 , bsize>;
using MP3x6   = MPNX<float, 18, bsize>;

struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;

  //  MP22I   hitidx;
  void load(MPTRK &dst){
    par.load(dst.par);
    cov.load(dst.cov);
    q.load(dst.q);    
    return;	  
  }
  void save(const MPTRK &src){
    par.save(src.par);
    cov.save(src.cov);
    q.save(src.q);
    return;
  }
};

struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;
  //
  void load(MPHIT &dst){
    pos.load(dst.pos);
    cov.load(dst.cov);
    return;
  }
  void save(const MPHIT &src){
    pos.save(src.pos);
    cov.save(src.cov);

    return;
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


template<typename MPTRKAllocator>
void prepareTracks(std::vector<MPTRK, MPTRKAllocator> &trcks, ATRK &inputtrk) {
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

template<typename MPHITAllocator>
void prepareHits(std::vector<MPHIT, MPHITAllocator> &hits, std::vector<AHIT>& inputhits) {
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

template<int N = 1>
inline void MultHelixProp(const MP6x6F &a, const MP6x6SF &b, MP6x6F &c) {//ok
#pragma unroll
  for (int it = 0;it < N; it++) {
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
  }
  return;
}

template<int N = 1>
inline void MultHelixPropTransp(const MP6x6F &a, const MP6x6F &b, MP6x6SF &c) {//

  for (int it = 0;it < N; it++) {
    
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
  }
  return;  
}

auto hipo = [](const float x, const float y) {return std::sqrt(x*x + y*y);};

template <int N = 1>
void KalmanUpdate(MP6x6SF &trkErr, MP6F &inPar, const MP3x3SF &hitErr, const MP3F &msP){	  
  
  MP1F    rotT00;
  MP1F    rotT01;
  MP2x2SF resErr_loc;
  //MP3x3SF resErr_glo;
    
  for (int it = 0;it < N; ++it) {   
    const auto msPX = msP(iparX, it);
    const auto msPY = msP(iparY, it);
    const auto inParX = inPar(iparX, it);
    const auto inParY = inPar(iparY, it);          
  
    const auto r = hipo(msPX, msPY);
    rotT00[it] = -(msPY + inParY) / (2*r);
    rotT01[it] =  (msPX + inParX) / (2*r);    
    
    resErr_loc[ 0*N+it] = (rotT00[it]*(trkErr[0*N+it] + hitErr[0*N+it]) +
                                    rotT01[it]*(trkErr[1*N+it] + hitErr[1*N+it]))*rotT00[it] +
                                   (rotT00[it]*(trkErr[1*N+it] + hitErr[1*N+it]) +
                                    rotT01[it]*(trkErr[2*N+it] + hitErr[2*N+it]))*rotT01[it];
    resErr_loc[ 1*N+it] = (trkErr[3*N+it] + hitErr[3*N+it])*rotT00[it] +
                                   (trkErr[4*N+it] + hitErr[4*N+it])*rotT01[it];
    resErr_loc[ 2*N+it] = (trkErr[5*N+it] + hitErr[5*N+it]);
  } 
  
  for (int it=0;it<N;++it) {
  
    const float det = (float)resErr_loc[0*N+it] * resErr_loc[2*N+it] -
                       (float)resErr_loc[1*N+it] * resErr_loc[1*N+it];
    const float s   = 1.f / det;
    const float tmp = s * resErr_loc[2*N+it];
    resErr_loc[1*N+it] *= -s;
    resErr_loc[2*N+it]  = s * resErr_loc[0*N+it];
    resErr_loc[0*N+it]  = tmp;  
  }     
  
  MP3x6 kGain;
  
#pragma omp simd
  for (int it=0; it<N; ++it) {
    kGain[ 0*N+it] = trkErr[ 0*N+it]*(rotT00[it]*resErr_loc[ 0*N+it]) +
	                        trkErr[ 1*N+it]*(rotT01[it]*resErr_loc[ 0*N+it]) +
	                        trkErr[ 3*N+it]*resErr_loc[ 1*N+it];
    kGain[ 1*N+it] = trkErr[ 0*N+it]*(rotT00[it]*resErr_loc[ 1*N+it]) +
	                        trkErr[ 1*N+it]*(rotT01[it]*resErr_loc[ 1*N+it]) +
	                        trkErr[ 3*N+it]*resErr_loc[ 2*N+it];
    kGain[ 2*N+it] = 0;
    kGain[ 3*N+it] = trkErr[ 1*N+it]*(rotT00[it]*resErr_loc[ 0*N+it]) +
	                        trkErr[ 2*N+it]*(rotT01[it]*resErr_loc[ 0*N+it]) +
	                        trkErr[ 4*N+it]*resErr_loc[ 1*N+it];
    kGain[ 4*N+it] = trkErr[ 1*N+it]*(rotT00[it]*resErr_loc[ 1*N+it]) +
	                        trkErr[ 2*N+it]*(rotT01[it]*resErr_loc[ 1*N+it]) +
	                        trkErr[ 4*N+it]*resErr_loc[ 2*N+it];
    kGain[ 5*N+it] = 0;
    kGain[ 6*N+it] = trkErr[ 3*N+it]*(rotT00[it]*resErr_loc[ 0*N+it]) +
	                        trkErr[ 4*N+it]*(rotT01[it]*resErr_loc[ 0*N+it]) +
	                        trkErr[ 5*N+it]*resErr_loc[ 1*N+it];
    kGain[ 7*N+it] = trkErr[ 3*N+it]*(rotT00[it]*resErr_loc[ 1*N+it]) +
	                        trkErr[ 4*N+it]*(rotT01[it]*resErr_loc[ 1*N+it]) +
	                        trkErr[ 5*N+it]*resErr_loc[ 2*N+it];
    kGain[ 8*N+it] = 0;
    kGain[ 9*N+it] = trkErr[ 6*N+it]*(rotT00[it]*resErr_loc[ 0*N+it]) +
	                        trkErr[ 7*N+it]*(rotT01[it]*resErr_loc[ 0*N+it]) +
	                        trkErr[ 8*N+it]*resErr_loc[ 1*N+it];
    kGain[10*N+it] = trkErr[ 6*N+it]*(rotT00[it]*resErr_loc[ 1*N+it]) +
	                        trkErr[ 7*N+it]*(rotT01[it]*resErr_loc[ 1*N+it]) +
	                        trkErr[ 8*N+it]*resErr_loc[ 2*N+it];
    kGain[11*N+it] = 0;
    kGain[12*N+it] = trkErr[10*N+it]*(rotT00[it]*resErr_loc[ 0*N+it]) +
	                        trkErr[11*N+it]*(rotT01[it]*resErr_loc[ 0*N+it]) +
	                        trkErr[12*N+it]*resErr_loc[ 1*N+it];
    kGain[13*N+it] = trkErr[10*N+it]*(rotT00[it]*resErr_loc[ 1*N+it]) +
	                        trkErr[11*N+it]*(rotT01[it]*resErr_loc[ 1*N+it]) +
	                        trkErr[12*N+it]*resErr_loc[ 2*N+it];
    kGain[14*N+it] = 0;
    kGain[15*N+it] = trkErr[15*N+it]*(rotT00[it]*resErr_loc[ 0*N+it]) +
	                        trkErr[16*N+it]*(rotT01[it]*resErr_loc[ 0*N+it]) +
	                        trkErr[17*N+it]*resErr_loc[ 1*N+it];
    kGain[16*N+it] = trkErr[15*N+it]*(rotT00[it]*resErr_loc[ 1*N+it]) +
	                        trkErr[16*N+it]*(rotT01[it]*resErr_loc[ 1*N+it]) +
	                        trkErr[17*N+it]*resErr_loc[ 2*N+it];
    kGain[17*N+it] = 0;  
  }  
     
  MP2F res_loc;   
  for (int it = 0; it < N; ++it) {
    const auto msPX = msP(iparX, it);
    const auto msPY = msP(iparY, it);
    const auto msPZ = msP(iparZ, it);    
    const auto inParX = inPar(iparX, it);
    const auto inParY = inPar(iparY, it);     
    const auto inParZ = inPar(iparZ, it); 
    
    const auto inParIpt   = inPar(iparIpt, it);
    const auto inParPhi   = inPar(iparPhi, it);
    const auto inParTheta = inPar(iparTheta, it);            
    
    res_loc[0*N+it] =  rotT00[it]*(msPX - inParX) + rotT01[it]*(msPY - inParY);
    res_loc[1*N+it] =  msPZ - inParZ;

    inPar(iparX,it)     = inParX + kGain[ 0*N+it] * res_loc[ 0*N+it] + kGain[ 1*N+it] * res_loc[ 1*N+it];
    inPar(iparY,it)     = inParY + kGain[ 3*N+it] * res_loc[ 0*N+it] + kGain[ 4*N+it] * res_loc[ 1*N+it];
    inPar(iparZ,it)     = inParZ + kGain[ 6*N+it] * res_loc[ 0*N+it] + kGain[ 7*N+it] * res_loc[ 1*N+it];
    inPar(iparIpt,it)   = inParIpt + kGain[ 9*N+it] * res_loc[ 0*N+it] + kGain[10*N+it] * res_loc[ 1*N+it];
    inPar(iparPhi,it)   = inParPhi + kGain[12*N+it] * res_loc[ 0*N+it] + kGain[13*N+it] * res_loc[ 1*N+it];
    inPar(iparTheta,it) = inParTheta + kGain[15*N+it] * res_loc[ 0*N+it] + kGain[16*N+it] * res_loc[ 1*N+it];     
  }

   MP6x6SF newErr;
   for (int it=0;it<N;++it)   {

     newErr[ 0*N+it] = kGain[ 0*N+it]*rotT00[it]*trkErr[ 0*N+it] +
                         kGain[ 0*N+it]*rotT01[it]*trkErr[ 1*N+it] +
                         kGain[ 1*N+it]*trkErr[ 3*N+it];
     newErr[ 1*N+it] = kGain[ 3*N+it]*rotT00[it]*trkErr[ 0*N+it] +
                         kGain[ 3*N+it]*rotT01[it]*trkErr[ 1*N+it] +
                         kGain[ 4*N+it]*trkErr[ 3*N+it];
     newErr[ 2*N+it] = kGain[ 3*N+it]*rotT00[it]*trkErr[ 1*N+it] +
                         kGain[ 3*N+it]*rotT01[it]*trkErr[ 2*N+it] +
                         kGain[ 4*N+it]*trkErr[ 4*N+it];
     newErr[ 3*N+it] = kGain[ 6*N+it]*rotT00[it]*trkErr[ 0*N+it] +
                         kGain[ 6*N+it]*rotT01[it]*trkErr[ 1*N+it] +
                         kGain[ 7*N+it]*trkErr[ 3*N+it];
     newErr[ 4*N+it] = kGain[ 6*N+it]*rotT00[it]*trkErr[ 1*N+it] +
                         kGain[ 6*N+it]*rotT01[it]*trkErr[ 2*N+it] +
                         kGain[ 7*N+it]*trkErr[ 4*N+it];
     newErr[ 5*N+it] = kGain[ 6*N+it]*rotT00[it]*trkErr[ 3*N+it] +
                         kGain[ 6*N+it]*rotT01[it]*trkErr[ 4*N+it] +
                         kGain[ 7*N+it]*trkErr[ 5*N+it];
     newErr[ 6*N+it] = kGain[ 9*N+it]*rotT00[it]*trkErr[ 0*N+it] +
                         kGain[ 9*N+it]*rotT01[it]*trkErr[ 1*N+it] +
                         kGain[10*N+it]*trkErr[ 3*N+it];
     newErr[ 7*N+it] = kGain[ 9*N+it]*rotT00[it]*trkErr[ 1*N+it] +
                         kGain[ 9*N+it]*rotT01[it]*trkErr[ 2*N+it] +
                         kGain[10*N+it]*trkErr[ 4*N+it];
     newErr[ 8*N+it] = kGain[ 9*N+it]*rotT00[it]*trkErr[ 3*N+it] +
                         kGain[ 9*N+it]*rotT01[it]*trkErr[ 4*N+it] +
                         kGain[10*N+it]*trkErr[ 5*N+it];
     newErr[ 9*N+it] = kGain[ 9*N+it]*rotT00[it]*trkErr[ 6*N+it] +
                         kGain[ 9*N+it]*rotT01[it]*trkErr[ 7*N+it] +
                         kGain[10*N+it]*trkErr[ 8*N+it];
     newErr[10*N+it] = kGain[12*N+it]*rotT00[it]*trkErr[ 0*N+it] +
                         kGain[12*N+it]*rotT01[it]*trkErr[ 1*N+it] +
                         kGain[13*N+it]*trkErr[ 3*N+it];
     newErr[11*N+it] = kGain[12*N+it]*rotT00[it]*trkErr[ 1*N+it] +
                         kGain[12*N+it]*rotT01[it]*trkErr[ 2*N+it] +
                         kGain[13*N+it]*trkErr[ 4*N+it];
     newErr[12*N+it] = kGain[12*N+it]*rotT00[it]*trkErr[ 3*N+it] +
                         kGain[12*N+it]*rotT01[it]*trkErr[ 4*N+it] +
                         kGain[13*N+it]*trkErr[ 5*N+it];
     newErr[13*N+it] = kGain[12*N+it]*rotT00[it]*trkErr[ 6*N+it] +
                         kGain[12*N+it]*rotT01[it]*trkErr[ 7*N+it] +
                         kGain[13*N+it]*trkErr[ 8*N+it];
     newErr[14*N+it] = kGain[12*N+it]*rotT00[it]*trkErr[10*N+it] +
                         kGain[12*N+it]*rotT01[it]*trkErr[11*N+it] +
                         kGain[13*N+it]*trkErr[12*N+it];
     newErr[15*N+it] = kGain[15*N+it]*rotT00[it]*trkErr[ 0*N+it] +
                         kGain[15*N+it]*rotT01[it]*trkErr[ 1*N+it] +
                         kGain[16*N+it]*trkErr[ 3*N+it];
     newErr[16*N+it] = kGain[15*N+it]*rotT00[it]*trkErr[ 1*N+it] +
                         kGain[15*N+it]*rotT01[it]*trkErr[ 2*N+it] +
                         kGain[16*N+it]*trkErr[ 4*N+it];
     newErr[17*N+it] = kGain[15*N+it]*rotT00[it]*trkErr[ 3*N+it] +
                         kGain[15*N+it]*rotT01[it]*trkErr[ 4*N+it] +
                         kGain[16*N+it]*trkErr[ 5*N+it];
     newErr[18*N+it] = kGain[15*N+it]*rotT00[it]*trkErr[ 6*N+it] +
                         kGain[15*N+it]*rotT01[it]*trkErr[ 7*N+it] +
                         kGain[16*N+it]*trkErr[ 8*N+it];
     newErr[19*N+it] = kGain[15*N+it]*rotT00[it]*trkErr[10*N+it] +
                         kGain[15*N+it]*rotT01[it]*trkErr[11*N+it] +
                         kGain[16*N+it]*trkErr[12*N+it];
     newErr[20*N+it] = kGain[15*N+it]*rotT00[it]*trkErr[15*N+it] +
                         kGain[15*N+it]*rotT01[it]*trkErr[16*N+it] +
                         kGain[16*N+it]*trkErr[17*N+it];     
 #pragma unroll
     for (int i = 0; i < 21; i++){
       trkErr[ i*N+it] = trkErr[ i*N+it] - newErr[ i*N+it];
     }
   }
   //
   return;                 
}
                  

constexpr float kfact= 100/(-0.299792458f*3.8112f);
constexpr int Niter=5;

template <int N = 1>
void propagateToR(const MP6x6SF &inErr, const MP6F &inPar, const MP1I &inChg, 
                  const MP3F &msP, MP6x6SF &outErr, MP6F &outPar) {
  //aux objects  
  MP6x6F errorProp;
  MP6x6F temp;
  
  auto PosInMtrx = [=] (int i, int j, int D, int bsz = 1) constexpr {return bsz*(i*D+j);};
  
  auto sincos4 = [] (const float x, float& sin, float& cos) {
    const float x2 = x*x;
    cos  = 1.f - 0.5f*x2 + 0.04166667f*x2*x2;
    sin  = x - 0.16666667f*x*x2;
  };

  
  for (int it = 0; it < N; ++it) {
    //initialize erroProp to identity matrix
    //for (size_t i=0;i<6;++i) errorProp.data[bsize*PosInMtrx(i,i,6) + it] = 1.f; 
    errorProp[PosInMtrx(0,0,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(1,1,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(2,2,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(3,3,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(4,4,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(5,5,6, N) + it] = 1.0f;
    //
    const float xin = inPar(iparX, it);
    const float yin = inPar(iparY, it);     
    const float zin = inPar(iparZ, it); 
    
    const float iptin   = inPar(iparIpt,   it);
    const float phiin   = inPar(iparPhi,   it);
    const float thetain = inPar(iparTheta, it); 
    //
    float r0 = hipo(xin, yin);
    const float k = inChg[it]*kfact;
    
    const float xmsP = msP(iparX, it);
    const float ymsP = msP(iparY, it);
    
    const float r = hipo(xmsP, ymsP);    
    
    outPar(iparX,it) = xin;
    outPar(iparY,it) = yin;
    outPar(iparZ,it) = zin;

    outPar(iparIpt,it)   = iptin;
    outPar(iparPhi,it)   = phiin;
    outPar(iparTheta,it) = thetain;
 
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
      const float xout = outPar(iparX, it);
      const float yout = outPar(iparY, it);     
      
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
      outPar(iparX,it) = xout + k*(pxin*sina - pyin*(1.f-cosa));
      outPar(iparY,it) = yout + k*(pyin*sina + pxin*(1.f-cosa));
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
 
    errorProp[PosInMtrx(0,0,6, N) + it] = 1.f+k*dadx*(cosPorT*cosa-sinPorT*sina)*pt;
    errorProp[PosInMtrx(0,1,6, N) + it] =     k*dady*(cosPorT*cosa-sinPorT*sina)*pt;
    errorProp[PosInMtrx(0,2,6, N) + it] = 0.f;
    errorProp[PosInMtrx(0,3,6, N) + it] = k*(cosPorT*(iptin*dadipt*cosa-sina)+sinPorT*((1.f-cosa)-iptin*dadipt*sina))*pt*pt;
    errorProp[PosInMtrx(0,4,6, N) + it] = k*(cosPorT*dadphi*cosa - sinPorT*dadphi*sina - sinPorT*sina + cosPorT*cosa - cosPorT)*pt;
    errorProp[PosInMtrx(0,5,6, N) + it] = 0.f;

    errorProp[PosInMtrx(1,0,6, N) + it] =     k*dadx*(sinPorT*cosa+cosPorT*sina)*pt;
    errorProp[PosInMtrx(1,1,6, N) + it] = 1.f+k*dady*(sinPorT*cosa+cosPorT*sina)*pt;
    errorProp[PosInMtrx(1,2,6, N) + it] = 0.f;
    errorProp[PosInMtrx(1,3,6, N) + it] = k*(sinPorT*(iptin*dadipt*cosa-sina)+cosPorT*(iptin*dadipt*sina-(1.f-cosa)))*pt*pt;
    errorProp[PosInMtrx(1,4,6, N) + it] = k*(sinPorT*dadphi*cosa + cosPorT*dadphi*sina + sinPorT*cosa + cosPorT*sina - sinPorT)*pt;
    errorProp[PosInMtrx(1,5,6, N) + it] = 0.f;

    //no trig approx here, theta can be large
    cosPorT=std::cos(thetain);
    sinPorT=std::sin(thetain);
    //redefine sinPorT as 1./sinPorT to reduce the number of temporaries
    sinPorT = 1.f/sinPorT;

    outPar(iparZ,it) = zin + k*alpha*cosPorT*pt*sinPorT;    

    errorProp[PosInMtrx(2,0,6, N) + it] = k*cosPorT*dadx*pt*sinPorT;
    errorProp[PosInMtrx(2,1,6, N) + it] = k*cosPorT*dady*pt*sinPorT;
    errorProp[PosInMtrx(2,2,6, N) + it] = 1.f;
    errorProp[PosInMtrx(2,3,6, N) + it] = k*cosPorT*(iptin*dadipt-alpha)*pt*pt*sinPorT;
    errorProp[PosInMtrx(2,4,6, N) + it] = k*dadphi*cosPorT*pt*sinPorT;
    errorProp[PosInMtrx(2,5,6, N) + it] =-k*alpha*pt*sinPorT*sinPorT;   
    //
    outPar(iparIpt,it) = iptin;
 
    errorProp[PosInMtrx(3,0,6, N) + it] = 0.f;
    errorProp[PosInMtrx(3,1,6, N) + it] = 0.f;
    errorProp[PosInMtrx(3,2,6, N) + it] = 0.f;
    errorProp[PosInMtrx(3,3,6, N) + it] = 1.f;
    errorProp[PosInMtrx(3,4,6, N) + it] = 0.f;
    errorProp[PosInMtrx(3,5,6, N) + it] = 0.f; 
    
    outPar(iparPhi,it) = phiin+alpha;
   
    errorProp[PosInMtrx(4,0,6, N) + it] = dadx;
    errorProp[PosInMtrx(4,1,6, N) + it] = dady;
    errorProp[PosInMtrx(4,2,6, N) + it] = 0.f;
    errorProp[PosInMtrx(4,3,6, N) + it] = dadipt;
    errorProp[PosInMtrx(4,4,6, N) + it] = 1.f+dadphi;
    errorProp[PosInMtrx(4,5,6, N) + it] = 0.f; 
  
    outPar(iparTheta,it) = thetain;        

    errorProp[PosInMtrx(5,0,6, N) + it] = 0.f;
    errorProp[PosInMtrx(5,1,6, N) + it] = 0.f;
    errorProp[PosInMtrx(5,2,6, N) + it] = 0.f;
    errorProp[PosInMtrx(5,3,6, N) + it] = 0.f;
    errorProp[PosInMtrx(5,4,6, N) + it] = 0.f;
    errorProp[PosInMtrx(5,5,6, N) + it] = 1.f; 
                                 
  }
  
  MultHelixProp<N>(errorProp, inErr, temp);
  MultHelixPropTransp<N>(errorProp, temp, outErr);  
  
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
   for (size_t lay=0; lay<nlayer; lay++){
     printf("hit in layer=%lu, pos: x=%f, y=%f, z=%f, r=%f \n", lay, inputhits[lay].pos[0], inputhits[lay].pos[1], inputhits[lay].pos[2], sqrtf(inputhits[lay].pos[0]*inputhits[lay].pos[0] + inputhits[lay].pos[1]*inputhits[lay].pos[1]));
   }
   
   printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
   printf("NITER=%d\n", NITER);

   long setup_start, setup_stop;
   struct timeval timecheck;
   //
   sycl::queue cq(sycl::gpu_selector{});
   //
   cl::sycl::usm_allocator<MPTRK, cl::sycl::usm::alloc::shared> MPTRKAllocator(cq);
   cl::sycl::usm_allocator<MPHIT, cl::sycl::usm::alloc::shared> MPHITAllocator(cq);
   //
   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
   //
   std::vector<MPTRK, decltype(MPTRKAllocator)> trcks(nevts*nb, MPTRKAllocator); 
   prepareTracks<decltype(MPTRKAllocator)>(trcks, inputtrk);
   //
   std::vector<MPHIT, decltype(MPHITAllocator)> hits(nlayer*nevts*nb, MPHITAllocator);
   prepareHits<decltype(MPHITAllocator)>(hits, inputhits);
   //
   std::vector<MPTRK, decltype(MPTRKAllocator)> outtrcks(nevts*nb, MPTRKAllocator);
   
   auto policy = oneapi::dpl::execution::make_device_policy(cq);
   //auto policy = oneapi::dpl::execution::device_policy{sycl::cpu_selector{}};
   //auto policy = oneapi::dpl::execution::device_policy{sycl::gpu_selector{}};

   auto p2r_kernels = [=,btracksPtr    = trcks.data(),
                         outtracksPtr  = outtrcks.data(),
                         bhitsPtr      = hits.data()] (const int i) {
                         //  
                         MPTRK btracks;
                         MPTRK obtracks;
                         MPHIT bhits;
                         //
                         btracksPtr[i].load(btracks);
                         //
                         for(int layer=0; layer<nlayer; ++layer) {
                           //
                           bhitsPtr[layer+nlayer*i].load(bhits);
                           //
                           propagateToR<bsize>(btracks.cov, btracks.par, btracks.q, bhits.pos, obtracks.cov, obtracks.par);
                           KalmanUpdate<bsize>(obtracks.cov, obtracks.par, bhits.cov, bhits.pos);
                           //
                         }
                         //
                         outtracksPtr[i].save(obtracks);
                       };

   const int outer_loop_range = nevts*nb;

   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");

   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nevts*nb*sizeof(MPHIT));

   // A warmup run to migrate data on the device:
   std::for_each(policy,
                 counting_iterator(0),
                 counting_iterator(outer_loop_range),
                 p2r_kernels);

   auto wall_start = std::chrono::high_resolution_clock::now();

   for(int itr=0; itr<NITER; itr++) {
     oneapi::dpl::experimental::for_each_async(policy,
                                               counting_iterator(0),
                                               counting_iterator(outer_loop_range),
                                               p2r_kernels);
   } //end of itr loop

   cq.wait();

   auto wall_stop = std::chrono::high_resolution_clock::now();

   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;   

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
