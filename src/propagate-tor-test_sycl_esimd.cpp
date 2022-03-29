/*
export PSTL_USAGE_WARNINGS=1
export ONEDPL_USE_DPCPP_BACKEND=1

clang++ -fsycl -O3 -std=c++17 src/propagate-toz-test_sycl_esimd.cpp -o test-sycl.exe -Dntrks=8192 -Dnevts=100 -DNITER=5 -Dbsize=1 -Dnlayer=20

*/

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

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel;

#ifndef bsize
constexpr int bSize = 8;
#else
constexpr int bSize = bsize;
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

template <typename T, int N, int bSize_>
struct MPNX {
   std::array<T,N*bSize_> data;
   //basic accessors
   const T& operator[](const int idx) const {return data[idx];}
   T& operator[](const int idx) {return data[idx];}
   const T& operator()(const int m, const int b) const {return data[m*bSize+b];}
   T& operator()(const int m, const int b) {return data[m*bSize+b];}
   //
   //std::enable_if_t<std::is_arithmetic_v<T>,void> load(MPNX& dst) const{
   void load(MPNX& dst) const{
     for (int it=0;it<bSize_;++it) {
     //const int l = it+ib*bsize+ie*nb*bsize;
       for (int ip=0;ip<N;++ip) {    	
    	 dst.data[it + ip*bSize_] = this->operator()(ip, it);  
       }
     }//
     
     return;
   }
   
   [[intel::sycl_explicit_simd]] void simd_load(MPNX<simd<T, bSize_>, N, 1>& dst){
     //
     {
     //const int l = it+ib*bsize+ie*nb*bsize;
       for (int ip=0;ip<N;++ip) {    	
    	 dst.data[ip] = block_load<T, bSize_>(&data[ip*bSize_]); //this->operator()(ip, 0);  
       }
     }//
     
     return;
   }

   //std::enable_if_t<std::is_arithmetic_v<T>,void> save(const MPNX& src) {
   void save(const MPNX& src) {
     for (int it=0;it<bSize_;++it) {
     //const int l = it+ib*bsize+ie*nb*bsize;
       for (int ip=0;ip<N;++ip) {    	
    	 this->operator()(ip, it) = src.data[it + ip*bSize_];  
       }
     }//
     
     return;
   }
//
   [[intel::sycl_explicit_simd]] void simd_save(const MPNX<simd<T, bSize_>, N, 1>& src) {
     {
     //const int l = it+ib*bsize+ie*nb*bsize;
       for (int ip=0;ip<N;++ip) {    	
    	 block_store<T, bSize_>(&data[ip*bSize_], src.data[ip]); 
       }
     }//
     
     return;
   }

};

using MP1I    = MPNX<int,   1 , bSize>;
using MP1F    = MPNX<float, 1 , bSize>;
using MP2F    = MPNX<float, 3 , bSize>;
using MP3F    = MPNX<float, 3 , bSize>;
using MP6F    = MPNX<float, 6 , bSize>;
using MP2x2SF = MPNX<float, 3 , bSize>;
using MP3x3SF = MPNX<float, 6 , bSize>;
using MP6x6SF = MPNX<float, 21, bSize>;
using MP6x6F  = MPNX<float, 36, bSize>;
using MP3x3   = MPNX<float, 9 , bSize>;
using MP3x6   = MPNX<float, 18, bSize>;

// Native fields:
using MP1I_    = MPNX<simd<int, bSize>,   1 , 1>;
using MP1F_    = MPNX<simd<float, bSize>, 1 , 1>;
using MP2F_    = MPNX<simd<float, bSize>, 3 , 1>;
using MP3F_    = MPNX<simd<float, bSize>, 3 , 1>;
using MP6F_    = MPNX<simd<float, bSize>, 6 , 1>;
using MP2x2SF_ = MPNX<simd<float, bSize>, 3 , 1>;
using MP3x3SF_ = MPNX<simd<float, bSize>, 6 , 1>;
using MP6x6SF_ = MPNX<simd<float, bSize>, 21, 1>;
using MP6x6F_  = MPNX<simd<float, bSize>, 36, 1>;
using MP3x3_   = MPNX<simd<float, bSize>, 9 , 1>;
using MP3x6_   = MPNX<simd<float, bSize>, 18, 1>;

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

  //  MP22I   hitidx;
  void load(MPTRK &dst){
    par.load(dst.par);
    cov.load(dst.cov);
    q.load(dst.q);    
    return;	  
  }
  //
  void simd_load(MPTRK_ &dst){
    //
    par.simd_load(dst.par);
    cov.simd_load(dst.cov);
    q.simd_load(dst.q);
    //   
    return;	  
  }
  //
  void save(const MPTRK &src){
    par.save(src.par);
    cov.save(src.cov);
    q.save(src.q);
    return;
  }
  //
  void simd_save(const MPTRK_ &src){
    //
    par.simd_save(src.par);
    cov.simd_save(src.cov);
    q.simd_save(src.q);
    //
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
  
  void simd_load(MPHIT_ &dst){
    //
    pos.simd_load(dst.pos);
    cov.simd_load(dst.cov);
    //
    return;
  }
  
  void save(const MPHIT &src){
    pos.save(src.pos);
    cov.save(src.cov);

    return;
  }
  
  void simd_save(const MPHIT_ &src){
    //
    pos.simd_save(src.pos);
    cov.simd_save(src.cov);
    //
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
    W = std::pow (U1, 2) + std::pow (U2, 2);
  }
  while (W >= 1 || W == 0); 
  mult = std::sqrt ((-2 * std::log (W)) / W);
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

[[intel::sycl_explicit_simd]] inline void MultHelixProp(const MP6x6F_ &a, const MP6x6SF_ &b, MP6x6F_ &c) {
  
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
  
  return;
}

[[intel::sycl_explicit_simd]] inline void MultHelixPropTransp(const MP6x6F_ &a, const MP6x6F_ &b, MP6x6SF_ &c) {//  
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
    
  return;  
}


template <int N = bSize>
[[intel::sycl_explicit_simd]] void KalmanUpdate(MP6x6SF_ &trkErr, MP6F_ &inPar, const MP3x3SF_ &hitErr, const MP3F_ &msP){	  
  
  using FloatN = simd<float, N>;
  
  MP1F_    rotT00;
  MP1F_    rotT01;
  MP2x2SF_ resErr_loc;
  //MP3x3SF_ resErr_glo;
    
  {   
    const FloatN msPX = msP[iparX];
    const FloatN msPY = msP[iparY];
    const FloatN inParX = inPar[iparX];
    const FloatN inParY = inPar[iparY];  
    
    FloatN tmp1 = msPX*msPX + msPY*msPY; 
    FloatN tmp2 = esimd::sqrt(tmp1);
    
    tmp1 = esimd::inv(tmp2);       
  
    const FloatN inv_2r = 0.5*tmp1;

    rotT00[0] = -(msPY + inParY) * inv_2r;
    rotT01[0] =  (msPX + inParX) * inv_2r;    
    
    resErr_loc[ 0] = (rotT00[0]*(trkErr[0] + hitErr[0]) +
                                    rotT01[0]*(trkErr[1] + hitErr[1]))*rotT00[0] +
                                   (rotT00[0]*(trkErr[1] + hitErr[1]) +
                                    rotT01[0]*(trkErr[2] + hitErr[2]))*rotT01[0];
    resErr_loc[ 1] = (trkErr[3] + hitErr[3])*rotT00[0] +
                                   (trkErr[4] + hitErr[4])*rotT01[0];
    resErr_loc[ 2] = (trkErr[5] + hitErr[5]);
  } 
  
  {
  
    const FloatN det = resErr_loc[0] * resErr_loc[2] -
                       resErr_loc[1] * resErr_loc[1];
    const FloatN s   = esimd::inv( det );
    const FloatN tmp = s * resErr_loc[2];
    resErr_loc[1]  = -(s * resErr_loc[1]);
    resErr_loc[2]  =  (s * resErr_loc[0]);
    resErr_loc[0]  = tmp;  
  }     
  
  MP3x6_ kGain;
  
  {
    kGain[ 0] = trkErr[ 0]*(rotT00[0]*resErr_loc[ 0]) +
	                        trkErr[ 1]*(rotT01[0]*resErr_loc[ 0]) +
	                        trkErr[ 3]*resErr_loc[ 1];
    kGain[ 1] = trkErr[ 0]*(rotT00[0]*resErr_loc[ 1]) +
	                        trkErr[ 1]*(rotT01[0]*resErr_loc[ 1]) +
	                        trkErr[ 3]*resErr_loc[ 2];
    kGain[ 2] = 0.f;
    kGain[ 3] = trkErr[ 1]*(rotT00[0]*resErr_loc[ 0]) +
	                        trkErr[ 2]*(rotT01[0]*resErr_loc[ 0]) +
	                        trkErr[ 4]*resErr_loc[ 1];
    kGain[ 4] = trkErr[ 1]*(rotT00[0]*resErr_loc[ 1]) +
	                        trkErr[ 2]*(rotT01[0]*resErr_loc[ 1]) +
	                        trkErr[ 4]*resErr_loc[ 2];
    kGain[ 5] = 0.f;
    kGain[ 6] = trkErr[ 3]*(rotT00[0]*resErr_loc[ 0]) +
	                        trkErr[ 4]*(rotT01[0]*resErr_loc[ 0]) +
	                        trkErr[ 5]*resErr_loc[ 1];
    kGain[ 7] = trkErr[ 3]*(rotT00[0]*resErr_loc[ 1]) +
	                        trkErr[ 4]*(rotT01[0]*resErr_loc[ 1]) +
	                        trkErr[ 5]*resErr_loc[ 2];
    kGain[ 8] = 0.f;
    kGain[ 9] = trkErr[ 6]*(rotT00[0]*resErr_loc[ 0]) +
	                        trkErr[ 7]*(rotT01[0]*resErr_loc[ 0]) +
	                        trkErr[ 8]*resErr_loc[ 1];
    kGain[10] = trkErr[ 6]*(rotT00[0]*resErr_loc[ 1]) +
	                        trkErr[ 7]*(rotT01[0]*resErr_loc[ 1]) +
	                        trkErr[ 8]*resErr_loc[ 2];
    kGain[11] = 0.f;
    kGain[12] = trkErr[10]*(rotT00[0]*resErr_loc[ 0]) +
	                        trkErr[11]*(rotT01[0]*resErr_loc[ 0]) +
	                        trkErr[12]*resErr_loc[ 1];
    kGain[13] = trkErr[10]*(rotT00[0]*resErr_loc[ 1]) +
	                        trkErr[11]*(rotT01[0]*resErr_loc[ 1]) +
	                        trkErr[12]*resErr_loc[ 2];
    kGain[14] = 0.f;
    kGain[15] = trkErr[15]*(rotT00[0]*resErr_loc[ 0]) +
	                        trkErr[16]*(rotT01[0]*resErr_loc[ 0]) +
	                        trkErr[17]*resErr_loc[ 1];
    kGain[16] = trkErr[15]*(rotT00[0]*resErr_loc[ 1]) +
	                        trkErr[16]*(rotT01[0]*resErr_loc[ 1]) +
	                        trkErr[17]*resErr_loc[ 2];
    kGain[17] = 0.f;  
  }  
     
  MP2F_ res_loc;   
  {
    const FloatN msPX = msP[iparX];
    const FloatN msPY = msP[iparY];
    const FloatN msPZ = msP[iparZ];    
    const FloatN inParX = inPar[iparX];
    const FloatN inParY = inPar[iparY];     
    const FloatN inParZ = inPar[iparZ]; 
    
    const FloatN inParIpt   = inPar[iparIpt];
    const FloatN inParPhi   = inPar[iparPhi];
    const FloatN inParTheta = inPar[iparTheta];            
    
    res_loc[0] =  rotT00[0]*(msPX - inParX) + rotT01[0]*(msPY - inParY);
    res_loc[1] =  msPZ - inParZ;

    inPar[iparX]     = inParX     + kGain[ 0] * res_loc[ 0] + kGain[ 1] * res_loc[ 1];
    inPar[iparY]     = inParY     + kGain[ 3] * res_loc[ 0] + kGain[ 4] * res_loc[ 1];
    inPar[iparZ]     = inParZ     + kGain[ 6] * res_loc[ 0] + kGain[ 7] * res_loc[ 1];
    inPar[iparIpt]   = inParIpt   + kGain[ 9] * res_loc[ 0] + kGain[10] * res_loc[ 1];
    inPar[iparPhi]   = inParPhi   + kGain[12] * res_loc[ 0] + kGain[13] * res_loc[ 1];
    inPar[iparTheta] = inParTheta + kGain[15] * res_loc[ 0] + kGain[16] * res_loc[ 1];     
  }

   MP6x6SF_ newErr;
   {

     newErr[ 0] = kGain[ 0]*rotT00[0]*trkErr[ 0] +
                         kGain[ 0]*rotT01[0]*trkErr[ 1] +
                         kGain[ 1]*trkErr[ 3];
     newErr[ 1] = kGain[ 3]*rotT00[0]*trkErr[ 0] +
                         kGain[ 3]*rotT01[0]*trkErr[ 1] +
                         kGain[ 4]*trkErr[ 3];
     newErr[ 2] = kGain[ 3]*rotT00[0]*trkErr[ 1] +
                         kGain[ 3]*rotT01[0]*trkErr[ 2] +
                         kGain[ 4]*trkErr[ 4];
     newErr[ 3] = kGain[ 6]*rotT00[0]*trkErr[ 0] +
                         kGain[ 6]*rotT01[0]*trkErr[ 1] +
                         kGain[ 7]*trkErr[ 3];
     newErr[ 4] = kGain[ 6]*rotT00[0]*trkErr[ 1] +
                         kGain[ 6]*rotT01[0]*trkErr[ 2] +
                         kGain[ 7]*trkErr[ 4];
     newErr[ 5] = kGain[ 6]*rotT00[0]*trkErr[ 3] +
                         kGain[ 6]*rotT01[0]*trkErr[ 4] +
                         kGain[ 7]*trkErr[ 5];
     newErr[ 6] = kGain[ 9]*rotT00[0]*trkErr[ 0] +
                         kGain[ 9]*rotT01[0]*trkErr[ 1] +
                         kGain[10]*trkErr[ 3];
     newErr[ 7] = kGain[ 9]*rotT00[0]*trkErr[ 1] +
                         kGain[ 9]*rotT01[0]*trkErr[ 2] +
                         kGain[10]*trkErr[ 4];
     newErr[ 8] = kGain[ 9]*rotT00[0]*trkErr[ 3] +
                         kGain[ 9]*rotT01[0]*trkErr[ 4] +
                         kGain[10]*trkErr[ 5];
     newErr[ 9] = kGain[ 9]*rotT00[0]*trkErr[ 6] +
                         kGain[ 9]*rotT01[0]*trkErr[ 7] +
                         kGain[10]*trkErr[ 8];
     newErr[10] = kGain[12]*rotT00[0]*trkErr[ 0] +
                         kGain[12]*rotT01[0]*trkErr[ 1] +
                         kGain[13]*trkErr[ 3];
     newErr[11] = kGain[12]*rotT00[0]*trkErr[ 1] +
                         kGain[12]*rotT01[0]*trkErr[ 2] +
                         kGain[13]*trkErr[ 4];
     newErr[12] = kGain[12]*rotT00[0]*trkErr[ 3] +
                         kGain[12]*rotT01[0]*trkErr[ 4] +
                         kGain[13]*trkErr[ 5];
     newErr[13] = kGain[12]*rotT00[0]*trkErr[ 6] +
                         kGain[12]*rotT01[0]*trkErr[ 7] +
                         kGain[13]*trkErr[ 8];
     newErr[14] = kGain[12]*rotT00[0]*trkErr[10] +
                         kGain[12]*rotT01[0]*trkErr[11] +
                         kGain[13]*trkErr[12];
     newErr[15] = kGain[15]*rotT00[0]*trkErr[ 0] +
                         kGain[15]*rotT01[0]*trkErr[ 1] +
                         kGain[16]*trkErr[ 3];
     newErr[16] = kGain[15]*rotT00[0]*trkErr[ 1] +
                         kGain[15]*rotT01[0]*trkErr[ 2] +
                         kGain[16]*trkErr[ 4];
     newErr[17] = kGain[15]*rotT00[0]*trkErr[ 3] +
                         kGain[15]*rotT01[0]*trkErr[ 4] +
                         kGain[16]*trkErr[ 5];
     newErr[18] = kGain[15]*rotT00[0]*trkErr[ 6] +
                         kGain[15]*rotT01[0]*trkErr[ 7] +
                         kGain[16]*trkErr[ 8];
     newErr[19] = kGain[15]*rotT00[0]*trkErr[10] +
                         kGain[15]*rotT01[0]*trkErr[11] +
                         kGain[16]*trkErr[12];
     newErr[20] = kGain[15]*rotT00[0]*trkErr[15] +
                         kGain[15]*rotT01[0]*trkErr[16] +
                         kGain[16]*trkErr[17];     

     for (int i = 0; i < 21; i++){
       trkErr[ i] = trkErr[ i] - newErr[ i];
     }
   }
   //
   return;                 
}            

constexpr auto kfact= 100/(-0.299792458f*3.8112f);
//constexpr float kfact= 100/3.8f;
constexpr int Niter=5;//5

template<int N = bSize>
[[intel::sycl_explicit_simd]] void propagateToR(const MP6x6SF_ &inErr, const MP6F_ &inPar, const MP1I_ &inChg, 
                  const MP3F_ &msP, MP6x6SF_ &outErr, MP6F_ &outPar) { 
                  
  using FloatN = simd<float, N>;                
  //aux objects  
  MP6x6F_ errorProp;
  MP6x6F_ temp;
  
  const FloatN zero = 0.f;
  
  auto PosInMtrx = [] (int i, int j, int nd) constexpr {return (i*nd+j);};
  
  auto sincos4 = [=] (const FloatN x, FloatN& sin, FloatN& cos) {
    const FloatN x2 = x*x;
    //
    const FloatN c0(+1.0f);
    const FloatN c1(-0.5f);
    const FloatN c2(+0.04166667f);
    const FloatN c3(-0.16666667f);
    // 
    cos  = c0 + c1*x2 + c2*x2*x2;
    sin  = x * (c0 + c3*x2);
  };
  
  auto hipo = [](const FloatN x, const FloatN y) {return esimd::sqrt(x*x + y*y);};
                  
  { 
    errorProp[PosInMtrx(0,0,6)] = 1.f;
    errorProp[PosInMtrx(1,1,6)] = 1.f;
    errorProp[PosInMtrx(2,2,6)] = 1.f;
    errorProp[PosInMtrx(3,3,6)] = 1.f;
    errorProp[PosInMtrx(4,4,6)] = 1.f;
    errorProp[PosInMtrx(5,5,6)] = 1.f;
    //
    const FloatN xin = inPar[iparX];
    const FloatN yin = inPar[iparY];     
    const FloatN zin = inPar[iparZ]; 
    
    const FloatN iptin   = inPar[iparIpt];
    const FloatN phiin   = inPar[iparPhi];
    const FloatN thetain = inPar[iparTheta]; 
    //
    FloatN r0 = hipo(xin, yin);
    
    simd_mask<N> predmsk = r0 > 0.f;
    //do basic regularization:
    FloatN invr0(1.0f);
    invr0.merge(zero, predmsk);//replace one's with zero's
    r0 = r0 + invr0;//replaced zero by one. keep the remaining unchanged  
    //
    invr0 = esimd::inv(r0); 
    
    const FloatN k = inChg[0]*kfact;
    
    const FloatN xmsP = msP[iparX];
    const FloatN ymsP = msP[iparY];
    
    const FloatN r = hipo(xmsP, ymsP);    
    
    outPar[iparX] = xin;
    outPar[iparY] = yin;
    outPar[iparZ] = zin;

    outPar[iparIpt]   = iptin;
    outPar[iparPhi]   = phiin;
    outPar[iparTheta] = thetain;

    const FloatN kinv  = esimd::inv(k);
    const FloatN pt    = esimd::inv(iptin);
    
    FloatN D = 0.f, cosa = 0.f, sina = 0.f, id = 0.f;
    //no trig approx here, phi can be large
    FloatN cosPorT = esimd::cos(phiin);
    FloatN sinPorT = esimd::sin(phiin);
    FloatN pxin = cosPorT*pt;
    FloatN pyin = sinPorT*pt;
    
    //derivatives initialized to value for first iteration, i.e. distance = r-r0in
    //FloatN dDdx = r0 > 0.f ? -xin/r0 : 0.f;
    //FloatN dDdy = r0 > 0.f ? -yin/r0 : 0.f;
    FloatN dDdx(0.f);
    FloatN dDdy(0.f);
    
    FloatN scaled_invr0 = -(xin*invr0);
    
    dDdx.merge(scaled_invr0, predmsk);
    
    scaled_invr0 = -(yin*invr0);
    
    dDdy.merge(scaled_invr0, predmsk);
    //
    FloatN dDdipt = 0.f;
    FloatN dDdphi = 0.f; 
    
    const FloatN iptinxkinv = iptin*kinv; 

#pragma unroll  
    for (int i = 0; i < Niter; ++i)
    {
     //compute distance and path for the current iteration
      const FloatN xout = outPar[iparX];
      const FloatN yout = outPar[iparY];     
      
      r0 = hipo(xout, yout);
      id = (r-r0);
      D+=id;
      
      const FloatN idxiptinxkinv = id*iptinxkinv; 
      sincos4(idxiptinxkinv, sina, cosa);

      //update derivatives on total distance
      if (i+1 != Niter) { 
 
	//const FloatN oor0 = (r0>0.f && std::abs(r-r0)<0.0001f) ? esimd::inv(r0) : 0.f;//?
	predmsk = (r0 > 0.f);
	//
        invr0 = 1.0f;
        invr0.merge(zero, predmsk);//replace one's with zero's
        r0 = r0 + invr0;//replaced zero by one. keep the remaining unchanged  
        //
        invr0 = esimd::inv(r0); 
	//
	predmsk = predmsk && (esimd::abs(id) < 0.0001f); 
	//
	FloatN oor0(0.f);
	oor0.merge(invr0, predmsk);

	const FloatN dadipt = id*kinv;

	const FloatN dadx = -xout*iptinxkinv*oor0;
	const FloatN dady = -yout*iptinxkinv*oor0;

	const FloatN pxca = pxin*cosa;
	const FloatN pxsa = pxin*sina;
	const FloatN pyca = pyin*cosa;
	const FloatN pysa = pyin*sina;

	FloatN tmp = k*dadx;
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
      outPar[iparX] = xout + k*(pxin*sina - pyin*(1.f-cosa));
      outPar[iparY] = yout + k*(pyin*sina + pxin*(1.f-cosa));
      const FloatN pxinold = pxin;//copy before overwriting
      pxin = pxin*cosa - pyin*sina;
      pyin = pyin*cosa + pxinold*sina;
  
    }

    const FloatN alpha  = D*iptinxkinv;
    const FloatN dadx   = dDdx*iptinxkinv;
    const FloatN dady   = dDdy*iptinxkinv;
    const FloatN dadipt = (iptin*dDdipt + D)*kinv;
    const FloatN dadphi = dDdphi*iptinxkinv;

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
    cosPorT=esimd::cos(thetain);
    sinPorT=esimd::sin(thetain);
    //redefine sinPorT as 1./sinPorT to reduce the number of temporaries
    sinPorT = esimd::inv(sinPorT);

    outPar[iparZ] = zin + k*alpha*cosPorT*pt*sinPorT;    

    errorProp[PosInMtrx(2,0,6)] = k*cosPorT*dadx*pt*sinPorT;
    errorProp[PosInMtrx(2,1,6)] = k*cosPorT*dady*pt*sinPorT;
    errorProp[PosInMtrx(2,2,6)] = 1.f;
    errorProp[PosInMtrx(2,3,6)] = k*cosPorT*(iptin*dadipt-alpha)*pt*pt*sinPorT;
    errorProp[PosInMtrx(2,4,6)] = k*dadphi*cosPorT*pt*sinPorT;
    errorProp[PosInMtrx(2,5,6)] =-k*alpha*pt*sinPorT*sinPorT;   
    //
    outPar[iparIpt] = iptin;
 
    errorProp[PosInMtrx(3,0,6)] = 0.f;
    errorProp[PosInMtrx(3,1,6)] = 0.f;
    errorProp[PosInMtrx(3,2,6)] = 0.f;
    errorProp[PosInMtrx(3,3,6)] = 1.f;
    errorProp[PosInMtrx(3,4,6)] = 0.f;
    errorProp[PosInMtrx(3,5,6)] = 0.f; 
    
    outPar[iparPhi] = phiin+alpha;
    
    errorProp[PosInMtrx(4,0,6)] = dadx;
    errorProp[PosInMtrx(4,1,6)] = dady;
    errorProp[PosInMtrx(4,2,6)] = 0.f;
    errorProp[PosInMtrx(4,3,6)] = dadipt;
    errorProp[PosInMtrx(4,4,6)] = 1.f+dadphi;
    errorProp[PosInMtrx(4,5,6)] = 0.f; 
  
    outPar[iparTheta] = thetain;        

    errorProp[PosInMtrx(5,0,6)] = 0.f;
    errorProp[PosInMtrx(5,1,6)] = 0.f;
    errorProp[PosInMtrx(5,2,6)] = 0.f;
    errorProp[PosInMtrx(5,3,6)] = 0.f;
    errorProp[PosInMtrx(5,4,6)] = 0.f;
    errorProp[PosInMtrx(5,5,6)] = 1.f; 
  }

  MultHelixProp(errorProp, inErr, temp);
  MultHelixPropTransp(errorProp, temp, outErr);
  
  return;
}

class ESIMDSelector : public device_selector {
  // Require GPU device unless HOST is requested in SYCL_DEVICE_FILTER env
  virtual int operator()(const sycl::device &device) const {
    if (const char *dev_filter = getenv("SYCL_DEVICE_FILTER")) {
      std::string filter_string(dev_filter);
      if (filter_string.find("gpu") != std::string::npos)
        return device.is_gpu() ? 1000 : -1;
      if (filter_string.find("host") != std::string::npos)
        return device.is_host() ? 1000 : -1;
      std::cerr
          << "Supported 'SYCL_DEVICE_FILTER' env var values are 'gpu' and "
             "'host', '"
          << filter_string << "' does not contain such substrings.\n";
      return -1;
    }
    // If "SYCL_DEVICE_FILTER" not defined, only allow gpu device
    return device.is_gpu() ? 1000 : -1;
  }
};

auto exception_handler = [](exception_list l) {
  for (auto ep : l) {
    try {
      std::rethrow_exception(ep);
    } catch (cl::sycl::exception &e0) {
      std::cout << "sycl::exception: " << e0.what() << std::endl;
    } catch (std::exception &e) {
      std::cout << "std::exception: " << e.what() << std::endl;
    } catch (...) {
      std::cout << "generic exception\n";
    }
  }
};


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
   sycl::queue cq(ESIMDSelector{}, exception_handler); //(sycl::gpu_selector{});
   //
   cl::sycl::usm_allocator<MPTRK, cl::sycl::usm::alloc::shared, 16u> MPTRKAllocator(cq);
   cl::sycl::usm_allocator<MPHIT, cl::sycl::usm::alloc::shared, 16u> MPHITAllocator(cq);
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
 
   constexpr unsigned outer_loop_range = nevts*ntrks;
   //  
   constexpr unsigned GroupSize = 4;
   // We need that many task groups
   cl::sycl::range<1> GroupRange{outer_loop_range / bSize};
   // We need that many tasks in each group
   cl::sycl::range<1> TaskRange{GroupSize};
   //
   cl::sycl::nd_range<1> Range{GroupRange, TaskRange};
 
   auto p2r_kernels = [=,btracksPtr    = trcks.data(),
                         outtracksPtr  = outtrcks.data(),
                         bhitsPtr      = hits.data()] (const nd_item<1> ndi) [[intel::sycl_explicit_simd]] {
                         //  
                         const int i = ndi.get_global_id(0);
                         //
                         MPTRK_ btracks;
                         MPTRK_ obtracks;
                         MPHIT_ bhits;
                         //
                         btracksPtr[i].simd_load(btracks);
                         //
                         for(int layer=0; layer<nlayer; ++layer) {
                           //
                           bhitsPtr[layer+nlayer*i].simd_load(bhits);
                           //
                           propagateToR<bSize>(btracks.cov, btracks.par, btracks.q, bhits.pos, obtracks.cov, obtracks.par);
                           KalmanUpdate<bSize>(obtracks.cov, obtracks.par, bhits.cov, bhits.pos);
                           //
                         }
                         //
                         outtracksPtr[i].simd_save(obtracks);
                       };

   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");

   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nevts*nb*sizeof(MPHIT));

   // A warmup run to migrate data on the device:
   cq.submit([&](sycl::handler &h){
       h.parallel_for(Range, p2r_kernels);
     });

   cq.wait();  

   auto wall_start = std::chrono::high_resolution_clock::now();

   for(int itr=0; itr<NITER; itr++) {
     cq.submit([&](sycl::handler &h){
       h.parallel_for(Range, p2r_kernels);
     });
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
#if 0
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
#endif
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
