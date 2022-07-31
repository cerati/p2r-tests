/*
see README.txt for instructions
*/

#include <cudaCheck.h>
#include <cuda_profiler_api.h>
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#ifndef EXCLUDE_H2D_TRANSFER
#define MEASURE_H2D_TRANSFER
#endif
#ifndef EXCLUDE_D2H_TRANSFER
#define MEASURE_D2H_TRANSFER
#endif

#ifndef ntrks
#define ntrks 8192 
#endif

#define nb    (ntrks)

#ifndef nevts
#define nevts 100
#endif
#define smear 0.00001 

#ifndef NITER
#define NITER 5
#endif
#ifndef NWARMUP
#define NWARMUP 2
#endif

#ifndef nlayer
#define nlayer 20
#endif

#ifndef num_streams
#define num_streams 1
#endif

#ifndef threadsperblockx
#define threadsperblockx 32
#endif
#ifndef blockspergrid
#define blockspergrid (nevts*nb)
#endif

#ifndef nthreads
#define nthreads 1
#endif

#define HOSTDEV __host__ __device__

HOSTDEV constexpr size_t PosInMtrx(size_t i, size_t j, size_t D) {
  return i*D+j;
}

HOSTDEV size_t SymOffsets33(size_t i) {
  const size_t offs[9] = {0, 1, 3, 1, 2, 4, 3, 4, 5};
  return offs[i];
}

HOSTDEV size_t SymOffsets66(size_t i) {
  const size_t offs[36] = {0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20};
  return offs[i];
}

struct ATRK {
  float par[6];
  float cov[21];
  int q;
  //  int hitidx[22];
};

struct AHIT {
  float pos[3];
  float cov[6];
};

struct MP1I {
  int data[1];
};

struct MP22I {
  int data[22];
};

struct MP1F {
  float data[1];
};

struct MP2F {
  float data[2];
};

struct MP3F {
  float data[3];
};

struct MP6F {
  float data[6];
};

struct MP3x3 {
  float data[9];
};
struct MP3x6 {
  float data[18];
};

struct MP2x2SF {
  float data[3];
};

struct MP3x3SF {
  float data[6];
};

struct MP6x6SF {
  float data[21];
};

struct MP6x6F {
  float data[36];
};

struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;
  //  MP22I   hitidx;
};

struct MPHIT {
  MP3F    pos;
  MP3x3SF cov;
};

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

HOSTDEV MPTRK* bTk(MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

HOSTDEV const MPTRK* bTk(const MPTRK* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

HOSTDEV int q(const MP1I* bq, size_t it){
  return (*bq).data[it];
}
//
HOSTDEV float par(const MP6F* bpars, size_t it, size_t ipar){
  return (*bpars).data[it + ipar];
}
HOSTDEV float x    (const MP6F* bpars, size_t it){ return par(bpars, it, 0); }
HOSTDEV float y    (const MP6F* bpars, size_t it){ return par(bpars, it, 1); }
HOSTDEV float z    (const MP6F* bpars, size_t it){ return par(bpars, it, 2); }
HOSTDEV float ipt  (const MP6F* bpars, size_t it){ return par(bpars, it, 3); }
HOSTDEV float phi  (const MP6F* bpars, size_t it){ return par(bpars, it, 4); }
HOSTDEV float theta(const MP6F* bpars, size_t it){ return par(bpars, it, 5); }
//
HOSTDEV float par(const MPTRK* btracks, size_t it, size_t ipar){
  return par(&(*btracks).par,it,ipar);
}
HOSTDEV float x    (const MPTRK* btracks, size_t it){ return par(btracks, it, 0); }
HOSTDEV float y    (const MPTRK* btracks, size_t it){ return par(btracks, it, 1); }
HOSTDEV float z    (const MPTRK* btracks, size_t it){ return par(btracks, it, 2); }
HOSTDEV float ipt  (const MPTRK* btracks, size_t it){ return par(btracks, it, 3); }
HOSTDEV float phi  (const MPTRK* btracks, size_t it){ return par(btracks, it, 4); }
HOSTDEV float theta(const MPTRK* btracks, size_t it){ return par(btracks, it, 5); }
//
HOSTDEV float par(const MPTRK* tracks, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk;
  const MPTRK* btracks = bTk(tracks, ev, ib);
  size_t it = 0 ;
  return par(btracks, it, ipar);
}
HOSTDEV float x    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 0); }
HOSTDEV float y    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 1); }
HOSTDEV float z    (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 2); }
HOSTDEV float ipt  (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 3); }
HOSTDEV float phi  (const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 4); }
HOSTDEV float theta(const MPTRK* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 5); }
//
HOSTDEV void setpar(MP6F* bpars, size_t it, size_t ipar, float val){
  (*bpars).data[it + ipar] = val;
}
HOSTDEV void setx    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 0, val); }
HOSTDEV void sety    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 1, val); }
HOSTDEV void setz    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 2, val); }
HOSTDEV void setipt  (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 3, val); }
HOSTDEV void setphi  (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 4, val); }
HOSTDEV void settheta(MP6F* bpars, size_t it, float val){ setpar(bpars, it, 5, val); }
//
HOSTDEV void setpar(MPTRK* btracks, size_t it, size_t ipar, float val){
  setpar(&(*btracks).par,it,ipar,val);
}
HOSTDEV void setx    (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 0, val); }
HOSTDEV void sety    (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 1, val); }
HOSTDEV void setz    (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 2, val); }
HOSTDEV void setipt  (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 3, val); }
HOSTDEV void setphi  (MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 4, val); }
HOSTDEV void settheta(MPTRK* btracks, size_t it, float val){ setpar(btracks, it, 5, val); }

HOSTDEV const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
HOSTDEV const MPHIT* bHit(const MPHIT* hits, size_t ev, size_t ib,size_t lay) {
return &(hits[lay + (ib*nlayer) +(ev*nlayer*nb)]);
}
//
HOSTDEV float pos(const MP3F* hpos, size_t it, size_t ipar){
  return (*hpos).data[it + ipar];
}
HOSTDEV float x(const MP3F* hpos, size_t it)    { return pos(hpos, it, 0); }
HOSTDEV float y(const MP3F* hpos, size_t it)    { return pos(hpos, it, 1); }
HOSTDEV float z(const MP3F* hpos, size_t it)    { return pos(hpos, it, 2); }
//
HOSTDEV float pos(const MPHIT* hits, size_t it, size_t ipar){
  return pos(&(*hits).pos,it,ipar);
}
HOSTDEV float x(const MPHIT* hits, size_t it)    { return pos(hits, it, 0); }
HOSTDEV float y(const MPHIT* hits, size_t it)    { return pos(hits, it, 1); }
HOSTDEV float z(const MPHIT* hits, size_t it)    { return pos(hits, it, 2); }
//
HOSTDEV float pos(const MPHIT* hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk;
  const MPHIT* bhits = bHit(hits, ev, ib);
  size_t it = 0 ;
  return pos(bhits,it,ipar);
}
HOSTDEV float x(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 0); }
HOSTDEV float y(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 1); }
HOSTDEV float z(const MPHIT* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 2); }

MPTRK* prepareTracks(ATRK inputtrk) {
  MPTRK* result ; 
  cudaCheck(cudaMallocHost((void**)&result,nevts*nb*sizeof(MPTRK)));
  // store in element order for bunches of  matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
	      //par
	      for (size_t ip=0;ip<6;++ip) {
	        result[ib + nb*ie].par.data[ip] = (1+smear*randn(0,1))*inputtrk.par[ip];
	      }
	      //cov, scaled by factor 100 
	      for (size_t ip=0;ip<21;++ip) {
	        result[ib + nb*ie].cov.data[ip] = (1+smear*randn(0,1))*inputtrk.cov[ip]*100;
	      }
	      //q
	      result[ib + nb*ie].q.data[0] = inputtrk.q;//can't really smear this or fit will be wrong
        //if((ib + nb*ie)%10==0 ) printf("prep trk index = %i ,track = (%.3f)\n ", ib+nb*ie);
    }
  }
  return result;
}

MPHIT* prepareHits(std::vector<AHIT>& inputhits) {
  MPHIT* result;  //fixme, align?
  cudaCheck(cudaMallocHost((void**)&result,nlayer*nevts*nb*sizeof(MPHIT)));
  // store in element order for bunches of  matrices (a la matriplex)
  for (size_t lay=0;lay<nlayer;++lay) {

    size_t mylay = lay;
    if (lay>=inputhits.size()) {
      // int wraplay = inputhits.size()/lay;
      exit(1);
    }
    AHIT& inputhit = inputhits[mylay];

    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        	//pos
        	for (size_t ip=0;ip<3;++ip) {
        	  result[lay+nlayer*(ib + nb*ie)].pos.data[ip] = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
        	  result[lay+nlayer*(ib + nb*ie)].cov.data[ip] = (1+smear*randn(0,1))*inputhit.cov[ip];
        	}
      }
    }
  }
  return result;
}

__forceinline__ __device__ void MultHelixProp(const MP6x6F* A, const MP6x6SF* B, MP6x6F* C) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
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

__forceinline__ __device__ void MultHelixPropTransp(const MP6x6F* A, const MP6x6F* B, MP6x6SF* C) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
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


__forceinline__ __device__ void KalmanGainInv(const MP6x6SF* A, const MP3x3SF* B, MP3x3* C) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
  {
    double det =
      ((a[0]+b[0])*(((a[ 6]+b[ 3]) *(a[11]+b[5])) - ((a[7]+b[4]) *(a[7]+b[4])))) -
      ((a[1]+b[1])*(((a[ 1]+b[ 1]) *(a[11]+b[5])) - ((a[7]+b[4]) *(a[2]+b[2])))) +
      ((a[2]+b[2])*(((a[ 1]+b[ 1]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[6]+b[3]))));
    double invdet = 1.0/det;

    c[ 0] =  invdet*(((a[ 6]+b[ 3]) *(a[11]+b[5])) - ((a[7]+b[4]) *(a[7]+b[4])));
    c[ 1] =  -1*invdet*(((a[ 1]+b[ 1]) *(a[11]+b[5])) - ((a[2]+b[2]) *(a[7]+b[4])));
    c[ 2] =  invdet*(((a[ 1]+b[ 1]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[7]+b[4])));
    c[ 3] =  -1*invdet*(((a[ 1]+b[ 1]) *(a[11]+b[5])) - ((a[7]+b[4]) *(a[2]+b[2])));
    c[ 4] =  invdet*(((a[ 0]+b[ 0]) *(a[11]+b[5])) - ((a[2]+b[2]) *(a[2]+b[2])));
    c[ 5] =  -1*invdet*(((a[ 0]+b[ 0]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[1]+b[1])));
    c[ 6] =  invdet*(((a[ 1]+b[ 1]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[6]+b[3])));
    c[ 7] =  -1*invdet*(((a[ 0]+b[ 0]) *(a[7]+b[4])) - ((a[2]+b[2]) *(a[1]+b[1])));
    c[ 8] =  invdet*(((a[ 0]+b[ 0]) *(a[6]+b[3])) - ((a[1]+b[1]) *(a[1]+b[1])));
  }
}
__forceinline__ __device__ void KalmanGain(const MP6x6SF* A, const MP3x3* B, MP3x6* C) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
  {
    c[ 0] = a[0]*b[0] + a[1]*b[3] + a[2]*b[6];
    c[ 1] = a[0]*b[1] + a[1]*b[4] + a[2]*b[7];
    c[ 2] = a[0]*b[2] + a[1]*b[5] + a[2]*b[8];
    c[ 3] = a[1]*b[0] + a[6]*b[3] + a[7]*b[6];
    c[ 4] = a[1]*b[1] + a[6]*b[4] + a[7]*b[7];
    c[ 5] = a[1]*b[2] + a[6]*b[5] + a[7]*b[8];
    c[ 6] = a[2]*b[0] + a[7]*b[3] + a[11]*b[6];
    c[ 7] = a[2]*b[1] + a[7]*b[4] + a[11]*b[7];
    c[ 8] = a[2]*b[2] + a[7]*b[5] + a[11]*b[8];
    c[ 9] = a[3]*b[0] + a[8]*b[3] + a[12]*b[6];
    c[ 10] = a[3]*b[1] + a[8]*b[4] + a[12]*b[7];
    c[ 11] = a[3]*b[2] + a[8]*b[5] + a[12]*b[8];
    c[ 12] = a[4]*b[0] + a[9]*b[3] + a[13]*b[6];
    c[ 13] = a[4]*b[1] + a[9]*b[4] + a[13]*b[7];
    c[ 14] = a[4]*b[2] + a[9]*b[5] + a[13]*b[8];
    c[ 15] = a[5]*b[0] + a[10]*b[3] + a[14]*b[6];
    c[ 16] = a[5]*b[1] + a[10]*b[4] + a[14]*b[7];
    c[ 17] = a[5]*b[2] + a[10]*b[5] + a[14]*b[8];
  }
}

HOSTDEV inline float hipo(float x, float y)
{
  return std::sqrt(x*x + y*y);
}

__device__ void KalmanUpdate(MP6x6SF* trkErr, MP6F* inPar, const MP3x3SF* hitErr, const MP3F* msP){

    MP1F    rotT00;
    MP1F    rotT01;
    MP2x2SF resErr_loc;

    const float r = hipo(x(msP,0), y(msP,0));
    rotT00.data[0] = -(y(msP,0) + y(inPar,0)) / (2*r);
    rotT01.data[0] =  (x(msP,0) + x(inPar,0)) / (2*r);    
    
    resErr_loc.data[0] = (rotT00.data[0]*(trkErr->data[0] + hitErr->data[0]) +
                                 rotT01.data[0]*(trkErr->data[1] + hitErr->data[1]))*rotT00.data[0] +
                                (rotT00.data[0]*(trkErr->data[1] + hitErr->data[1]) +
                                 rotT01.data[0]*(trkErr->data[2] + hitErr->data[2]))*rotT01.data[0];
    resErr_loc.data[1] = (trkErr->data[3] + hitErr->data[3])*rotT00.data[0] +
                                (trkErr->data[4] + hitErr->data[4])*rotT01.data[0];
    resErr_loc.data[2] = (trkErr->data[5] + hitErr->data[5]);

    const double det = (double)resErr_loc.data[0] * resErr_loc.data[2] -
                       (double)resErr_loc.data[1] * resErr_loc.data[1];
    const float s   = 1.f / det;
    const float tmp = s * resErr_loc.data[2];
    resErr_loc.data[1] *= -s;
    resErr_loc.data[2]  = s * resErr_loc.data[0];
    resErr_loc.data[0]  = tmp;

    MP3x6 kGain;
      kGain.data[ 0] = trkErr->data[ 0]*(rotT00.data[0]*resErr_loc.data[0]) +
	                    trkErr->data[ 1]*(rotT01.data[0]*resErr_loc.data[0]) +
	                    trkErr->data[ 3]*resErr_loc.data[1];
      kGain.data[ 1] = trkErr->data[ 0]*(rotT00.data[0]*resErr_loc.data[1]) +
	                    trkErr->data[ 1]*(rotT01.data[0]*resErr_loc.data[1]) +
	                    trkErr->data[ 3]*resErr_loc.data[ 2];
      kGain.data[ 2] = 0;
      kGain.data[ 3] = trkErr->data[ 1]*(rotT00.data[0]*resErr_loc.data[0]) +
	                    trkErr->data[ 2]*(rotT01.data[0]*resErr_loc.data[0]) +
	                    trkErr->data[ 4]*resErr_loc.data[ 1];
      kGain.data[ 4] = trkErr->data[ 1]*(rotT00.data[0]*resErr_loc.data[1]) +
	                    trkErr->data[ 2]*(rotT01.data[0]*resErr_loc.data[1]) +
	                    trkErr->data[ 4]*resErr_loc.data[ 2];
      kGain.data[ 5] = 0;
      kGain.data[ 6] = trkErr->data[ 3]*(rotT00.data[0]*resErr_loc.data[0]) +
	                    trkErr->data[ 4]*(rotT01.data[0]*resErr_loc.data[0]) +
	                    trkErr->data[ 5]*resErr_loc.data[ 1];
      kGain.data[ 7] = trkErr->data[ 3]*(rotT00.data[0]*resErr_loc.data[1]) +
	                    trkErr->data[ 4]*(rotT01.data[0]*resErr_loc.data[1]) +
	                    trkErr->data[ 5]*resErr_loc.data[ 2];
      kGain.data[ 8] = 0;
      kGain.data[ 9] = trkErr->data[ 6]*(rotT00.data[0]*resErr_loc.data[0]) +
	                    trkErr->data[ 7]*(rotT01.data[0]*resErr_loc.data[0]) +
	                    trkErr->data[ 8]*resErr_loc.data[ 1];
      kGain.data[10] = trkErr->data[ 6]*(rotT00.data[0]*resErr_loc.data[ 1]) +
	                    trkErr->data[ 7]*(rotT01.data[0]*resErr_loc.data[ 1]) +
	                    trkErr->data[ 8]*resErr_loc.data[ 2];
      kGain.data[11] = 0;
      kGain.data[12] = trkErr->data[10]*(rotT00.data[0]*resErr_loc.data[0]) +
	                    trkErr->data[11]*(rotT01.data[0]*resErr_loc.data[0]) +
	                    trkErr->data[12]*resErr_loc.data[ 1];
      kGain.data[13] = trkErr->data[10]*(rotT00.data[0]*resErr_loc.data[ 1]) +
	                    trkErr->data[11]*(rotT01.data[0]*resErr_loc.data[ 1]) +
	                    trkErr->data[12]*resErr_loc.data[ 2];
      kGain.data[14] = 0;
      kGain.data[15] = trkErr->data[15]*(rotT00.data[0]*resErr_loc.data[ 0]) +
	                    trkErr->data[16]*(rotT01.data[0]*resErr_loc.data[ 0]) +
	                    trkErr->data[17]*resErr_loc.data[ 1];
      kGain.data[16] = trkErr->data[15]*(rotT00.data[0]*resErr_loc.data[ 1]) +
	                    trkErr->data[16]*(rotT01.data[0]*resErr_loc.data[ 1]) +
	                    trkErr->data[17]*resErr_loc.data[ 2];
      kGain.data[17] = 0;

    MP2F res_loc;
     res_loc.data[0] =  rotT00.data[0]*(x(msP,0) - x(inPar,0)) + rotT01.data[0]*(y(msP,0) - y(inPar,0));
     res_loc.data[1] =  z(msP,0) - z(inPar,0);

     setx(inPar, 0, x(inPar, 0) + kGain.data[ 0] * res_loc.data[ 0] + kGain.data[ 1] * res_loc.data[ 1]);
     sety(inPar, 0, y(inPar, 0) + kGain.data[ 3] * res_loc.data[ 0] + kGain.data[ 4] * res_loc.data[ 1]);
     setz(inPar, 0, z(inPar, 0) + kGain.data[ 6] * res_loc.data[ 0] + kGain.data[ 7] * res_loc.data[ 1]);
     setipt(inPar, 0, ipt(inPar, 0) + kGain.data[ 9] * res_loc.data[ 0] + kGain.data[10] * res_loc.data[ 1]);
     setphi(inPar, 0, phi(inPar, 0) + kGain.data[12] * res_loc.data[ 0] + kGain.data[13] * res_loc.data[ 1]);
     settheta(inPar, 0, theta(inPar, 0) + kGain.data[15] * res_loc.data[0] + kGain.data[16] * res_loc.data[ 1]);
  
    MP6x6SF newErr;

     newErr.data[ 0] =     kGain.data[ 0]*rotT00.data[0]*trkErr->data[ 0] +
                            kGain.data[ 0]*rotT01.data[0]*trkErr->data[ 1] +
                            kGain.data[ 1]*trkErr->data[ 3];
     newErr.data[ 1] =     kGain.data[ 3]*rotT00.data[0]*trkErr->data[ 0] +
                            kGain.data[ 3]*rotT01.data[0]*trkErr->data[ 1] +
                            kGain.data[ 4]*trkErr->data[ 3];
     newErr.data[ 2] =     kGain.data[ 3]*rotT00.data[0]*trkErr->data[ 1] +
                            kGain.data[ 3]*rotT01.data[0]*trkErr->data[ 2] +
                            kGain.data[ 4]*trkErr->data[ 4];
     newErr.data[ 3] =     kGain.data[ 6]*rotT00.data[0]*trkErr->data[ 0] +
                            kGain.data[ 6]*rotT01.data[0]*trkErr->data[ 1] +
                            kGain.data[ 7]*trkErr->data[ 3];
     newErr.data[ 4] =     kGain.data[ 6]*rotT00.data[0]*trkErr->data[ 1] +
                            kGain.data[ 6]*rotT01.data[0]*trkErr->data[ 2] +
                            kGain.data[ 7]*trkErr->data[ 4];
     newErr.data[ 5] =     kGain.data[ 6]*rotT00.data[0]*trkErr->data[ 3] +
                            kGain.data[ 6]*rotT01.data[0]*trkErr->data[ 4] +
                            kGain.data[ 7]*trkErr->data[ 5];
     newErr.data[ 6] =     kGain.data[ 9]*rotT00.data[0]*trkErr->data[ 0] +
                            kGain.data[ 9]*rotT01.data[0]*trkErr->data[ 1] +
                            kGain.data[10]*trkErr->data[ 3];
     newErr.data[ 7] =     kGain.data[ 9]*rotT00.data[0]*trkErr->data[ 1] +
                            kGain.data[ 9]*rotT01.data[0]*trkErr->data[ 2] +
                            kGain.data[10]*trkErr->data[ 4];
     newErr.data[ 8] =     kGain.data[ 9]*rotT00.data[0]*trkErr->data[ 3] +
                            kGain.data[ 9]*rotT01.data[0]*trkErr->data[ 4] +
                            kGain.data[10]*trkErr->data[ 5];
     newErr.data[ 9] =     kGain.data[ 9]*rotT00.data[0]*trkErr->data[ 6] +
                            kGain.data[ 9]*rotT01.data[0]*trkErr->data[ 7] +
                            kGain.data[10]*trkErr->data[ 8];
     newErr.data[10] =     kGain.data[12]*rotT00.data[0]*trkErr->data[ 0] +
                            kGain.data[12]*rotT01.data[0]*trkErr->data[ 1] +
                            kGain.data[13]*trkErr->data[ 3];
     newErr.data[11] =     kGain.data[12]*rotT00.data[0]*trkErr->data[ 1] +
                            kGain.data[12]*rotT01.data[0]*trkErr->data[ 2] +
                            kGain.data[13]*trkErr->data[ 4];
     newErr.data[12] =     kGain.data[12]*rotT00.data[0]*trkErr->data[ 3] +
                            kGain.data[12]*rotT01.data[0]*trkErr->data[ 4] +
                            kGain.data[13]*trkErr->data[ 5];
     newErr.data[13] =     kGain.data[12]*rotT00.data[0]*trkErr->data[ 6] +
                            kGain.data[12]*rotT01.data[0]*trkErr->data[ 7] +
                            kGain.data[13]*trkErr->data[ 8];
     newErr.data[14] =     kGain.data[12]*rotT00.data[0]*trkErr->data[10] +
                            kGain.data[12]*rotT01.data[0]*trkErr->data[11] +
                            kGain.data[13]*trkErr->data[12];
     newErr.data[15] =     kGain.data[15]*rotT00.data[0]*trkErr->data[ 0] +
                            kGain.data[15]*rotT01.data[0]*trkErr->data[ 1] +
                            kGain.data[16]*trkErr->data[ 3];
     newErr.data[16] =     kGain.data[15]*rotT00.data[0]*trkErr->data[ 1] +
                            kGain.data[15]*rotT01.data[0]*trkErr->data[ 2] +
                            kGain.data[16]*trkErr->data[ 4];
     newErr.data[17] =     kGain.data[15]*rotT00.data[0]*trkErr->data[ 3] +
                            kGain.data[15]*rotT01.data[0]*trkErr->data[ 4] +
                            kGain.data[16]*trkErr->data[ 5];
     newErr.data[18] =     kGain.data[15]*rotT00.data[0]*trkErr->data[ 6] +
                            kGain.data[15]*rotT01.data[0]*trkErr->data[ 7] +
                            kGain.data[16]*trkErr->data[ 8];
     newErr.data[19] =     kGain.data[15]*rotT00.data[0]*trkErr->data[10] +
                            kGain.data[15]*rotT01.data[0]*trkErr->data[11] +
                            kGain.data[16]*trkErr->data[12];
     newErr.data[20] =     kGain.data[15]*rotT00.data[0]*trkErr->data[15] +
                            kGain.data[15]*rotT01.data[0]*trkErr->data[16] +
                            kGain.data[16]*trkErr->data[17];

     newErr.data[ 0] = trkErr->data[ 0] - newErr.data[ 0];
     newErr.data[ 1] = trkErr->data[ 1] - newErr.data[ 1];
     newErr.data[ 2] = trkErr->data[ 2] - newErr.data[ 2];
     newErr.data[ 3] = trkErr->data[ 3] - newErr.data[ 3];
     newErr.data[ 4] = trkErr->data[ 4] - newErr.data[ 4];
     newErr.data[ 5] = trkErr->data[ 5] - newErr.data[ 5];
     newErr.data[ 6] = trkErr->data[ 6] - newErr.data[ 6];
     newErr.data[ 7] = trkErr->data[ 7] - newErr.data[ 7];
     newErr.data[ 8] = trkErr->data[ 8] - newErr.data[ 8];
     newErr.data[ 9] = trkErr->data[ 9] - newErr.data[ 9];
     newErr.data[10] = trkErr->data[10] - newErr.data[10];
     newErr.data[11] = trkErr->data[11] - newErr.data[11];
     newErr.data[12] = trkErr->data[12] - newErr.data[12];
     newErr.data[13] = trkErr->data[13] - newErr.data[13];
     newErr.data[14] = trkErr->data[14] - newErr.data[14];
     newErr.data[15] = trkErr->data[15] - newErr.data[15];
     newErr.data[16] = trkErr->data[16] - newErr.data[16];
     newErr.data[17] = trkErr->data[17] - newErr.data[17];
     newErr.data[18] = trkErr->data[18] - newErr.data[18];
     newErr.data[19] = trkErr->data[19] - newErr.data[19];
     newErr.data[20] = trkErr->data[20] - newErr.data[20];

  /*
  MPlexLH K;           // kalman gain, fixme should be L2
  KalmanHTG(rotT00, rotT01, resErr_loc, tempHH); // intermediate term to get kalman gain (H^T*G)
  KalmanGain(psErr, tempHH, K);

  MPlexHV res_glo;   //position residual in global coordinates
  SubtractFirst3(msPar, psPar, res_glo);
  MPlex2V res_loc;   //position residual in local coordinates
  RotateResidulsOnTangentPlane(rotT00,rotT01,res_glo,res_loc);

  //    Chi2Similarity(res_loc, resErr_loc, outChi2);

  MultResidualsAdd(K, psPar, res_loc, outPar);
  MPlexLL tempLL;
  squashPhiMPlex(outPar,N_proc); // ensure phi is between |pi|
  KHMult(K, rotT00, rotT01, tempLL);
  KHC(tempLL, psErr, outErr);
  outErr.Subtract(psErr, outErr);
  */
  
  //(*trkErr) = (*newErr);
}

HOSTDEV inline void sincos4(const float x, float& sin, float& cos)
{
   // Had this writen with explicit division by factorial.
   // The *whole* fitting test ran like 2.5% slower on MIC, sigh.

   const float x2 = x*x;
   cos  = 1.f - 0.5f*x2 + 0.04166667f*x2*x2;
   sin  = x - 0.16666667f*x*x2;
}

constexpr float kfact= 100/(-0.299792458*3.8112);
constexpr int Niter=5;
__device__ void propagateToR(const MP6x6SF* inErr, const MP6F* inPar, const MP1I* inChg, 
                  const MP3F* msP, MP6x6SF* outErr, MP6F* outPar) {
  
  MP6x6F errorProp, temp;
    //initialize erroProp to identity matrix
    for (size_t i=0;i<6;++i) errorProp.data[PosInMtrx(i,i,6)] = 1.f;
    
    float r0 = hipo(x(inPar,0), y(inPar,0));
    const float k = q(inChg,0) * kfact;
    const float r = hipo(x(msP,0), y(msP,0));

    const float xin     = x(inPar,0);
    const float yin     = y(inPar,0);
    const float iptin   = ipt(inPar,0);
    const float phiin   = phi(inPar,0);
    const float thetain = theta(inPar,0);

    //initialize outPar to inPar
    setx(outPar,0, xin);
    sety(outPar,0, yin);
    setz(outPar,0, z(inPar,0));
    setipt(outPar,0, iptin);
    setphi(outPar,0, phiin);
    settheta(outPar,0, thetain);

    const float kinv  = 1.f/k;
    const float pt = 1.f/iptin;

    float D = 0., cosa = 0., sina = 0., id = 0.;
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
      r0 = hipo(x(outPar,0), y(outPar,0));
      id = (r-r0);
      D+=id;
      sincos4(id*iptin*kinv, sina, cosa);

      //update derivatives on total distance
      if (i+1 != Niter) {

	const float xtmp = x(outPar,0);
	const float ytmp = y(outPar,0);
	const float oor0 = (r0>0.f && std::abs(r-r0)<0.0001f) ? 1.f/r0 : 0.f;

	const float dadipt = id*kinv;

	const float dadx = -xtmp*iptin*kinv*oor0;
	const float dady = -ytmp*iptin*kinv*oor0;

	const float pxca = pxin*cosa;
	const float pxsa = pxin*sina;
	const float pyca = pyin*cosa;
	const float pysa = pyin*sina;

	float tmp = k*dadx;
	dDdx   -= ( xtmp*(1.f + tmp*(pxca - pysa)) + ytmp*tmp*(pyca + pxsa) )*oor0;
	tmp = k*dady;
	dDdy   -= ( xtmp*tmp*(pxca - pysa) + ytmp*(1.f + tmp*(pyca + pxsa)) )*oor0;
	//now r0 depends on ipt and phi as well
	tmp = dadipt*iptin;
	dDdipt -= k*( xtmp*(pxca*tmp - pysa*tmp - pyca - pxsa + pyin) +
		      ytmp*(pyca*tmp + pxsa*tmp - pysa + pxca - pxin))*pt*oor0;
	dDdphi += k*( xtmp*(pysa - pxin + pxca) - ytmp*(pxsa - pyin + pyca))*oor0;
      }

      //update parameters
      setx(outPar,0, x(outPar,0) + k*(pxin*sina - pyin*(1.f-cosa)));
      sety(outPar,0, y(outPar,0) + k*(pyin*sina + pxin*(1.f-cosa)));
      const float pxinold = pxin;//copy before overwriting
      pxin = pxin*cosa - pyin*sina;
      pyin = pyin*cosa + pxinold*sina;
    }

    const float alpha  = D*iptin*kinv;
    const float dadx   = dDdx*iptin*kinv;
    const float dady   = dDdy*iptin*kinv;
    const float dadipt = (iptin*dDdipt + D)*kinv;
    const float dadphi = dDdphi*iptin*kinv;

    sincos4(alpha, sina, cosa);

    errorProp.data[PosInMtrx(0,0,6) ] = 1.f+k*dadx*(cosPorT*cosa-sinPorT*sina)*pt;
    errorProp.data[PosInMtrx(0,1,6) ] =     k*dady*(cosPorT*cosa-sinPorT*sina)*pt;
    errorProp.data[PosInMtrx(0,2,6) ] = 0.f;
    errorProp.data[PosInMtrx(0,3,6) ] = k*(cosPorT*(iptin*dadipt*cosa-sina)+sinPorT*((1.f-cosa)-iptin*dadipt*sina))*pt*pt;
    errorProp.data[PosInMtrx(0,4,6) ] = k*(cosPorT*dadphi*cosa - sinPorT*dadphi*sina - sinPorT*sina + cosPorT*cosa - cosPorT)*pt;
    errorProp.data[PosInMtrx(0,5,6) ] = 0.f;

    errorProp.data[PosInMtrx(1,0,6) ] =     k*dadx*(sinPorT*cosa+cosPorT*sina)*pt;
    errorProp.data[PosInMtrx(1,1,6) ] = 1.f+k*dady*(sinPorT*cosa+cosPorT*sina)*pt;
    errorProp.data[PosInMtrx(1,2,6) ] = 0.f;
    errorProp.data[PosInMtrx(1,3,6) ] = k*(sinPorT*(iptin*dadipt*cosa-sina)+cosPorT*(iptin*dadipt*sina-(1.f-cosa)))*pt*pt;
    errorProp.data[PosInMtrx(1,4,6) ] = k*(sinPorT*dadphi*cosa + cosPorT*dadphi*sina + sinPorT*cosa + cosPorT*sina - sinPorT)*pt;
    errorProp.data[PosInMtrx(1,5,6) ] = 0.f;

    //no trig approx here, theta can be large
    cosPorT=std::cos(thetain);
    sinPorT=std::sin(thetain);
    //redefine sinPorT as 1./sinPorT to reduce the number of temporaries
    sinPorT = 1.f/sinPorT;

    setz(outPar,0, z(inPar,0) + k*alpha*cosPorT*pt*sinPorT);

    errorProp.data[PosInMtrx(2,0,6)] = k*cosPorT*dadx*pt*sinPorT;
    errorProp.data[PosInMtrx(2,1,6)] = k*cosPorT*dady*pt*sinPorT;
    errorProp.data[PosInMtrx(2,2,6)] = 1.f;
    errorProp.data[PosInMtrx(2,3,6)] = k*cosPorT*(iptin*dadipt-alpha)*pt*pt*sinPorT;
    errorProp.data[PosInMtrx(2,4,6)] = k*dadphi*cosPorT*pt*sinPorT;
    errorProp.data[PosInMtrx(2,5,6)] =-k*alpha*pt*sinPorT*sinPorT;

    setipt(outPar,0, iptin);

    errorProp.data[PosInMtrx(3,0,6)] = 0.f;
    errorProp.data[PosInMtrx(3,1,6)] = 0.f;
    errorProp.data[PosInMtrx(3,2,6)] = 0.f;
    errorProp.data[PosInMtrx(3,3,6)] = 1.f;
    errorProp.data[PosInMtrx(3,4,6)] = 0.f;
    errorProp.data[PosInMtrx(3,5,6)] = 0.f;

    setphi(outPar,0, phi(inPar,0)+alpha );

    errorProp.data[PosInMtrx(4,0,6)] = dadx;
    errorProp.data[PosInMtrx(4,1,6)] = dady;
    errorProp.data[PosInMtrx(4,2,6)] = 0.f;
    errorProp.data[PosInMtrx(4,3,6)] = dadipt;
    errorProp.data[PosInMtrx(4,4,6)] = 1.f+dadphi;
    errorProp.data[PosInMtrx(4,5,6)] = 0.f;

    settheta(outPar,0, thetain);

    errorProp.data[PosInMtrx(5,0,6)] = 0.f;
    errorProp.data[PosInMtrx(5,1,6)] = 0.f;
    errorProp.data[PosInMtrx(5,2,6)] = 0.f;
    errorProp.data[PosInMtrx(5,3,6)] = 0.f;
    errorProp.data[PosInMtrx(5,4,6)] = 0.f;
    errorProp.data[PosInMtrx(5,5,6)] = 1.f;

  MultHelixProp(&errorProp, inErr, &temp);
  MultHelixPropTransp(&errorProp, &temp, outErr);
}

__device__ __constant__ int ie_range = (int) nevts/num_streams; 
__global__ void GPUsequence(MPTRK *trk, MPHIT *hit, MPTRK *outtrk,  const int stream, const int length){

   const int end = (stream < num_streams) ?
     nb*nevts / num_streams : // for "full" streams
     nb*nevts % num_streams; // possible remainder

    auto i = threadIdx.x + blockIdx.x * blockDim.x;

    while(i<length){
        MPTRK obtracks;

        const MPTRK btracks = (trk[i]);

        for (int layer=0;layer<nlayer;++layer){	
            const MPHIT bhits = (hit[layer+nlayer*i]);
            propagateToR(&(btracks).cov,&(btracks).par,&(btracks).q,&(bhits).pos,&(obtracks).cov, &(obtracks).par);
            KalmanUpdate(&(obtracks).cov,&(obtracks).par,&(bhits).cov,&(bhits).pos);
        }
        outtrk[i] = obtracks;
        i += gridDim.x * blockDim.x;
    }
    return;
    //if((index)%100==0 ) printf("index = %i ,(block,grid)=(%i,%i), track = (%.3f)\n ", index,blockDim.x,gridDim.x,&(*btracks).par.data[8]);
}

int main (int argc, char* argv[]) {

   printf("Streams: %d, blocks: %d, threads(x): (%d)   \n",num_streams,blockspergrid,threadsperblockx);
   
   #include "input_track.h"

   std::vector<AHIT> inputhits{inputhit21,inputhit20,inputhit19,inputhit18,inputhit17,inputhit16,inputhit15,inputhit14,
                               inputhit13,inputhit12,inputhit11,inputhit10,inputhit09,inputhit08,inputhit07,inputhit06,
                               inputhit05,inputhit04,inputhit03,inputhit02,inputhit01,inputhit00};

   printf("track in pos: x=%f, y=%f, z=%f, r=%f, pt=%f, phi=%f, theta=%f \n", inputtrk.par[0], inputtrk.par[1], inputtrk.par[2],
	  sqrtf(inputtrk.par[0]*inputtrk.par[0] + inputtrk.par[1]*inputtrk.par[1]),
	  1./inputtrk.par[3], inputtrk.par[4], inputtrk.par[5]);
   printf("track in cov: xx=%.2e, yy=%.2e, zz=%.2e \n", inputtrk.cov[SymOffsets66(PosInMtrx(0,0,6))],
	                                       inputtrk.cov[SymOffsets66(PosInMtrx(1,1,6))],
	                                       inputtrk.cov[SymOffsets66(PosInMtrx(2,2,6))]);
   for (size_t lay=0; lay<nlayer; lay++){
     printf("hit in layer=%lu, pos: x=%f, y=%f, z=%f, r=%f \n", lay, inputhits[lay].pos[0], inputhits[lay].pos[1], inputhits[lay].pos[2], sqrtf(inputhits[lay].pos[0]*inputhits[lay].pos[0] + inputhits[lay].pos[1]*inputhits[lay].pos[1]));
   }
   
   printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
   printf("NITER=%d\n", NITER);
   long setup_start, setup_stop;
   struct timeval timecheck;

   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
   MPTRK* trk = prepareTracks(inputtrk);
   MPHIT* hit = prepareHits(inputhits);
   MPTRK* outtrk ;
   cudaMallocHost((void**)&outtrk,nevts*nb*sizeof(MPTRK)); 
   //device pointers
   MPTRK* trk_dev;
   MPHIT* hit_dev;
   MPTRK* outtrk_dev;
   cudaMalloc((MPTRK**)&trk_dev,nevts*nb*sizeof(MPTRK));
   cudaMalloc((MPHIT**)&hit_dev,nlayer*nevts*nb*sizeof(MPHIT));
   cudaMalloc((MPTRK**)&outtrk_dev,nevts*nb*sizeof(MPTRK));
   dim3 grid(blockspergrid,1,1);
   dim3 block(threadsperblockx,1,1); 

   //for (size_t ie=0;ie<nevts;++ie) {
   //  for (size_t it=0;it<ntrks;++it) {
   //    float x_ = x(trk,ie,it);
   //    float y_ = y(trk,ie,it);
   //    float z_ = z(trk,ie,it);
   //    float r_ = sqrtf(x_*x_ + y_*y_);
   //    if((it+ie*ntrks)%10==0) printf("iTrk = %i,  track (x,y,z,r)=(%.3f,%.3f,%.3f,%.3f) \n", it+ie*ntrks, x_,y_,z_,r_);
   //  }
   //}

   int device = -1;
   cudaGetDevice(&device);

  printf(" Runing on cudaDevice: %d\n", device);

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf(" Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
  }

  int stream_chunk = ((int)(nevts*nb/num_streams));
  int stream_remainder = ((int)((nevts*nb)%num_streams));
  int stream_range;
  if (stream_remainder == 0){ stream_range =num_streams;}
  else{stream_range = num_streams+1;}
  cudaStream_t streams[stream_range];
  for (int s = 0; s<stream_range;s++){
    //cudaStreamCreateWithFlags(&streams[s],cudaStreamNonBlocking);
    cudaStreamCreate(&streams[s]);
  }


   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");
 
   printf("Number of struct MPTRK trk[] = %d\n", nevts*nb);
   printf("Number of struct MPTRK outtrk[] = %d\n", nevts*nb);
   printf("Number of struct struct MPHIT hit[] = %d\n", nevts*nb);
  
   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(struct MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(struct MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nlayer*nevts*nb*sizeof(struct MPHIT));


   //WIP: attempt to increase shared memory size 
   //size_t maxbytes = 100000; 
   //cudaFuncSetAttribute(GPUsequence, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
   //cudaFuncSetAttribute(GPUsequence, cudaFuncAttributePreferredSharedMemoryCarveout, maxbytes);

   //MP6x6SF * newErr;
   //MP6x6F * errorProp;
   //cudaMalloc(&newErr,sizeof(MP6x6SF)*blockspergrid);
   //cudaMalloc(&errorProp,sizeof(MP6x6F)*blockspergrid);

   auto chunkSize = [&](int s) {
     return s < num_streams ? stream_chunk : stream_remainder;
   };
   auto forStream = [&](auto ptr, int s) {
     return ptr + s*stream_chunk;
   };
   auto transferAsyncTrk = [&](int s) {
     cudaCheck(cudaMemcpyAsync(forStream(trk_dev, s), forStream(trk, s), chunkSize(s)*sizeof(MPTRK), cudaMemcpyHostToDevice, streams[s]));
   };
   auto transferAsyncHit = [&](int s) {
     cudaCheck(cudaMemcpyAsync(forStream(hit_dev, s), forStream(hit, s), chunkSize(s)*nlayer*sizeof(MPHIT), cudaMemcpyHostToDevice, streams[s]));
   };
   auto transfer_backAsync = [&](int s) {
     cudaCheck(cudaMemcpyAsync(forStream(outtrk, s), forStream(outtrk_dev, s), chunkSize(s)*sizeof(MPTRK), cudaMemcpyDeviceToHost, streams[s]));
   };

   auto doWork = [&](const char* msg, int nIters) {
     std::cout<<msg<<std::endl;
     double wall_time = 0;

     const int phys_length      = nevts*ntrks;
#ifdef MEASURE_H2D_TRANSFER
     for(int itr=0; itr<nIters; itr++) {
       auto wall_start = std::chrono::high_resolution_clock::now();
       for (int s = 0; s<num_streams;s++) {
         transferAsyncTrk(s);
         transferAsyncHit(s);
       }

       for (int s = 0; s<num_streams;s++) {
         GPUsequence<<<grid,block,0,streams[s]>>>(forStream(trk_dev, s), forStream(hit_dev, s), forStream(outtrk_dev, s),s,phys_length);
       }

#ifdef MEASURE_D2H_TRANSFER
       for (int s = 0; s<num_streams;s++) {
         transfer_backAsync(s);
       }
#endif // MEASURE_D2H_TRANSFER
       cudaDeviceSynchronize();
       auto wall_stop = std::chrono::high_resolution_clock::now();
       wall_time += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_stop-wall_start).count()) / 1e6;
     }
#else // not MEASURE_H2D_TRANSFER
     for (int s = 0; s<num_streams;s++) {
       transferAsyncTrk(s);
       transferAsyncHit(s);
     }
     cudaDeviceSynchronize();
     for(int itr=0; itr<nIters; itr++) {
       auto wall_start = std::chrono::high_resolution_clock::now();
       for (int s = 0; s<num_streams;s++) {
         GPUsequence<<<grid,block,0,streams[s]>>>(forStream(trk_dev, s), forStream(hit_dev, s), forStream(outtrk_dev, s), s,phys_length);
       }

#ifdef MEASURE_D2H_TRANSFER
       for (int s = 0; s<num_streams;s++) {
         transfer_backAsync(s);
       }
#endif // MEASURE_D2H_TRANSFER
       cudaDeviceSynchronize();
       auto wall_stop = std::chrono::high_resolution_clock::now();
       wall_time += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_stop-wall_start).count()) / 1e6;
     }
#endif // MEASURE_H2D_TRANSFER

#ifndef MEASURE_D2H_TRANSFER
     for (int s = 0; s<num_streams;s++) {
       transfer_backAsync(s);
     }
     cudaDeviceSynchronize();
#endif

     return wall_time;
   };

   doWork("Warming up", NWARMUP);
   auto wall_time = doWork("Launching", NITER);

   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i  %f 0 %f %i\n",int(NITER),nevts, ntrks,  nb, wall_time, (setup_stop-setup_start)*0.001, nthreads);


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
       //if((it+ie*ntrks)%10==0) printf("iTrk = %i,  track (x,y,z,r)=(%.3f,%.3f,%.3f,%.3f) \n", it+ie*ntrks, x_,y_,z_,r_);
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

   cudaFreeHost(trk);
   cudaFreeHost(hit);
   cudaFreeHost(outtrk);
   cudaFree(trk_dev);
   cudaFree(hit_dev);
   cudaFree(outtrk_dev);

   return 0;
}
