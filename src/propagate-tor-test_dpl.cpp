/*
example :: dpcpp -std=c++17 -O3 propagate-tor-test_dpl.cpp  -dpl -o propagate-tor-test_dpl.exe
*/

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>//<= why does it need tbb?
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/random>


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

#ifndef bsize
#define bsize 1
#endif
#ifndef ntrks
#define ntrks 9600//8192
#endif

#define nb    (ntrks/bsize)

#ifndef nevts
#define nevts 100
#endif
#define smear 0.1

#ifndef NITER
#define NITER 5
#endif
#ifndef nlayer
#define nlayer 20
#endif

#include <CL/sycl.hpp>
using oneapi::dpl::counting_iterator;


auto PosInMtrx = [](const int &&i, const int &&j, const int &&D, const int block_size = 1) constexpr {return block_size*(i*D+j);};

enum class FieldOrder{P2R_TRACKBLK_EVENT_LAYER_MATIDX_ORDER,
                      P2R_TRACKBLK_EVENT_MATIDX_LAYER_ORDER,
                      P2R_MATIDX_LAYER_TRACKBLK_EVENT_ORDER};

using IntAllocator   = cl::sycl::usm_allocator<int, cl::sycl::usm::alloc::shared>;
using FloatAllocator = cl::sycl::usm_allocator<float, cl::sycl::usm::alloc::shared>;

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


template <typename T, typename Allocator, int n, int bSize>
struct MPNX {
   using DataType = T;
   using AllocType= Allocator;

   static constexpr int N    = n;
   static constexpr int BS   = bSize;

   const int nTrks;//note that bSize is a tuning parameter!
   const int nEvts;
   const int nLayers;

   std::vector<T, Allocator> data;

   MPNX(const Allocator alloc) : nTrks(bSize), nEvts(0), nLayers(0), data(n*bSize, alloc){}

   MPNX(const int ntrks_, const int nevts_, const Allocator alloc, const int nlayers_ = 1) :
      nTrks(ntrks_),
      nEvts(nevts_),
      nLayers(nlayers_),
      data(n*nTrks*nEvts*nLayers, alloc){
   }

   MPNX(const std::vector<T, Allocator> data_, const int ntrks_, const int nevts_, const int nlayers_ = 1) :
      nTrks(ntrks_),
      nEvts(nevts_),
      nLayers(nlayers_),
      data(data_) {
     if(data_.size() > n*nTrks*nEvts*nLayers) {std::cerr << "Incorrect dim parameters."; }
   }
};

using MP1I    = MPNX<int,  IntAllocator,   1 , bsize>;
using MP1F    = MPNX<float,FloatAllocator, 1 , bsize>;
using MP2F    = MPNX<float,FloatAllocator, 2 , bsize>;
using MP3F    = MPNX<float,FloatAllocator, 3 , bsize>;
using MP6F    = MPNX<float,FloatAllocator, 6 , bsize>;
using MP3x3   = MPNX<float,FloatAllocator, 9 , bsize>;
using MP3x6   = MPNX<float,FloatAllocator, 18, bsize>;
using MP2x2SF = MPNX<float,FloatAllocator, 3 , bsize>;
using MP3x3SF = MPNX<float,FloatAllocator, 6 , bsize>;
using MP6x6SF = MPNX<float,FloatAllocator, 21, bsize>;
using MP6x6F  = MPNX<float,FloatAllocator, 36, bsize>;


template <typename MPNTp, FieldOrder Order = FieldOrder::P2R_MATIDX_LAYER_TRACKBLK_EVENT_ORDER>
struct MPNXAccessor {
   typedef typename MPNTp::DataType T;

   static constexpr int bsz = MPNTp::BS;
   static constexpr int n   = MPNTp::N;//matrix linear dim (total number of els)

   const int nTrkB;
   const int nEvts;
   const int nLayers;

   const int NevtsNtbBsz;

   const int stride;
   
   const int thread_stride;

   T* data_; //accessor field only for the data access, not allocated here

   MPNXAccessor() : nTrkB(0), nEvts(0), nLayers(0), NevtsNtbBsz(0), stride(0), thread_stride(0), data_(nullptr){}
   MPNXAccessor(const MPNTp &v) :
        nTrkB(v.nTrks / bsz),
        nEvts(v.nEvts),
        nLayers(v.nLayers),
        NevtsNtbBsz(nEvts*nTrkB*bsz),
        stride(Order == FieldOrder::P2R_TRACKBLK_EVENT_LAYER_MATIDX_ORDER ? bsz*nTrkB*nEvts*nLayers  :
              (Order == FieldOrder::P2R_TRACKBLK_EVENT_MATIDX_LAYER_ORDER ? bsz*nTrkB*nEvts*n : n*bsz*nLayers)),
        thread_stride(Order == FieldOrder::P2R_TRACKBLK_EVENT_LAYER_MATIDX_ORDER ? stride  :
              (Order == FieldOrder::P2R_TRACKBLK_EVENT_MATIDX_LAYER_ORDER ? NevtsNtbBsz : bsz)),              
        data_(const_cast<T*>(v.data.data())){
	 }

   T& operator[](const int idx) const {return data_[idx];}

   T& operator()(const int mat_idx, const int trkev_idx, const int b_idx, const int layer_idx) const {
     if      constexpr (Order == FieldOrder::P2R_TRACKBLK_EVENT_LAYER_MATIDX_ORDER)
       return data_[mat_idx*stride + layer_idx*NevtsNtbBsz + trkev_idx*bsz + b_idx];//using defualt order batch id (the fastest) > track id > event id > layer id (the slowest)
     else if constexpr (Order == FieldOrder::P2R_TRACKBLK_EVENT_MATIDX_LAYER_ORDER)
       return data_[layer_idx*stride + mat_idx*NevtsNtbBsz + trkev_idx*bsz + b_idx];
     else //(Order == FieldOrder::P2Z_MATIDX_LAYER_TRACKBLK_EVENT_ORDER)
       return data_[trkev_idx*stride+layer_idx*n*bsz+mat_idx*bsz+b_idx];
   }//i is the internal dof index

   T& operator()(const int thrd_idx, const int blk_offset) const { return data_[thrd_idx*thread_stride + blk_offset];}//

   int GetThreadOffset(const int thrd_idx, const int layer_idx = 0) const {
     if      constexpr (Order == FieldOrder::P2R_TRACKBLK_EVENT_LAYER_MATIDX_ORDER)
       return (layer_idx*NevtsNtbBsz + thrd_idx*bsz);//using defualt order batch id (the fastest) > track id > event id > layer id (the slowest)
     else if constexpr (Order == FieldOrder::P2R_TRACKBLK_EVENT_MATIDX_LAYER_ORDER)
       return (layer_idx*stride + thrd_idx*bsz);
     else //(Order == FieldOrder::P2Z_MATIDX_LAYER_TRACKBLK_EVENT_ORDER)
       return (thrd_idx*stride+layer_idx*n*bsz);
   }
};

struct MPTRK {
  using MP6F_alloc   = typename MP6F::AllocType;
  using MP6x6SF_alloc= typename MP6x6SF::AllocType;
  using MP1I_alloc   = typename MP1I::AllocType;
  
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;

  MPTRK(sycl::queue cqueue) : par(MP6F_alloc(cqueue)), cov(MP6x6SF_alloc(cqueue)), q(MP1I_alloc(cqueue)){}
  
  MPTRK(const int ntrks_, const int nevts_, sycl::queue cqueue) : par(ntrks_, nevts_, MP6F_alloc(cqueue)), cov(ntrks_, nevts_, MP6x6SF_alloc(cqueue)), q(ntrks_, nevts_, MP6x6SF_alloc(cqueue)) {}

  //  MP22I   hitidx;
};

template <FieldOrder Order>
struct MPTRKAccessor {
  using MP6FAccessor   = MPNXAccessor<MP6F,    Order>;
  using MP6x6SFAccessor= MPNXAccessor<MP6x6SF, Order>;
  using MP1IAccessor   = MPNXAccessor<MP1I,    Order>;

  MP6FAccessor    par;
  MP6x6SFAccessor cov;
  MP1IAccessor    q;

  MPTRKAccessor() : par(), cov(), q() {}
  MPTRKAccessor(const MPTRK &in) : par(in.par), cov(in.cov), q(in.q) {}
};

template<FieldOrder order>
std::shared_ptr<MPTRK> prepareTracksN(struct ATRK inputtrk, sycl::queue cqueue) {

  auto result = std::make_shared<MPTRK>(ntrks, nevts, cqueue);
  //create an accessor field:
  std::unique_ptr<MPTRKAccessor<order>> rA(new MPTRKAccessor<order>(*result));

  // store in element order for bunches of bsize matrices (a la matriplex)
  for (int ie=0;ie<nevts;++ie) {
    for (int ib=0;ib<nb;++ib) {
      for (int it=0;it<bsize;++it) {
        //const int l = it+ib*bsize+ie*nb*bsize;
        const int tid = ib+ie*nb;
    	  //par
    	  for (int ip=0;ip<6;++ip) {
          rA->par(ip, tid, it, 0) = (1+smear*randn(0,1))*inputtrk.par[ip];
    	  }
    	  //cov
    	  for (int ip=0;ip<21;++ip) {
          rA->cov(ip, tid, it, 0) = (1+smear*randn(0,1))*inputtrk.cov[ip];
    	  }
    	  //q
        rA->q(0, tid, it, 0) = inputtrk.q-2*ceil(-0.5 + (float)rand() / RAND_MAX);
      }
    }
  }
  return std::move(result);
}



struct MPHIT {
  using MP3F_alloc   = typename MP3F::AllocType;
  using MP3x3SF_alloc= typename MP3x3SF::AllocType;

  MP3F    pos;
  MP3x3SF cov;

  MPHIT(sycl::queue cqueue) : pos(MP3F_alloc(cqueue)), cov(MP3x3SF_alloc(cqueue)){}
  MPHIT(const int ntrks_, const int nevts_, const int nlayers_, sycl::queue cqueue) : pos(ntrks_, nevts_, MP3F_alloc(cqueue), nlayers_), cov(ntrks_, nevts_, MP3x3SF_alloc(cqueue), nlayers_) {}

};

template <FieldOrder Order>
struct MPHITAccessor {
  using MP3FAccessor   = MPNXAccessor<MP3F,    Order>;
  using MP3x3SFAccessor= MPNXAccessor<MP3x3SF, Order>;

  MP3FAccessor    pos;
  MP3x3SFAccessor cov;

  MPHITAccessor() : pos(), cov() {}
  MPHITAccessor(const MPHIT &in) : pos(in.pos), cov(in.cov) {}
};

template<FieldOrder order>
std::shared_ptr<MPHIT> prepareHitsN(struct AHIT inputhit, sycl::queue cqueue) {
  auto result = std::make_shared<MPHIT>(ntrks, nevts, nlayer, cqueue);
  //create an accessor field:
  std::unique_ptr<MPHITAccessor<order>> rA(new MPHITAccessor<order>(*result));

  // store in element order for bunches of bsize matrices (a la matriplex)
  for (int lay=0;lay<nlayer;++lay) {
    for (int ie=0;ie<nevts;++ie) {
      for (int ib=0;ib<nb;++ib) {
        for (int it=0;it<bsize;++it) {
          //const int l = it + ib*bsize + ie*nb*bsize + lay*nb*bsize*nevts;
          const int tid = ib + ie*nb;
        	//pos
        	for (int ip=0;ip<3;++ip) {
            rA->pos(ip, tid, it, lay) = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (int ip=0;ip<6;++ip) {
            rA->cov(ip, tid, it, lay) = (1+smear*randn(0,1))*inputhit.cov[ip];
        	}
        }
      }
    }
  }
  return std::move(result);
}

//Pure static version:

template <typename T, int N, int bSize>
struct MPNX_ {
   std::array<T,N*bSize> data;
   //basic accessors
   const T& operator[](const int idx) const {return data[idx];}
   T& operator[](const int idx) {return data[idx];}
};

using MP1I_    = MPNX_<int,   1 , bsize>;
using MP1F_    = MPNX_<float, 1 , bsize>;
using MP2F_    = MPNX_<float, 3 , bsize>;
using MP3F_    = MPNX_<float, 3 , bsize>;
using MP6F_    = MPNX_<float, 6 , bsize>;
using MP2x2SF_ = MPNX_<float, 3 , bsize>;
using MP3x3SF_ = MPNX_<float, 6 , bsize>;
using MP6x6SF_ = MPNX_<float, 21, bsize>;
using MP6x6F_  = MPNX_<float, 36, bsize>;
using MP3x3_   = MPNX_<float, 9 , bsize>;
using MP3x6_   = MPNX_<float, 18, bsize>;

struct MPTRK_ {
  MP6F_    par;
  MP6x6SF_ cov;
  MP1I_    q;
  //  MP22I   hitidx;
};

struct MPHIT_ {
  MP3F_    pos;
  MP3x3SF_ cov;
};

//////////////////////////////////////////////////////////////////////////////////////

MPTRK_* bTk(MPTRK_* tracks, int ev, int ib) {
  return &(tracks[ib + nb*ev]);
}

const MPTRK_* bTk(const MPTRK_* tracks, int ev, int ib) {
  return &(tracks[ib + nb*ev]);
}

float q(const MP1I_* bq, int it){
  return (*bq).data[it];
}
//
float par(const MP6F_* bpars, int it, int ipar){
  return (*bpars).data[it + ipar*bsize];
}
float x    (const MP6F_* bpars, int it){ return par(bpars, it, 0); }
float y    (const MP6F_* bpars, int it){ return par(bpars, it, 1); }
float z    (const MP6F_* bpars, int it){ return par(bpars, it, 2); }
float ipt  (const MP6F_* bpars, int it){ return par(bpars, it, 3); }
float phi  (const MP6F_* bpars, int it){ return par(bpars, it, 4); }
float theta(const MP6F_* bpars, int it){ return par(bpars, it, 5); }
//
float par(const MPTRK_* btracks, int it, int ipar){
  return par(&(*btracks).par,it,ipar);
}
float x    (const MPTRK_* btracks, int it){ return par(btracks, it, 0); }
float y    (const MPTRK_* btracks, int it){ return par(btracks, it, 1); }
float z    (const MPTRK_* btracks, int it){ return par(btracks, it, 2); }
float ipt  (const MPTRK_* btracks, int it){ return par(btracks, it, 3); }
float phi  (const MPTRK_* btracks, int it){ return par(btracks, it, 4); }
float theta(const MPTRK_* btracks, int it){ return par(btracks, it, 5); }
//
float par(const MPTRK_* tracks, int ev, int tk, int ipar){
  int ib = tk/bsize;
  const MPTRK_* btracks = bTk(tracks, ev, ib);
  int it = tk % bsize;
  return par(btracks, it, ipar);
}
float x    (const MPTRK_* tracks, int ev, int tk){ return par(tracks, ev, tk, 0); }
float y    (const MPTRK_* tracks, int ev, int tk){ return par(tracks, ev, tk, 1); }
float z    (const MPTRK_* tracks, int ev, int tk){ return par(tracks, ev, tk, 2); }
float ipt  (const MPTRK_* tracks, int ev, int tk){ return par(tracks, ev, tk, 3); }
float phi  (const MPTRK_* tracks, int ev, int tk){ return par(tracks, ev, tk, 4); }
float theta(const MPTRK_* tracks, int ev, int tk){ return par(tracks, ev, tk, 5); }
//

const MPHIT_* bHit(const MPHIT_* hits, int ev, int ib) {
  return &(hits[ib + nb*ev]);
}
const MPHIT_* bHit(const MPHIT_* hits, int ev, int ib,int lay) {
return &(hits[lay + (ib*nlayer) +(ev*nlayer*nb)]);
}
//
float pos(const MP3F_* hpos, int it, int ipar){
  return (*hpos).data[it + ipar*bsize];
}
float x(const MP3F_* hpos, int it)    { return pos(hpos, it, 0); }
float y(const MP3F_* hpos, int it)    { return pos(hpos, it, 1); }
float z(const MP3F_* hpos, int it)    { return pos(hpos, it, 2); }
//
float pos(const MPHIT_* hits, int it, int ipar){
  return pos(&(*hits).pos,it,ipar);
}
float x(const MPHIT_* hits, int it)    { return pos(hits, it, 0); }
float y(const MPHIT_* hits, int it)    { return pos(hits, it, 1); }
float z(const MPHIT_* hits, int it)    { return pos(hits, it, 2); }
//
float pos(const MPHIT_* hits, int ev, int tk, int ipar){
  int ib = tk/bsize;
  const MPHIT_* bhits = bHit(hits, ev, ib);
  int it = tk % bsize;
  return pos(bhits,it,ipar);
}
float x(const MPHIT_* hits, int ev, int tk)    { return pos(hits, ev, tk, 0); }
float y(const MPHIT_* hits, int ev, int tk)    { return pos(hits, ev, tk, 1); }
float z(const MPHIT_* hits, int ev, int tk)    { return pos(hits, ev, tk, 2); }

template<FieldOrder order>
void convertTracks(MPTRK_* out,  const MPTRK* inp) {
  //create an accessor field:
  std::unique_ptr<MPTRKAccessor<order>> inpA(new MPTRKAccessor<order>(*inp));
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (int ie=0;ie<nevts;++ie) {
    for (int ib=0;ib<nb;++ib) {
      for (int it=0;it<bsize;++it) {
        //const int l = it+ib*bsize+ie*nb*bsize;
        const int tid = ib+ie*nb;
    	  //par
    	  for (int ip=0;ip<6;++ip) {
    	    out[tid].par.data[it + ip*bsize] = inpA->par(ip, tid, it, 0);
    	  }
    	  //cov
    	  for (int ip=0;ip<21;++ip) {
    	    out[tid].cov.data[it + ip*bsize] = inpA->cov(ip, tid, it, 0);
    	  }
    	  //q
    	  out[tid].q.data[it] = inpA->q(0, tid, it, 0);//fixme check
      }
    }
  }
  return;
}

template<FieldOrder order>
void convertHits(MPHIT_* out, const MPHIT* inp) {
  //create an accessor field:
  std::unique_ptr<MPHITAccessor<order>> inpA(new MPHITAccessor<order>(*inp));
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (int lay=0;lay<nlayer;++lay) {
    for (int ie=0;ie<nevts;++ie) {
      for (int ib=0;ib<nb;++ib) {
        for (int it=0;it<bsize;++it) {
          //const int l = it + ib*bsize + ie*nb*bsize + lay*nb*bsize*nevts;
          const int tid = ib + ie*nb;
        	//pos
        	for (int ip=0;ip<3;++ip) {
            out[lay+nlayer*tid].pos.data[it + ip*bsize] = inpA->pos(ip, tid, it, lay);
        	}
        	//cov
        	for (int ip=0;ip<6;++ip) {
            out[lay+nlayer*tid].cov.data[it + ip*bsize] = inpA->cov(ip, tid, it, lay);
        	}
        }
      }
    }
  }
  return;
}

MPHIT_* prepareHits(struct AHIT inputhit) {
  MPHIT_* result = new MPHIT_[nlayer*nevts*nb];

  // store in element order for bunches of bsize matrices (a la matriplex)
  for (int lay=0;lay<nlayer;++lay) {
    for (int ie=0;ie<nevts;++ie) {
      for (int ib=0;ib<nb;++ib) {
        for (int it=0;it<bsize;++it) {
        	//pos
        	for (int ip=0;ip<3;++ip) {
        	  result[lay+nlayer*(ib + nb*ie)].pos.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (int ip=0;ip<6;++ip) {
        	  result[lay+nlayer*(ib + nb*ie)].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.cov[ip];
        	}
        }
      }
    }
  }
  return result;
}


////////////////////////////////////////////////////////////////////////
///MAIN subroutines

template<typename MP6x6SFAccessor_, int bsz = 1>
inline void MultHelixProp(const MP6x6F_ &a, const MP6x6SFAccessor_ &b, MP6x6F_ &c, const int tid) {

  const float offset = b.GetThreadOffset(tid);

  for (int it = 0;it < bsz; it++) {
    const float boffset = offset+it;
    c[ 0*bsz+it] = a[ 0*bsz+it]*b( 0, boffset) + a[ 1*bsz+it]*b( 1, boffset) + a[ 3*bsz+it]*b( 6, boffset) + a[ 4*bsz+it]*b(10, boffset);
    c[ 1*bsz+it] = a[ 0*bsz+it]*b( 1, boffset) + a[ 1*bsz+it]*b( 2, boffset) + a[ 3*bsz+it]*b( 7, boffset) + a[ 4*bsz+it]*b(11, boffset);
    c[ 2*bsz+it] = a[ 0*bsz+it]*b( 3, boffset) + a[ 1*bsz+it]*b( 4, boffset) + a[ 3*bsz+it]*b( 8, boffset) + a[ 4*bsz+it]*b(12, boffset);
    c[ 3*bsz+it] = a[ 0*bsz+it]*b( 6, boffset) + a[ 1*bsz+it]*b( 7, boffset) + a[ 3*bsz+it]*b( 9, boffset) + a[ 4*bsz+it]*b(13, boffset);
    c[ 4*bsz+it] = a[ 0*bsz+it]*b(10, boffset) + a[ 1*bsz+it]*b(11, boffset) + a[ 3*bsz+it]*b(13, boffset) + a[ 4*bsz+it]*b(14, boffset);
    c[ 5*bsz+it] = a[ 0*bsz+it]*b(15, boffset) + a[ 1*bsz+it]*b(16, boffset) + a[ 3*bsz+it]*b(18, boffset) + a[ 4*bsz+it]*b(19, boffset);
    c[ 6*bsz+it] = a[ 6*bsz+it]*b( 0, boffset) + a[ 7*bsz+it]*b( 1, boffset) + a[ 9*bsz+it]*b( 6, boffset) + a[10*bsz+it]*b(10, boffset);
    c[ 7*bsz+it] = a[ 6*bsz+it]*b( 1, boffset) + a[ 7*bsz+it]*b( 2, boffset) + a[ 9*bsz+it]*b( 7, boffset) + a[10*bsz+it]*b(11, boffset);
    c[ 8*bsz+it] = a[ 6*bsz+it]*b( 3, boffset) + a[ 7*bsz+it]*b( 4, boffset) + a[ 9*bsz+it]*b( 8, boffset) + a[10*bsz+it]*b(12, boffset);
    c[ 9*bsz+it] = a[ 6*bsz+it]*b( 6, boffset) + a[ 7*bsz+it]*b( 7, boffset) + a[ 9*bsz+it]*b( 9, boffset) + a[10*bsz+it]*b(13, boffset);
    c[10*bsz+it] = a[ 6*bsz+it]*b(10, boffset) + a[ 7*bsz+it]*b(11, boffset) + a[ 9*bsz+it]*b(13, boffset) + a[10*bsz+it]*b(14, boffset);
    c[11*bsz+it] = a[ 6*bsz+it]*b(15, boffset) + a[ 7*bsz+it]*b(16, boffset) + a[ 9*bsz+it]*b(18, boffset) + a[10*bsz+it]*b(19, boffset);
    
    c[12*bsz+it] = a[12*bsz+it]*b( 0, boffset) + a[13*bsz+it]*b( 1, boffset) + b( 3, boffset) + a[15*bsz+it]*b( 6, boffset) + a[16*bsz+it]*b(10, boffset) + a[17*bsz+it]*b(15, boffset);
    c[13*bsz+it] = a[12*bsz+it]*b( 1, boffset) + a[13*bsz+it]*b( 2, boffset) + b( 4, boffset) + a[15*bsz+it]*b( 7, boffset) + a[16*bsz+it]*b(11, boffset) + a[17*bsz+it]*b(16, boffset);
    c[14*bsz+it] = a[12*bsz+it]*b( 3, boffset) + a[13*bsz+it]*b( 4, boffset) + b( 5, boffset) + a[15*bsz+it]*b( 8, boffset) + a[16*bsz+it]*b(12, boffset) + a[17*bsz+it]*b(17, boffset);
    c[15*bsz+it] = a[12*bsz+it]*b( 6, boffset) + a[13*bsz+it]*b( 7, boffset) + b( 8, boffset) + a[15*bsz+it]*b( 9, boffset) + a[16*bsz+it]*b(13, boffset) + a[17*bsz+it]*b(18, boffset);
    c[16*bsz+it] = a[12*bsz+it]*b(10, boffset) + a[13*bsz+it]*b(11, boffset) + b(12, boffset) + a[15*bsz+it]*b(13, boffset) + a[16*bsz+it]*b(14, boffset) + a[17*bsz+it]*b(19, boffset);
    c[17*bsz+it] = a[12*bsz+it]*b(15, boffset) + a[13*bsz+it]*b(16, boffset) + b(17, boffset) + a[15*bsz+it]*b(18, boffset) + a[16*bsz+it]*b(19, boffset) + a[17*bsz+it]*b(20, boffset);
    
    c[18*bsz+it] = a[18*bsz+it]*b( 0, boffset) + a[19*bsz+it]*b( 1, boffset) + a[21*bsz+it]*b( 6, boffset) + a[22*bsz+it]*b(10, boffset);
    c[19*bsz+it] = a[18*bsz+it]*b( 1, boffset) + a[19*bsz+it]*b( 2, boffset) + a[21*bsz+it]*b( 7, boffset) + a[22*bsz+it]*b(11, boffset);
    c[20*bsz+it] = a[18*bsz+it]*b( 3, boffset) + a[19*bsz+it]*b( 4, boffset) + a[21*bsz+it]*b( 8, boffset) + a[22*bsz+it]*b(12, boffset);
    c[21*bsz+it] = a[18*bsz+it]*b( 6, boffset) + a[19*bsz+it]*b( 7, boffset) + a[21*bsz+it]*b( 9, boffset) + a[22*bsz+it]*b(13, boffset);
    c[22*bsz+it] = a[18*bsz+it]*b(10, boffset) + a[19*bsz+it]*b(11, boffset) + a[21*bsz+it]*b(13, boffset) + a[22*bsz+it]*b(14, boffset);
    c[23*bsz+it] = a[18*bsz+it]*b(15, boffset) + a[19*bsz+it]*b(16, boffset) + a[21*bsz+it]*b(18, boffset) + a[22*bsz+it]*b(19, boffset);
    c[24*bsz+it] = a[24*bsz+it]*b( 0, boffset) + a[25*bsz+it]*b( 1, boffset) + a[27*bsz+it]*b( 6, boffset) + a[28*bsz+it]*b(10, boffset);
    c[25*bsz+it] = a[24*bsz+it]*b( 1, boffset) + a[25*bsz+it]*b( 2, boffset) + a[27*bsz+it]*b( 7, boffset) + a[28*bsz+it]*b(11, boffset);
    c[26*bsz+it] = a[24*bsz+it]*b( 3, boffset) + a[25*bsz+it]*b( 4, boffset) + a[27*bsz+it]*b( 8, boffset) + a[28*bsz+it]*b(12, boffset);
    c[27*bsz+it] = a[24*bsz+it]*b( 6, boffset) + a[25*bsz+it]*b( 7, boffset) + a[27*bsz+it]*b( 9, boffset) + a[28*bsz+it]*b(13, boffset);
    c[28*bsz+it] = a[24*bsz+it]*b(10, boffset) + a[25*bsz+it]*b(11, boffset) + a[27*bsz+it]*b(13, boffset) + a[28*bsz+it]*b(14, boffset);
    c[29*bsz+it] = a[24*bsz+it]*b(15, boffset) + a[25*bsz+it]*b(16, boffset) + a[27*bsz+it]*b(18, boffset) + a[28*bsz+it]*b(19, boffset);
    c[30*bsz+it] = b(15, boffset);
    c[31*bsz+it] = b(16, boffset);
    c[32*bsz+it] = b(17, boffset);
    c[33*bsz+it] = b(18, boffset);
    c[34*bsz+it] = b(19, boffset);
    c[35*bsz+it] = b(20, boffset);    
  }
  return;
}

template<typename MP6x6SFAccessor_, int bsz = 1>
inline void MultHelixPropTransp(const MP6x6F_ &a, const MP6x6F_ &b, MP6x6SFAccessor_ &c, const int tid) {

  const float offset = c.GetThreadOffset(tid);
  
  for (int it = 0;it < bsz; it++) {
    const float boffset = offset+it;
    
    c( 0, boffset) = b[ 0*bsz+it]*a[ 0*bsz+it] + b[ 1*bsz+it]*a[ 1*bsz+it] + b[ 3*bsz+it]*a[ 3*bsz+it] + b[ 4*bsz+it]*a[ 4*bsz+it];
    c( 1, boffset) = b[ 6*bsz+it]*a[ 0*bsz+it] + b[ 7*bsz+it]*a[ 1*bsz+it] + b[ 9*bsz+it]*a[ 3*bsz+it] + b[10*bsz+it]*a[ 4*bsz+it];
    c( 2, boffset) = b[ 6*bsz+it]*a[ 6*bsz+it] + b[ 7*bsz+it]*a[ 7*bsz+it] + b[ 9*bsz+it]*a[ 9*bsz+it] + b[10*bsz+it]*a[10*bsz+it];
    c( 3, boffset) = b[12*bsz+it]*a[ 0*bsz+it] + b[13*bsz+it]*a[ 1*bsz+it] + b[15*bsz+it]*a[ 3*bsz+it] + b[16*bsz+it]*a[ 4*bsz+it];
    c( 4, boffset) = b[12*bsz+it]*a[ 6*bsz+it] + b[13*bsz+it]*a[ 7*bsz+it] + b[15*bsz+it]*a[ 9*bsz+it] + b[16*bsz+it]*a[10*bsz+it];
    c( 5, boffset) = b[12*bsz+it]*a[12*bsz+it] + b[13*bsz+it]*a[13*bsz+it] + b[14*bsz+it] + b[15*bsz+it]*a[15*bsz+it] + b[16*bsz+it]*a[16*bsz+it] + b[17*bsz+it]*a[17*bsz+it];
    c( 6, boffset) = b[18*bsz+it]*a[ 0*bsz+it] + b[19*bsz+it]*a[ 1*bsz+it] + b[21*bsz+it]*a[ 3*bsz+it] + b[22*bsz+it]*a[ 4*bsz+it];
    c( 7, boffset) = b[18*bsz+it]*a[ 6*bsz+it] + b[19*bsz+it]*a[ 7*bsz+it] + b[21*bsz+it]*a[ 9*bsz+it] + b[22*bsz+it]*a[10*bsz+it];
    c( 8, boffset) = b[18*bsz+it]*a[12*bsz+it] + b[19*bsz+it]*a[13*bsz+it] + b[20*bsz+it] + b[21*bsz+it]*a[15*bsz+it] + b[22*bsz+it]*a[16*bsz+it] + b[23*bsz+it]*a[17*bsz+it];
    c( 9, boffset) = b[18*bsz+it]*a[18*bsz+it] + b[19*bsz+it]*a[19*bsz+it] + b[21*bsz+it]*a[21*bsz+it] + b[22*bsz+it]*a[22*bsz+it];
    c(10, boffset) = b[24*bsz+it]*a[ 0*bsz+it] + b[25*bsz+it]*a[ 1*bsz+it] + b[27*bsz+it]*a[ 3*bsz+it] + b[28*bsz+it]*a[ 4*bsz+it];
    c(11, boffset) = b[24*bsz+it]*a[ 6*bsz+it] + b[25*bsz+it]*a[ 7*bsz+it] + b[27*bsz+it]*a[ 9*bsz+it] + b[28*bsz+it]*a[10*bsz+it];
    c(12, boffset) = b[24*bsz+it]*a[12*bsz+it] + b[25*bsz+it]*a[13*bsz+it] + b[26*bsz+it] + b[27*bsz+it]*a[15*bsz+it] + b[28*bsz+it]*a[16*bsz+it] + b[29*bsz+it]*a[17*bsz+it];
    c(13, boffset) = b[24*bsz+it]*a[18*bsz+it] + b[25*bsz+it]*a[19*bsz+it] + b[27*bsz+it]*a[21*bsz+it] + b[28*bsz+it]*a[22*bsz+it];
    c(14, boffset) = b[24*bsz+it]*a[24*bsz+it] + b[25*bsz+it]*a[25*bsz+it] + b[27*bsz+it]*a[27*bsz+it] + b[28*bsz+it]*a[28*bsz+it];
    c(15, boffset) = b[30*bsz+it]*a[ 0*bsz+it] + b[31*bsz+it]*a[ 1*bsz+it] + b[33*bsz+it]*a[ 3*bsz+it] + b[34*bsz+it]*a[ 4*bsz+it];
    c(16, boffset) = b[30*bsz+it]*a[ 6*bsz+it] + b[31*bsz+it]*a[ 7*bsz+it] + b[33*bsz+it]*a[ 9*bsz+it] + b[34*bsz+it]*a[10*bsz+it];
    c(17, boffset) = b[30*bsz+it]*a[12*bsz+it] + b[31*bsz+it]*a[13*bsz+it] + b[32*bsz+it] + b[33*bsz+it]*a[15*bsz+it] + b[34*bsz+it]*a[16*bsz+it] + b[35*bsz+it]*a[17*bsz+it];
    c(18, boffset) = b[30*bsz+it]*a[18*bsz+it] + b[31*bsz+it]*a[19*bsz+it] + b[33*bsz+it]*a[21*bsz+it] + b[34*bsz+it]*a[22*bsz+it];
    c(19, boffset) = b[30*bsz+it]*a[24*bsz+it] + b[31*bsz+it]*a[25*bsz+it] + b[33*bsz+it]*a[27*bsz+it] + b[34*bsz+it]*a[28*bsz+it];
    c(20, boffset) = b[35*bsz+it];
  }
  return;  
}

template<typename AccessorTp1, typename AccessorTp2, int bsz = 1>
inline void KalmanGainInv(const AccessorTp1 &a, const AccessorTp2 &b, MP3x3_ &c, const int tid, const int lay) {

  const float a_offset_ = a.GetThreadOffset(tid);
  const float b_offset_ = b.GetThreadOffset(tid, lay);
  
  for (int it = 0; it < bsz; ++it)
  {
    const float a_offset = a_offset_+it;
    const float b_offset = b_offset_+it;

    float det =
        ((a(0, a_offset)+b(0, b_offset))*(((a(6, a_offset)+b(3, b_offset)) *(a(11,a_offset)+b(5, b_offset))) - ((a(7, a_offset)+b(4, b_offset)) *(a(7, a_offset)+b(4, b_offset))))) -
        ((a(1, a_offset)+b(1, b_offset))*(((a(1, a_offset)+b(1, b_offset)) *(a(11,a_offset)+b(5, b_offset))) - ((a(7, a_offset)+b(4, b_offset)) *(a(2, a_offset)+b(2, b_offset))))) +
        ((a(2, a_offset)+b(2, b_offset))*(((a(1, a_offset)+b(1, b_offset)) *(a(7, a_offset)+b(4, b_offset))) - ((a(2, a_offset)+b(2, b_offset)) *(a(6, a_offset)+b(3, b_offset)))));

    float invdet = 1.0f / det;

    c[0*bsz+it] =   invdet*(((a(6, a_offset)+b(3, b_offset)) *(a(11,a_offset)+b(5, b_offset))) - ((a(7, a_offset)+b(4, b_offset)) *(a(7, a_offset)+b(4, b_offset))));
    c[1*bsz+it] =  -invdet*(((a(1, a_offset)+b(1, b_offset)) *(a(11,a_offset)+b(5, b_offset))) - ((a(2, a_offset)+b(2, b_offset)) *(a(7, a_offset)+b(4, b_offset))));
    c[2*bsz+it] =   invdet*(((a(1, a_offset)+b(1, b_offset)) *(a(7, a_offset)+b(4, b_offset))) - ((a(2, a_offset)+b(2, b_offset)) *(a(7, a_offset)+b(4, b_offset))));
    c[3*bsz+it] =  -invdet*(((a(1, a_offset)+b(1, b_offset)) *(a(11,a_offset)+b(5, b_offset))) - ((a(7, a_offset)+b(4, b_offset)) *(a(2, a_offset)+b(2, b_offset))));
    c[4*bsz+it] =   invdet*(((a(0, a_offset)+b(0, b_offset)) *(a(11,a_offset)+b(5, b_offset))) - ((a(2, a_offset)+b(2, b_offset)) *(a(2, a_offset)+b(2, b_offset))));
    c[5*bsz+it] =  -invdet*(((a(0, a_offset)+b(0, b_offset)) *(a(7, a_offset)+b(4, b_offset))) - ((a(2, a_offset)+b(2, b_offset)) *(a(1, a_offset)+b(1, b_offset))));
    c[6*bsz+it] =   invdet*(((a(1, a_offset)+b(1, b_offset)) *(a(7, a_offset)+b(4, b_offset))) - ((a(2, a_offset)+b(2, b_offset)) *(a(6, a_offset)+b(3, b_offset))));
    c[7*bsz+it] =  -invdet*(((a(0, a_offset)+b(0, b_offset)) *(a(7, a_offset)+b(4, b_offset))) - ((a(2, a_offset)+b(2, b_offset)) *(a(1, a_offset)+b(1, b_offset))));
    c[8*bsz+it] =   invdet*(((a(0, a_offset)+b(0, b_offset)) *(a(6, a_offset)+b(3, b_offset))) - ((a(1, a_offset)+b(1, b_offset)) *(a(1, a_offset)+b(1, b_offset))));

  }
}

template<typename AccessorTp, int bsz = 1>
inline void KalmanGain(const AccessorTp &a, const MP3x3_ &b, MP3x6_ &c, const int tid) {
  const float a_offset_= a.GetThreadOffset(tid);

  for (int it = 0; it < bsz; ++it)
  {
    const float a_offset = a_offset_+it;

    c[ 0*bsz+it] = a(0, a_offset)*b[0*bsz+it] + a( 1, a_offset)*b[3*bsz+it] + a( 2, a_offset)*b[6*bsz+it];
    c[ 1*bsz+it] = a(0, a_offset)*b[1*bsz+it] + a( 1, a_offset)*b[4*bsz+it] + a( 2, a_offset)*b[7*bsz+it];
    c[ 2*bsz+it] = a(0, a_offset)*b[2*bsz+it] + a( 1, a_offset)*b[5*bsz+it] + a( 2, a_offset)*b[8*bsz+it];
    c[ 3*bsz+it] = a(1, a_offset)*b[0*bsz+it] + a( 6, a_offset)*b[3*bsz+it] + a( 7, a_offset)*b[6*bsz+it];
    c[ 4*bsz+it] = a(1, a_offset)*b[1*bsz+it] + a( 6, a_offset)*b[4*bsz+it] + a( 7, a_offset)*b[7*bsz+it];
    c[ 5*bsz+it] = a(1, a_offset)*b[2*bsz+it] + a( 6, a_offset)*b[5*bsz+it] + a( 7, a_offset)*b[8*bsz+it];
    c[ 6*bsz+it] = a(2, a_offset)*b[0*bsz+it] + a( 7, a_offset)*b[3*bsz+it] + a(11, a_offset)*b[6*bsz+it];
    c[ 7*bsz+it] = a(2, a_offset)*b[1*bsz+it] + a( 7, a_offset)*b[4*bsz+it] + a(11, a_offset)*b[7*bsz+it];
    c[ 8*bsz+it] = a(2, a_offset)*b[2*bsz+it] + a( 7, a_offset)*b[5*bsz+it] + a(11, a_offset)*b[8*bsz+it];
    c[ 9*bsz+it] = a(3, a_offset)*b[0*bsz+it] + a( 8, a_offset)*b[3*bsz+it] + a(12, a_offset)*b[6*bsz+it];
    c[10*bsz+it] = a(3, a_offset)*b[1*bsz+it] + a( 8, a_offset)*b[4*bsz+it] + a(12, a_offset)*b[7*bsz+it];
    c[11*bsz+it] = a(3, a_offset)*b[2*bsz+it] + a( 8, a_offset)*b[5*bsz+it] + a(12, a_offset)*b[8*bsz+it];
    c[12*bsz+it] = a(4, a_offset)*b[0*bsz+it] + a( 9, a_offset)*b[3*bsz+it] + a(13, a_offset)*b[6*bsz+it];
    c[13*bsz+it] = a(4, a_offset)*b[1*bsz+it] + a( 9, a_offset)*b[4*bsz+it] + a(13, a_offset)*b[7*bsz+it];
    c[14*bsz+it] = a(4, a_offset)*b[2*bsz+it] + a( 9, a_offset)*b[5*bsz+it] + a(13, a_offset)*b[8*bsz+it];
    c[15*bsz+it] = a(5, a_offset)*b[0*bsz+it] + a(10, a_offset)*b[3*bsz+it] + a(14, a_offset)*b[6*bsz+it];
    c[16*bsz+it] = a(5, a_offset)*b[1*bsz+it] + a(10, a_offset)*b[4*bsz+it] + a(14, a_offset)*b[7*bsz+it];
    c[17*bsz+it] = a(5, a_offset)*b[2*bsz+it] + a(10, a_offset)*b[5*bsz+it] + a(14, a_offset)*b[8*bsz+it];

  }
}


auto hipo = [](const float x, const float y) {return std::sqrt(x*x + y*y);};

template <class MPTRKAccessors, class MPHITAccessors, int bsz = 1>
void KalmanUpdate(MPTRKAccessors       obtracksPtr,
		  const MPHITAccessors bhitsPtr,
		  const int tid,
		  const int lay) {
  using MP6Faccessor    = typename MPTRKAccessors::MP6FAccessor;
  using MP6x6SFaccessor = typename MPTRKAccessors::MP6x6SFAccessor;
  using MP3x3SFaccessor = typename MPHITAccessors::MP3x3SFAccessor;
  using MP3Faccessor    = typename MPHITAccessors::MP3FAccessor;
  
  const MP3x3SFaccessor &hitErr   = bhitsPtr.cov;
  const MP3Faccessor    &msP      = bhitsPtr.pos;

  MP6x6SFaccessor  &trkErr = obtracksPtr.cov;
  MP6Faccessor     &inPar  = obtracksPtr.par;		  
  
  MP1F_    rotT00;
  MP1F_    rotT01;
  MP2x2SF_ resErr_loc;
  //MP3x3SF_ resErr_glo;
  
  const float terr_offset = trkErr.GetThreadOffset(tid);
  const float ipar_offset = inPar.GetThreadOffset(tid); 
  const float herr_offset = hitErr.GetThreadOffset(tid, lay);   
  
  for (int it = 0;it < bsz; ++it) {
    const float terr_blk_offset = terr_offset+it;
    const float herr_blk_offset = herr_offset+it;    
    const float ipar_blk_offset = ipar_offset+it;
    
    const float msPX = msP(iparX, tid, it, lay);
    const float msPY = msP(iparY, tid, it, lay);
    const float inParX = inPar(iparX, ipar_blk_offset);
    const float inParY = inPar(iparY, ipar_blk_offset);          
  
    const float r = hipo(msPX, msPY);
    rotT00[it] = -(msPY + inParY) / (2*r);
    rotT01[it] =  (msPX + inParX) / (2*r);    
    
    resErr_loc[ 0*bsz+it] = (rotT00[it]*(trkErr(0, terr_blk_offset) + hitErr(0, herr_blk_offset)) +
                                    rotT01[it]*(trkErr(1, terr_blk_offset) + hitErr(1, herr_blk_offset)))*rotT00[it] +
                                   (rotT00[it]*(trkErr(1, terr_blk_offset) + hitErr(1, herr_blk_offset)) +
                                    rotT01[it]*(trkErr(2, terr_blk_offset) + hitErr(2, herr_blk_offset)))*rotT01[it];
    resErr_loc[ 1*bsz+it] = (trkErr(3, terr_blk_offset) + hitErr(3, herr_blk_offset))*rotT00[it] +
                                   (trkErr(4, terr_blk_offset) + hitErr(4, herr_blk_offset))*rotT01[it];
    resErr_loc[ 2*bsz+it] = (trkErr(5, terr_blk_offset) + hitErr(5, herr_blk_offset));
  } 
  
  for (int it=0;it<bsz;++it) {
  
    const float det = (float)resErr_loc[0*bsz+it] * resErr_loc[2*bsz+it] -
                      (float)resErr_loc[1*bsz+it] * resErr_loc[1*bsz+it];
    const float s   = 1.f / det;
    const float tmp = s * resErr_loc[2*bsz+it];
    resErr_loc[1*bsz+it] *= -s;
    resErr_loc[2*bsz+it]  = s * resErr_loc[0*bsz+it];
    resErr_loc[0*bsz+it]  = tmp;
  }     
  
  MP3x6_ kGain;
  
#pragma omp simd
  for (int it=0; it<bsz; ++it) {  
    const float terr_blk_offset = terr_offset+it;
    //
    kGain[ 0*bsz+it] = trkErr( 0, terr_blk_offset)*(rotT00[it]*resErr_loc[ 0*bsz+it]) +
	                        trkErr( 1, terr_blk_offset)*(rotT01[it]*resErr_loc[ 0*bsz+it]) +
	                        trkErr( 3, terr_blk_offset)*resErr_loc[ 1*bsz+it];
    kGain[ 1*bsz+it] = trkErr( 0, terr_blk_offset)*(rotT00[it]*resErr_loc[ 1*bsz+it]) +
	                        trkErr( 1, terr_blk_offset)*(rotT01[it]*resErr_loc[ 1*bsz+it]) +
	                        trkErr( 3, terr_blk_offset)*resErr_loc[ 2*bsz+it];
    kGain[ 2*bsz+it] = 0;
    kGain[ 3*bsz+it] = trkErr( 1, terr_blk_offset)*(rotT00[it]*resErr_loc[ 0*bsz+it]) +
	                        trkErr( 2, terr_blk_offset)*(rotT01[it]*resErr_loc[ 0*bsz+it]) +
	                        trkErr( 4, terr_blk_offset)*resErr_loc[ 1*bsz+it];
    kGain[ 4*bsz+it] = trkErr( 1, terr_blk_offset)*(rotT00[it]*resErr_loc[ 1*bsz+it]) +
	                        trkErr( 2, terr_blk_offset)*(rotT01[it]*resErr_loc[ 1*bsz+it]) +
	                        trkErr( 4, terr_blk_offset)*resErr_loc[ 2*bsz+it];
    kGain[ 5*bsz+it] = 0;
    kGain[ 6*bsz+it] = trkErr( 3, terr_blk_offset)*(rotT00[it]*resErr_loc[ 0*bsz+it]) +
	                        trkErr( 4, terr_blk_offset)*(rotT01[it]*resErr_loc[ 0*bsz+it]) +
	                        trkErr( 5, terr_blk_offset)*resErr_loc[ 1*bsz+it];
    kGain[ 7*bsz+it] = trkErr( 3, terr_blk_offset)*(rotT00[it]*resErr_loc[ 1*bsz+it]) +
	                        trkErr( 4, terr_blk_offset)*(rotT01[it]*resErr_loc[ 1*bsz+it]) +
	                        trkErr( 5, terr_blk_offset)*resErr_loc[ 2*bsz+it];
    kGain[ 8*bsz+it] = 0;
    kGain[ 9*bsz+it] = trkErr( 6, terr_blk_offset)*(rotT00[it]*resErr_loc[ 0*bsz+it]) +
	                        trkErr( 7, terr_blk_offset)*(rotT01[it]*resErr_loc[ 0*bsz+it]) +
	                        trkErr( 8, terr_blk_offset)*resErr_loc[ 1*bsz+it];
    kGain[10*bsz+it] = trkErr( 6, terr_blk_offset)*(rotT00[it]*resErr_loc[ 1*bsz+it]) +
	                        trkErr( 7, terr_blk_offset)*(rotT01[it]*resErr_loc[ 1*bsz+it]) +
	                        trkErr( 8, terr_blk_offset)*resErr_loc[ 2*bsz+it];
    kGain[11*bsz+it] = 0;
    kGain[12*bsz+it] = trkErr(10, terr_blk_offset)*(rotT00[it]*resErr_loc[ 0*bsz+it]) +
	                        trkErr(11, terr_blk_offset)*(rotT01[it]*resErr_loc[ 0*bsz+it]) +
	                        trkErr(12, terr_blk_offset)*resErr_loc[ 1*bsz+it];
    kGain[13*bsz+it] = trkErr(10, terr_blk_offset)*(rotT00[it]*resErr_loc[ 1*bsz+it]) +
	                        trkErr(11, terr_blk_offset)*(rotT01[it]*resErr_loc[ 1*bsz+it]) +
	                        trkErr(12, terr_blk_offset)*resErr_loc[ 2*bsz+it];
    kGain[14*bsz+it] = 0;
    kGain[15*bsz+it] = trkErr(15, terr_blk_offset)*(rotT00[it]*resErr_loc[ 0*bsz+it]) +
	                        trkErr(16, terr_blk_offset)*(rotT01[it]*resErr_loc[ 0*bsz+it]) +
	                        trkErr(17, terr_blk_offset)*resErr_loc[ 1*bsz+it];
    kGain[16*bsz+it] = trkErr(15, terr_blk_offset)*(rotT00[it]*resErr_loc[ 1*bsz+it]) +
	                        trkErr(16, terr_blk_offset)*(rotT01[it]*resErr_loc[ 1*bsz+it]) +
	                        trkErr(17, terr_blk_offset)*resErr_loc[ 2*bsz+it];
    kGain[17*bsz+it] = 0;
  }  
     
  MP2F_ res_loc;   
  for (int it = 0; it < bsz; ++it) { 
    const float ipar_blk_offset = ipar_offset+it;
    
    const float msPX = msP(iparX, tid, it, lay);
    const float msPY = msP(iparY, tid, it, lay);
    const float msPZ = msP(iparZ, tid, it, lay);    
    const float inParX = inPar(iparX, ipar_blk_offset);
    const float inParY = inPar(iparY, ipar_blk_offset);     
    const float inParZ = inPar(iparZ, ipar_blk_offset); 
    
    const float inParIpt   = inPar(iparIpt, ipar_blk_offset);
    const float inParPhi   = inPar(iparPhi, ipar_blk_offset);
    const float inParTheta = inPar(iparTheta, ipar_blk_offset);            
    
    res_loc[0*bsz+it] =  rotT00[it]*(msPX - inParX) + rotT01[it]*(msPY - inParY);
    res_loc[1*bsz+it] =  msPZ - inParZ;

    inPar(iparX,ipar_blk_offset)     = inParX + kGain[ 0*bsz+it] * res_loc[ 0*bsz+it] + kGain[ 1*bsz+it] * res_loc[ 1*bsz+it];
    inPar(iparY,ipar_blk_offset)     = inParY + kGain[ 3*bsz+it] * res_loc[ 0*bsz+it] + kGain[ 4*bsz+it] * res_loc[ 1*bsz+it];
    inPar(iparZ,ipar_blk_offset)     = inParZ + kGain[ 6*bsz+it] * res_loc[ 0*bsz+it] + kGain[ 7*bsz+it] * res_loc[ 1*bsz+it];
    inPar(iparIpt,ipar_blk_offset)   = inParIpt + kGain[ 9*bsz+it] * res_loc[ 0*bsz+it] + kGain[10*bsz+it] * res_loc[ 1*bsz+it];
    inPar(iparPhi,ipar_blk_offset)   = inParPhi + kGain[12*bsz+it] * res_loc[ 0*bsz+it] + kGain[13*bsz+it] * res_loc[ 1*bsz+it];
    inPar(iparTheta,ipar_blk_offset) = inParTheta + kGain[15*bsz+it] * res_loc[ 0*bsz+it] + kGain[16*bsz+it] * res_loc[ 1*bsz+it];    
  }

   MP6x6SF_ newErr;
   for (int it=0;it<bsize;++it)   {
     const float terr_blk_offset = terr_offset+it;

     newErr[ 0*bsz+it] = kGain[ 0*bsz+it]*rotT00[it]*trkErr( 0, terr_blk_offset) +
                         kGain[ 0*bsz+it]*rotT01[it]*trkErr( 1, terr_blk_offset) +
                         kGain[ 1*bsz+it]*trkErr( 3, terr_blk_offset);
     newErr[ 1*bsz+it] = kGain[ 3*bsz+it]*rotT00[it]*trkErr( 0, terr_blk_offset) +
                         kGain[ 3*bsz+it]*rotT01[it]*trkErr( 1, terr_blk_offset) +
                         kGain[ 4*bsz+it]*trkErr( 3, terr_blk_offset);
     newErr[ 2*bsz+it] = kGain[ 3*bsz+it]*rotT00[it]*trkErr( 1, terr_blk_offset) +
                         kGain[ 3*bsz+it]*rotT01[it]*trkErr( 2, terr_blk_offset) +
                         kGain[ 4*bsz+it]*trkErr( 4, terr_blk_offset);
     newErr[ 3*bsz+it] = kGain[ 6*bsz+it]*rotT00[it]*trkErr( 0, terr_blk_offset) +
                         kGain[ 6*bsz+it]*rotT01[it]*trkErr( 1, terr_blk_offset) +
                         kGain[ 7*bsz+it]*trkErr( 3, terr_blk_offset);
     newErr[ 4*bsz+it] = kGain[ 6*bsz+it]*rotT00[it]*trkErr( 1, terr_blk_offset) +
                         kGain[ 6*bsz+it]*rotT01[it]*trkErr( 2, terr_blk_offset) +
                         kGain[ 7*bsz+it]*trkErr( 4, terr_blk_offset);
     newErr[ 5*bsz+it] = kGain[ 6*bsz+it]*rotT00[it]*trkErr( 3, terr_blk_offset) +
                         kGain[ 6*bsz+it]*rotT01[it]*trkErr( 4, terr_blk_offset) +
                         kGain[ 7*bsz+it]*trkErr( 5, terr_blk_offset);
     newErr[ 6*bsz+it] = kGain[ 9*bsz+it]*rotT00[it]*trkErr( 0, terr_blk_offset) +
                         kGain[ 9*bsz+it]*rotT01[it]*trkErr( 1, terr_blk_offset) +
                         kGain[10*bsz+it]*trkErr( 3, terr_blk_offset);
     newErr[ 7*bsz+it] = kGain[ 9*bsz+it]*rotT00[it]*trkErr( 1, terr_blk_offset) +
                         kGain[ 9*bsz+it]*rotT01[it]*trkErr( 2, terr_blk_offset) +
                         kGain[10*bsz+it]*trkErr( 4, terr_blk_offset);
     newErr[ 8*bsz+it] = kGain[ 9*bsz+it]*rotT00[it]*trkErr( 3, terr_blk_offset) +
                         kGain[ 9*bsz+it]*rotT01[it]*trkErr( 4, terr_blk_offset) +
                         kGain[10*bsz+it]*trkErr( 5, terr_blk_offset);
     newErr[ 9*bsz+it] = kGain[ 9*bsz+it]*rotT00[it]*trkErr( 6, terr_blk_offset) +
                         kGain[ 9*bsz+it]*rotT01[it]*trkErr( 7, terr_blk_offset) +
                         kGain[10*bsz+it]*trkErr( 8, terr_blk_offset);
     newErr[10*bsz+it] = kGain[12*bsz+it]*rotT00[it]*trkErr( 0, terr_blk_offset) +
                         kGain[12*bsz+it]*rotT01[it]*trkErr( 1, terr_blk_offset) +
                         kGain[13*bsz+it]*trkErr( 3, terr_blk_offset);
     newErr[11*bsz+it] = kGain[12*bsz+it]*rotT00[it]*trkErr( 1, terr_blk_offset) +
                         kGain[12*bsz+it]*rotT01[it]*trkErr( 2, terr_blk_offset) +
                         kGain[13*bsz+it]*trkErr( 4, terr_blk_offset);
     newErr[12*bsz+it] = kGain[12*bsz+it]*rotT00[it]*trkErr( 3, terr_blk_offset) +
                         kGain[12*bsz+it]*rotT01[it]*trkErr( 4, terr_blk_offset) +
                         kGain[13*bsz+it]*trkErr( 5, terr_blk_offset);
     newErr[13*bsz+it] = kGain[12*bsz+it]*rotT00[it]*trkErr( 6, terr_blk_offset) +
                         kGain[12*bsz+it]*rotT01[it]*trkErr( 7, terr_blk_offset) +
                         kGain[13*bsz+it]*trkErr( 8, terr_blk_offset);
     newErr[14*bsz+it] = kGain[12*bsz+it]*rotT00[it]*trkErr(10, terr_blk_offset) +
                         kGain[12*bsz+it]*rotT01[it]*trkErr(11, terr_blk_offset) +
                         kGain[13*bsz+it]*trkErr(12, terr_blk_offset);
     newErr[15*bsz+it] = kGain[15*bsz+it]*rotT00[it]*trkErr( 0, terr_blk_offset) +
                         kGain[15*bsz+it]*rotT01[it]*trkErr( 1, terr_blk_offset) +
                         kGain[16*bsz+it]*trkErr( 3, terr_blk_offset);
     newErr[16*bsz+it] = kGain[15*bsz+it]*rotT00[it]*trkErr( 1, terr_blk_offset) +
                         kGain[15*bsz+it]*rotT01[it]*trkErr( 2, terr_blk_offset) +
                         kGain[16*bsz+it]*trkErr( 4, terr_blk_offset);
     newErr[17*bsz+it] = kGain[15*bsz+it]*rotT00[it]*trkErr( 3, terr_blk_offset) +
                         kGain[15*bsz+it]*rotT01[it]*trkErr( 4, terr_blk_offset) +
                         kGain[16*bsz+it]*trkErr( 5, terr_blk_offset);
     newErr[18*bsz+it] = kGain[15*bsz+it]*rotT00[it]*trkErr( 6, terr_blk_offset) +
                         kGain[15*bsz+it]*rotT01[it]*trkErr( 7, terr_blk_offset) +
                         kGain[16*bsz+it]*trkErr( 8, terr_blk_offset);
     newErr[19*bsz+it] = kGain[15*bsz+it]*rotT00[it]*trkErr(10, terr_blk_offset) +
                         kGain[15*bsz+it]*rotT01[it]*trkErr(11, terr_blk_offset) +
                         kGain[16*bsz+it]*trkErr(12, terr_blk_offset);
     newErr[20*bsz+it] = kGain[15*bsz+it]*rotT00[it]*trkErr(15, terr_blk_offset) +
                         kGain[15*bsz+it]*rotT01[it]*trkErr(16, terr_blk_offset) +
                         kGain[16*bsz+it]*trkErr(17, terr_blk_offset);     

     newErr[ 0*bsize+it] = trkErr( 0, terr_blk_offset) - newErr[ 0*bsize+it];
     newErr[ 1*bsize+it] = trkErr( 1, terr_blk_offset) - newErr[ 1*bsize+it];
     newErr[ 2*bsize+it] = trkErr( 2, terr_blk_offset) - newErr[ 2*bsize+it];
     newErr[ 3*bsize+it] = trkErr( 3, terr_blk_offset) - newErr[ 3*bsize+it];
     newErr[ 4*bsize+it] = trkErr( 4, terr_blk_offset) - newErr[ 4*bsize+it];
     newErr[ 5*bsize+it] = trkErr( 5, terr_blk_offset) - newErr[ 5*bsize+it];
     newErr[ 6*bsize+it] = trkErr( 6, terr_blk_offset) - newErr[ 6*bsize+it];
     newErr[ 7*bsize+it] = trkErr( 7, terr_blk_offset) - newErr[ 7*bsize+it];
     newErr[ 8*bsize+it] = trkErr( 8, terr_blk_offset) - newErr[ 8*bsize+it];
     newErr[ 9*bsize+it] = trkErr( 9, terr_blk_offset) - newErr[ 9*bsize+it];
     newErr[10*bsize+it] = trkErr(10, terr_blk_offset) - newErr[10*bsize+it];
     newErr[11*bsize+it] = trkErr(11, terr_blk_offset) - newErr[11*bsize+it];
     newErr[12*bsize+it] = trkErr(12, terr_blk_offset) - newErr[12*bsize+it];
     newErr[13*bsize+it] = trkErr(13, terr_blk_offset) - newErr[13*bsize+it];
     newErr[14*bsize+it] = trkErr(14, terr_blk_offset) - newErr[14*bsize+it];
     newErr[15*bsize+it] = trkErr(15, terr_blk_offset) - newErr[15*bsize+it];
     newErr[16*bsize+it] = trkErr(16, terr_blk_offset) - newErr[16*bsize+it];
     newErr[17*bsize+it] = trkErr(17, terr_blk_offset) - newErr[17*bsize+it];
     newErr[18*bsize+it] = trkErr(18, terr_blk_offset) - newErr[18*bsize+it];
     newErr[19*bsize+it] = trkErr(19, terr_blk_offset) - newErr[19*bsize+it];
     newErr[20*bsize+it] = trkErr(20, terr_blk_offset) - newErr[20*bsize+it];
     //fake transfer?
     trkErr( 0, terr_blk_offset) = newErr[ 0*bsize+it];
     trkErr( 1, terr_blk_offset) = newErr[ 1*bsize+it];
     trkErr( 2, terr_blk_offset) = newErr[ 2*bsize+it];
     trkErr( 3, terr_blk_offset) = newErr[ 3*bsize+it];
     trkErr( 4, terr_blk_offset) = newErr[ 4*bsize+it];
     trkErr( 5, terr_blk_offset) = newErr[ 5*bsize+it];
     trkErr( 6, terr_blk_offset) = newErr[ 6*bsize+it];
     trkErr( 7, terr_blk_offset) = newErr[ 7*bsize+it];
     trkErr( 8, terr_blk_offset) = newErr[ 8*bsize+it];
     trkErr( 9, terr_blk_offset) = newErr[ 9*bsize+it];
     trkErr(10, terr_blk_offset) = newErr[10*bsize+it];
     trkErr(11, terr_blk_offset) = newErr[11*bsize+it];
     trkErr(12, terr_blk_offset) = newErr[12*bsize+it];
     trkErr(13, terr_blk_offset) = newErr[13*bsize+it];
     trkErr(14, terr_blk_offset) = newErr[14*bsize+it];
     trkErr(15, terr_blk_offset) = newErr[15*bsize+it];
     trkErr(16, terr_blk_offset) = newErr[16*bsize+it];
     trkErr(17, terr_blk_offset) = newErr[17*bsize+it];
     trkErr(18, terr_blk_offset) = newErr[18*bsize+it];
     trkErr(19, terr_blk_offset) = newErr[19*bsize+it];
     trkErr(20, terr_blk_offset) = newErr[20*bsize+it];
   }
   // 
                 
}
                  

auto sincos4 = [](const float x, float& sin, float& cos) {
   // Had this writen with explicit division by factorial.
   // The *whole* fitting test ran like 2.5% slower on MIC, sigh.
   const float x2 = x*x;
   cos  = 1.f - 0.5f*x2 + 0.04166667f*x2*x2;
   sin  = x - 0.16666667f*x*x2;
};

constexpr float kfact= 100/3.8f;
constexpr int Niter=5;

template <class MPTRKAccessors, class MPHITAccessors, int bsz = 1>
void propagateToR(MPTRKAccessors       obtracks,
                  const MPTRKAccessors btracks,
                  const MPHITAccessors bhits,
                  const int tid,
                  const int lay) {

  using MP6Faccessor    = typename MPTRKAccessors::MP6FAccessor;
  using MP1Iaccessor    = typename MPTRKAccessors::MP1IAccessor;
  using MP6x6SFaccessor = typename MPTRKAccessors::MP6x6SFAccessor;
  using MP3Faccessor    = typename MPHITAccessors::MP3FAccessor;
  
  const MP6Faccessor &inPar    = btracks.par;
  const MP1Iaccessor &inChg    = btracks.q  ;
  const MP6x6SFaccessor &inErr = btracks.cov;

  const MP3Faccessor &msP      = bhits.pos;

  MP6x6SFaccessor &outErr    = obtracks.cov;
  MP6Faccessor    &outPar    = obtracks.par;

  const float par_offset = inPar.GetThreadOffset(tid);
  
  MP6x6F_ errorProp;
  MP6x6F_ temp;
  
  for (int it = 0; it < bsz; ++it) {
    const float par_blk_offset  = par_offset+it;
    	
    //initialize erroProp to identity matrix
    //for (int i=0;i<6;++i) errorProp.data[bsize*PosInMtrx(i,i,6) + it] = 1.f; 
    errorProp[PosInMtrx(0,0,6, bsz) + it] = 1.0f;
    errorProp[PosInMtrx(1,1,6, bsz) + it] = 1.0f;
    errorProp[PosInMtrx(2,2,6, bsz) + it] = 1.0f;
    errorProp[PosInMtrx(3,3,6, bsz) + it] = 1.0f;
    errorProp[PosInMtrx(4,4,6, bsz) + it] = 1.0f;
    errorProp[PosInMtrx(5,5,6, bsz) + it] = 1.0f;
    //
    const float xin = inPar(iparX, par_blk_offset);
    const float yin = inPar(iparY, par_blk_offset);     
    const float zin = inPar(iparZ, par_blk_offset); 
    
    const float iptin   = inPar(iparIpt,   par_blk_offset);
    const float phiin   = inPar(iparPhi,   par_blk_offset);
    const float thetain = inPar(iparTheta, par_blk_offset); 
    //
    float r0 = hipo(xin, yin);
    const float k = inChg(0, tid, it, 0)*kfact;
    
    const float xmsP = msP(iparX, tid, it, lay);
    const float ymsP = msP(iparY, tid, it, lay);
    
    const float r = hipo(xmsP, ymsP);    
    
    outPar(iparX,par_blk_offset) = xin;
    outPar(iparY,par_blk_offset) = yin;
    outPar(iparZ,par_blk_offset) = zin;

    outPar(iparIpt,par_blk_offset)   = iptin;
    outPar(iparPhi,par_blk_offset)   = phiin;
    outPar(iparTheta,par_blk_offset) = thetain;
    
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
      const float xout = outPar(iparX, par_blk_offset);
      const float yout = outPar(iparY, par_blk_offset);     
  
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
      outPar(iparX,par_blk_offset) = xout + k*(pxin*sina - pyin*(1.f-cosa));
      outPar(iparY,par_blk_offset) = yout + k*(pyin*sina + pxin*(1.f-cosa));
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
    
    errorProp[PosInMtrx(0,0,6, bsz) + it] = 1.f+k*dadx*(cosPorT*cosa-sinPorT*sina)*pt;
    errorProp[PosInMtrx(0,1,6, bsz) + it] =     k*dady*(cosPorT*cosa-sinPorT*sina)*pt;
    errorProp[PosInMtrx(0,2,6, bsz) + it] = 0.f;
    errorProp[PosInMtrx(0,3,6, bsz) + it] = k*(cosPorT*(iptin*dadipt*cosa-sina)+sinPorT*((1.f-cosa)-iptin*dadipt*sina))*pt*pt;
    errorProp[PosInMtrx(0,4,6, bsz) + it] = k*(cosPorT*dadphi*cosa - sinPorT*dadphi*sina - sinPorT*sina + cosPorT*cosa - cosPorT)*pt;
    errorProp[PosInMtrx(0,5,6, bsz) + it] = 0.f;

    errorProp[PosInMtrx(1,0,6, bsz) + it] =     k*dadx*(sinPorT*cosa+cosPorT*sina)*pt;
    errorProp[PosInMtrx(1,1,6, bsz) + it] = 1.f+k*dady*(sinPorT*cosa+cosPorT*sina)*pt;
    errorProp[PosInMtrx(1,2,6, bsz) + it] = 0.f;
    errorProp[PosInMtrx(1,3,6, bsz) + it] = k*(sinPorT*(iptin*dadipt*cosa-sina)+cosPorT*(iptin*dadipt*sina-(1.f-cosa)))*pt*pt;
    errorProp[PosInMtrx(1,4,6, bsz) + it] = k*(sinPorT*dadphi*cosa + cosPorT*dadphi*sina + sinPorT*cosa + cosPorT*sina - sinPorT)*pt;
    errorProp[PosInMtrx(1,5,6, bsz) + it] = 0.f;

    //no trig approx here, theta can be large
    cosPorT=std::cos(thetain);
    sinPorT=std::sin(thetain);
    //redefine sinPorT as 1./sinPorT to reduce the number of temporaries
    sinPorT = 1.f/sinPorT;

    outPar(iparZ,par_blk_offset) = zin + k*alpha*cosPorT*pt*sinPorT;    

    errorProp[PosInMtrx(2,0,6, bsz) + it] = k*cosPorT*dadx*pt*sinPorT;
    errorProp[PosInMtrx(2,1,6, bsz) + it] = k*cosPorT*dady*pt*sinPorT;
    errorProp[PosInMtrx(2,2,6, bsz) + it] = 1.f;
    errorProp[PosInMtrx(2,3,6, bsz) + it] = k*cosPorT*(iptin*dadipt-alpha)*pt*pt*sinPorT;
    errorProp[PosInMtrx(2,4,6, bsz) + it] = k*dadphi*cosPorT*pt*sinPorT;
    errorProp[PosInMtrx(2,5,6, bsz) + it] =-k*alpha*pt*sinPorT*sinPorT;   
    //
    outPar(iparIpt,par_blk_offset) = iptin;
    
    errorProp[PosInMtrx(3,0,6, bsz) + it] = 0.f;
    errorProp[PosInMtrx(3,1,6, bsz) + it] = 0.f;
    errorProp[PosInMtrx(3,2,6, bsz) + it] = 0.f;
    errorProp[PosInMtrx(3,3,6, bsz) + it] = 1.f;
    errorProp[PosInMtrx(3,4,6, bsz) + it] = 0.f;
    errorProp[PosInMtrx(3,5,6, bsz) + it] = 0.f; 
    
    outPar(iparPhi,par_blk_offset) = phiin+alpha;
    
    errorProp[PosInMtrx(4,0,6, bsz) + it] = dadx;
    errorProp[PosInMtrx(4,1,6, bsz) + it] = dady;
    errorProp[PosInMtrx(4,2,6, bsz) + it] = 0.f;
    errorProp[PosInMtrx(4,3,6, bsz) + it] = dadipt;
    errorProp[PosInMtrx(4,4,6, bsz) + it] = 1.f+dadphi;
    errorProp[PosInMtrx(4,5,6, bsz) + it] = 0.f; 
    
    outPar(iparTheta,par_blk_offset) = thetain;        

    errorProp[PosInMtrx(5,0,6, bsz) + it] = 0.f;
    errorProp[PosInMtrx(5,1,6, bsz) + it] = 0.f;
    errorProp[PosInMtrx(5,2,6, bsz) + it] = 0.f;
    errorProp[PosInMtrx(5,3,6, bsz) + it] = 0.f;
    errorProp[PosInMtrx(5,4,6, bsz) + it] = 0.f;
    errorProp[PosInMtrx(5,5,6, bsz) + it] = 1.f;                                    
  }
  
  MultHelixProp<MP6x6SFaccessor, bsz>(errorProp, inErr, temp, tid);
  MultHelixPropTransp<MP6x6SFaccessor, bsz>(errorProp, temp, outErr, tid);  
  
  return;
}


int main (int argc, char* argv[]) {

   int itr;
   struct ATRK inputtrk = {
     {-12.806846618652344, -7.723824977874756, 38.13014221191406,0.23732035065189902, -2.613372802734375, 0.35594117641448975},
     {6.290299552347278e-07,4.1375109560704004e-08,7.526661534029699e-07,2.0973730840978533e-07,1.5431574240665213e-07,9.626245400795597e-08,-2.804026640189443e-06,
      6.219111130687595e-06,2.649119409845118e-07,0.00253512163402557,-2.419662877381737e-07,4.3124190760040646e-07,3.1068903991780678e-09,0.000923913115050627,
      0.00040678296006807003,-7.755406890332818e-07,1.68539375883925e-06,6.676875566525437e-08,0.0008420574605423793,7.356584799406111e-05,0.0002306247719158348},
     1
   };

   struct AHIT inputhit = {
     {-20.7824649810791, -12.24150276184082, 57.8067626953125},
     {2.545517190810642e-06,-2.6680759219743777e-06,2.8030024168401724e-06,0.00014160551654640585,0.00012282167153898627,11.385087966918945}
   };

   printf("track in pos: %f, %f, %f \n", inputtrk.par[0], inputtrk.par[1], inputtrk.par[2]);
   printf("track in cov: %.2e, %.2e, %.2e \n", inputtrk.cov[SymOffsets66[0]],
	                                       inputtrk.cov[SymOffsets66[(1*6+1)]],
	                                       inputtrk.cov[SymOffsets66[(2*6+2)]]);
   printf("hit in pos: %f %f %f \n", inputhit.pos[0], inputhit.pos[1], inputhit.pos[2]);

   printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
   printf("NITER=%d\n", NITER);

   long setup_start, setup_stop;
   struct timeval timecheck;

   constexpr auto order = FieldOrder::P2R_TRACKBLK_EVENT_LAYER_MATIDX_ORDER;
   //constexpr auto order = FieldOrder::P2R_MATIDX_LAYER_TRACKBLK_EVENT_ORDER;

   sycl::queue cq; //(sycl::gpu_selector{});
  
   FloatAllocator alloc_f32(cq);
   IntAllocator   alloc_int(cq); 

   using MPTRKAccessorTp = MPTRKAccessor<order>;
   using MPHITAccessorTp = MPHITAccessor<order>;

   MPHIT_* hit    = prepareHits(inputhit);
   MPTRK_* outtrk = (MPTRK_*) malloc(nevts*nb*sizeof(MPTRK_));

   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   auto trkNPtr = prepareTracksN<order>(inputtrk, cq);//1
   std::unique_ptr<MPTRKAccessorTp> trkNaccPtr(new MPTRKAccessorTp(*trkNPtr));
   auto &trkNacc = *trkNaccPtr;

   auto hitNPtr = prepareHitsN<order>(inputhit, cq);//2
   std::unique_ptr<MPHITAccessorTp> hitNaccPtr(new MPHITAccessorTp(*hitNPtr));
   auto &hitNacc = *hitNaccPtr;

   std::unique_ptr<MPTRK> outtrkNPtr(new MPTRK(ntrks, nevts, cq));//3
   std::unique_ptr<MPTRKAccessorTp> outtrkNaccPtr(new MPTRKAccessorTp(*outtrkNPtr));
   auto &outtrkNacc = *outtrkNaccPtr;

   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");

   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nevts*nb*sizeof(MPHIT));
  
   auto policy = oneapi::dpl::execution::make_device_policy(cq);
   //auto policy = oneapi::dpl::execution::dpcpp_default;  
   
   std::cout << "Begin warming up..." << std::endl;
   //
   std::vector<float, decltype(alloc_f32)> x_(10*nevts*nb, alloc_f32);
   std::vector<float, decltype(alloc_f32)> y_(10*nevts*nb, alloc_f32);

   auto warm_start = std::chrono::high_resolution_clock::now();

   std::fill(policy, x_.begin(), x_.end(), 1.0f);
   std::fill(policy, y_.begin(), y_.end(), 2.0f);

   double sum = 0.0;
   for(int i = 0; i < 1000;i++)
   {
     float lsum = oneapi::dpl::transform_reduce(policy,
                                x_.begin(),
                                x_.end(),
                                y_.begin(),
                                0.0f, std::plus<float>(),
                                [=](const auto &xi, const auto &yi) { return xi*yi;} );
     sum += lsum;
   }
   auto warm_stop = std::chrono::high_resolution_clock::now();
   auto warm_diff = warm_stop - warm_start;
   auto warm_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(warm_diff).count()) / 1e6;

   std::cout << "..done. Warmup time: " << warm_time << " secs. " << std::endl;

   auto wall_start = std::chrono::high_resolution_clock::now();

   for(itr=0; itr<NITER; itr++) {

     const int outer_loop_range = nevts*nb;

     oneapi::dpl::for_each(policy,
                           counting_iterator(0),
                           counting_iterator(outer_loop_range),

                   [=] (const auto i) {
                     for(int layer=0; layer<nlayer; ++layer) {
                       propagateToR<MPTRKAccessorTp, MPHITAccessorTp, bsize>(outtrkNacc, trkNacc, hitNacc, i, layer);
                       KalmanUpdate<MPTRKAccessorTp, MPHITAccessorTp, bsize>(outtrkNacc, hitNacc, i, layer);
                     }
                   });
   } //end of itr loop

   auto wall_stop = std::chrono::high_resolution_clock::now();

   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, (setup_stop-setup_start)*0.001, -1);


   convertTracks<order>(outtrk, outtrkNPtr.get());
   convertHits<order>(hit, hitNPtr.get());

   float avgx = 0, avgy = 0, avgz = 0, avgr = 0;
   float avgpt = 0, avgphi = 0, avgtheta = 0;
   float avgdx = 0, avgdy = 0, avgdz = 0, avgdr = 0;
   for (int ie=0;ie<nevts;++ie) {
     for (int it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
       float r_ = sqrtf(x_*x_ + y_*y_);
       float pt_ = 1./ipt(outtrk,ie,it);
       float phi_ = phi(outtrk,ie,it);
       float theta_ = theta(outtrk,ie,it);
       avgpt += pt_;
       avgphi += phi_;
       avgtheta += theta_;
       avgx += x_;
       avgy += y_;
       avgz += z_;
       avgr += r_;
       float hx_ = x(hit,ie,it);
       float hy_ = y(hit,ie,it);
       float hz_ = z(hit,ie,it);
       float hr_ = sqrtf(hx_*hx_ + hy_*hy_);
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
   for (int ie=0;ie<nevts;++ie) {
     for (int it=0;it<ntrks;++it) {
       float x_ = x(outtrk,ie,it);
       float y_ = y(outtrk,ie,it);
       float z_ = z(outtrk,ie,it);
       float r_ = sqrtf(x_*x_ + y_*y_);
       stdx += (x_-avgx)*(x_-avgx);
       stdy += (y_-avgy)*(y_-avgy);
       stdz += (z_-avgz)*(z_-avgz);
       stdr += (r_-avgr)*(r_-avgr);
       float hx_ = x(hit,ie,it);
       float hy_ = y(hit,ie,it);
       float hz_ = z(hit,ie,it);
       float hr_ = sqrtf(hx_*hx_ + hy_*hy_);
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


   delete [] hit;
   delete [] outtrk;


   return 0;
}
