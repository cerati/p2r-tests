/*
see README.txt for instructions
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <tbb/tbb.h>
#include <iostream>
#include <chrono>
#include <iomanip>

#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>
#include <execution>

#ifndef bsize
#define bsize 128
#endif
#ifndef ntrks
#define ntrks 9600
#endif

#define nb    ntrks/bsize

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

#ifndef nthreads
#define nthreads 1
#endif

#if defined(__NVCOMPILER_CUDA__)

#include <thrust/iterator/counting_iterator.h>
using namespace thrust;

#else

#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

#include <tbb/tbb.h>
using namespace tbb;
constexpr int alloc_align  = (2*1024*1024);

#endif 

template<typename Tp>
struct AlignedAllocator {
   public:

   typedef Tp value_type;

   AlignedAllocator () {};
   AlignedAllocator(const AlignedAllocator&) { }

   template<typename Tp1> constexpr AlignedAllocator(const AlignedAllocator<Tp1>&) { }

   ~AlignedAllocator() { }
     
   Tp* address(Tp& x) const { return &x; }

   std::size_t  max_size() const throw() { return size_t(-1) / sizeof(Tp); }

   [[nodiscard]] Tp* allocate(std::size_t n){

     Tp* ptr = nullptr;
#ifdef __NVCOMPILER_CUDA__
     auto err = cudaMallocManaged((void **)&ptr,n*sizeof(Tp));

     if( err != cudaSuccess ) {
       ptr = (Tp *) NULL;
       std::cerr << " cudaMallocManaged failed for " << n*sizeof(Tp) << " bytes " <<cudaGetErrorString(err)<< std::endl;
       assert(0);
     }
#else
     //ptr = (Tp*)aligned_malloc(alloc_align, n*sizeof(Tp));
#if defined(__INTEL_COMPILER)
     ptr = (Tp*)malloc(bytes);
#else
     ptr = (Tp*)_mm_malloc(n*sizeof(Tp),alloc_align);
#endif
     if(!ptr) throw std::bad_alloc();
#endif

     return ptr;
   }

   void deallocate( Tp* p, std::size_t n) noexcept {
#ifdef __NVCOMPILER_CUDA__
     cudaFree((void *)p);
#else

#if defined(__INTEL_COMPILER)
     free((void*)p);
#else
     _mm_free((void *)p);
#endif

#endif
   }
};

auto PosInMtrx = [](const size_t &&i, const size_t &&j, const size_t &&D, const size_t block_size = 1) constexpr {return block_size*(i*D+j);};

enum class FieldOrder{P2R_TRACKBLK_EVENT_LAYER_MATIDX_ORDER,
                      P2R_TRACKBLK_EVENT_MATIDX_LAYER_ORDER,
                      P2R_MATIDX_LAYER_TRACKBLK_EVENT_ORDER};

using IntAllocator   = AlignedAllocator<int>;
using FloatAllocator = AlignedAllocator<float>;

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

   static constexpr int N    = n;
   static constexpr int BS   = bSize;

   const int nTrks;//note that bSize is a tuning parameter!
   const int nEvts;
   const int nLayers;

   std::vector<T, Allocator> data;

   MPNX() : nTrks(bSize), nEvts(0), nLayers(0), data(n*bSize){}

   MPNX(const int ntrks_, const int nevts_, const int nlayers_ = 1) :
      nTrks(ntrks_),
      nEvts(nevts_),
      nLayers(nlayers_),
      data(n*nTrks*nEvts*nLayers){
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

   T* data_; //accessor field only for the data access, not allocated here

   MPNXAccessor() : nTrkB(0), nEvts(0), nLayers(0), NevtsNtbBsz(0), stride(0), data_(nullptr){}
   MPNXAccessor(const MPNTp &v) :
        nTrkB(v.nTrks / bsz),
        nEvts(v.nEvts),
        nLayers(v.nLayers),
        NevtsNtbBsz(nEvts*nTrkB*bsz),
        stride(Order == FieldOrder::P2R_TRACKBLK_EVENT_LAYER_MATIDX_ORDER ? bsz*nTrkB*nEvts*nLayers  :
              (Order == FieldOrder::P2R_TRACKBLK_EVENT_MATIDX_LAYER_ORDER ? bsz*nTrkB*nEvts*n : n*bsz*nLayers)),
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

   T& operator()(const int thrd_idx, const int stride, const int blk_offset) const { return data_[thrd_idx*stride + blk_offset];}//

   int GetThreadOffset(const int thrd_idx, const int layer_idx = 0) const {
     if      constexpr (Order == FieldOrder::P2R_TRACKBLK_EVENT_LAYER_MATIDX_ORDER)
       return (layer_idx*NevtsNtbBsz + thrd_idx*bsz);//using defualt order batch id (the fastest) > track id > event id > layer id (the slowest)
     else if constexpr (Order == FieldOrder::P2R_TRACKBLK_EVENT_MATIDX_LAYER_ORDER)
       return (layer_idx*stride + thrd_idx*bsz);
     else //(Order == FieldOrder::P2Z_MATIDX_LAYER_TRACKBLK_EVENT_ORDER)
       return (thrd_idx*stride+layer_idx*n*bsz);
   }

   int GetThreadStride() const {
     if      constexpr (Order == FieldOrder::P2R_TRACKBLK_EVENT_LAYER_MATIDX_ORDER)
       return stride;//using defualt order batch id (the fastest) > track id > event id > layer id (the slowest)
     else if constexpr (Order == FieldOrder::P2R_TRACKBLK_EVENT_MATIDX_LAYER_ORDER)
       return NevtsNtbBsz;
     else //(Order == FieldOrder::P2Z_MATIDX_LAYER_TRACKBLK_EVENT_ORDER)
       return bsz;
   }

};

struct MPTRK {
  MP6F    par;
  MP6x6SF cov;
  MP1I    q;

  MPTRK() : par(), cov(), q() {}
  MPTRK(const int ntrks_, const int nevts_) : par(ntrks_, nevts_), cov(ntrks_, nevts_), q(ntrks_, nevts_) {}

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
std::shared_ptr<MPTRK> prepareTracksN(struct ATRK inputtrk) {

  auto result = std::make_shared<MPTRK>(ntrks, nevts);
  //create an accessor field:
  std::unique_ptr<MPTRKAccessor<order>> rA(new MPTRKAccessor<order>(*result));

  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
        //const int l = it+ib*bsize+ie*nb*bsize;
        const int tid = ib+ie*nb;
    	  //par
    	  for (size_t ip=0;ip<6;++ip) {
          rA->par(ip, tid, it, 0) = (1+smear*randn(0,1))*inputtrk.par[ip];
    	  }
    	  //cov
    	  for (size_t ip=0;ip<21;++ip) {
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
  MP3F    pos;
  MP3x3SF cov;

  MPHIT() : pos(), cov(){}
  MPHIT(const int ntrks_, const int nevts_, const int nlayers_) : pos(ntrks_, nevts_, nlayers_), cov(ntrks_, nevts_, nlayers_) {}

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
std::shared_ptr<MPHIT> prepareHitsN(struct AHIT inputhit) {
  auto result = std::make_shared<MPHIT>(ntrks, nevts, nlayer);
  //create an accessor field:
  std::unique_ptr<MPHITAccessor<order>> rA(new MPHITAccessor<order>(*result));

  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t lay=0;lay<nlayer;++lay) {
    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        for (size_t it=0;it<bsize;++it) {
          //const int l = it + ib*bsize + ie*nb*bsize + lay*nb*bsize*nevts;
          const int tid = ib + ie*nb;
        	//pos
        	for (size_t ip=0;ip<3;++ip) {
            rA->pos(ip, tid, it, lay) = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
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
using MP3F_    = MPNX_<float, 3 , bsize>;
using MP6F_    = MPNX_<float, 6 , bsize>;
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

MPTRK_* bTk(MPTRK_* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

const MPTRK_* bTk(const MPTRK_* tracks, size_t ev, size_t ib) {
  return &(tracks[ib + nb*ev]);
}

float q(const MP1I_* bq, size_t it){
  return (*bq).data[it];
}
//
float par(const MP6F_* bpars, size_t it, size_t ipar){
  return (*bpars).data[it + ipar*bsize];
}
float x    (const MP6F_* bpars, size_t it){ return par(bpars, it, 0); }
float y    (const MP6F_* bpars, size_t it){ return par(bpars, it, 1); }
float z    (const MP6F_* bpars, size_t it){ return par(bpars, it, 2); }
float ipt  (const MP6F_* bpars, size_t it){ return par(bpars, it, 3); }
float phi  (const MP6F_* bpars, size_t it){ return par(bpars, it, 4); }
float theta(const MP6F_* bpars, size_t it){ return par(bpars, it, 5); }
//
float par(const MPTRK_* btracks, size_t it, size_t ipar){
  return par(&(*btracks).par,it,ipar);
}
float x    (const MPTRK_* btracks, size_t it){ return par(btracks, it, 0); }
float y    (const MPTRK_* btracks, size_t it){ return par(btracks, it, 1); }
float z    (const MPTRK_* btracks, size_t it){ return par(btracks, it, 2); }
float ipt  (const MPTRK_* btracks, size_t it){ return par(btracks, it, 3); }
float phi  (const MPTRK_* btracks, size_t it){ return par(btracks, it, 4); }
float theta(const MPTRK_* btracks, size_t it){ return par(btracks, it, 5); }
//
float par(const MPTRK_* tracks, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPTRK_* btracks = bTk(tracks, ev, ib);
  size_t it = tk % bsize;
  return par(btracks, it, ipar);
}
float x    (const MPTRK_* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 0); }
float y    (const MPTRK_* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 1); }
float z    (const MPTRK_* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 2); }
float ipt  (const MPTRK_* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 3); }
float phi  (const MPTRK_* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 4); }
float theta(const MPTRK_* tracks, size_t ev, size_t tk){ return par(tracks, ev, tk, 5); }
//
void setpar(MP6F* bpars, size_t it, size_t ipar, float val){
  (*bpars).data[it + ipar*bsize] = val;
}
void setx    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 0, val); }
void sety    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 1, val); }
void setz    (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 2, val); }
void setipt  (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 3, val); }
void setphi  (MP6F* bpars, size_t it, float val){ setpar(bpars, it, 4, val); }
void settheta(MP6F* bpars, size_t it, float val){ setpar(bpars, it, 5, val); }
//
void setpar(MPTRK_* btracks, size_t it, size_t ipar, float val){
  setpar(&(*btracks).par,it,ipar,val);
}
void setx    (MPTRK_* btracks, size_t it, float val){ setpar(btracks, it, 0, val); }
void sety    (MPTRK_* btracks, size_t it, float val){ setpar(btracks, it, 1, val); }
void setz    (MPTRK_* btracks, size_t it, float val){ setpar(btracks, it, 2, val); }
void setipt  (MPTRK_* btracks, size_t it, float val){ setpar(btracks, it, 3, val); }
void setphi  (MPTRK_* btracks, size_t it, float val){ setpar(btracks, it, 4, val); }
void settheta(MPTRK_* btracks, size_t it, float val){ setpar(btracks, it, 5, val); }

const MPHIT_* bHit(const MPHIT_* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
const MPHIT_* bHit(const MPHIT_* hits, size_t ev, size_t ib,size_t lay) {
return &(hits[lay + (ib*nlayer) +(ev*nlayer*nb)]);
}
//
float pos(const MP3F_* hpos, size_t it, size_t ipar){
  return (*hpos).data[it + ipar*bsize];
}
float x(const MP3F_* hpos, size_t it)    { return pos(hpos, it, 0); }
float y(const MP3F_* hpos, size_t it)    { return pos(hpos, it, 1); }
float z(const MP3F_* hpos, size_t it)    { return pos(hpos, it, 2); }
//
float pos(const MPHIT_* hits, size_t it, size_t ipar){
  return pos(&(*hits).pos,it,ipar);
}
float x(const MPHIT_* hits, size_t it)    { return pos(hits, it, 0); }
float y(const MPHIT_* hits, size_t it)    { return pos(hits, it, 1); }
float z(const MPHIT_* hits, size_t it)    { return pos(hits, it, 2); }
//
float pos(const MPHIT_* hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPHIT_* bhits = bHit(hits, ev, ib);
  size_t it = tk % bsize;
  return pos(bhits,it,ipar);
}
float x(const MPHIT_* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 0); }
float y(const MPHIT_* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 1); }
float z(const MPHIT_* hits, size_t ev, size_t tk)    { return pos(hits, ev, tk, 2); }
/*
MPTRK* prepareTracks(ATRK inputtrk) {
  MPTRK* result = (MPTRK*) malloc(nevts*nb*sizeof(MPTRK)); //fixme, align?
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
	      //par
	      for (size_t ip=0;ip<6;++ip) {
	        result[ib + nb*ie].par.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.par[ip];
	      }
	      //cov
	      for (size_t ip=0;ip<21;++ip) {
	        result[ib + nb*ie].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputtrk.cov[ip];
	      }
	      //q
	      result[ib + nb*ie].q.data[it] = inputtrk.q-2*ceil(-0.5 + (float)rand() / RAND_MAX);//fixme check
      }
    }
  }
  return result;
}

MPHIT* prepareHits(AHIT inputhit) {
  MPHIT* result = (MPHIT*) malloc(nlayer*nevts*nb*sizeof(MPHIT));  //fixme, align?
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t lay=0;lay<nlayer;++lay) {
    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        for (size_t it=0;it<bsize;++it) {
        	//pos
        	for (size_t ip=0;ip<3;++ip) {
        	  result[lay+nlayer*(ib + nb*ie)].pos.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
        	  result[lay+nlayer*(ib + nb*ie)].cov.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.cov[ip];
        	}
        }
      }
    }
  }
  return result;
}
*/

template<FieldOrder order>
void convertTracks(MPTRK_* out,  const MPTRK* inp) {
  //create an accessor field:
  std::unique_ptr<MPTRKAccessor<order>> inpA(new MPTRKAccessor<order>(*inp));
  // store in element order for bunches of bsize matrices (a la matriplex)
  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
        //const int l = it+ib*bsize+ie*nb*bsize;
        const int tid = ib+ie*nb;
    	  //par
    	  for (size_t ip=0;ip<6;++ip) {
    	    out[tid].par.data[it + ip*bsize] = inpA->par(ip, tid, it, 0);
    	  }
    	  //cov
    	  for (size_t ip=0;ip<21;++ip) {
    	    out[tid].cov.data[it + ip*bsize] = inpA->cov(ip, tid, it, 0);
    	  }
    	  //q
    	  out[tid].q.data[it] = inpA->q(0, tid, it);//fixme check
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
  for (size_t lay=0;lay<nlayer;++lay) {
    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        for (size_t it=0;it<bsize;++it) {
          //const int l = it + ib*bsize + ie*nb*bsize + lay*nb*bsize*nevts;
          const int tid = ib + ie*nb;
        	//pos
        	for (size_t ip=0;ip<3;++ip) {
            out[lay+nlayer*tid].pos.data[it + ip*bsize] = inpA->pos(ip, tid, it, lay);
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
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
  for (size_t lay=0;lay<nlayer;++lay) {
    for (size_t ie=0;ie<nevts;++ie) {
      for (size_t ib=0;ib<nb;++ib) {
        for (size_t it=0;it<bsize;++it) {
        	//pos
        	for (size_t ip=0;ip<3;++ip) {
        	  result[lay+nlayer*(ib + nb*ie)].pos.data[it + ip*bsize] = (1+smear*randn(0,1))*inputhit.pos[ip];
        	}
        	//cov
        	for (size_t ip=0;ip<6;++ip) {
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

template<typename MP6x6SFAccessor_, size_t bsz = 1>
inline void MultHelixProp(const MP6x6F_ &a, const MP6x6SFAccessor_ &b, MP6x6F_ &c, const int tid) {

  const auto stride = b.GetThreadStride();
  const auto offset = b.GetThreadOffset(tid);

  for (int it = 0;it < bsz; it++) {
    const auto boffset = offset+it;
    c[ 0*bsz+it] = a[ 0*bsz+it]*b( 0, stride, boffset) + a[ 1*bsz+it]*b( 1, stride, boffset) + a[ 3*bsz+it]*b( 6, stride, boffset) + a[ 4*bsz+it]*b(10, stride, boffset);
    c[ 1*bsz+it] = a[ 0*bsz+it]*b( 1, stride, boffset) + a[ 1*bsz+it]*b( 2, stride, boffset) + a[ 3*bsz+it]*b( 7, stride, boffset) + a[ 4*bsz+it]*b(11, stride, boffset);
    c[ 2*bsz+it] = a[ 0*bsz+it]*b( 3, stride, boffset) + a[ 1*bsz+it]*b( 4, stride, boffset) + a[ 3*bsz+it]*b( 8, stride, boffset) + a[ 4*bsz+it]*b(12, stride, boffset);
    c[ 3*bsz+it] = a[ 0*bsz+it]*b( 6, stride, boffset) + a[ 1*bsz+it]*b( 7, stride, boffset) + a[ 3*bsz+it]*b( 9, stride, boffset) + a[ 4*bsz+it]*b(13, stride, boffset);
    c[ 4*bsz+it] = a[ 0*bsz+it]*b(10, stride, boffset) + a[ 1*bsz+it]*b(11, stride, boffset) + a[ 3*bsz+it]*b(13, stride, boffset) + a[ 4*bsz+it]*b(14, stride, boffset);
    c[ 5*bsz+it] = a[ 0*bsz+it]*b(15, stride, boffset) + a[ 1*bsz+it]*b(16, stride, boffset) + a[ 3*bsz+it]*b(18, stride, boffset) + a[ 4*bsz+it]*b(19, stride, boffset);
    c[ 6*bsz+it] = a[ 6*bsz+it]*b( 0, stride, boffset) + a[ 7*bsz+it]*b( 1, stride, boffset) + a[ 9*bsz+it]*b( 6, stride, boffset) + a[10*bsz+it]*b(10, stride, boffset);
    c[ 7*bsz+it] = a[ 6*bsz+it]*b( 1, stride, boffset) + a[ 7*bsz+it]*b( 2, stride, boffset) + a[ 9*bsz+it]*b( 7, stride, boffset) + a[10*bsz+it]*b(11, stride, boffset);
    c[ 8*bsz+it] = a[ 6*bsz+it]*b( 3, stride, boffset) + a[ 7*bsz+it]*b( 4, stride, boffset) + a[ 9*bsz+it]*b( 8, stride, boffset) + a[10*bsz+it]*b(12, stride, boffset);
    c[ 9*bsz+it] = a[ 6*bsz+it]*b( 6, stride, boffset) + a[ 7*bsz+it]*b( 7, stride, boffset) + a[ 9*bsz+it]*b( 9, stride, boffset) + a[10*bsz+it]*b(13, stride, boffset);
    c[10*bsz+it] = a[ 6*bsz+it]*b(10, stride, boffset) + a[ 7*bsz+it]*b(11, stride, boffset) + a[ 9*bsz+it]*b(13, stride, boffset) + a[10*bsz+it]*b(14, stride, boffset);
    c[11*bsz+it] = a[ 6*bsz+it]*b(15, stride, boffset) + a[ 7*bsz+it]*b(16, stride, boffset) + a[ 9*bsz+it]*b(18, stride, boffset) + a[10*bsz+it]*b(19, stride, boffset);
    
    c[12*bsz+it] = a[12*bsz+it]*b( 0, stride, boffset) + a[13*bsz+it]*b( 1, stride, boffset) + b( 3, stride, boffset) + a[15*bsz+it]*b( 6, stride, boffset) + a[16*bsz+it]*b(10, stride, boffset) + a[17*bsz+it]*b(15, stride, boffset);
    c[13*bsz+it] = a[12*bsz+it]*b( 1, stride, boffset) + a[13*bsz+it]*b( 2, stride, boffset) + b( 4, stride, boffset) + a[15*bsz+it]*b( 7, stride, boffset) + a[16*bsz+it]*b(11, stride, boffset) + a[17*bsz+it]*b(16, stride, boffset);
    c[14*bsz+it] = a[12*bsz+it]*b( 3, stride, boffset) + a[13*bsz+it]*b( 4, stride, boffset) + b( 5, stride, boffset) + a[15*bsz+it]*b( 8, stride, boffset) + a[16*bsz+it]*b(12, stride, boffset) + a[17*bsz+it]*b(17, stride, boffset);
    c[15*bsz+it] = a[12*bsz+it]*b( 6, stride, boffset) + a[13*bsz+it]*b( 7, stride, boffset) + b( 8, stride, boffset) + a[15*bsz+it]*b( 9, stride, boffset) + a[16*bsz+it]*b(13, stride, boffset) + a[17*bsz+it]*b(18, stride, boffset);
    c[16*bsz+it] = a[12*bsz+it]*b(10, stride, boffset) + a[13*bsz+it]*b(11, stride, boffset) + b(12, stride, boffset) + a[15*bsz+it]*b(13, stride, boffset) + a[16*bsz+it]*b(14, stride, boffset) + a[17*bsz+it]*b(19, stride, boffset);
    c[17*bsz+it] = a[12*bsz+it]*b(15, stride, boffset) + a[13*bsz+it]*b(16, stride, boffset) + b(17, stride, boffset) + a[15*bsz+it]*b(18, stride, boffset) + a[16*bsz+it]*b(19, stride, boffset) + a[17*bsz+it]*b(20, stride, boffset);
    
    c[18*bsz+it] = a[18*bsz+it]*b( 0, stride, boffset) + a[19*bsz+it]*b( 1, stride, boffset) + a[21*bsz+it]*b( 6, stride, boffset) + a[22*bsz+it]*b(10, stride, boffset);
    c[19*bsz+it] = a[18*bsz+it]*b( 1, stride, boffset) + a[19*bsz+it]*b( 2, stride, boffset) + a[21*bsz+it]*b( 7, stride, boffset) + a[22*bsz+it]*b(11, stride, boffset);
    c[20*bsz+it] = a[18*bsz+it]*b( 3, stride, boffset) + a[19*bsz+it]*b( 4, stride, boffset) + a[21*bsz+it]*b( 8, stride, boffset) + a[22*bsz+it]*b(12, stride, boffset);
    c[21*bsz+it] = a[18*bsz+it]*b( 6, stride, boffset) + a[19*bsz+it]*b( 7, stride, boffset) + a[21*bsz+it]*b( 9, stride, boffset) + a[22*bsz+it]*b(13, stride, boffset);
    c[22*bsz+it] = a[18*bsz+it]*b(10, stride, boffset) + a[19*bsz+it]*b(11, stride, boffset) + a[21*bsz+it]*b(13, stride, boffset) + a[22*bsz+it]*b(14, stride, boffset);
    c[23*bsz+it] = a[18*bsz+it]*b(15, stride, boffset) + a[19*bsz+it]*b(16, stride, boffset) + a[21*bsz+it]*b(18, stride, boffset) + a[22*bsz+it]*b(19, stride, boffset);
    c[24*bsz+it] = a[24*bsz+it]*b( 0, stride, boffset) + a[25*bsz+it]*b( 1, stride, boffset) + a[27*bsz+it]*b( 6, stride, boffset) + a[28*bsz+it]*b(10, stride, boffset);
    c[25*bsz+it] = a[24*bsz+it]*b( 1, stride, boffset) + a[25*bsz+it]*b( 2, stride, boffset) + a[27*bsz+it]*b( 7, stride, boffset) + a[28*bsz+it]*b(11, stride, boffset);
    c[26*bsz+it] = a[24*bsz+it]*b( 3, stride, boffset) + a[25*bsz+it]*b( 4, stride, boffset) + a[27*bsz+it]*b( 8, stride, boffset) + a[28*bsz+it]*b(12, stride, boffset);
    c[27*bsz+it] = a[24*bsz+it]*b( 6, stride, boffset) + a[25*bsz+it]*b( 7, stride, boffset) + a[27*bsz+it]*b( 9, stride, boffset) + a[28*bsz+it]*b(13, stride, boffset);
    c[28*bsz+it] = a[24*bsz+it]*b(10, stride, boffset) + a[25*bsz+it]*b(11, stride, boffset) + a[27*bsz+it]*b(13, stride, boffset) + a[28*bsz+it]*b(14, stride, boffset);
    c[29*bsz+it] = a[24*bsz+it]*b(15, stride, boffset) + a[25*bsz+it]*b(16, stride, boffset) + a[27*bsz+it]*b(18, stride, boffset) + a[28*bsz+it]*b(19, stride, boffset);
    c[30*bsz+it] = b(15, stride, boffset);
    c[31*bsz+it] = b(16, stride, boffset);
    c[32*bsz+it] = b(17, stride, boffset);
    c[33*bsz+it] = b(18, stride, boffset);
    c[34*bsz+it] = b(19, stride, boffset);
    c[35*bsz+it] = b(20, stride, boffset);    
  }
  return;
}

template<typename MP6x6SFAccessor_, size_t bsz = 1>
inline void MultHelixPropTransp(const MP6x6F_ &a, const MP6x6F_ &b, MP6x6SFAccessor_ &c, const int tid) {

  const auto stride = c.GetThreadStride();
  const auto offset = c.GetThreadOffset(tid);
  
  for (int it = 0;it < bsz; it++) {
    const auto boffset = offset+it;
    
    c( 0, stride, boffset) = b[ 0*bsz+it]*a[ 0*bsz+it] + b[ 1*bsz+it]*a[ 1*bsz+it] + b[ 3*bsz+it]*a[ 3*bsz+it] + b[ 4*bsz+it]*a[ 4*bsz+it];
    c( 1, stride, boffset) = b[ 6*bsz+it]*a[ 0*bsz+it] + b[ 7*bsz+it]*a[ 1*bsz+it] + b[ 9*bsz+it]*a[ 3*bsz+it] + b[10*bsz+it]*a[ 4*bsz+it];
    c( 2, stride, boffset) = b[ 6*bsz+it]*a[ 6*bsz+it] + b[ 7*bsz+it]*a[ 7*bsz+it] + b[ 9*bsz+it]*a[ 9*bsz+it] + b[10*bsz+it]*a[10*bsz+it];
    c( 3, stride, boffset) = b[12*bsz+it]*a[ 0*bsz+it] + b[13*bsz+it]*a[ 1*bsz+it] + b[15*bsz+it]*a[ 3*bsz+it] + b[16*bsz+it]*a[ 4*bsz+it];
    c( 4, stride, boffset) = b[12*bsz+it]*a[ 6*bsz+it] + b[13*bsz+it]*a[ 7*bsz+it] + b[15*bsz+it]*a[ 9*bsz+it] + b[16*bsz+it]*a[10*bsz+it];
    c( 5, stride, boffset) = b[12*bsz+it]*a[12*bsz+it] + b[13*bsz+it]*a[13*bsz+it] + b[14*bsz+it] + b[15*bsz+it]*a[15*bsz+it] + b[16*bsz+it]*a[16*bsz+it] + b[17*bsz+it]*a[17*bsz+it];
    c( 6, stride, boffset) = b[18*bsz+it]*a[ 0*bsz+it] + b[19*bsz+it]*a[ 1*bsz+it] + b[21*bsz+it]*a[ 3*bsz+it] + b[22*bsz+it]*a[ 4*bsz+it];
    c( 7, stride, boffset) = b[18*bsz+it]*a[ 6*bsz+it] + b[19*bsz+it]*a[ 7*bsz+it] + b[21*bsz+it]*a[ 9*bsz+it] + b[22*bsz+it]*a[10*bsz+it];
    c( 8, stride, boffset) = b[18*bsz+it]*a[12*bsz+it] + b[19*bsz+it]*a[13*bsz+it] + b[20*bsz+it] + b[21*bsz+it]*a[15*bsz+it] + b[22*bsz+it]*a[16*bsz+it] + b[23*bsz+it]*a[17*bsz+it];
    c( 9, stride, boffset) = b[18*bsz+it]*a[18*bsz+it] + b[19*bsz+it]*a[19*bsz+it] + b[21*bsz+it]*a[21*bsz+it] + b[22*bsz+it]*a[22*bsz+it];
    c(10, stride, boffset) = b[24*bsz+it]*a[ 0*bsz+it] + b[25*bsz+it]*a[ 1*bsz+it] + b[27*bsz+it]*a[ 3*bsz+it] + b[28*bsz+it]*a[ 4*bsz+it];
    c(11, stride, boffset) = b[24*bsz+it]*a[ 6*bsz+it] + b[25*bsz+it]*a[ 7*bsz+it] + b[27*bsz+it]*a[ 9*bsz+it] + b[28*bsz+it]*a[10*bsz+it];
    c(12, stride, boffset) = b[24*bsz+it]*a[12*bsz+it] + b[25*bsz+it]*a[13*bsz+it] + b[26*bsz+it] + b[27*bsz+it]*a[15*bsz+it] + b[28*bsz+it]*a[16*bsz+it] + b[29*bsz+it]*a[17*bsz+it];
    c(13, stride, boffset) = b[24*bsz+it]*a[18*bsz+it] + b[25*bsz+it]*a[19*bsz+it] + b[27*bsz+it]*a[21*bsz+it] + b[28*bsz+it]*a[22*bsz+it];
    c(14, stride, boffset) = b[24*bsz+it]*a[24*bsz+it] + b[25*bsz+it]*a[25*bsz+it] + b[27*bsz+it]*a[27*bsz+it] + b[28*bsz+it]*a[28*bsz+it];
    c(15, stride, boffset) = b[30*bsz+it]*a[ 0*bsz+it] + b[31*bsz+it]*a[ 1*bsz+it] + b[33*bsz+it]*a[ 3*bsz+it] + b[34*bsz+it]*a[ 4*bsz+it];
    c(16, stride, boffset) = b[30*bsz+it]*a[ 6*bsz+it] + b[31*bsz+it]*a[ 7*bsz+it] + b[33*bsz+it]*a[ 9*bsz+it] + b[34*bsz+it]*a[10*bsz+it];
    c(17, stride, boffset) = b[30*bsz+it]*a[12*bsz+it] + b[31*bsz+it]*a[13*bsz+it] + b[32*bsz+it] + b[33*bsz+it]*a[15*bsz+it] + b[34*bsz+it]*a[16*bsz+it] + b[35*bsz+it]*a[17*bsz+it];
    c(18, stride, boffset) = b[30*bsz+it]*a[18*bsz+it] + b[31*bsz+it]*a[19*bsz+it] + b[33*bsz+it]*a[21*bsz+it] + b[34*bsz+it]*a[22*bsz+it];
    c(19, stride, boffset) = b[30*bsz+it]*a[24*bsz+it] + b[31*bsz+it]*a[25*bsz+it] + b[33*bsz+it]*a[27*bsz+it] + b[34*bsz+it]*a[28*bsz+it];
    c(20, stride, boffset) = b[35*bsz+it];
  }
  return;  
}

void KalmanGainInv(const MP6x6SF* A, const MP3x3SF* B, MP3x3* C) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
#pragma omp simd
  for (int n = 0; n < N; ++n)
  {
    double det =
      ((a[0*N+n]+b[0*N+n])*(((a[ 6*N+n]+b[ 3*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[7*N+n]+b[4*N+n])))) -
      ((a[1*N+n]+b[1*N+n])*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[2*N+n]+b[2*N+n])))) +
      ((a[2*N+n]+b[2*N+n])*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[6*N+n]+b[3*N+n]))));
    double invdet = 1.0/det;

    c[ 0*N+n] =  invdet*(((a[ 6*N+n]+b[ 3*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[7*N+n]+b[4*N+n])));
    c[ 1*N+n] =  -1*invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[7*N+n]+b[4*N+n])));
    c[ 2*N+n] =  invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[7*N+n]+b[4*N+n])));
    c[ 3*N+n] =  -1*invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[7*N+n]+b[4*N+n]) *(a[2*N+n]+b[2*N+n])));
    c[ 4*N+n] =  invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[11*N+n]+b[5*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[2*N+n]+b[2*N+n])));
    c[ 5*N+n] =  -1*invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[1*N+n]+b[1*N+n])));
    c[ 6*N+n] =  invdet*(((a[ 1*N+n]+b[ 1*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[6*N+n]+b[3*N+n])));
    c[ 7*N+n] =  -1*invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[7*N+n]+b[4*N+n])) - ((a[2*N+n]+b[2*N+n]) *(a[1*N+n]+b[1*N+n])));
    c[ 8*N+n] =  invdet*(((a[ 0*N+n]+b[ 0*N+n]) *(a[6*N+n]+b[3*N+n])) - ((a[1*N+n]+b[1*N+n]) *(a[1*N+n]+b[1*N+n])));
  }
}
void KalmanGain(const MP6x6SF* A, const MP3x3* B, MP3x6* C) {
  const float* a = (*A).data; //ASSUME_ALIGNED(a, 64);
  const float* b = (*B).data; //ASSUME_ALIGNED(b, 64);
  float* c = (*C).data;       //ASSUME_ALIGNED(c, 64);
#pragma omp simd
  for (int n = 0; n < N; ++n)
  {
    c[ 0*N+n] = a[0*N+n]*b[0*N+n] + a[1*N+n]*b[3*N+n] + a[2*N+n]*b[6*N+n];
    c[ 1*N+n] = a[0*N+n]*b[1*N+n] + a[1*N+n]*b[4*N+n] + a[2*N+n]*b[7*N+n];
    c[ 2*N+n] = a[0*N+n]*b[2*N+n] + a[1*N+n]*b[5*N+n] + a[2*N+n]*b[8*N+n];
    c[ 3*N+n] = a[1*N+n]*b[0*N+n] + a[6*N+n]*b[3*N+n] + a[7*N+n]*b[6*N+n];
    c[ 4*N+n] = a[1*N+n]*b[1*N+n] + a[6*N+n]*b[4*N+n] + a[7*N+n]*b[7*N+n];
    c[ 5*N+n] = a[1*N+n]*b[2*N+n] + a[6*N+n]*b[5*N+n] + a[7*N+n]*b[8*N+n];
    c[ 6*N+n] = a[2*N+n]*b[0*N+n] + a[7*N+n]*b[3*N+n] + a[11*N+n]*b[6*N+n];
    c[ 7*N+n] = a[2*N+n]*b[1*N+n] + a[7*N+n]*b[4*N+n] + a[11*N+n]*b[7*N+n];
    c[ 8*N+n] = a[2*N+n]*b[2*N+n] + a[7*N+n]*b[5*N+n] + a[11*N+n]*b[8*N+n];
    c[ 9*N+n] = a[3*N+n]*b[0*N+n] + a[8*N+n]*b[3*N+n] + a[12*N+n]*b[6*N+n];
    c[ 10*N+n] = a[3*N+n]*b[1*N+n] + a[8*N+n]*b[4*N+n] + a[12*N+n]*b[7*N+n];
    c[ 11*N+n] = a[3*N+n]*b[2*N+n] + a[8*N+n]*b[5*N+n] + a[12*N+n]*b[8*N+n];
    c[ 12*N+n] = a[4*N+n]*b[0*N+n] + a[9*N+n]*b[3*N+n] + a[13*N+n]*b[6*N+n];
    c[ 13*N+n] = a[4*N+n]*b[1*N+n] + a[9*N+n]*b[4*N+n] + a[13*N+n]*b[7*N+n];
    c[ 14*N+n] = a[4*N+n]*b[2*N+n] + a[9*N+n]*b[5*N+n] + a[13*N+n]*b[8*N+n];
    c[ 15*N+n] = a[5*N+n]*b[0*N+n] + a[10*N+n]*b[3*N+n] + a[14*N+n]*b[6*N+n];
    c[ 16*N+n] = a[5*N+n]*b[1*N+n] + a[10*N+n]*b[4*N+n] + a[14*N+n]*b[7*N+n];
    c[ 17*N+n] = a[5*N+n]*b[2*N+n] + a[10*N+n]*b[5*N+n] + a[14*N+n]*b[8*N+n];
  }
}

inline float hipo(float x, float y)
{
  return std::sqrt(x*x + y*y);
}

void KalmanUpdate(MP6x6SF* trkErr, MP6F* inPar, const MP3x3SF* hitErr, const MP3F* msP){
  
  MP1F rotT00;
  MP1F rotT01;
  MP2x2SF resErr_loc;
  MP3x3SF resErr_glo;
#pragma omp simd
  for (size_t it=0;it<bsize;++it) {
    const float r = hipo(x(msP,it), y(msP,it));
    rotT00.data[it] = -(y(msP,it) + y(inPar,it)) / (2*r);
    rotT01.data[it] =  (x(msP,it) + x(inPar,it)) / (2*r);    
    
    resErr_loc.data[ 0*bsize+it] = (rotT00.data[it]*(trkErr->data[0*bsize+it] + hitErr->data[0*bsize+it]) +
                                    rotT01.data[it]*(trkErr->data[1*bsize+it] + hitErr->data[1*bsize+it]))*rotT00.data[it] +
                                   (rotT00.data[it]*(trkErr->data[1*bsize+it] + hitErr->data[1*bsize+it]) +
                                    rotT01.data[it]*(trkErr->data[2*bsize+it] + hitErr->data[2*bsize+it]))*rotT01.data[it];
    resErr_loc.data[ 1*bsize+it] = (trkErr->data[3*bsize+it] + hitErr->data[3*bsize+it])*rotT00.data[it] +
                                   (trkErr->data[4*bsize+it] + hitErr->data[4*bsize+it])*rotT01.data[it];
    resErr_loc.data[ 2*bsize+it] = (trkErr->data[5*bsize+it] + hitErr->data[5*bsize+it]);
  }

  #pragma omp simd
  for (size_t it=0;it<bsize;++it)
  {
    const double det = (double)resErr_loc.data[0*bsize+it] * resErr_loc.data[2*bsize+it] -
                       (double)resErr_loc.data[1*bsize+it] * resErr_loc.data[1*bsize+it];
    const float s   = 1.f / det;
    const float tmp = s * resErr_loc.data[2*bsize+it];
    resErr_loc.data[1*bsize+it] *= -s;
    resErr_loc.data[2*bsize+it]  = s * resErr_loc.data[0*bsize+it];
    resErr_loc.data[0*bsize+it]  = tmp;
  }

   MP3x6 kGain;
#pragma omp simd
   for (size_t it=0;it<bsize;++it)
   {
      kGain.data[ 0*bsize+it] = trkErr->data[ 0*bsize+it]*(rotT00.data[it]*resErr_loc.data[ 0*bsize+it]) +
	                        trkErr->data[ 1*bsize+it]*(rotT01.data[it]*resErr_loc.data[ 0*bsize+it]) +
	                        trkErr->data[ 3*bsize+it]*resErr_loc.data[ 1*bsize+it];
      kGain.data[ 1*bsize+it] = trkErr->data[ 0*bsize+it]*(rotT00.data[it]*resErr_loc.data[ 1*bsize+it]) +
	                        trkErr->data[ 1*bsize+it]*(rotT01.data[it]*resErr_loc.data[ 1*bsize+it]) +
	                        trkErr->data[ 3*bsize+it]*resErr_loc.data[ 2*bsize+it];
      kGain.data[ 2*bsize+it] = 0;
      kGain.data[ 3*bsize+it] = trkErr->data[ 1*bsize+it]*(rotT00.data[it]*resErr_loc.data[ 0*bsize+it]) +
	                        trkErr->data[ 2*bsize+it]*(rotT01.data[it]*resErr_loc.data[ 0*bsize+it]) +
	                        trkErr->data[ 4*bsize+it]*resErr_loc.data[ 1*bsize+it];
      kGain.data[ 4*bsize+it] = trkErr->data[ 1*bsize+it]*(rotT00.data[it]*resErr_loc.data[ 1*bsize+it]) +
	                        trkErr->data[ 2*bsize+it]*(rotT01.data[it]*resErr_loc.data[ 1*bsize+it]) +
	                        trkErr->data[ 4*bsize+it]*resErr_loc.data[ 2*bsize+it];
      kGain.data[ 5*bsize+it] = 0;
      kGain.data[ 6*bsize+it] = trkErr->data[ 3*bsize+it]*(rotT00.data[it]*resErr_loc.data[ 0*bsize+it]) +
	                        trkErr->data[ 4*bsize+it]*(rotT01.data[it]*resErr_loc.data[ 0*bsize+it]) +
	                        trkErr->data[ 5*bsize+it]*resErr_loc.data[ 1*bsize+it];
      kGain.data[ 7*bsize+it] = trkErr->data[ 3*bsize+it]*(rotT00.data[it]*resErr_loc.data[ 1*bsize+it]) +
	                        trkErr->data[ 4*bsize+it]*(rotT01.data[it]*resErr_loc.data[ 1*bsize+it]) +
	                        trkErr->data[ 5*bsize+it]*resErr_loc.data[ 2*bsize+it];
      kGain.data[ 8*bsize+it] = 0;
      kGain.data[ 9*bsize+it] = trkErr->data[ 6*bsize+it]*(rotT00.data[it]*resErr_loc.data[ 0*bsize+it]) +
	                        trkErr->data[ 7*bsize+it]*(rotT01.data[it]*resErr_loc.data[ 0*bsize+it]) +
	                        trkErr->data[ 8*bsize+it]*resErr_loc.data[ 1*bsize+it];
      kGain.data[10*bsize+it] = trkErr->data[ 6*bsize+it]*(rotT00.data[it]*resErr_loc.data[ 1*bsize+it]) +
	                        trkErr->data[ 7*bsize+it]*(rotT01.data[it]*resErr_loc.data[ 1*bsize+it]) +
	                        trkErr->data[ 8*bsize+it]*resErr_loc.data[ 2*bsize+it];
      kGain.data[11*bsize+it] = 0;
      kGain.data[12*bsize+it] = trkErr->data[10*bsize+it]*(rotT00.data[it]*resErr_loc.data[ 0*bsize+it]) +
	                        trkErr->data[11*bsize+it]*(rotT01.data[it]*resErr_loc.data[ 0*bsize+it]) +
	                        trkErr->data[12*bsize+it]*resErr_loc.data[ 1*bsize+it];
      kGain.data[13*bsize+it] = trkErr->data[10*bsize+it]*(rotT00.data[it]*resErr_loc.data[ 1*bsize+it]) +
	                        trkErr->data[11*bsize+it]*(rotT01.data[it]*resErr_loc.data[ 1*bsize+it]) +
	                        trkErr->data[12*bsize+it]*resErr_loc.data[ 2*bsize+it];
      kGain.data[14*bsize+it] = 0;
      kGain.data[15*bsize+it] = trkErr->data[15*bsize+it]*(rotT00.data[it]*resErr_loc.data[ 0*bsize+it]) +
	                        trkErr->data[16*bsize+it]*(rotT01.data[it]*resErr_loc.data[ 0*bsize+it]) +
	                        trkErr->data[17*bsize+it]*resErr_loc.data[ 1*bsize+it];
      kGain.data[16*bsize+it] = trkErr->data[15*bsize+it]*(rotT00.data[it]*resErr_loc.data[ 1*bsize+it]) +
	                        trkErr->data[16*bsize+it]*(rotT01.data[it]*resErr_loc.data[ 1*bsize+it]) +
	                        trkErr->data[17*bsize+it]*resErr_loc.data[ 2*bsize+it];
      kGain.data[17*bsize+it] = 0;
   }

   MP2F res_loc;
#pragma omp simd
   for (size_t it=0;it<bsize;++it)
   {
     res_loc.data[0*bsize+it] =  rotT00.data[it]*(x(msP,it) - x(inPar,it)) + rotT01.data[it]*(y(msP,it) - y(inPar,it));
     res_loc.data[1*bsize+it] =  z(msP,it) - z(inPar,it);

     setx(inPar, it, x(inPar, it) + kGain.data[ 0*bsize+it] * res_loc.data[ 0*bsize+it] + kGain.data[ 1*bsize+it] * res_loc.data[ 1*bsize+it]);
     sety(inPar, it, y(inPar, it) + kGain.data[ 3*bsize+it] * res_loc.data[ 0*bsize+it] + kGain.data[ 4*bsize+it] * res_loc.data[ 1*bsize+it]);
     setz(inPar, it, z(inPar, it) + kGain.data[ 6*bsize+it] * res_loc.data[ 0*bsize+it] + kGain.data[ 7*bsize+it] * res_loc.data[ 1*bsize+it]);
     setipt(inPar, it, ipt(inPar, it) + kGain.data[ 9*bsize+it] * res_loc.data[ 0*bsize+it] + kGain.data[10*bsize+it] * res_loc.data[ 1*bsize+it]);
     setphi(inPar, it, phi(inPar, it) + kGain.data[12*bsize+it] * res_loc.data[ 0*bsize+it] + kGain.data[13*bsize+it] * res_loc.data[ 1*bsize+it]);
     settheta(inPar, it, theta(inPar, it) + kGain.data[15*bsize+it] * res_loc.data[ 0*bsize+it] + kGain.data[16*bsize+it] * res_loc.data[ 1*bsize+it]);
   }

   MP6x6SF newErr;
#pragma omp simd
   for (size_t it=0;it<bsize;++it)
   {
     newErr.data[ 0*bsize+it] = kGain.data[ 0*bsize+it]*rotT00.data[it]*trkErr->data[ 0*bsize+it] +
                                kGain.data[ 0*bsize+it]*rotT01.data[it]*trkErr->data[ 1*bsize+it] +
                                kGain.data[ 1*bsize+it]*trkErr->data[ 3*bsize+it];
     newErr.data[ 1*bsize+it] = kGain.data[ 3*bsize+it]*rotT00.data[it]*trkErr->data[ 0*bsize+it] +
                                kGain.data[ 3*bsize+it]*rotT01.data[it]*trkErr->data[ 1*bsize+it] +
                                kGain.data[ 4*bsize+it]*trkErr->data[ 3*bsize+it];
     newErr.data[ 2*bsize+it] = kGain.data[ 3*bsize+it]*rotT00.data[it]*trkErr->data[ 1*bsize+it] +
                                kGain.data[ 3*bsize+it]*rotT01.data[it]*trkErr->data[ 2*bsize+it] +
                                kGain.data[ 4*bsize+it]*trkErr->data[ 4*bsize+it];
     newErr.data[ 3*bsize+it] = kGain.data[ 6*bsize+it]*rotT00.data[it]*trkErr->data[ 0*bsize+it] +
                                kGain.data[ 6*bsize+it]*rotT01.data[it]*trkErr->data[ 1*bsize+it] +
                                kGain.data[ 7*bsize+it]*trkErr->data[ 3*bsize+it];
     newErr.data[ 4*bsize+it] = kGain.data[ 6*bsize+it]*rotT00.data[it]*trkErr->data[ 1*bsize+it] +
                                kGain.data[ 6*bsize+it]*rotT01.data[it]*trkErr->data[ 2*bsize+it] +
                                kGain.data[ 7*bsize+it]*trkErr->data[ 4*bsize+it];
     newErr.data[ 5*bsize+it] = kGain.data[ 6*bsize+it]*rotT00.data[it]*trkErr->data[ 3*bsize+it] +
                                kGain.data[ 6*bsize+it]*rotT01.data[it]*trkErr->data[ 4*bsize+it] +
                                kGain.data[ 7*bsize+it]*trkErr->data[ 5*bsize+it];
     newErr.data[ 6*bsize+it] = kGain.data[ 9*bsize+it]*rotT00.data[it]*trkErr->data[ 0*bsize+it] +
                                kGain.data[ 9*bsize+it]*rotT01.data[it]*trkErr->data[ 1*bsize+it] +
                                kGain.data[10*bsize+it]*trkErr->data[ 3*bsize+it];
     newErr.data[ 7*bsize+it] = kGain.data[ 9*bsize+it]*rotT00.data[it]*trkErr->data[ 1*bsize+it] +
                                kGain.data[ 9*bsize+it]*rotT01.data[it]*trkErr->data[ 2*bsize+it] +
                                kGain.data[10*bsize+it]*trkErr->data[ 4*bsize+it];
     newErr.data[ 8*bsize+it] = kGain.data[ 9*bsize+it]*rotT00.data[it]*trkErr->data[ 3*bsize+it] +
                                kGain.data[ 9*bsize+it]*rotT01.data[it]*trkErr->data[ 4*bsize+it] +
                                kGain.data[10*bsize+it]*trkErr->data[ 5*bsize+it];
     newErr.data[ 9*bsize+it] = kGain.data[ 9*bsize+it]*rotT00.data[it]*trkErr->data[ 6*bsize+it] +
                                kGain.data[ 9*bsize+it]*rotT01.data[it]*trkErr->data[ 7*bsize+it] +
                                kGain.data[10*bsize+it]*trkErr->data[ 8*bsize+it];
     newErr.data[10*bsize+it] = kGain.data[12*bsize+it]*rotT00.data[it]*trkErr->data[ 0*bsize+it] +
                                kGain.data[12*bsize+it]*rotT01.data[it]*trkErr->data[ 1*bsize+it] +
                                kGain.data[13*bsize+it]*trkErr->data[ 3*bsize+it];
     newErr.data[11*bsize+it] = kGain.data[12*bsize+it]*rotT00.data[it]*trkErr->data[ 1*bsize+it] +
                                kGain.data[12*bsize+it]*rotT01.data[it]*trkErr->data[ 2*bsize+it] +
                                kGain.data[13*bsize+it]*trkErr->data[ 4*bsize+it];
     newErr.data[12*bsize+it] = kGain.data[12*bsize+it]*rotT00.data[it]*trkErr->data[ 3*bsize+it] +
                                kGain.data[12*bsize+it]*rotT01.data[it]*trkErr->data[ 4*bsize+it] +
                                kGain.data[13*bsize+it]*trkErr->data[ 5*bsize+it];
     newErr.data[13*bsize+it] = kGain.data[12*bsize+it]*rotT00.data[it]*trkErr->data[ 6*bsize+it] +
                                kGain.data[12*bsize+it]*rotT01.data[it]*trkErr->data[ 7*bsize+it] +
                                kGain.data[13*bsize+it]*trkErr->data[ 8*bsize+it];
     newErr.data[14*bsize+it] = kGain.data[12*bsize+it]*rotT00.data[it]*trkErr->data[10*bsize+it] +
                                kGain.data[12*bsize+it]*rotT01.data[it]*trkErr->data[11*bsize+it] +
                                kGain.data[13*bsize+it]*trkErr->data[12*bsize+it];
     newErr.data[15*bsize+it] = kGain.data[15*bsize+it]*rotT00.data[it]*trkErr->data[ 0*bsize+it] +
                                kGain.data[15*bsize+it]*rotT01.data[it]*trkErr->data[ 1*bsize+it] +
                                kGain.data[16*bsize+it]*trkErr->data[ 3*bsize+it];
     newErr.data[16*bsize+it] = kGain.data[15*bsize+it]*rotT00.data[it]*trkErr->data[ 1*bsize+it] +
                                kGain.data[15*bsize+it]*rotT01.data[it]*trkErr->data[ 2*bsize+it] +
                                kGain.data[16*bsize+it]*trkErr->data[ 4*bsize+it];
     newErr.data[17*bsize+it] = kGain.data[15*bsize+it]*rotT00.data[it]*trkErr->data[ 3*bsize+it] +
                                kGain.data[15*bsize+it]*rotT01.data[it]*trkErr->data[ 4*bsize+it] +
                                kGain.data[16*bsize+it]*trkErr->data[ 5*bsize+it];
     newErr.data[18*bsize+it] = kGain.data[15*bsize+it]*rotT00.data[it]*trkErr->data[ 6*bsize+it] +
                                kGain.data[15*bsize+it]*rotT01.data[it]*trkErr->data[ 7*bsize+it] +
                                kGain.data[16*bsize+it]*trkErr->data[ 8*bsize+it];
     newErr.data[19*bsize+it] = kGain.data[15*bsize+it]*rotT00.data[it]*trkErr->data[10*bsize+it] +
                                kGain.data[15*bsize+it]*rotT01.data[it]*trkErr->data[11*bsize+it] +
                                kGain.data[16*bsize+it]*trkErr->data[12*bsize+it];
     newErr.data[20*bsize+it] = kGain.data[15*bsize+it]*rotT00.data[it]*trkErr->data[15*bsize+it] +
                                kGain.data[15*bsize+it]*rotT01.data[it]*trkErr->data[16*bsize+it] +
                                kGain.data[16*bsize+it]*trkErr->data[17*bsize+it];

     newErr.data[ 0*bsize+it] = trkErr->data[ 0*bsize+it] - newErr.data[ 0*bsize+it];
     newErr.data[ 1*bsize+it] = trkErr->data[ 1*bsize+it] - newErr.data[ 1*bsize+it];
     newErr.data[ 2*bsize+it] = trkErr->data[ 2*bsize+it] - newErr.data[ 2*bsize+it];
     newErr.data[ 3*bsize+it] = trkErr->data[ 3*bsize+it] - newErr.data[ 3*bsize+it];
     newErr.data[ 4*bsize+it] = trkErr->data[ 4*bsize+it] - newErr.data[ 4*bsize+it];
     newErr.data[ 5*bsize+it] = trkErr->data[ 5*bsize+it] - newErr.data[ 5*bsize+it];
     newErr.data[ 6*bsize+it] = trkErr->data[ 6*bsize+it] - newErr.data[ 6*bsize+it];
     newErr.data[ 7*bsize+it] = trkErr->data[ 7*bsize+it] - newErr.data[ 7*bsize+it];
     newErr.data[ 8*bsize+it] = trkErr->data[ 8*bsize+it] - newErr.data[ 8*bsize+it];
     newErr.data[ 9*bsize+it] = trkErr->data[ 9*bsize+it] - newErr.data[ 9*bsize+it];
     newErr.data[10*bsize+it] = trkErr->data[10*bsize+it] - newErr.data[10*bsize+it];
     newErr.data[11*bsize+it] = trkErr->data[11*bsize+it] - newErr.data[11*bsize+it];
     newErr.data[12*bsize+it] = trkErr->data[12*bsize+it] - newErr.data[12*bsize+it];
     newErr.data[13*bsize+it] = trkErr->data[13*bsize+it] - newErr.data[13*bsize+it];
     newErr.data[14*bsize+it] = trkErr->data[14*bsize+it] - newErr.data[14*bsize+it];
     newErr.data[15*bsize+it] = trkErr->data[15*bsize+it] - newErr.data[15*bsize+it];
     newErr.data[16*bsize+it] = trkErr->data[16*bsize+it] - newErr.data[16*bsize+it];
     newErr.data[17*bsize+it] = trkErr->data[17*bsize+it] - newErr.data[17*bsize+it];
     newErr.data[18*bsize+it] = trkErr->data[18*bsize+it] - newErr.data[18*bsize+it];
     newErr.data[19*bsize+it] = trkErr->data[19*bsize+it] - newErr.data[19*bsize+it];
     newErr.data[20*bsize+it] = trkErr->data[20*bsize+it] - newErr.data[20*bsize+it];
   }

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
  
  trkErr = &newErr;
}

inline void sincos4(const float x, float& sin, float& cos)
{
   // Had this writen with explicit division by factorial.
   // The *whole* fitting test ran like 2.5% slower on MIC, sigh.

   const float x2 = x*x;
   cos  = 1.f - 0.5f*x2 + 0.04166667f*x2*x2;
   sin  = x - 0.16666667f*x*x2;
}

constexpr float kfact= 100/3.8;
constexpr int Niter=5;
void propagateToR(const MP6x6SF* inErr, const MP6F* inPar, const MP1I* inChg, 
                  const MP3F* msP, MP6x6SF* outErr, MP6F* outPar) {
  
  MP6x6F errorProp, temp;
#pragma omp simd
  for (size_t it=0;it<bsize;++it) {	
    //initialize erroProp to identity matrix
    for (size_t i=0;i<6;++i) errorProp.data[bsize*PosInMtrx(i,i,6) + it] = 1.f;
    
    float r0 = hipo(x(inPar,it), y(inPar,it));
    const float k = q(inChg,it) * kfact;
    const float r = hipo(x(msP,it), y(msP,it));

    const float xin     = x(inPar,it);
    const float yin     = y(inPar,it);
    const float iptin   = ipt(inPar,it);
    const float phiin   = phi(inPar,it);
    const float thetain = theta(inPar,it);

    //initialize outPar to inPar
    setx(outPar,it, xin);
    sety(outPar,it, yin);
    setz(outPar,it, z(inPar,it));
    setipt(outPar,it, iptin);
    setphi(outPar,it, phiin);
    settheta(outPar,it, thetain);

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

    for (int i = 0; i < Niter; ++i)
    {
      //compute distance and path for the current iteration
      r0 = hipo(x(outPar,it), y(outPar,it));
      id = (r-r0);
      D+=id;
      sincos4(id*iptin*kinv, sina, cosa);

      //update derivatives on total distance
      if (i+1 != Niter) {

	const float xtmp = x(outPar,it);
	const float ytmp = y(outPar,it);
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
      setx(outPar,it, x(outPar,it) + k*(pxin*sina - pyin*(1.f-cosa)));
      sety(outPar,it, y(outPar,it) + k*(pyin*sina + pxin*(1.f-cosa)));
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

    errorProp.data[bsize*PosInMtrx(0,0,6) + it] = 1.f+k*dadx*(cosPorT*cosa-sinPorT*sina)*pt;
    errorProp.data[bsize*PosInMtrx(0,1,6) + it] =     k*dady*(cosPorT*cosa-sinPorT*sina)*pt;
    errorProp.data[bsize*PosInMtrx(0,2,6) + it] = 0.f;
    errorProp.data[bsize*PosInMtrx(0,3,6) + it] = k*(cosPorT*(iptin*dadipt*cosa-sina)+sinPorT*((1.f-cosa)-iptin*dadipt*sina))*pt*pt;
    errorProp.data[bsize*PosInMtrx(0,4,6) + it] = k*(cosPorT*dadphi*cosa - sinPorT*dadphi*sina - sinPorT*sina + cosPorT*cosa - cosPorT)*pt;
    errorProp.data[bsize*PosInMtrx(0,5,6) + it] = 0.f;

    errorProp.data[bsize*PosInMtrx(1,0,6) + it] =     k*dadx*(sinPorT*cosa+cosPorT*sina)*pt;
    errorProp.data[bsize*PosInMtrx(1,1,6) + it] = 1.f+k*dady*(sinPorT*cosa+cosPorT*sina)*pt;
    errorProp.data[bsize*PosInMtrx(1,2,6) + it] = 0.f;
    errorProp.data[bsize*PosInMtrx(1,3,6) + it] = k*(sinPorT*(iptin*dadipt*cosa-sina)+cosPorT*(iptin*dadipt*sina-(1.f-cosa)))*pt*pt;
    errorProp.data[bsize*PosInMtrx(1,4,6) + it] = k*(sinPorT*dadphi*cosa + cosPorT*dadphi*sina + sinPorT*cosa + cosPorT*sina - sinPorT)*pt;
    errorProp.data[bsize*PosInMtrx(1,5,6) + it] = 0.f;

    //no trig approx here, theta can be large
    cosPorT=std::cos(thetain);
    sinPorT=std::sin(thetain);
    //redefine sinPorT as 1./sinPorT to reduce the number of temporaries
    sinPorT = 1.f/sinPorT;

    setz(outPar,it, z(inPar,it) + k*alpha*cosPorT*pt*sinPorT);

    errorProp.data[bsize*PosInMtrx(2,0,6) + it] = k*cosPorT*dadx*pt*sinPorT;
    errorProp.data[bsize*PosInMtrx(2,1,6) + it] = k*cosPorT*dady*pt*sinPorT;
    errorProp.data[bsize*PosInMtrx(2,2,6) + it] = 1.f;
    errorProp.data[bsize*PosInMtrx(2,3,6) + it] = k*cosPorT*(iptin*dadipt-alpha)*pt*pt*sinPorT;
    errorProp.data[bsize*PosInMtrx(2,4,6) + it] = k*dadphi*cosPorT*pt*sinPorT;
    errorProp.data[bsize*PosInMtrx(2,5,6) + it] =-k*alpha*pt*sinPorT*sinPorT;

    setipt(outPar,it, iptin);

    errorProp.data[bsize*PosInMtrx(3,0,6) + it] = 0.f;
    errorProp.data[bsize*PosInMtrx(3,1,6) + it] = 0.f;
    errorProp.data[bsize*PosInMtrx(3,2,6) + it] = 0.f;
    errorProp.data[bsize*PosInMtrx(3,3,6) + it] = 1.f;
    errorProp.data[bsize*PosInMtrx(3,4,6) + it] = 0.f;
    errorProp.data[bsize*PosInMtrx(3,5,6) + it] = 0.f;

    setphi(outPar,it, phi(inPar,it)+alpha );

    errorProp.data[bsize*PosInMtrx(4,0,6) + it] = dadx;
    errorProp.data[bsize*PosInMtrx(4,1,6) + it] = dady;
    errorProp.data[bsize*PosInMtrx(4,2,6) + it] = 0.f;
    errorProp.data[bsize*PosInMtrx(4,3,6) + it] = dadipt;
    errorProp.data[bsize*PosInMtrx(4,4,6) + it] = 1.f+dadphi;
    errorProp.data[bsize*PosInMtrx(4,5,6) + it] = 0.f;

    settheta(outPar,it, thetain);

    errorProp.data[bsize*PosInMtrx(5,0,6) + it] = 0.f;
    errorProp.data[bsize*PosInMtrx(5,1,6) + it] = 0.f;
    errorProp.data[bsize*PosInMtrx(5,2,6) + it] = 0.f;
    errorProp.data[bsize*PosInMtrx(5,3,6) + it] = 0.f;
    errorProp.data[bsize*PosInMtrx(5,4,6) + it] = 0.f;
    errorProp.data[bsize*PosInMtrx(5,5,6) + it] = 1.f;
  }

  MultHelixProp(&errorProp, inErr, &temp);
  MultHelixPropTransp(&errorProp, &temp, outErr);
}

int main (int argc, char* argv[]) {

   int itr;
   ATRK inputtrk = {
     {-12.806846618652344, -7.723824977874756, 38.13014221191406,0.23732035065189902, -2.613372802734375, 0.35594117641448975},
     {6.290299552347278e-07,4.1375109560704004e-08,7.526661534029699e-07,2.0973730840978533e-07,1.5431574240665213e-07,9.626245400795597e-08,-2.804026640189443e-06,
      6.219111130687595e-06,2.649119409845118e-07,0.00253512163402557,-2.419662877381737e-07,4.3124190760040646e-07,3.1068903991780678e-09,0.000923913115050627,
      0.00040678296006807003,-7.755406890332818e-07,1.68539375883925e-06,6.676875566525437e-08,0.0008420574605423793,7.356584799406111e-05,0.0002306247719158348},
     1
   };

   AHIT inputhit = {
     {-20.7824649810791, -12.24150276184082, 57.8067626953125},
     {2.545517190810642e-06,-2.6680759219743777e-06,2.8030024168401724e-06,0.00014160551654640585,0.00012282167153898627,11.385087966918945}
   };

   printf("track in pos: x=%f, y=%f, z=%f, r=%f \n", inputtrk.par[0], inputtrk.par[1], inputtrk.par[2], sqrtf(inputtrk.par[0]*inputtrk.par[0] + inputtrk.par[1]*inputtrk.par[1]));
   printf("track in cov: xx=%.2e, yy=%.2e, zz=%.2e \n", inputtrk.cov[SymOffsets66(PosInMtrx(0,0,6))],
	                                       inputtrk.cov[SymOffsets66(PosInMtrx(1,1,6))],
	                                       inputtrk.cov[SymOffsets66(PosInMtrx(2,2,6))]);
   printf("hit in pos: x=%f, y=%f, z=%f, r=%f \n", inputhit.pos[0], inputhit.pos[1], inputhit.pos[2], sqrtf(inputhit.pos[0]*inputhit.pos[0] + inputhit.pos[1]*inputhit.pos[1]));
   
   printf("produce nevts=%i ntrks=%i smearing by=%f \n", nevts, ntrks, smear);
   printf("NITER=%d\n", NITER);
   long setup_start, setup_stop;
   struct timeval timecheck;

   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
   MPTRK* trk = prepareTracks(inputtrk);
   MPHIT* hit = prepareHits(inputhit);
   MPTRK* outtrk = (MPTRK*) malloc(nevts*nb*sizeof(MPTRK));
   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");
   

   task_scheduler_init init(nthreads);

   auto wall_start = std::chrono::high_resolution_clock::now();

   for(itr=0; itr<NITER; itr++) {
      parallel_for(blocked_range<size_t>(0,nevts,4),[&](blocked_range<size_t> iex){
      for(size_t ie =iex.begin(); ie<iex.end();++ie){
        parallel_for(blocked_range<size_t>(0,nb,4),[&](blocked_range<size_t> ibx){
        for(size_t ib =ibx.begin(); ib<ibx.end();++ib){
          const MPTRK* btracks = bTk(trk, ie, ib);
          MPTRK* obtracks = bTk(outtrk, ie, ib);
          for(size_t layer=0; layer<nlayer; ++layer) {
            const MPHIT* bhits = bHit(hit, ie, ib, layer);
            propagateToR(&(*btracks).cov, &(*btracks).par, &(*btracks).q, &(*bhits).pos, &(*obtracks).cov, &(*obtracks).par); // vectorized function
            KalmanUpdate(&(*obtracks).cov,&(*obtracks).par,&(*bhits).cov,&(*bhits).pos);
          }
        }});
      }});
   } //end of itr loop
   auto wall_stop = std::chrono::high_resolution_clock::now();

   auto wall_diff = wall_stop - wall_start;
   auto wall_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
   printf("setup time time=%f (s)\n", (setup_stop-setup_start)*0.001);
   printf("done ntracks=%i tot time=%f (s) time/trk=%e (s)\n", nevts*ntrks*int(NITER), wall_time, wall_time/(nevts*ntrks*int(NITER)));
   printf("formatted %i %i %i %i %i %f 0 %f %i\n",int(NITER),nevts, ntrks, bsize, nb, wall_time, (setup_stop-setup_start)*0.001, nthreads);

   float avgx = 0, avgy = 0, avgz = 0, avgr = 0;
   float avgpt = 0, avgphi = 0, avgtheta = 0;
   float avgdx = 0, avgdy = 0, avgdz = 0, avgdr = 0;
   for (size_t ie=0;ie<nevts;++ie) {
     for (size_t it=0;it<ntrks;++it) {
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

   free(trk);
   free(hit);
   free(outtrk);

   return 0;
}
