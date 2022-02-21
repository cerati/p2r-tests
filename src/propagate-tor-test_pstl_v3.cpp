/*
nvc++ -O2 -std=c++17 -stdpar=gpu -gpu=cc75 -gpu=managed -gpu=fma -gpu=fastmath -gpu=autocollapse -gpu=loadcache:L1 -gpu=unroll  src/propagate-tor-test_pstl.cpp   -o ./propagate_nvcpp_pstl
nvc++ -O2 -std=c++17 -stdpar=multicore src/propagate-tor-test_pstl.cpp   -o ./propagate_nvcpp_pstl 
g++ -O3 -I. -fopenmp -mavx512f -std=c++17 src/propagate-tor-test_pstl.cpp -lm -lgomp -Lpath-to-tbb-lib -ltbb  -o ./propagate_gcc_pstl
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>

#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>
#include <execution>
#include <random>

#ifndef bsize
#if defined(__NVCOMPILER_CUDA__)
#define bsize 1
#else
#define bsize 128
#endif//__NVCOMPILER_CUDA__
#endif
#ifndef ntrks
#define ntrks 9600//8192
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

#ifndef __NVCOMPILER_CUDA__

#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

constexpr int alloc_align  = (2*1024*1024);

#endif

namespace impl {

   template <typename IntType>
   class counting_iterator {
       static_assert(std::numeric_limits<IntType>::is_integer, "Cannot instantiate counting_iterator with a non-integer type");
     public:
       using value_type = IntType;
       using difference_type = typename std::make_signed<IntType>::type;
       using pointer = IntType*;
       using reference = IntType&;
       using iterator_category = std::random_access_iterator_tag;

       counting_iterator() : value(0) { }
       explicit counting_iterator(IntType v) : value(v) { }

       value_type operator*() const { return value; }
       value_type operator[](difference_type n) const { return value + n; }

       counting_iterator& operator++() { ++value; return *this; }
       counting_iterator operator++(int) {
         counting_iterator result{value};
         ++value;
         return result;
       }  
       counting_iterator& operator--() { --value; return *this; }
       counting_iterator operator--(int) {
         counting_iterator result{value};
         --value;
         return result;
       }
       counting_iterator& operator+=(difference_type n) { value += n; return *this; }
       counting_iterator& operator-=(difference_type n) { value -= n; return *this; }

       friend counting_iterator operator+(counting_iterator const& i, difference_type n)          { return counting_iterator(i.value + n);  }
       friend counting_iterator operator+(difference_type n, counting_iterator const& i)          { return counting_iterator(i.value + n);  }
       friend difference_type   operator-(counting_iterator const& x, counting_iterator const& y) { return x.value - y.value;  }
       friend counting_iterator operator-(counting_iterator const& i, difference_type n)          { return counting_iterator(i.value - n);  }

       friend bool operator==(counting_iterator const& x, counting_iterator const& y) { return x.value == y.value;  }
       friend bool operator!=(counting_iterator const& x, counting_iterator const& y) { return x.value != y.value;  }
       friend bool operator<(counting_iterator const& x, counting_iterator const& y)  { return x.value < y.value; }
       friend bool operator<=(counting_iterator const& x, counting_iterator const& y) { return x.value <= y.value; }
       friend bool operator>(counting_iterator const& x, counting_iterator const& y)  { return x.value > y.value; }
       friend bool operator>=(counting_iterator const& x, counting_iterator const& y) { return x.value >= y.value; }

     private:
       IntType value;
   };

} //impl


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
#if defined( __NVCOMPILER_CUDA__) || defined(__INTEL_COMPILER)
	 ptr = (Tp*) malloc(n*sizeof(Tp));
#elif !defined(DPCPP_BACKEND)
         ptr = (Tp*)_mm_malloc(n*sizeof(Tp),alloc_align);
#endif
	 if(!ptr) throw std::bad_alloc();

         return ptr;
       }

      void deallocate( Tp* p, std::size_t n) noexcept {
#if defined(__NVCOMPILER_CUDA__) || defined(__INTEL_COMPILER)
         free((void *)p);
#elif !defined(DPCPP_BACKEND)
         _mm_free((void *)p);
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


template <typename T, int N, int bSize>
struct MPNX_ {
   std::array<T,N*bSize> data;
   //basic accessors
   const T& operator[](const int idx) const {return data[idx];}
   T& operator[](const int idx) {return data[idx];}
   const T& operator()(const int m, const int b) const {return data[m*bSize+b];}
   T& operator()(const int m, const int b) {return data[m*bSize+b];}
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
   
   void load(MPNX_<T, n, bsz>& dest, const int tid, const int layer = 0) const {
      auto tid_offset = GetThreadOffset(tid, layer);
#pragma unroll 
      for(int it = 0; it < bsz; it++){
#pragma unroll
        for(int id = 0; id < n; id++){
          dest(id, it) = this->operator()(id, tid_offset+it);
        }
      }
      return;
   }
   void save(const MPNX_<T, n, bsz>& src, const int tid, const int layer = 0){
      auto tid_offset = GetThreadOffset(tid, layer); 
#pragma unroll
      for(int it = 0; it < bsz; it++){
#pragma unroll
        for(int id = 0; id < n; id++){
           this->operator()(id, tid_offset+it) = src(id, it);
        }
      }
      return;
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
  
  void load(MPTRK_ &dst, const int tid, const int layer = 0) const {
    this->par.load(dst.par, tid, layer);
    this->cov.load(dst.cov, tid, layer);
    this->q.load(dst.q, tid, layer);
    
    return;
  }
  
  void save(MPTRK_ &src, const int tid, const int layer = 0) {
    this->par.save(src.par, tid, layer);
    this->cov.save(src.cov, tid, layer);
    this->q.save(src.q, tid, layer);
    
    return;
  }


};

template<FieldOrder order>
std::shared_ptr<MPTRK> prepareTracksN(struct ATRK inputtrk) {

  auto result = std::make_shared<MPTRK>(ntrks, nevts);
  //create an accessor field:
  std::unique_ptr<MPTRKAccessor<order>> rA(new MPTRKAccessor<order>(*result));

  // store in element order for bunches of bsize matrices (a la matriplex)
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(1);
  std::uniform_real_distribution<float> urdist(0.0,1.0);

  for (size_t ie=0;ie<nevts;++ie) {
    for (size_t ib=0;ib<nb;++ib) {
      for (size_t it=0;it<bsize;++it) {
        //const int l = it+ib*bsize+ie*nb*bsize;
        const int tid = ib+ie*nb;
    	//par
    	for (size_t ip=0;ip<6;++ip) {
          rA->par(ip, tid, it, 0) = (1.f+smear*urdist(gen))*inputtrk.par[ip];
    	}
    	//cov
    	for (size_t ip=0;ip<21;++ip) {
          rA->cov(ip, tid, it, 0) = (1.f+smear*urdist(gen))*inputtrk.cov[ip]*100;
    	}
    	//q
        rA->q(0, tid, it, 0) = inputtrk.q;
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
  
  void load(MPHIT_ &dst, const int tid, const int layer = 0) const {
    this->pos.load(dst.pos, tid, layer);
    this->cov.load(dst.cov, tid, layer);
    
    return;
  }
  
  void save(MPHIT_ &src, const int tid, const int layer = 0) {
    this->pos.save(src.pos, tid, layer);
    this->cov.save(src.cov, tid, layer);
    
    return;
  }
  
};

template<FieldOrder order>
std::shared_ptr<MPHIT> prepareHitsN(std::vector<AHIT>& inputhits) {
  auto result = std::make_shared<MPHIT>(ntrks, nevts, nlayer);
  //create an accessor field:
  std::unique_ptr<MPHITAccessor<order>> rA(new MPHITAccessor<order>(*result));

  // store in element order for bunches of bsize matrices (a la matriplex)
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(2);
  std::uniform_real_distribution<float> urdist(0.0,1.0);

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
          //const int l = it + ib*bsize + ie*nb*bsize + lay*nb*bsize*nevts;
          const int tid = ib + ie*nb;
          //pos
          for (size_t ip=0;ip<3;++ip) {
            rA->pos(ip, tid, it, lay) = (1+smear*urdist(gen))*inputhit.pos[ip];
          }
          //cov
          for (size_t ip=0;ip<6;++ip) {
            rA->cov(ip, tid, it, lay) = (1+smear*urdist(gen))*inputhit.cov[ip];
          }
        }
      }
    }
  }
  return std::move(result);
}

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

const MPHIT_* bHit(const MPHIT_* hits, size_t ev, size_t ib) {
  return &(hits[ib + nb*ev]);
}
const MPHIT_* bHit(const MPHIT_* hits, size_t ev, size_t ib,size_t lay) {
return &(hits[lay + (ib*nlayer) +(ev*nlayer*nb)]);
}
//
float Pos(const MP3F_* hpos, size_t it, size_t ipar){
  return (*hpos).data[it + ipar*bsize];
}
float x(const MP3F_* hpos, size_t it)    { return Pos(hpos, it, 0); }
float y(const MP3F_* hpos, size_t it)    { return Pos(hpos, it, 1); }
float z(const MP3F_* hpos, size_t it)    { return Pos(hpos, it, 2); }
//
float Pos(const MPHIT_* hits, size_t it, size_t ipar){
  return Pos(&(*hits).pos,it,ipar);
}
float x(const MPHIT_* hits, size_t it)    { return Pos(hits, it, 0); }
float y(const MPHIT_* hits, size_t it)    { return Pos(hits, it, 1); }
float z(const MPHIT_* hits, size_t it)    { return Pos(hits, it, 2); }
//
float Pos(const MPHIT_* hits, size_t ev, size_t tk, size_t ipar){
  size_t ib = tk/bsize;
  const MPHIT_* bhits = bHit(hits, ev, ib);
  size_t it = tk % bsize;
  return Pos(bhits,it,ipar);
}
float x(const MPHIT_* hits, size_t ev, size_t tk)    { return Pos(hits, ev, tk, 0); }
float y(const MPHIT_* hits, size_t ev, size_t tk)    { return Pos(hits, ev, tk, 1); }
float z(const MPHIT_* hits, size_t ev, size_t tk)    { return Pos(hits, ev, tk, 2); }

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

MPHIT_* prepareHits(std::vector<AHIT>& inputhits) {
  MPHIT_* result = new MPHIT_[nlayer*nevts*nb];
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

template<size_t N = 1>
inline void MultHelixProp(const MP6x6F_ &a, const MP6x6SF_ &b, MP6x6F_ &c) {//ok
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

template<size_t N = 1>
inline void MultHelixPropTransp(const MP6x6F_ &a, const MP6x6F_ &b, MP6x6SF_ &c) {//

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

template <size_t N = 1>
void KalmanUpdate(MP6x6SF_ &trkErr_, MP6F_ &inPar_, const MP3x3SF_ &hitErr_, const MP3F_ &msP_){	  
  
  MP1F_    rotT00;
  MP1F_    rotT01;
  MP2x2SF_ resErr_loc;
  //MP3x3SF_ resErr_glo;
    
  for (size_t it = 0;it < N; ++it) {   
    const auto msPX = msP_(iparX, it);
    const auto msPY = msP_(iparY, it);
    const auto inParX = inPar_(iparX, it);
    const auto inParY = inPar_(iparY, it);          
  
    const auto r = hipo(msPX, msPY);
    rotT00[it] = -(msPY + inParY) / (2*r);
    rotT01[it] =  (msPX + inParX) / (2*r);    
    
    resErr_loc[ 0*N+it] = (rotT00[it]*(trkErr_[0*N+it] + hitErr_[0*N+it]) +
                                    rotT01[it]*(trkErr_[1*N+it] + hitErr_[1*N+it]))*rotT00[it] +
                                   (rotT00[it]*(trkErr_[1*N+it] + hitErr_[1*N+it]) +
                                    rotT01[it]*(trkErr_[2*N+it] + hitErr_[2*N+it]))*rotT01[it];
    resErr_loc[ 1*N+it] = (trkErr_[3*N+it] + hitErr_[3*N+it])*rotT00[it] +
                                   (trkErr_[4*N+it] + hitErr_[4*N+it])*rotT01[it];
    resErr_loc[ 2*N+it] = (trkErr_[5*N+it] + hitErr_[5*N+it]);
  } 
  
  for (size_t it=0;it<N;++it) {
  
    const double det = (double)resErr_loc[0*N+it] * resErr_loc[2*N+it] -
                       (double)resErr_loc[1*N+it] * resErr_loc[1*N+it];
    const float s   = 1.f / det;
    const float tmp = s * resErr_loc[2*N+it];
    resErr_loc[1*N+it] *= -s;
    resErr_loc[2*N+it]  = s * resErr_loc[0*N+it];
    resErr_loc[0*N+it]  = tmp;  
  }     
  
  MP3x6_ kGain;
  
#pragma omp simd
  for (size_t it=0; it<N; ++it) {
    kGain[ 0*N+it] = trkErr_[ 0*N+it]*(rotT00[it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 1*N+it]*(rotT01[it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 3*N+it]*resErr_loc[ 1*N+it];
    kGain[ 1*N+it] = trkErr_[ 0*N+it]*(rotT00[it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 1*N+it]*(rotT01[it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 3*N+it]*resErr_loc[ 2*N+it];
    kGain[ 2*N+it] = 0;
    kGain[ 3*N+it] = trkErr_[ 1*N+it]*(rotT00[it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 2*N+it]*(rotT01[it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 4*N+it]*resErr_loc[ 1*N+it];
    kGain[ 4*N+it] = trkErr_[ 1*N+it]*(rotT00[it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 2*N+it]*(rotT01[it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 4*N+it]*resErr_loc[ 2*N+it];
    kGain[ 5*N+it] = 0;
    kGain[ 6*N+it] = trkErr_[ 3*N+it]*(rotT00[it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 4*N+it]*(rotT01[it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 5*N+it]*resErr_loc[ 1*N+it];
    kGain[ 7*N+it] = trkErr_[ 3*N+it]*(rotT00[it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 4*N+it]*(rotT01[it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 5*N+it]*resErr_loc[ 2*N+it];
    kGain[ 8*N+it] = 0;
    kGain[ 9*N+it] = trkErr_[ 6*N+it]*(rotT00[it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 7*N+it]*(rotT01[it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[ 8*N+it]*resErr_loc[ 1*N+it];
    kGain[10*N+it] = trkErr_[ 6*N+it]*(rotT00[it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 7*N+it]*(rotT01[it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[ 8*N+it]*resErr_loc[ 2*N+it];
    kGain[11*N+it] = 0;
    kGain[12*N+it] = trkErr_[10*N+it]*(rotT00[it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[11*N+it]*(rotT01[it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[12*N+it]*resErr_loc[ 1*N+it];
    kGain[13*N+it] = trkErr_[10*N+it]*(rotT00[it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[11*N+it]*(rotT01[it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[12*N+it]*resErr_loc[ 2*N+it];
    kGain[14*N+it] = 0;
    kGain[15*N+it] = trkErr_[15*N+it]*(rotT00[it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[16*N+it]*(rotT01[it]*resErr_loc[ 0*N+it]) +
	                        trkErr_[17*N+it]*resErr_loc[ 1*N+it];
    kGain[16*N+it] = trkErr_[15*N+it]*(rotT00[it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[16*N+it]*(rotT01[it]*resErr_loc[ 1*N+it]) +
	                        trkErr_[17*N+it]*resErr_loc[ 2*N+it];
    kGain[17*N+it] = 0;  
  }  
     
  MP2F_ res_loc;   
  for (size_t it = 0; it < N; ++it) {
    const auto msPX = msP_(iparX, it);
    const auto msPY = msP_(iparY, it);
    const auto msPZ = msP_(iparZ, it);    
    const auto inParX = inPar_(iparX, it);
    const auto inParY = inPar_(iparY, it);     
    const auto inParZ = inPar_(iparZ, it); 
    
    const auto inParIpt   = inPar_(iparIpt, it);
    const auto inParPhi   = inPar_(iparPhi, it);
    const auto inParTheta = inPar_(iparTheta, it);            
    
    res_loc[0*N+it] =  rotT00[it]*(msPX - inParX) + rotT01[it]*(msPY - inParY);
    res_loc[1*N+it] =  msPZ - inParZ;

    inPar_(iparX,it)     = inParX + kGain[ 0*N+it] * res_loc[ 0*N+it] + kGain[ 1*N+it] * res_loc[ 1*N+it];
    inPar_(iparY,it)     = inParY + kGain[ 3*N+it] * res_loc[ 0*N+it] + kGain[ 4*N+it] * res_loc[ 1*N+it];
    inPar_(iparZ,it)     = inParZ + kGain[ 6*N+it] * res_loc[ 0*N+it] + kGain[ 7*N+it] * res_loc[ 1*N+it];
    inPar_(iparIpt,it)   = inParIpt + kGain[ 9*N+it] * res_loc[ 0*N+it] + kGain[10*N+it] * res_loc[ 1*N+it];
    inPar_(iparPhi,it)   = inParPhi + kGain[12*N+it] * res_loc[ 0*N+it] + kGain[13*N+it] * res_loc[ 1*N+it];
    inPar_(iparTheta,it) = inParTheta + kGain[15*N+it] * res_loc[ 0*N+it] + kGain[16*N+it] * res_loc[ 1*N+it];     
  }

   MP6x6SF_ newErr;
   for (size_t it=0;it<bsize;++it)   {

     newErr[ 0*N+it] = kGain[ 0*N+it]*rotT00[it]*trkErr_[ 0*N+it] +
                         kGain[ 0*N+it]*rotT01[it]*trkErr_[ 1*N+it] +
                         kGain[ 1*N+it]*trkErr_[ 3*N+it];
     newErr[ 1*N+it] = kGain[ 3*N+it]*rotT00[it]*trkErr_[ 0*N+it] +
                         kGain[ 3*N+it]*rotT01[it]*trkErr_[ 1*N+it] +
                         kGain[ 4*N+it]*trkErr_[ 3*N+it];
     newErr[ 2*N+it] = kGain[ 3*N+it]*rotT00[it]*trkErr_[ 1*N+it] +
                         kGain[ 3*N+it]*rotT01[it]*trkErr_[ 2*N+it] +
                         kGain[ 4*N+it]*trkErr_[ 4*N+it];
     newErr[ 3*N+it] = kGain[ 6*N+it]*rotT00[it]*trkErr_[ 0*N+it] +
                         kGain[ 6*N+it]*rotT01[it]*trkErr_[ 1*N+it] +
                         kGain[ 7*N+it]*trkErr_[ 3*N+it];
     newErr[ 4*N+it] = kGain[ 6*N+it]*rotT00[it]*trkErr_[ 1*N+it] +
                         kGain[ 6*N+it]*rotT01[it]*trkErr_[ 2*N+it] +
                         kGain[ 7*N+it]*trkErr_[ 4*N+it];
     newErr[ 5*N+it] = kGain[ 6*N+it]*rotT00[it]*trkErr_[ 3*N+it] +
                         kGain[ 6*N+it]*rotT01[it]*trkErr_[ 4*N+it] +
                         kGain[ 7*N+it]*trkErr_[ 5*N+it];
     newErr[ 6*N+it] = kGain[ 9*N+it]*rotT00[it]*trkErr_[ 0*N+it] +
                         kGain[ 9*N+it]*rotT01[it]*trkErr_[ 1*N+it] +
                         kGain[10*N+it]*trkErr_[ 3*N+it];
     newErr[ 7*N+it] = kGain[ 9*N+it]*rotT00[it]*trkErr_[ 1*N+it] +
                         kGain[ 9*N+it]*rotT01[it]*trkErr_[ 2*N+it] +
                         kGain[10*N+it]*trkErr_[ 4*N+it];
     newErr[ 8*N+it] = kGain[ 9*N+it]*rotT00[it]*trkErr_[ 3*N+it] +
                         kGain[ 9*N+it]*rotT01[it]*trkErr_[ 4*N+it] +
                         kGain[10*N+it]*trkErr_[ 5*N+it];
     newErr[ 9*N+it] = kGain[ 9*N+it]*rotT00[it]*trkErr_[ 6*N+it] +
                         kGain[ 9*N+it]*rotT01[it]*trkErr_[ 7*N+it] +
                         kGain[10*N+it]*trkErr_[ 8*N+it];
     newErr[10*N+it] = kGain[12*N+it]*rotT00[it]*trkErr_[ 0*N+it] +
                         kGain[12*N+it]*rotT01[it]*trkErr_[ 1*N+it] +
                         kGain[13*N+it]*trkErr_[ 3*N+it];
     newErr[11*N+it] = kGain[12*N+it]*rotT00[it]*trkErr_[ 1*N+it] +
                         kGain[12*N+it]*rotT01[it]*trkErr_[ 2*N+it] +
                         kGain[13*N+it]*trkErr_[ 4*N+it];
     newErr[12*N+it] = kGain[12*N+it]*rotT00[it]*trkErr_[ 3*N+it] +
                         kGain[12*N+it]*rotT01[it]*trkErr_[ 4*N+it] +
                         kGain[13*N+it]*trkErr_[ 5*N+it];
     newErr[13*N+it] = kGain[12*N+it]*rotT00[it]*trkErr_[ 6*N+it] +
                         kGain[12*N+it]*rotT01[it]*trkErr_[ 7*N+it] +
                         kGain[13*N+it]*trkErr_[ 8*N+it];
     newErr[14*N+it] = kGain[12*N+it]*rotT00[it]*trkErr_[10*N+it] +
                         kGain[12*N+it]*rotT01[it]*trkErr_[11*N+it] +
                         kGain[13*N+it]*trkErr_[12*N+it];
     newErr[15*N+it] = kGain[15*N+it]*rotT00[it]*trkErr_[ 0*N+it] +
                         kGain[15*N+it]*rotT01[it]*trkErr_[ 1*N+it] +
                         kGain[16*N+it]*trkErr_[ 3*N+it];
     newErr[16*N+it] = kGain[15*N+it]*rotT00[it]*trkErr_[ 1*N+it] +
                         kGain[15*N+it]*rotT01[it]*trkErr_[ 2*N+it] +
                         kGain[16*N+it]*trkErr_[ 4*N+it];
     newErr[17*N+it] = kGain[15*N+it]*rotT00[it]*trkErr_[ 3*N+it] +
                         kGain[15*N+it]*rotT01[it]*trkErr_[ 4*N+it] +
                         kGain[16*N+it]*trkErr_[ 5*N+it];
     newErr[18*N+it] = kGain[15*N+it]*rotT00[it]*trkErr_[ 6*N+it] +
                         kGain[15*N+it]*rotT01[it]*trkErr_[ 7*N+it] +
                         kGain[16*N+it]*trkErr_[ 8*N+it];
     newErr[19*N+it] = kGain[15*N+it]*rotT00[it]*trkErr_[10*N+it] +
                         kGain[15*N+it]*rotT01[it]*trkErr_[11*N+it] +
                         kGain[16*N+it]*trkErr_[12*N+it];
     newErr[20*N+it] = kGain[15*N+it]*rotT00[it]*trkErr_[15*N+it] +
                         kGain[15*N+it]*rotT01[it]*trkErr_[16*N+it] +
                         kGain[16*N+it]*trkErr_[17*N+it];     
 #pragma unroll
     for (int i = 0; i < 21; i++){
       trkErr_[ i*N+it] = trkErr_[ i*N+it] - newErr[ i*N+it];
     }
   }
   //
   return;                 
}
                  

auto sincos4 = [](const float x, float& sin, float& cos) {
   // Had this writen with explicit division by factorial.
   // The *whole* fitting test ran like 2.5% slower on MIC, sigh.
   const float x2 = x*x;
   cos  = 1.f - 0.5f*x2 + 0.04166667f*x2*x2;
   sin  = x - 0.16666667f*x*x2;
};

constexpr float kfact= 100/(-0.299792458*3.8112);
constexpr int Niter=5;

template <size_t N = 1>
void propagateToR(const MP6x6SF_ &inErr_, const MP6F_ &inPar_, const MP1I_ &inChg_, 
                  const MP3F_ &msP_, MP6x6SF_ &outErr_, MP6F_ &outPar_) {
  //aux objects  
  MP6x6F_ errorProp;
  MP6x6F_ temp;
  
  for (size_t it = 0; it < N; ++it) {
    //initialize erroProp to identity matrix
    //for (size_t i=0;i<6;++i) errorProp.data[bsize*PosInMtrx(i,i,6) + it] = 1.f; 
    errorProp[PosInMtrx(0,0,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(1,1,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(2,2,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(3,3,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(4,4,6, N) + it] = 1.0f;
    errorProp[PosInMtrx(5,5,6, N) + it] = 1.0f;
    //
    const auto xin = inPar_(iparX, it);
    const auto yin = inPar_(iparY, it);     
    const auto zin = inPar_(iparZ, it); 
    
    const auto iptin   = inPar_(iparIpt,   it);
    const auto phiin   = inPar_(iparPhi,   it);
    const auto thetain = inPar_(iparTheta, it); 
    //
    auto r0 = hipo(xin, yin);
    const auto k = inChg_[it]*kfact;//?
    
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
  }
  
  MultHelixProp<N>(errorProp, inErr_, temp);
  MultHelixPropTransp<N>(errorProp, temp, outErr_);  
  
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
#if defined(__NVCOMPILER_CUDA__)
   constexpr auto order = FieldOrder::P2R_TRACKBLK_EVENT_LAYER_MATIDX_ORDER;
#else
   constexpr auto order = FieldOrder::P2R_MATIDX_LAYER_TRACKBLK_EVENT_ORDER;
#endif
   using MPTRKAccessorTp = MPTRKAccessor<order>;
   using MPHITAccessorTp = MPHITAccessor<order>;

   MPHIT_* hit    = prepareHits(inputhits);
   MPTRK_* outtrk = (MPTRK_*) malloc(nevts*nb*sizeof(MPTRK_));

   gettimeofday(&timecheck, NULL);
   setup_start = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   auto trkNPtr = prepareTracksN<order>(inputtrk);
   std::unique_ptr<MPTRKAccessorTp> trkNaccPtr(new MPTRKAccessorTp(*trkNPtr));

   auto hitNPtr = prepareHitsN<order>(inputhits);
   std::unique_ptr<MPHITAccessorTp> hitNaccPtr(new MPHITAccessorTp(*hitNPtr));

   std::unique_ptr<MPTRK> outtrkNPtr(new MPTRK(ntrks, nevts));
   std::unique_ptr<MPTRKAccessorTp> outtrkNaccPtr(new MPTRKAccessorTp(*outtrkNPtr));

   gettimeofday(&timecheck, NULL);
   setup_stop = (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;

   printf("done preparing!\n");

   printf("Size of struct MPTRK trk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct MPTRK outtrk[] = %ld\n", nevts*nb*sizeof(MPTRK));
   printf("Size of struct struct MPHIT hit[] = %ld\n", nevts*nb*sizeof(MPHIT));

   auto policy = std::execution::par_unseq;

   auto wall_start = std::chrono::high_resolution_clock::now();

   for(int itr=0; itr<NITER; itr++) {

     const int outer_loop_range = nevts*nb;

     std::for_each(policy,
                   impl::counting_iterator(0),
                   impl::counting_iterator(outer_loop_range),
                   [=,&btrakcsAccessor    = *trkNaccPtr,
                      &bhitAccessor       = *hitNaccPtr,
                      &outtrakcsAccessor  = *outtrkNaccPtr] (const auto i) {
                     //  
                     MPTRK_ btracks;
                     MPTRK_ obtracks;
                     MPHIT_ bhits;   
 
                     for(int layer=0; layer<nlayer; ++layer) {  
                       //
                       btrakcsAccessor.load(btracks, i);
                       bhitAccessor.load(bhits, i, layer);
                       //
                       propagateToR<bsize>(btracks.cov, btracks.par, btracks.q, bhits.pos, obtracks.cov, obtracks.par);
                       KalmanUpdate<bsize>(obtracks.cov, obtracks.par, bhits.cov, bhits.pos);
                       //
                       outtrakcsAccessor.save(obtracks, i);
                     }
                   });
#if defined(__NVCOMPILER_CUDA__) 
      //convertTracks<order>(outtrk, outtrkNPtr.get());
#endif

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
       if((it+ie*ntrks)%100000==0) printf("iTrk = %i,  track (x,y,z,r)=(%.6f,%.6f,%.6f,%.6f) \n", it+ie*ntrks, x_,y_,z_,r_);
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


   delete [] hit;
   delete [] outtrk;


   return 0;
}
