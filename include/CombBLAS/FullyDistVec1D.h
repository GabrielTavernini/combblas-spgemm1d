#ifndef _FULLYDISTVEC1D_H_
#define _FULLYDISTVEC1D_H_

#include "CommGrid1D.h"
#include "MPIOp.h"
#include <vector>
#include <memory>
#include <numeric>

namespace combblas {

template <class IT, class NT, class DER>
class SpParMat1D;


template<class IT, class NT>
class FullyDistVec1D{
public: 
    FullyDistVec1D(IT glen, NT init_val);
    FullyDistVec1D(std::vector<IT> blocksizevec, NT init_val);
    FullyDistVec1D(const FullyDistVec1D<IT, NT> & rhs) = default;
    

    IT TotalLength() const ;
    int Owner(IT elemidx) const ;
    inline IT MyLocLength(){return arr_.size();}


    ///////////////////////
    // reduce
    ///////////////////////
    template <class _BinaryOperation>
    inline NT Reduce(_BinaryOperation __binary_op, NT identity) const
    {
        // std::accumulate returns identity for empty sequences
        NT localsum = std::accumulate( arr_.begin(), arr_.end(), identity, __binary_op);
        NT totalsum = identity;
        MPI_Allreduce( &localsum, &totalsum, 1, MPIType<NT>(), MPIOp<_BinaryOperation, NT>::op(), grid1d_->GetWorld());
        return totalsum;
    }


    ///////////////////////
    // opertaor overload
    ///////////////////////
    inline FullyDistVec1D<IT,NT> &  operator=(NT val) // assign fixed value
    {
        #pragma omp simd
        for(IT i=0; i < arr_.size(); ++i)
            arr_[i] = val;
        return *this;
    }
    inline FullyDistVec1D<IT,NT> & operator-=(const FullyDistVec1D<IT,NT> & rhs)
    {
        #pragma omp simd
        for(IT i=0; i < arr_.size(); ++i)
            arr_[i] -= rhs.arr_[i];
        return *this;
    }

    ///////////////////////
    // getter of data member
    ///////////////////////
    inline std::vector<IT> getblocksizevec() const { return blocksizevec_; }
    inline std::vector<IT> getblocksizeprefix() const { return blocksizeprefix_; }
    inline std::shared_ptr<CommGrid1D> getcommgrid() const { return grid1d_; }
    inline const std::vector<NT>& GetLocVec() const { return arr_; }
    ////////////////
    // Friends
    ////////////////
    template<class IU, class NU, class DER>
    friend class SpParMat1D;

    std::vector<NT> arr_;
    IT gidx_start;
    IT gidx_end;
    std::vector<IT> blocksizevec_;
    std::vector<IT> blocksizeprefix_;
    std::shared_ptr<CommGrid1D> grid1d_;

};

/////////////////////////////
// Constructor
/////////////////////////////
template<class IT, class NT>
FullyDistVec1D<IT,NT>::FullyDistVec1D(IT glen, NT init_val)
{
    this->grid1d_= std::make_shared<CommGrid1D>(MPI_COMM_WORLD);
    IT nprocs    = grid1d_->GetSize();
    IT myrank    = grid1d_->GetRank();
    IT blocksize = glen / nprocs;
    for(int i=0; i<nprocs-1; i++){
        blocksizevec_.push_back(blocksize);
    }
    blocksizevec_.push_back(glen - (nprocs-1)*blocksize);
    blocksizeprefix_ = std::vector<IT>(nprocs,0);
    std::partial_sum(blocksizevec_.begin(), blocksizevec_.end()-1, blocksizeprefix_.begin()+1);
    this->arr_.resize(blocksizevec_[myrank],init_val);
}

template<class IT, class NT>
FullyDistVec1D<IT,NT>::FullyDistVec1D(std::vector<IT> blocksizevec, NT init_val)
{
    grid1d_= std::make_shared<CommGrid1D>(MPI_COMM_WORLD);
    blocksizevec_ = blocksizevec;
    IT nprocs    = grid1d_->GetSize();
    IT myrank    = grid1d_->GetRank();
    blocksizeprefix_ = std::vector<IT>(nprocs,0);
    std::partial_sum(blocksizevec_.begin(), blocksizevec_.end()-1, blocksizeprefix_.begin()+1);
    arr_.resize(blocksizevec_[myrank],init_val);
}

template<class IT, class NT>
IT FullyDistVec1D<IT,NT>::TotalLength() const 
{
    IT totallength = 0;
    IT localsize = arr_.size();
    MPI_Allreduce(&localsize, &totallength, 1, MPIType<IT>(), MPI_SUM, MPI_COMM_WORLD);
    return totallength;
}
template<class IT, class NT>
int FullyDistVec1D<IT,NT>::Owner(IT elemidx) const 
{
    int nblocks=blocksizeprefix_.size();
    int owner = 0;
    while(owner <= nblocks-1 && blocksizeprefix_[owner] <= elemidx) owner++;
    owner-=1;
    return owner;
}



}

#endif