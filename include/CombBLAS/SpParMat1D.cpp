/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */



#include "SpParMat1D.h"
#include "CombBLAS/CommGrid.h"
#include "CombBLAS/CommGrid1D.h"
#include "CombBLAS/Deleter.h"
#include "CombBLAS/SpCCols.h"
#include "CombBLAS/SpDCCols.h"
#include "CombBLAS/SpDefs.h"
#include "CombBLAS/SpHelper.h"
#include "CombBLAS/SpMat.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/SpTuples.h"
#include "CombBLAS/csc.h"
#include "CombBLAS/dcsc.h"
#include "CombBLAS/mtSpGEMM.h"
#include "CommGrid1D.h"
#include "ParFriends.h"
#include "Operations.h"
#include "FileHeader.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <ios>
#include <memory>
#include <numeric>
#include <omp.h>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
extern "C" {
#include "mmio.h"
}
#include <sys/types.h>
#include <sys/stat.h>

#include <mpi.h>
#include <fstream>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include "CombBLAS/CombBLAS.h"
#include <unistd.h>
#include <memory.h>
using std::cout;
using std::endl;
using std::shared_ptr;
using std::make_shared;
using std::tuple;
using std::string;
using std::vector;

namespace combblas
{

template <class IT, class NT, class DER>
SpParMat1D<IT,NT,DER>::SpParMat1D (std::shared_ptr<CommGrid1D> grid)
{
    grid1d_ = grid;
    spSeq_ = nullptr;
}

template <class IT, class NT, class DER>
SpParMat1D<IT,NT,DER>::SpParMat1D (DER * userseq, std::shared_ptr<CommGrid1D> grid)
{
    grid1d_ = grid;
    spSeq_ = shared_ptr<DER>(userseq);
    int nprocs = grid1d_->GetSize();
    int myrank = grid1d_->GetRank();
    IT localm = spSeq_->getnrow();
    // localm is actaully global m
    fillblocksizevec(rowblocksizevec_, nprocs, localm);
    // get column blocksize information.
    blocksizevec_.resize(nprocs);
    IT localn = spSeq_->getncol(); // here we must gather localn through all processes in order to generate prefix.
    MPI_Allgather(&localn, 1, MPI_LONG_LONG, blocksizevec_.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
    generateblocksizeprefix(blocksizevec_,blocksizeprefix_);
    generateblocksizeprefix(rowblocksizevec_,rowblocksizeprefix_);
}

template <class IT, class NT, class DER>
SpParMat1D<IT,NT,DER>::SpParMat1D(IT gnrows, IT gncols){
    grid1d_ = make_shared<CommGrid1D>(MPI_COMM_WORLD);
    int myrank = grid1d_->GetRank();
    int nprocs = grid1d_->GetSize();
    fillblocksizevec(blocksizevec_, nprocs, gncols);
    fillblocksizevec(rowblocksizevec_, nprocs, gnrows);
    generateblocksizeprefix(blocksizevec_,blocksizeprefix_);
    generateblocksizeprefix(rowblocksizevec_,rowblocksizeprefix_);
    SpTuples<IT, NT> tuples = SpTuples<IT, NT>(0, gnrows, blocksizevec_[myrank], nullptr, false);
    spSeq_ = make_shared<DER>(tuples, false);
}

template <class IT, class NT, class DER>
SpParMat1D<IT,NT,DER>::SpParMat1D(IT gnrows, IT gncols, DER * userseq){
    grid1d_ = make_shared<CommGrid1D>(MPI_COMM_WORLD);
    int myrank = grid1d_->GetRank();
    int nprocs = grid1d_->GetSize();
    fillblocksizevec(blocksizevec_, nprocs, gncols);
    fillblocksizevec(rowblocksizevec_, nprocs, gnrows);
    generateblocksizeprefix(blocksizevec_,blocksizeprefix_);
    generateblocksizeprefix(rowblocksizevec_,rowblocksizeprefix_);
    spSeq_ = shared_ptr<DER>(userseq);
    IT localcol = spSeq_->getncol();
    IT localrow = spSeq_->getnrow();
}

template <class IT, class NT, class DER>
SpParMat1D<IT,NT,DER>::SpParMat1D (
    const IT gnrows, const IT gncols, const std::vector<IT> blocksizevec, DER * userseq, 
    std::shared_ptr<CommGrid1D> grid1d, const std::vector<IT> rowblocksizevec)
{
    // double t0 = MPI_Wtime();
    grid1d_ = grid1d;
    int nprocs = grid1d_->GetSize();
    int myrank = grid1d_->GetRank();
    // double kkt0 = MPI_Wtime();
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,part1.1,create grid"<<kkt0-t0<<std::endl;
    if(blocksizevec.size() != 0){
        // user provided
        blocksizevec_ = blocksizevec;
    }else{
        // default seperation
        fillblocksizevec(blocksizevec_,nprocs, gncols);
    }
    // double kkt1 = MPI_Wtime();
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,part1.2,fill blocksize"<<kkt1-kkt0<<std::endl;
    if(rowblocksizevec.size() != 0){
        // user provided
        rowblocksizevec_ = rowblocksizevec;
    }else{
        // default seperation
        fillblocksizevec(rowblocksizevec_,nprocs, gnrows);
    }
    // double tt0 = MPI_Wtime();
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,part1.3,fill rowblocksize"<<tt0-kkt1<<std::endl;
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,part1,get blocksize"<<tt0-t0<<std::endl;
    generateblocksizeprefix(blocksizevec_,blocksizeprefix_);
    generateblocksizeprefix(rowblocksizevec_,rowblocksizeprefix_);
    // double tt1 = MPI_Wtime();
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,part2,get prefix"<<tt1-tt0<<std::endl;
    SpTuples<IT, NT> tuples = SpTuples<IT, NT>(0, gnrows, blocksizevec_[myrank], nullptr, false);
    spSeq_ = make_shared<DER>(tuples, false);
    // double t1 = MPI_Wtime();
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,part3,create tuples"<<t1-tt1<<std::endl;
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,grow,gcol,bsize,rbs,"<<t1-t0<<std::endl;
    spSeq_ = shared_ptr<DER>(userseq);
}

template <class IT, class NT, class DER>
SpParMat1D<IT,NT,DER>::SpParMat1D(IT gnrows, IT gncols, std::vector<IT> blocksizevec,std::vector<IT> rowblocksizevec){
    // double t0 = MPI_Wtime();
    grid1d_ = make_shared<CommGrid1D>(MPI_COMM_WORLD);
    int nprocs = grid1d_->GetSize();
    int myrank = grid1d_->GetRank();
    // double kkt0 = MPI_Wtime();
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,part1.1,create grid"<<kkt0-t0<<std::endl;
    if(blocksizevec.size() != 0){
        // user provided
        blocksizevec_ = blocksizevec;
    }else{
        // default seperation
        fillblocksizevec(blocksizevec_,nprocs, gncols);
    }
    // double kkt1 = MPI_Wtime();
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,part1.2,fill blocksize"<<kkt1-kkt0<<std::endl;
    if(rowblocksizevec.size() != 0){
        // user provided
        rowblocksizevec_ = rowblocksizevec;
    }else{
        // default seperation
        fillblocksizevec(rowblocksizevec_,nprocs, gnrows);
    }
    // double tt0 = MPI_Wtime();
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,part1.3,fill rowblocksize"<<tt0-kkt1<<std::endl;
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,part1,get blocksize"<<tt0-t0<<std::endl;
    generateblocksizeprefix(blocksizevec_,blocksizeprefix_);
    generateblocksizeprefix(rowblocksizevec_,rowblocksizeprefix_);
    // double tt1 = MPI_Wtime();
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,part2,get prefix"<<tt1-tt0<<std::endl;
    SpTuples<IT, NT> tuples = SpTuples<IT, NT>(0, gnrows, blocksizevec_[myrank], nullptr, false);
    spSeq_ = make_shared<DER>(tuples, false);
    // double t1 = MPI_Wtime();
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,part3,create tuples"<<t1-tt1<<std::endl;
    // if(grid1d_->GetRank() == 0)std::cerr<<"callconst,grow,gcol,bsize,rbs,"<<t1-t0<<std::endl;
}

template <class IT, class NT, class DER>
SpParMat1D<IT,NT,DER>::SpParMat1D(IT gnrows, IT gncols, std::vector<IT> blocksizevec, DER * userseq, std::vector<IT> rowblocksizevec)
:SpParMat1D(gnrows, gncols,blocksizevec, rowblocksizevec){
    spSeq_ = shared_ptr<DER>(userseq);
}

template <class IT, class NT, class DER>
SpParMat1D< IT,NT,DER >::SpParMat1D (const SpParMat1D < IT,NT,DER > & A1D)
{
    this->blocksizevec_ = A1D.getblocksizevec();
    this->blocksizeprefix_ = A1D.getblocksizeprefix();
    this->rowblocksizevec_ = A1D.getrowblocksizevec();
    this->rowblocksizeprefix_ = A1D.getrowblocksizeprefix();
    this->grid1d_ = make_shared<CommGrid1D>(A1D.getgrid()->GetWorld());
    this->spSeq_ = make_shared<DER>(*A1D.seqptr());
}


template <class IT, class NT, class DER>
SpParMat1D<IT,NT,DER>::SpParMat1D(const SpParMat < IT,NT,DER > & A2D, const std::vector<IT> blocksizevec, const std::vector<IT> rowblocksizevec){
    /*global row and col information.*/
    IT gnrows = A2D.getnrow();
    IT gncols = A2D.getncol();
    /*construct blocksizevec and prefix.*/
    grid1d_ = make_shared<CommGrid1D>(MPI_COMM_WORLD);
    int myrank = grid1d_->GetRank();
    int nprocs = grid1d_->GetSize();
    
    if(blocksizevec.size() != 0){
        // user provided
        blocksizevec_ = blocksizevec;
        IT blocksum = 0;
        for(auto x : blocksizevec) blocksum += x;
        if(blocksum != gncols) {
            if(myrank==0)std::cerr<<"blocksizevec doen't equal total column size!!"<<std::endl;
            exit(0);
        }
    }else{
        // default seperation
        fillblocksizevec(blocksizevec_, nprocs, gncols);
    }
    
    
    if(rowblocksizevec.size() != 0){
        // user provided
        rowblocksizevec_ = rowblocksizevec;
        IT blocksum = 0;
        for(auto x : rowblocksizevec) blocksum += x;
        if(blocksum != gnrows) {
            if(myrank==0)std::cerr<<"rowblocksizevec doen't equal total row size!!"<<std::endl;
            exit(0);
        }
    }else{
        // default seperation
        fillblocksizevec(rowblocksizevec_, nprocs, gnrows);
    }
    generateblocksizeprefix(blocksizevec_,blocksizeprefix_);
    generateblocksizeprefix(rowblocksizevec_,rowblocksizeprefix_);
    /*redistribute the data from 2d to 1d.*/
    auto commGrid2D = A2D.getcommgrid();
    int pr2d = commGrid2D->GetGridRows();
    int pc2d = commGrid2D->GetGridCols();
    int rowrank2d = commGrid2D->GetRankInProcRow();
    int colrank2d = commGrid2D->GetRankInProcCol();
    IT m_perproc2d = gnrows / pr2d;
    IT n_perproc2d = gncols / pc2d;
    DER* spSeq2d = A2D.seqptr(); // local submatrix
    IT localRowStart2d = colrank2d * m_perproc2d; // first row in this process
    IT localColStart2d = rowrank2d * n_perproc2d; // first col in this process
    IT lcol1d;
    std::vector< std::vector< std::tuple<IT,IT, NT> > > sendTuples (nprocs);
    for(typename DER::SpColIter colit = spSeq2d->begcol(); colit != spSeq2d->endcol(); ++colit)
    {
        IT gcol = colit.colid() + localColStart2d;
        int owner = Owner(gcol, lcol1d);
        for(typename DER::SpColIter::NzIter nzit = spSeq2d->begnz(colit); nzit != spSeq2d->endnz(colit); ++nzit)
        {
            IT grow = nzit.rowid() + localRowStart2d;
            NT val = nzit.value();
            sendTuples[owner].push_back(std::make_tuple(grow, lcol1d, val));
        }
    }
    IT datasize;
    std::tuple<IT,IT,NT>* recvTuples = SpParHelper::ExchangeDataGeneral(sendTuples, commGrid2D->GetWorld(), datasize);
    SpTuples<IT, NT>spTuples(datasize, gnrows, blocksizevec_[myrank], recvTuples);
    spSeq_ = make_shared<DER>(spTuples, false);
}


template <class IT, class NT, class DER>
SpParMat1D<IT, NT, DER>::~SpParMat1D(){}



template <class IT, class NT, class DER>
void SpParMat1D<IT, NT, DER>::Transpose(std::vector<IT> blocksize, std::vector<IT> rowblocksize){
    // #if MAT1D_DEBUG_LEVEL >= 1
    // MPI_Timer timer;
    // timer.start("SpGEMM1D-Transpose");
    // #endif
    int nprocs = grid1d_->GetSize();
    int myrank = grid1d_->GetRank();
    // dimension check
    std::vector<IT> oldcolblock = blocksizevec_;
    std::vector<IT> oldrowblock = rowblocksizevec_;
    std::vector<IT> oldcolblockprefix = blocksizeprefix_;
    std::vector<IT> oldrowblockprefix = rowblocksizeprefix_;
    if(blocksize.size() == 0){
        blocksizevec_ = oldrowblock;
    }else{
        IT sum = std::accumulate(blocksize.begin(),blocksize.end(),(IT)0);
        if(sum != spSeq_->getnrow()){
            if(myrank == 0) std::cerr << "Transpose Input Error: blocksize dim check fails." << std::endl;
            exit(0);
        }
        blocksizevec_ = blocksize;
    }
    if(rowblocksize.size() == 0){
        rowblocksizevec_ = oldcolblock;
    }else{
        IT sum = std::accumulate(rowblocksize.begin(),rowblocksize.end(),(IT)0);
        IT gncols = getncol();
        if(sum != gncols){
            if(myrank == 0) std::cerr << "Transpose Input Error: rowblocksize dim check fails." << std::endl;
            exit(0);
        }
        rowblocksizevec_ = rowblocksize;
    }
    generateblocksizeprefix(blocksizevec_,blocksizeprefix_);
    generateblocksizeprefix(rowblocksizevec_,rowblocksizeprefix_);
    // transpose
    std::vector<std::vector<std::tuple<IT,IT,NT>>> sendbtuples(nprocs);
    for(typename DER::SpColIter colit = spSeq_->begcol(); colit != spSeq_->endcol(); ++colit)
    {
        IT gcol = colit.colid() + oldcolblockprefix[myrank];
        for(typename DER::SpColIter::NzIter nzit = spSeq_->begnz(colit); nzit != spSeq_->endnz(colit); ++nzit)
        {
            IT grow = nzit.rowid();
            IT newgrow = gcol; // transposed new index
            IT newgcol = grow;
            // now blocksizeprefix_ is the new one, we can use it to decide the new owner
            // grow is used because now grow will be new gcol 
            int owner = 0;
            while(owner <= nprocs-1 && blocksizeprefix_[owner] <= newgcol) owner++;
            owner--;
            sendbtuples[owner].push_back(std::make_tuple(newgrow,newgcol-blocksizeprefix_[owner],nzit.value()));
        }
    }
    IT datasize;
    std::tuple<IT,IT,NT> *recvoffbtuples = 
        SpParHelper::ExchangeDataGeneral(sendbtuples, MPI_COMM_WORLD, datasize);
    IT nrows = std::accumulate(rowblocksizevec_.begin(), rowblocksizevec_.end(),(IT)0);
    SpTuples<IT, NT>tmpsptuples = SpTuples<IT, NT>(datasize, nrows, blocksizevec_[myrank], recvoffbtuples);
    spSeq_ = make_shared<DER>(tmpsptuples, false); // cleared recvoffbtuples;
    // #if   MAT1D_DEBUG_LEVEL == 1
    // timer.stop("SpGEMM1D-Transpose");
    // #elif MAT1D_DEBUG_LEVEL == 2
    // timer.stop("SpGEMM1D-Transpose",true);
    // #endif
}

template<class IT, class NT, class DER>
void SpParMat1D<IT, NT, DER>::GetVertexWeight(std::vector<int64_t> & vwgt, int permutetype)
{
    // permutetype = 2: none, permutetype = 3: nnz, permutetype = 4: flops
    vwgt = std::vector<int64_t>(spSeq_->getncol(),1);
    if(permutetype == 2) return;
    Dcsc<IT, NT> * dcscptr = spSeq_->getDCSC();
    for(IT ji=0; ji < dcscptr->nzc; ji++)
    {
        IT jid = dcscptr->jc[ji];
        IT nnzpercol = dcscptr->cp[jid+1] - dcscptr->cp[jid];
        if     (permutetype == 3)    vwgt[jid] += nnzpercol;
        else if(permutetype == 4)    vwgt[jid] += (nnzpercol) * (nnzpercol);
    }
}

template <class IT, class NT, class DER>
void SpParMat1D<IT, NT, DER>::Redistrube(std::vector<IT> blocksize, std::vector<IT> rowblocksize)
{
    int myrank = grid1d_->GetRank();
    int nprocs = grid1d_->GetSize();
    
    if(blocksize.size() == 0 || rowblocksize.size() == 0){
        if(myrank==0) std::cerr << "Redistriute requires you to input non-empty blocksize and rowblocksize. " << std::endl;
        return;
    }
    std::vector<IT> oldblocksize(blocksizevec_.begin(), blocksizevec_.end());
    std::vector<IT> oldrowblocksize(rowblocksizevec_.begin(), rowblocksizevec_.end());
    std::vector<IT> oldblocksizeprefix(blocksizeprefix_.begin(), blocksizeprefix_.end());
    std::vector<IT> oldrowblocksizeprefix(rowblocksizeprefix_.begin(), rowblocksizeprefix_.end());
    blocksizevec_ = blocksize;
    rowblocksizevec_ = rowblocksize;
    generateblocksizeprefix(blocksizevec_, blocksizeprefix_);
    generateblocksizeprefix(rowblocksizevec_, rowblocksizeprefix_);
    IT lcol1d;
    std::vector< std::vector< std::tuple<IT,IT, NT> > > sendTuples (nprocs);
    for(typename DER::SpColIter colit = spSeq_->begcol(); colit != spSeq_->endcol(); ++colit)
    {
        IT gcol = colit.colid() + oldblocksizeprefix[myrank]; // global column id
        int nblocks=nprocs;
        int owner = 0;
        while(owner <= nblocks-1 && blocksizeprefix_[owner] <= gcol) owner++; // in new blocksizeprefix view
        owner-=1;
        lcol1d = gcol - blocksizeprefix_[owner];
        for(typename DER::SpColIter::NzIter nzit = spSeq_->begnz(colit); nzit != spSeq_->endnz(colit); ++nzit)
        {
            IT grow = nzit.rowid();
            NT val = nzit.value();
            sendTuples[owner].push_back(std::make_tuple(grow, lcol1d, val));
        }
    }
    IT gnrows = getnrow();
    IT datasize;
    std::tuple<IT,IT,NT>* recvTuples = SpParHelper::ExchangeDataGeneral(sendTuples, grid1d_->GetWorld(), datasize);
    SpTuples<IT, NT>spTuples(datasize, gnrows, blocksize[myrank], recvTuples);
    spSeq_ = make_shared<DER>(spTuples, false);
}

template <class IT, class NT,class DER>
IT SpParMat1D< IT,NT,DER >::getncol() const
{
    IT totalcols = 0;
    IT localcols = spSeq_->getncol();
    MPI_Allreduce( &localcols, &totalcols, 1, MPIType<IT>(), MPI_SUM, grid1d_->GetWorld());
    return totalcols;
}

template <class IT, class NT,class DER>
IT SpParMat1D< IT,NT,DER >::getnrow() const
{
    IT totalrows = 0;
    totalrows = spSeq_->getnrow();
    return totalrows;
}

template <class IT, class NT,class DER>
IT SpParMat1D<IT,NT,DER>::getnnz() const 
{
    IT totalnnz = 0;
    IT localnnz = spSeq_->getnnz();
    MPI_Allreduce( &localnnz, &totalnnz, 1, MPIType<IT>(), MPI_SUM, grid1d_->GetWorld());
    return totalnnz;
}

template <class IT, class NT,class DER>
IT SpParMat1D<IT,NT,DER>::getnzc() const 
{
    IT totalnzc = 0;
    IT localnzc = spSeq_->getnzc();
    MPI_Allreduce( &localnzc, &totalnzc, 1, MPIType<IT>(), MPI_SUM, grid1d_->GetWorld());
    return totalnzc;
}

template <class IT, class NT, class DER>
SpParMat1D< IT,NT,DER > & SpParMat1D< IT,NT,DER >::operator+=(const SpParMat1D< IT,NT,DER > & rhs)
{
    if(this != &rhs)
    {
        if(*grid1d_ == *rhs.grid1d_)
        {
            ( *(spSeq_) ) += ( *(rhs.seqptr()) );
        }
        else
        {
            std::cout << "Grids are not comparable for parallel addition (A+B)" << std::endl; 
        }
    }
    else
    {
        std::cout<< "Missing feature (A+A): Use multiply with 2 instead !"<<std::endl;	
    }
    return *this;
}

template <class IT, class NT, class DER>
bool SpParMat1D<IT,NT,DER>::operator== (const SpParMat1D<IT,NT,DER> & rhs) const
{
	int local = static_cast<int>((*spSeq_) == (*(rhs.seqptr())));
	int whole = 1;
	MPI_Allreduce( &local, &whole, 1, MPI_INT, MPI_BAND, grid1d_->GetWorld());
	return static_cast<bool>(whole);	
}

template <class IT, class NT, class DER>
void SpParMat1D<IT,NT,DER>::SubsRef_SR(
    const std::vector<IT> & fullrowperm, const std::vector<IT> & fullcolperm)
{
    std::vector<IT> forwardrow(fullrowperm.size()), forwardcol(fullcolperm.size());
    for(IT i=0; i<fullrowperm.size(); i++){
        forwardrow[fullrowperm[i]] = i;
    }
    for(IT i=0; i<fullcolperm.size(); i++){
        forwardcol[fullcolperm[i]] = i;
    }
    int nprocs = grid1d_->GetSize();
    int myrank = grid1d_->GetRank();
    std::vector< std::vector< std::tuple<IT,IT, NT> > > sendTuples (nprocs);
    for(typename DER::SpColIter colit = spSeq_->begcol(); colit != spSeq_->endcol(); ++colit)
    {
        IT gcol = colit.colid() + blocksizeprefix_[myrank];
        for(typename DER::SpColIter::NzIter nzit = spSeq_->begnz(colit); nzit != spSeq_->endnz(colit); ++nzit)
        {
            IT grow = nzit.rowid();
            IT newgcol = forwardcol[gcol];
            IT newgrow = forwardrow[grow];
            IT lcol;
            int owner = Owner(newgcol, lcol);
            NT val = nzit.value();
            sendTuples[owner].push_back(std::make_tuple(newgrow, lcol, val));
        }
    }
    IT datasize;
    std::tuple<IT,IT,NT>* recvTuples = 
        SpParHelper::ExchangeDataGeneral(sendTuples, grid1d_->GetWorld(), datasize);
    IT gnrows = spSeq_->getnrow();
    SpTuples<IT, NT>spTuples(datasize, gnrows, blocksizevec_[myrank], recvTuples);
    spSeq_ = make_shared<DER>(spTuples, false);
}
template <class IT, class NT, class DER>
void SpParMat1D<IT,NT,DER>::SubsRef_SR(
    const std::vector<IT> & fullrowperm, const std::vector<IT> & fullcolperm,
    const std::vector<IT> & blocksize, const std::vector<IT> & rowblocksize){
    std::vector<IT> forwardrow(fullrowperm.size()), forwardcol(fullcolperm.size());
    for(IT i=0; i<fullrowperm.size(); i++){
        forwardrow[fullrowperm[i]] = i;
    }
    for(IT i=0; i<fullcolperm.size(); i++){
        forwardcol[fullcolperm[i]] = i;
    }
    int nprocs = grid1d_->GetSize();
    int myrank = grid1d_->GetRank();
    // update old blocksize with new one
    std::vector<IT> oldblocksize(blocksizevec_.begin(), blocksizevec_.end());
    std::vector<IT> oldrowblocksize(rowblocksizevec_.begin(), rowblocksizevec_.end());
    std::vector<IT> oldblocksizeprefix(blocksizeprefix_.begin(), blocksizeprefix_.end());
    std::vector<IT> oldrowblocksizeprefix(rowblocksizeprefix_.begin(), rowblocksizeprefix_.end());
    blocksizevec_ = blocksize;
    rowblocksizevec_ = rowblocksize;
    generateblocksizeprefix(blocksizevec_, blocksizeprefix_);
    generateblocksizeprefix(rowblocksizevec_, rowblocksizeprefix_);
    std::vector< std::vector< std::tuple<IT,IT, NT> > > sendTuples (nprocs);
    for(typename DER::SpColIter colit = spSeq_->begcol(); colit != spSeq_->endcol(); ++colit)
    {
        IT gcol = colit.colid() + oldblocksizeprefix[myrank];
        for(typename DER::SpColIter::NzIter nzit = spSeq_->begnz(colit); nzit != spSeq_->endnz(colit); ++nzit)
        {
            IT grow = nzit.rowid();
            IT newgcol = forwardcol[gcol];
            IT newgrow = forwardrow[grow];
            // int owner = Owner(newgcol, lcol);
            int nblocks=nprocs;
            int owner = 0;
            while(owner <= nblocks-1 && blocksizeprefix_[owner] <= newgcol) owner++; // in new blocksizeprefix view
            owner-=1;
            IT lcol1d = newgcol - blocksizeprefix_[owner];
            NT val = nzit.value();
            sendTuples[owner].push_back(std::make_tuple(newgrow, lcol1d, val));
        }
    }
    IT datasize;
    std::tuple<IT,IT,NT>* recvTuples = 
        SpParHelper::ExchangeDataGeneral(sendTuples, grid1d_->GetWorld(), datasize);
    IT gnrows = spSeq_->getnrow();
    SpTuples<IT, NT>spTuples(datasize, gnrows, blocksizevec_[myrank], recvTuples);
    spSeq_ = make_shared<DER>(spTuples, false);
}

template <class IT, class NT, class DER>
std::vector<IT> SpParMat1D< IT,NT,DER >::BlockwiseNNZanalysis(std::string filename){
    int myrank = grid1d_->GetRank();
    int nprocs = grid1d_->GetSize();
    std::vector<std::vector<IT>> ret(nprocs,std::vector<IT>(nprocs,0));
    std::vector< std::vector< std::tuple<IT,IT, NT> > > sendTuples (nprocs);
    for(typename DER::SpColIter colit = spSeq_->begcol(); colit != spSeq_->endcol(); ++colit)
    {
        IT gcol = colit.colid() + blocksizeprefix_[myrank];
        for(typename DER::SpColIter::NzIter nzit = spSeq_->begnz(colit); nzit != spSeq_->endnz(colit); ++nzit)
        {
            IT grow = nzit.rowid();
            IT lcol;
            auto idx = BlockIndex(grow, gcol);
            NT val = nzit.value();
            ret[idx.first][idx.second]++;
        }
    }
    std::vector<IT> flatret;
    SpHelper::flatten(flatret, ret);
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    SpParHelper::AllreduceVector(flatret, MPI_SUM, comm);
    if(myrank==0){
        std::ofstream os(filename);
        if(os.is_open()){
            for(int i=0; i<nprocs; i++){
                for(int j=0; j<nprocs; j++){
                    os << std::setw(7) << flatret[i*16+j] << " ";
                }
                os << std::endl;
            }
        }
    }
    return flatret;
}


template <class IT, class NT, class DER>
template <typename NNT,typename NDER>
SpParMat1D<IT,NT,DER>::operator SpParMat1D<IT,NNT,NDER> () const
{
    if(grid1d_->GetRank()==0)std::cerr<<"using operator"<<std::endl;
    NDER * convert = new NDER(*spSeq_);
    exit(0);//TODO: Yuxi: fixed this.
    return SpParMat1D<IT,NNT,NDER> (convert, grid1d_);
}

//! Change index type as well
template <class IT, class NT, class DER>
template <typename NIT, typename NNT,typename NDER>
SpParMat1D<IT,NT,DER>::operator SpParMat1D<NIT,NNT,NDER> () const
{
    if(grid1d_->GetRank()==0)std::cerr<<"using operator"<<std::endl;
    if(std::is_floating_point<NT>()){
        if(grid1d_->GetRank()==0)std::cerr<<"NT is double"<<std::endl;
    }
    NDER * convert = new NDER(*spSeq_);
    exit(0);//TODO: Yuxi: fixed this.
    return SpParMat1D<NIT,NNT,NDER> (NULL, grid1d_);
}


/****************************************************************************/
/********************* SUBMATRIX SELECTION **********************************/
/****************************************************************************/



template <class IT, class NT, class DER>
template<class HANDLER>
void SpParMat1D< IT,NT,DER >::ParallelWriteMM(const std::string filename, bool onebased, HANDLER handler)
{
    int myrank = grid1d_->GetRank();
    int nprocs = grid1d_->GetSize();
    IT totalm = getnrow();
    IT totaln = getncol();
    IT totnnz = getnnz();
    if(myrank == 0) SpHelper::RemoveExistingFile(filename);
    MPI_Barrier(MPI_COMM_WORLD);
    std::stringstream ss;
    if(myrank == 0)
    {
        ss << "%%MatrixMarket matrix coordinate real general" << std::endl;
        ss << totalm << " " << totaln << " " << totnnz << std::endl;
    }
    
    IT entries =  spSeq_->getnnz();
    IT sizeuntil = 0;
    MPI_Exscan( &entries, &sizeuntil, 1, MPIType<IT>(), MPI_SUM, grid1d_->GetWorld() );
    if(myrank == 0) sizeuntil = 0;    // because MPI_Exscan says the recvbuf in process 0 is undefined
    
    IT roffset = 0;
    IT coffset = blocksizeprefix_[myrank];
    if(onebased)
    {
        roffset += 1;    // increment by 1
        coffset += 1;
    }
    
    for(typename DER::SpColIter colit = spSeq_->begcol(); colit != spSeq_->endcol(); ++colit)    // iterate over nonempty subcolumns
    {
        for(typename DER::SpColIter::NzIter nzit = spSeq_->begnz(colit); nzit != spSeq_->endnz(colit); ++nzit)
        {
            IT glrowid = nzit.rowid() + roffset;
            IT glcolid = colit.colid() + coffset;
            ss << glrowid << '\t';
            ss << glcolid << '\t';
            handler.save(ss, nzit.value(), glrowid, glcolid);
            ss << '\n';
        }
    }
    std::string text = ss.str();

    int64_t * bytes = new int64_t[nprocs];
    bytes[myrank] = text.size();
    MPI_Allgather(MPI_IN_PLACE, 1, MPIType<int64_t>(), bytes, 1, MPIType<int64_t>(), grid1d_->GetWorld());
    int64_t bytesuntil = std::accumulate(bytes, bytes+myrank, static_cast<int64_t>(0));
    int64_t bytestotal = std::accumulate(bytes, bytes+nprocs, static_cast<int64_t>(0));


    MPI_File thefile;
    MPI_File_open(grid1d_->GetWorld(), (char*) filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &thefile) ;
    int mpi_err = MPI_File_set_view(thefile, bytesuntil, MPI_CHAR, MPI_CHAR, (char*)"external32", MPI_INFO_NULL);
    if (mpi_err == 51) {
        // external32 datarep is not supported, use native instead
        MPI_File_set_view(thefile, bytesuntil, MPI_CHAR, MPI_CHAR, (char*)"native", MPI_INFO_NULL);
    }
 
    int64_t batchSize = 256 * 1024 * 1024;
    size_t localfileptr = 0;
    int64_t remaining = bytes[myrank];
    int64_t totalremaining = bytestotal;
    
    while(totalremaining > 0)
    {
    #ifdef COMBBLAS_DEBUG
        if(myrank == 0)
            std::cout << "Remaining " << totalremaining << " bytes to write in aggregate" << std::endl;
    #endif
        MPI_Status status;
        int curBatch = std::min(batchSize, remaining);
        MPI_File_write_all(thefile, text.c_str()+localfileptr, curBatch, MPI_CHAR, &status);
        int count;
        MPI_Get_count(&status, MPI_CHAR, &count); // known bug: https://github.com/pmodels/mpich/issues/2332
        assert( (curBatch == 0) || (count == curBatch) ); // count can return the previous/wrong value when 0 elements are written
        localfileptr += curBatch;
        remaining -= curBatch;
        MPI_Allreduce(&remaining, &totalremaining, 1, MPIType<int64_t>(), MPI_SUM, grid1d_->GetWorld());
    }
    MPI_File_close(&thefile);
    
    delete [] bytes;
}

template <class IT, class NT, class DER>
void SpParMat1D< IT,NT,DER >::ParallelReadMM(const string filename, const std::string blocksizevecfilename, bool onebased){
    int32_t type = -1;
    int32_t symmetric = 0;
    int64_t nrows, ncols, nonzeros;
    int64_t linesread = 0;
    
    FILE *f;
    int myrank = grid1d_->GetRank();
    int nprocs = grid1d_->GetSize();
    if(myrank == 0)
    {
        MM_typecode matcode;
        if ((f = fopen(filename.c_str(), "r")) == NULL)
        {
            printf("COMBBLAS: Matrix-market file %s can not be found\n", filename.c_str());
            MPI_Abort(MPI_COMM_WORLD, NOFILE);
        }
        if (mm_read_banner(f, &matcode) != 0)
        {
            printf("Could not process Matrix Market banner.\n");
            exit(1);
        }
        linesread++;
        
        if (mm_is_complex(matcode))
        {
            printf("Sorry, this application does not support complext types");
            printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        }
        else if(mm_is_real(matcode))
        {
            // std::cout << "Matrix is Float" << std::endl;
            type = 0;
        }
        else if(mm_is_integer(matcode))
        {
            std::cout << "Matrix is Integer" << std::endl;
            type = 1;
        }
        else if(mm_is_pattern(matcode))
        {
            std::cout << "Matrix is Boolean" << std::endl;
            type = 2;
        }
        if(mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
        {
            std::cout << "Matrix is symmetric" << std::endl;
            symmetric = 1;
        }
        int ret_code;
        if ((ret_code = mm_read_mtx_crd_size(f, &nrows, &ncols, &nonzeros, &linesread)) !=0)  // ABAB: mm_read_mtx_crd_size made 64-bit friendly
            exit(1);
    }
    MPI_Bcast(&type, 1, MPI_INT, 0, grid1d_->commWorld);
    MPI_Bcast(&symmetric, 1, MPI_INT, 0, grid1d_->commWorld);
    MPI_Bcast(&nrows, 1, MPIType<int64_t>(), 0, grid1d_->commWorld);
    MPI_Bcast(&ncols, 1, MPIType<int64_t>(), 0, grid1d_->commWorld);
    MPI_Bcast(&nonzeros, 1, MPIType<int64_t>(), 0, grid1d_->commWorld);
    
    // now we have nrows and ncols, generate blocksize 
    if(blocksizevecfilename == ""){
        if (ncols < nprocs){
            if(myrank==0) std::cerr << "ncol " << ncols << " is too small for nprocs " << nprocs << std::endl;
            exit(0);
        }
        fillblocksizevec(blocksizevec_, nprocs, ncols);
        fillblocksizevec(rowblocksizevec_,nprocs, nrows);
    }else{
        blocksizevec_ = SpHelper::ReadIntegersFromFile(blocksizevecfilename);
        IT sum = std::accumulate(blocksizevec_.begin(),blocksizevec_.end(),(IT)0);
        if(sum != ncols){
            if(myrank == 0) std::cerr << "dimension in " << blocksizevecfilename << " summation doesn't equal " << ncols << 
            ", as stated in the " << filename << std::endl;
            exit(0);
        }
        fillblocksizevec(rowblocksizevec_, nprocs, nrows); 
    }
    generateblocksizeprefix(blocksizevec_,blocksizeprefix_);
    generateblocksizeprefix(rowblocksizevec_,rowblocksizeprefix_);
    // Use fseek again to go backwards two bytes and check that byte with fgetc
    struct stat st;     // get file size
    if (stat(filename.c_str(), &st) == -1)
    {
        MPI_Abort(MPI_COMM_WORLD, NOFILE);
    }
    int64_t file_size = st.st_size;
    MPI_Offset fpos, end_fpos, endofheader;
    if(grid1d_->GetRank() == 0)    // the offset needs to be for this rank
    {
        fpos = ftell(f);
        endofheader =  fpos;
        MPI_Bcast(&endofheader, 1, MPIType<MPI_Offset>(), 0, grid1d_->commWorld);
        fclose(f);
    }
    else
    {
        MPI_Bcast(&endofheader, 1, MPIType<MPI_Offset>(), 0, grid1d_->commWorld);  // receive the file loc at the end of header
        fpos = endofheader + myrank * (file_size-endofheader) / nprocs;
    }
    if(myrank != (nprocs-1)) end_fpos = endofheader + (myrank + 1) * (file_size-endofheader) / nprocs;
    else end_fpos = file_size;

    MPI_File mpi_fh;
    MPI_File_open (grid1d_->commWorld, const_cast<char*>(filename.c_str()), MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_fh);

    typedef typename DER::LocalIT LIT;
    std::vector<LIT> rows;
    std::vector<LIT> cols;
    std::vector<NT> vals;

    std::vector<std::string> lines;
    bool finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, true, lines, myrank);
    int64_t entriesread = lines.size();
    SpHelper::ProcessLines(rows, cols, vals, lines, symmetric, type, onebased);
    MPI_Barrier(grid1d_->commWorld);

    while(!finished)
    {
        finished = SpParHelper::FetchBatch(mpi_fh, fpos, end_fpos, false, lines, myrank);
        entriesread += lines.size();
        SpHelper::ProcessLines(rows, cols, vals, lines, symmetric, type, onebased);
    }
    int64_t allentriesread;
    MPI_Reduce(&entriesread, &allentriesread, 1, MPIType<int64_t>(), MPI_SUM, 0, grid1d_->commWorld);
    std::vector< std::vector < std::tuple<LIT,LIT,NT> > > data(nprocs);
    LIT locsize = rows.size();   // remember: locsize != entriesread (unless the matrix is unsymmetric)
    for(LIT i=0; i<locsize; ++i)
    {
        LIT lrow, lcol;
        lrow = rows[i];
        lcol = cols[i];
        int owner = Owner(cols[i], lcol);
        data[owner].push_back(std::make_tuple(lrow,lcol,vals[i]));
    }
    std::vector<LIT>().swap(rows);
    std::vector<LIT>().swap(cols);
    std::vector<NT>().swap(vals);	

    IT datasize;
    std::tuple<IT,IT,NT>* recvTuples = SpParHelper::ExchangeDataGeneral(data, grid1d_->commWorld, datasize);
    SpTuples<IT, NT>spTuples(datasize, nrows, blocksizevec_[myrank], recvTuples);
    spSeq_ = make_shared<DER>(spTuples, false);
}
template <class IT, class NT, class DER>
void SpParMat1D< IT,NT,DER >::ParallelWriteMetisGraph(const std::string filename, std::string vwgt)
{
    int myrank = grid1d_->GetRank();
    int nprocs = grid1d_->GetSize();
    SpCCols<IT, NT> spcsc(*spSeq_.get());
    Csc<IT, NT> *cscptr = spcsc.GetCSC();
    IT localedges = 0;
    IT edges = 0;
    string localstr = "";
    IT colprefix = blocksizeprefix_[myrank];
    for(IT jlocal=0; jlocal < cscptr->n; jlocal++)
    {
        IT jglobal = jlocal + colprefix;
        IT rowstart = cscptr->jc[jlocal];
        IT rowend   = cscptr->jc[jlocal+1];
        if     (vwgt == "flops")    localstr += " " + std::to_string( (rowend - rowstart) * (rowend - rowstart) + 1);
        else if(vwgt == "nnz"  )    localstr += " " + std::to_string(rowend - rowstart);
        IT vtxedgecnt = 0;
        for(IT rowidx = rowstart; rowidx < rowend; rowidx++)
        {
            IT rowid = cscptr->ir[rowidx];
            if(rowid != jglobal){
                localstr += " " + std::to_string(rowid+1);
                // numx : potential use
                NT numx = cscptr->num[rowid];
                localedges++;
                vtxedgecnt++;
            }
        }
        localstr += " \n";
    }
    IT nvtx = getncol();
    MPI_Allreduce(&localedges, &edges, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    edges /= 2;
    if(myrank == 0) {
        if(vwgt == "none") localstr = std::to_string(nvtx) + " " + std::to_string(edges) + " 000\n" + localstr;
        else localstr = std::to_string(nvtx) + " " + std::to_string(edges) + " 010\n" + localstr;
    }
    /* Calculate offset */
    IT localcharsize = localstr.size();
    vector<IT> gcharsize(nprocs,0);
    MPI_Allgather(
        &localcharsize, 1, MPI_LONG_LONG, 
        gcharsize.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
    vector<IT> prefixsum(nprocs,0);
    partial_sum(gcharsize.begin(),gcharsize.end()-1,prefixsum.begin()+1);
    SpHelper::RemoveExistingFile(filename);
    MPI_Barrier(MPI_COMM_WORLD);
    /* Write to FS */
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    MPI_Offset offset = prefixsum[myrank];
    MPI_File_write_at(file, offset, localstr.c_str(), localstr.size(), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&file);
}


template <class IT, class NT, class DER>
void SpParMat1D< IT,NT,DER >::SequentialWriteMM(const std::string filename, bool onebased)
{   
    vector<IT> rows, cols;
    vector<NT> vals;
    SpHelper::RemoveExistingFile(filename);
    for(typename DER::SpColIter colit = spSeq_->begcol(); colit != spSeq_->endcol(); ++colit)
    {
        IT gcol = colit.colid();
        for(typename DER::SpColIter::NzIter nzit = spSeq_->begnz(colit); nzit != spSeq_->endnz(colit); ++nzit)
        {
            IT grow = nzit.rowid();
            NT val = nzit.value();
            rows.push_back(grow);
            cols.push_back(gcol);
            vals.push_back(val);
        }
    }
    IT locrow = spSeq_->getnrow();
    IT loccol = spSeq_->getncol();
    std::ofstream os(filename, std::ios_base::binary);
}

template <class IT, class NT, class DER>
void SpParMat1D< IT,NT,DER >::SequentialReadMM(const std::string filename, bool onebased){
    assert(onebased == true); // double check
    vector<IT> rows, cols;
    vector<NT> vals;
    IT gnrows, gncols;
    std::ifstream os(filename, std::ios_base::binary);

    // calculate blocksizevec
    int myrank = grid1d_->GetRank();
    int nprocs = grid1d_->GetSize();
    fillblocksizevec(blocksizevec_, nprocs, gncols);
    fillblocksizevec(rowblocksizevec_, nprocs, gnrows);
    generateblocksizeprefix(blocksizevec_,blocksizeprefix_);
    generateblocksizeprefix(rowblocksizevec_,rowblocksizeprefix_);

    std::vector<std::tuple<IT,IT,NT>> datatuple;
    for(IT i=0; i < rows.size(); i++){
        datatuple.push_back(std::make_tuple(rows[i],cols[i],vals[i]));
    }
    SpTuples<IT, NT> * newtuples = 
    new SpTuples<IT,NT>(datatuple.size(), gnrows, gncols, datatuple.data());
    newtuples->tuples_deleted = true;
    spSeq_.reset(new DER(*newtuples,false));
    delete newtuples;
}

template <class IT, class NT,class DER>
int SpParMat1D<IT,NT,DER>::Owner(IT gcol, IT & lcol) const {
    int nblocks=blocksizeprefix_.size();
    int owner = 0;
    while(owner <= nblocks-1 && blocksizeprefix_[owner] <= gcol) owner++;
    owner-=1;
    lcol = gcol - blocksizeprefix_[owner];
    return owner;
}

template <class IT, class NT, class DER>
std::pair<int,int> SpParMat1D< IT,NT,DER >::BlockIndex(const IT grow, const IT gcol) const {
    int bridx,bcidx,nblocks=blocksizeprefix_.size();
    int owner = 0;
    while(owner <= nblocks-1 && blocksizeprefix_[owner] <= gcol) owner++;
    owner-=1;
    bcidx = owner;
    owner = 0;
    while(owner <= nblocks-1 && rowblocksizeprefix_[owner] <= grow) owner++;
    owner-=1; 
    bridx = owner;
    return std::make_pair(bridx, bcidx);
}

template<class IT, class NT, class DER>
void SpParMat1D< IT,NT,DER >::PrintInfo(int level)
{
    int myrank = grid1d_->GetRank();
    int nprocs = grid1d_->GetSize();
    IT gnrows = getnrow();
    IT gncols = getncol();
    IT nnz = getnnz();
    if(level >= 0){
        if(myrank == 0){
            char printstr[2000];
            sprintf(
                printstr, "As a whole 1D matrix, nrows %ld ncols %ld nnz %ld memsize %f MB \n",
                gnrows, gncols, nnz, (double)nnz * (sizeof(IT) + sizeof(NT)) * 1e-6 
            );
            std::cerr << printstr;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(level >= 1){
        IT diagelems = 0, offdiagelems = 0;
        IT diagstart = rowblocksizeprefix_[myrank];
        IT diagend = (myrank != nprocs-1) ? rowblocksizeprefix_[myrank+1] : gnrows;
        for(typename DER::SpColIter colit = spSeq_->begcol(); colit != spSeq_->endcol(); ++colit)
        {
            for(typename DER::SpColIter::NzIter nzit = spSeq_->begnz(colit); nzit != spSeq_->endnz(colit); ++nzit)
            {
                IT grow = nzit.rowid();
                if(grow >= diagstart && grow < diagend){ diagelems++;}
            }
        }
        offdiagelems = spSeq_->getnnz() - diagelems;
        double dp = (double)(diagelems) / spSeq_->getnnz();
        double odp = (double)(offdiagelems) / spSeq_->getnnz();
        char printstr[2000];
        sprintf(
            printstr, "myrank %d, my local diag elems %ld (%.3f), my local offdiag elems %ld (%.3f)\n", 
            myrank, diagelems, dp, offdiagelems, odp);
        std::cerr << printstr;
    }
}


template <class IT, class NT, class DER>
template <typename _BinaryOperation>
FullyDistVec1D<IT,NT> 
SpParMat1D< IT,NT,DER >::Reduce(Dim dim, _BinaryOperation __binary_op, NT id) const
{
    
    std::vector<IT> blocksizevec;
    switch(dim)
    {
        case Column:
        {
            blocksizevec = this->blocksizevec_;
            break;
        }
        case Row:
        {
            blocksizevec = this->rowblocksizevec_;
            break;
        }
        default:
        {
            std::cout << "Unknown reduction dimension, returning empty vector" << std::endl;
            break;
        }
    }
    FullyDistVec1D<IT, NT> rvec(blocksizevec, (NT)0);
    Reduce(rvec, dim, __binary_op, id, myidentity<NT>());
    return rvec;
}

template <class IT, class NT, class DER>
template <typename _BinaryOperation, typename _UnaryOperation>
FullyDistVec1D<IT,NT> 
SpParMat1D< IT,NT,DER >::Reduce(Dim dim, _BinaryOperation __binary_op, NT id, _UnaryOperation __unary_op) const
{
    std::vector<IT> blocksizevec;

    switch(dim)
    {
        case Column:
        {
            blocksizevec = this->blocksizevec_;
            break;
        }
        case Row:
        {
            blocksizevec = this->rowblocksizevec_;
            break;
        }
        default:
        {
            std::cout << "Unknown reduction dimension, returning empty vector" << std::endl;
            break;
        }
    }
    FullyDistVec1D<IT, NT> rvec(blocksizevec, (NT)0);
    Reduce(rvec, dim, __binary_op, id, __unary_op);
    return rvec;
}

// Reduce with an input vector API
template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _BinaryOperation>
void 
SpParMat1D< IT,NT,DER >::Reduce(FullyDistVec1D<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, NT id) const
{
    Reduce(rvec, dim, __binary_op, id, myidentity<NT>());
}
template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation >
void 
SpParMat1D< IT,NT,DER >::Reduce(FullyDistVec1D<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op) const
{
    Reduce(rvec, dim, __binary_op, id, __unary_op, MPIOp<_BinaryOperation, VT>::op());
}

// Reduce Implementation
template <class IT, class NT, class DER>
template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation >
void 
SpParMat1D< IT,NT,DER >::Reduce(FullyDistVec1D<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op, MPI_Op mympiop) const
{
    switch (dim) {
        case Column:
        {
            // column reduction logics
            IT ncols = getncol();
            for(typename DER::SpColIter colit = spSeq_->begcol(); colit != spSeq_->endcol(); ++colit)	// iterate over a portion of columns
            {
                for(typename DER::SpColIter::NzIter nzit = spSeq_->begnz(colit); nzit != spSeq_->endnz(colit); ++nzit)	// all nonzeros in this column
                {
                    rvec.arr_[colit.colid()] = __binary_op(static_cast<VT>(__unary_op(nzit.value())), rvec.arr_[colit.colid()]);
                }
            }
            break;
        }
        case Row:
        {
            printf("not implemented!\n");
            break;
        }
    }
}





template <class IT, class NT, class DER>
void SpParMat1D< IT,NT,DER >::ExchangeDcscIndex(
    std::vector<std::vector<IT>> & allcpvec, std::vector<std::vector<IT>> & alljcvec)
{
    int nprocs = grid1d_->GetSize();
    allcpvec = std::vector<std::vector<IT>>(nprocs);
    alljcvec = std::vector<std::vector<IT>>(nprocs);
    // if(dcscA == NULL || dcscA == nullptr){
    //     std::cerr<<"I met a nullptr"<<std::endl;
    //     std::cerr<<"mynz is "<< spSeq_->getnnz() <<std::endl;
    // }
    std::vector<IT> localAcpvec, localAjcvec;
    if(spSeq_->getnnz() != 0){
        Dcsc<IT, NT> * dcscA =  spSeq_->GetDCSC();
        localAcpvec = std::vector<IT>(dcscA->cp, dcscA->cp+dcscA->nzc+1);
        localAjcvec = std::vector<IT>(dcscA->jc, dcscA->jc+dcscA->nzc);
    }
    SpParHelper::AllgatherVector(allcpvec, localAcpvec);
    SpParHelper::AllgatherVector(alljcvec, localAjcvec);
}


}
