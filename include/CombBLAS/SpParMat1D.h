#ifndef _SP_PAR_MAT_1D_H_
#define _SP_PAR_MAT_1D_H_

#include <cstdint>
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <mpi.h>
#include <tuple>
#include <vector>
#include <iterator>

#include "CombBLAS/Compare.h"
#include "SpMat.h"
#include "SpTuples.h"
#include "SpDCCols.h"
#include "CommGrid.h"
#include "CommGrid1D.h"

#include "MPIType.h"
#include "LocArr.h"
#include "SpDefs.h"
#include "Deleter.h"
#include "SpHelper.h"
#include "SpParHelper.h"
#include "FullyDistVec.h"
#include "Friends.h"
#include "Operations.h"
#include "DistEdgeList.h"
#include "mtSpGEMM.h"
#include "MultiwayMerge.h"
#include "CombBLAS.h"
#include "SpParMat1DFriends.h"
#include "FullyDistVec1D.h"

namespace combblas
{




template <class IT, class NT, class DER>
class SpParMat1D{
    
public:
/****************************************************************************/
/********************* CLASS STATIC FUNCTION ********************************/
/****************************************************************************/

// class static function, utility function
static void fillblocksizevec(std::vector<IT> & targetvec, int nprocs,  IT length){
    targetvec = std::vector<IT>(nprocs, 0); //assign memory
    // idea from metis [ParallelReadGraph function]
    IT i, k, l;
    for(i=0, k=length; i < nprocs; i++){
        l = k / (nprocs-i);
        targetvec[i] = l;
        k -= l;
    }
    // IT blocksize = length / nprocs;
    // for(int i=0; i<nprocs-1; i++){
    //     targetvec.push_back(blocksize);
    // }
    // targetvec.push_back(length - (nprocs-1)*blocksize);
}
static void generateblocksizeprefix(const std::vector<IT> & blocksize, std::vector<IT> & blocksizeprefix){
    int nprocs = blocksize.size();
    blocksizeprefix = std::vector<IT>(nprocs,0);
    std::partial_sum(blocksize.begin(), blocksize.end()-1, blocksizeprefix.begin()+1);
}


/****************************************************************************/
/********************* Constructor / Deconstructor **************************/
/****************************************************************************/
    /* Constructor and Deconstructor */
    SpParMat1D () = delete;
    SpParMat1D (std::shared_ptr<CommGrid1D> grid); // init with a commgrid.
    SpParMat1D (DER * userseq, std::shared_ptr<CommGrid1D> grid); // accept user seq and grid.
    SpParMat1D (const IT gnrows, const IT gncols); // we will assign blocksize to each process.
    SpParMat1D (const IT gnrows, const IT gncols, DER * userseq); // we will assign blocksize and data to each process.
    SpParMat1D (const IT gnrows, const IT gncols, const std::vector<IT> blocksizevec, const std::vector<IT> rowblocksizevec={}); // user defined blocksize.
    SpParMat1D (const IT gnrows, const IT gncols, const std::vector<IT> blocksizevec, DER * userseq, const std::vector<IT> rowblocksizevec={}); // user defined blocksize and data.
    SpParMat1D (const IT gnrows, const IT gncols, const std::vector<IT> blocksizevec, DER * userseq, std::shared_ptr<CommGrid1D> grid1d, const std::vector<IT> rowblocksizevec={}); // user defined blocksize and data.
    SpParMat1D (const SpParMat1D < IT,NT,DER > & A1D); // construct from a 1D matrix
    // SpParMat1D (const SpParMat < IT,NT,DER > & A2D); // receive data from a 2D matrix, we assign blocksize.
    SpParMat1D (const SpParMat < IT,NT,DER > & A2D, const std::vector<IT> blocksizevec={}, const std::vector<IT> rowblocksizevec={}); // receive data from a 2D matrix, user defeind blocksize.
    ~SpParMat1D ();
    /* interface to private members */
    IT getnrow() const;
    IT getncol() const;
    IT getnnz()  const;
    IT getnzc()  const;
    std::vector<IT> getblocksizevec() const {return this->blocksizevec_;};
    std::vector<IT> getblocksizeprefix() const {return this->blocksizeprefix_;};
    std::vector<IT> getrowblocksizevec() const {return this->rowblocksizevec_;};
    std::vector<IT> getrowblocksizeprefix() const {return this->rowblocksizeprefix_;};
    DER * seqptr() const { return spSeq_.get(); } 
    CommGrid1D* getgrid() const {return grid1d_.get();};
    std::shared_ptr<CommGrid1D> getsharedgrid()const{return grid1d_;}
    template <typename _UnaryOperation>
    void Apply(_UnaryOperation __unary_op)
    {
        spSeq_->Apply(__unary_op);
    }

    /**
     * @brief Transpose a 1D parititon matrix
     * one can pass new blocksize and row blocksize, but sum of blocksize should equal global rows.
     * sum of rowblocksize should equal global columns. otherwise blocksize will be current rowblocksizevec_,
     * rowblocksize will be current blocksizevec_.
     * @param blocksize 
     * @param rowblocksize 
     */
    void Transpose(std::vector<IT> blocksize = {}, std::vector<IT> rowblocksize = {});
    void Redistrube(std::vector<IT> blocksize = {}, std::vector<IT> rowblocksize = {});
    /* MPI Based I/O */
    void ParallelWriteMM(const std::string filename, bool onebased){ParallelWriteMM(filename,onebased,ScalarReadSaveHandler());};
    template<class HANDLER>
    void ParallelWriteMM(const std::string filename, bool onebased, HANDLER handler);
    void ParallelReadMM(const std::string filename, const std::string blocksizevecfilename = "", bool onebased=true);
    void SequentialWriteMM(const std::string filename, bool onebased);
    void SequentialReadMM(const std::string filename, bool onebased);
    void ParallelWriteMetisGraph(const std::string filename, std::string VertexWeight = "none");
    /* Graph Partition */
    void GetVertexWeight(std::vector<int64_t> & vwgt, int vwgttype);
    /* utility functions */
    int Owner(IT gcol, IT & lcol) const;
    int OutProductOwner(IT grow, IT gcol, IT & lrow, IT & lcol) const; 
    std::pair<int,int> BlockIndex(const IT grow, const IT gcol) const;
    bool InDiagblock(const IT grow, const IT gcol);
    std::vector<IT> BlockwiseNNZanalysis(std::string filename); // get NNZ elements inside each block defined by blocksizevec.
    void KeepDiagonalBlock(); // keep diagonal block, remove others.
    void RemoveDiagonalblock(); // remove diagonal block, keep others.

    template <typename _UnaryOperation>
    SpParMat1D<IT,NT,DER> Prune(_UnaryOperation __unary_op, bool inPlace = true) //<! Prune any nonzero entries for which the __unary_op evaluates to true (solely based on value)
    {
        if (inPlace)
        {
            spSeq_->Prune(__unary_op, inPlace);
            return SpParMat1D<IT,NT,DER>(grid1d_); // return blank to match signature
        }
        else
        {
            return SpParMat1D<IT,NT,DER>(spSeq_->Prune(__unary_op, inPlace), grid1d_);
        }
    }

    template <typename NNT, typename NDER> 
    operator SpParMat1D< IT,NNT,NDER > () const;    //!< Type conversion operator
    template <typename NIT, typename NNT, typename NDER> 
    operator SpParMat1D< NIT,NNT,NDER > () const;   //!< Type conversion operator (for indices as well)

    // ----------------------Util function For RDMA SpGEMM----------------------------
    
    // get everyone's cp and jc array. this requires seqptr is SpDCCols object.
    void ExchangeDcscIndex(std::vector<std::vector<IT>> & allcpvec, std::vector<std::vector<IT>> & alljcvec);

    

    // Reduce without an input vector API
    template <typename _BinaryOperation>
    FullyDistVec1D<IT,NT> 
    Reduce(Dim dim, _BinaryOperation __binary_op, NT id) const;

    template <typename _BinaryOperation, typename _UnaryOperation>
    FullyDistVec1D<IT,NT> 
    Reduce(Dim dim, _BinaryOperation __binary_op, NT id, _UnaryOperation __unary_op) const;


    // Reduce with an input vector API
    template <typename VT, typename GIT, typename _BinaryOperation>
    void 
    Reduce(FullyDistVec1D<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, NT id) const;

    template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation >
    void 
    Reduce(FullyDistVec1D<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op) const;

    // Reduce Implementation
    template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation >
    void 
    Reduce(FullyDistVec1D<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op, MPI_Op mympiop) const;

    /* operators overload */
    SpParMat1D< IT,NT,DER > & operator+=(const SpParMat1D< IT,NT,DER > & rhs);
    bool operator==(const SpParMat1D< IT,NT,DER > & rhs) const;
    SpParMat1D< IT,NT,DER > & operator-=(const SpParMat1D< IT,NT,DER > & rhs);
    SpParMat1D< IT,NT,DER > & operator*(NT scale_value);

    // current ri and ci has to be a permutation array of row and column index. we perform this operation only.
    void SubsRef_SR(const std::vector<IT> & Rperm, const std::vector<IT> & Cperm);
    void SubsRef_SR(
        const std::vector<IT> & Rperm, const std::vector<IT> & Cperm,
        const std::vector<IT> & blocksize, const std::vector<IT> & rowblocksize);


    // do permutation using ri and ci, but also, change the ditribution using blocksize and row blocksize,
    // this is only used in graph partition. the matrix should be a square matrix.
    void operator() (
        const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci, 
        const std::vector<IT> & blocksize, const std::vector<IT> & rowblocksize)
    {
        IT locmax_ri = 0;
        IT locmax_ci = 0;
        if(!ri.arr.empty())
            locmax_ri = *std::max_element(ri.arr.begin(), ri.arr.end());
        if(!ci.arr.empty())
            locmax_ci = *std::max_element(ci.arr.begin(), ci.arr.end());

        IT totalm = getnrow();
        IT totaln = getncol();
        if(locmax_ri > totalm || locmax_ci > totaln)	
        {
            throw outofrangeexception();
        }
        
        // let's gather ri and ci so that everyone has a full copy for simplicity.
        std::vector<IT> fullrowperm, fullcolperm;
        SpParHelper::AllgatherVector(fullrowperm, ri.GetLocVec());
        SpParHelper::AllgatherVector(fullcolperm, ci.GetLocVec());
        SubsRef_SR(fullrowperm, fullcolperm,blocksize, rowblocksize);
    }
    void operator() (const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci)
    {
        IT locmax_ri = 0;
        IT locmax_ci = 0;
        if(!ri.arr.empty())
            locmax_ri = *std::max_element(ri.arr.begin(), ri.arr.end());
        if(!ci.arr.empty())
            locmax_ci = *std::max_element(ci.arr.begin(), ci.arr.end());

        IT totalm = getnrow();
        IT totaln = getncol();
        if(locmax_ri > totalm || locmax_ci > totaln)	
        {
            throw outofrangeexception();
        }
        
        // let's gather ri and ci so that everyone has a full copy for simplicity.
        std::vector<IT> fullrowperm, fullcolperm;
        SpParHelper::AllgatherVector(fullrowperm, ri.GetLocVec());
        SpParHelper::AllgatherVector(fullcolperm, ci.GetLocVec());
        SubsRef_SR(fullrowperm, fullcolperm);
    }
    void operator() (const std::vector<IT> & ri, const std::vector<IT> & ci)
    {
        std::vector<IT> fullrowperm, fullcolperm;
        SpParHelper::AllgatherVector(fullrowperm, ri);
        SpParHelper::AllgatherVector(fullcolperm, ci);
        SubsRef_SR(ri, ci);
    }
    
    /****************************************************************************/
    /********************* SUBMATRIX SELECTION **********************************/
    /****************************************************************************/

    /**
    * Create a submatrix of size m x (sum of ci size acrouss p proces) on a p processor grid
    * Essentially fetches the columns ci[0], ci[1],... ci[ci.size()-1] from every submatrix
    */
    SpParMat1D<IT,NT,DER> SubsRefCol (const std::vector<IT> & ci) const;//!< Column indexing with special parallel semantics

    /* friend classes and functions */
    // Block Diagonal Implementation
    template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
    friend SpParMat1D<IU, NUO, UDERO> Mult_AnXBn_1D_BD(SpParMat1D<IU,NU1,UDERA> & A, SpParMat1D<IU,NU2,UDERB> & B, bool clearA, bool clearB );

    template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB> 
    friend SpParMat1D<IU, NUO, UDERO> Mult_AnXBn_1D_OP(SpParMat1D<IU,NU1,UDERA> & A, SpParMat1D<IU,NU2,UDERB> & B, bool clearA, bool clearB );

    template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
    friend SpParMat1D<IU, NUO, UDERO> Mult_AnXBn_1D_CbC_RDMA_FetchAll(SpParMat1D<IU,NU1,UDERA> & A, SpParMat1D<IU,NU2,UDERB> & B, bool clearA, bool clearB, bool msgen);
    
    template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
    friend SpParMat1D<IU, NUO, UDERO> Mult_AnXBn_1D_CbC_RDMA_Overlap(SpParMat1D<IU,NU1,UDERA> & A, SpParMat1D<IU,NU2,UDERB> & B, bool clearA, bool clearB);
    
    template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
    friend SpParMat1D<IU, NUO, UDERO> Mult_AnXBn_1D_CbC_RDMA_OverlapOpenMPTask(SpParMat1D<IU,NU1,UDERA> & A, SpParMat1D<IU,NU2,UDERB> & B, bool clearA, bool clearB);
    
    template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
    friend SpParMat1D<IU, NUO, UDERO> Mult_AnXBn_1D_CbC_RDMA_OverlapsqrtPstage(SpParMat1D<IU,NU1,UDERA> & A, SpParMat1D<IU,NU2,UDERB> & B, bool clearA, bool clearB);

    friend class SpParMat<IT, NT, DER>;

    friend class SpParHelper;

    void PrintInfo(int level=1);

    class ScalarReadSaveHandler
    {
    public:
        NT getNoNum(IT row, IT col) { return static_cast<NT>(1); }
        void binaryfill(FILE * rFile, IT & row, IT & col, NT & val) 
        {
            if (fread(&row, sizeof(IT), 1,rFile) != 1)
                std::cout << "binaryfill(): error reading row index" << std::endl;
            if (fread(&col, sizeof(IT), 1,rFile) != 1)
                std::cout << "binaryfill(): error reading col index" << std::endl;
            if (fread(&val, sizeof(NT), 1,rFile) != 1)
                std::cout << "binaryfill(): error reading value" << std::endl;
            return; 
        }
        size_t entrylength() { return 2*sizeof(IT)+sizeof(NT); }
        
        template <typename c, typename t>
        NT read(std::basic_istream<c,t>& is, IT row, IT col)
        {
            NT v;
            is >> v;
            return v;
        }

        template <typename c, typename t>
        void save(std::basic_ostream<c,t>& os, const NT& v, IT row, IT col)
        {
            os << v;
        }
    };
protected:
    std::shared_ptr<DER> spSeq_;
    std::shared_ptr<CommGrid1D> grid1d_;
    std::vector<IT> blocksizevec_; // col 
    std::vector<IT> blocksizeprefix_; // col
    std::vector<IT> rowblocksizevec_;
    std::vector<IT> rowblocksizeprefix_;
};



}

#include "SpParMat1D.cpp"

#endif
