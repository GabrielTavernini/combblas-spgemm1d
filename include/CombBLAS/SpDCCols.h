/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
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
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */


#ifndef _SP_DCCOLS_H
#define _SP_DCCOLS_H


#include <fstream>
#include <omp.h>
// #include "SpMat.h"	// Best to include the base class first
#include "SpHelper.h"
#include "StackEntry.h"
#include "dcsc.h"
#include "Isect.h"
#include "Semirings.h"
// #include "MemoryPool.h"
#include "LocArr.h"
// #include "Friends.h"
// #include "CombBLAS.h"
// #include "FullyDist.h"

#include <vector>

namespace combblas {

template<class A, class B, class C>
class SpMat;

class SpHelper;

template <class IT, class NT>
class SpDCCols: public SpMat<IT, NT, SpDCCols<IT, NT> >
{
public:
	typedef IT LocalIT;
	typedef NT LocalNT;

	// Constructors :
	SpDCCols ();
	SpDCCols (IT size, IT nRow, IT nCol, IT nzc);
	SpDCCols (const SpTuples<IT,NT> & rhs, bool transpose);
    SpDCCols (IT nRow, IT nCol, IT nnz1, const std::tuple<IT, IT, NT> * rhs, bool transpose);

	SpDCCols (const SpDCCols<IT,NT> & rhs);					// Actual copy constructor		
	~SpDCCols();

	template <typename NNT> operator SpDCCols<IT,NNT> () const;		//!< NNT: New numeric type
	template <typename NIT, typename NNT> operator SpDCCols<NIT,NNT> () const;		//!< NNT: New numeric type, NIT: New index type

	// Member Functions and Operators: 
	SpDCCols<IT,NT> & operator= (const SpDCCols<IT, NT> & rhs);
	SpDCCols<IT,NT> & operator+= (const SpDCCols<IT, NT> & rhs);
	SpDCCols<IT,NT> operator() (IT ri, IT ci) const;	
	SpDCCols<IT,NT> operator() (const std::vector<IT> & ri, const std::vector<IT> & ci) const;
	bool operator== (const SpDCCols<IT, NT> & rhs) const
	{
		if(rhs.nnz == 0 && nnz == 0)
			return true;
        if(m != rhs.m || n != rhs.n){
            // std::cerr << "dim not match: " << m << " vs " << rhs.m << ", "
            // << n << " vs " << rhs.n << " ." << std::endl;
            return false;
        }
        if(nnz != rhs.nnz){
            // std::cerr << "nnz not match: " << nnz << " vs " << rhs.nnz << std::endl;
            return false;
        }
		
			
		return ((*dcsc) == (*(rhs.dcsc)));
	}


    class SpColIter //! Iterate over (sparse) columns of the sparse matrix
    {
    public:
        class NzIter	//! Iterate over the nonzeros of the sparse column
        {
        public:
            NzIter(IT * ir = NULL, NT * num = NULL) : rid(ir), val(num) {}
            
            bool operator==(const NzIter & other)
            {
                return(rid == other.rid);	// compare pointers
            }
            bool operator!=(const NzIter & other)
            {
                return(rid != other.rid);
            }
            bool operator<(const NzIter & other)
            {
                return(rid < other.rid);
            }
            NzIter operator+(IT inc)
            {
                rid+=inc;
                val+=inc;
                return(*this);
            }
            NzIter operator-(IT inc)
            {
                rid-=inc;
                val-=inc;
                return(*this);
            }
            NzIter & operator++()	// prefix operator
            {
                ++rid;
                ++val;
                return(*this);
            }
            NzIter operator++(int)	// postfix operator
            {
                NzIter tmp(*this);
                ++(*this);
                return(tmp);
            }
            IT rowid() const	//!< Return the "local" rowid of the current nonzero entry.
            {
                return (*rid);
            }
            NT & value()		//!< value is returned by reference for possible updates
            {
                return (*val);
            }
        private:
            IT * rid;
            NT * val;
            
        };
        
        SpColIter(IT * cp = NULL, IT * jc = NULL) : cptr(cp), cid(jc) {}
        bool operator==(const SpColIter& other)
        {
            return(cptr == other.cptr);	// compare pointers
        }
        bool operator!=(const SpColIter& other)
        {
            return(cptr != other.cptr);
        }
        
        SpColIter& operator++()		// prefix operator
        {
            ++cptr;
            ++cid;
            return(*this);
        }
        SpColIter operator++(int)	// postfix operator (common)
        {
            SpColIter tmp(*this);
            ++(*this);
            return(tmp);
        }
        SpColIter operator+(IT inc)
        {
            cptr+=inc;
            cid+=inc;
            return(*this);
        }
        SpColIter operator-(IT inc)
        {
            cptr-=inc;
            cid-=inc;
            return(*this);
        }
        IT colid() const	//!< Return the "local" colid of the current column.
        {
            return (*cid);
        }
		IT colptr() const
		{
			return (*cptr);
		}
		IT colptrnext() const
		{
			return (*(cptr+1));
		}
		IT nnz() const
		{
			return (colptrnext() - colptr());
		}
  	private:
        IT * cptr;
		IT * cid;
   	};
	
	SpColIter begcol()
	{
		if( nnz > 0 )
			return SpColIter(dcsc->cp, dcsc->jc); 
		else	
			return SpColIter(NULL, NULL);
	}
    SpColIter begcol(int i)  // multithreaded version
    {
        if( dcscarr[i] )
            return SpColIter(dcscarr[i]->cp, dcscarr[i]->jc);
        else
            return SpColIter(NULL, NULL);
    }

	SpColIter endcol()
	{
		if( nnz > 0 )
			return SpColIter(dcsc->cp + dcsc->nzc, NULL);
		else
			return SpColIter(NULL, NULL);
	}
    
    SpColIter endcol(int i)  //multithreaded version
    {
        if( dcscarr[i] )
            return SpColIter(dcscarr[i]->cp + dcscarr[i]->nzc, NULL);
        else
            return SpColIter(NULL, NULL);
    }

	typename SpColIter::NzIter begnz(const SpColIter & ccol)	//!< Return the beginning iterator for the nonzeros of the current column
	{
		return typename SpColIter::NzIter( dcsc->ir + ccol.colptr(), dcsc->numx + ccol.colptr() );
	}

	typename SpColIter::NzIter endnz(const SpColIter & ccol)	//!< Return the ending iterator for the nonzeros of the current column
	{
		return typename SpColIter::NzIter( dcsc->ir + ccol.colptrnext(), NULL );
	}			

    typename SpColIter::NzIter begnz(const SpColIter & ccol, int i)	//!< multithreaded version
    {
        return typename SpColIter::NzIter( dcscarr[i]->ir + ccol.colptr(), dcscarr[i]->numx + ccol.colptr() );
    }
    
    typename SpColIter::NzIter endnz(const SpColIter & ccol, int i)	//!< multithreaded version
    {
        return typename SpColIter::NzIter( dcscarr[i]->ir + ccol.colptrnext(), NULL );
    }
    
	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{
		if(nnz > 0)
			dcsc->Apply(__unary_op);	
	}
	
    // ---------------IO-----------------------------------------------------

    void WriteMM(std::string filename, bool onebased);

    // ----------------Reduction Operation--------------------------------

    // count elements in a column
    void ProjectRow(std::vector<int> & reducevector);
    // count elements in a column
    void ProjectColumn(std::vector<IT> & reducevector);
    // sum elements in a column
    void ReduceColumn(std::vector<NT> & reducevector);
    // sum elements in a row
    void ReduceRow(std::vector<NT> & reducevector);
    
    

    

	template <typename _UnaryOperation, typename GlobalIT>
	SpDCCols<IT,NT>* PruneI(_UnaryOperation __unary_op, bool inPlace, GlobalIT rowOffset, GlobalIT colOffset);
	template <typename _UnaryOperation>
	SpDCCols<IT,NT>* Prune(_UnaryOperation __unary_op, bool inPlace);
    template <typename _BinaryOperation>
    SpDCCols<IT,NT>* PruneColumn(NT* pvals, _BinaryOperation __binary_op, bool inPlace);
    template <typename _BinaryOperation>
    SpDCCols<IT,NT>* PruneColumn(IT* pinds, NT* pvals, _BinaryOperation __binary_op, bool inPlace);

    void PruneColumnByIndex(const std::vector<IT>& ci);

	template <typename _BinaryOperation>
	void UpdateDense(NT ** array, _BinaryOperation __binary_op) const
	{
		if(nnz > 0 && dcsc != NULL)
			dcsc->UpdateDense(array, __binary_op);
	}

	void EWiseScale(NT ** scaler, IT m_scaler, IT n_scaler);
	void EWiseMult (const SpDCCols<IT,NT> & rhs, bool exclude);
	void SetDifference (const SpDCCols<IT,NT> & rhs);
	
	void Transpose(); //!< Mutator version, replaces the calling object 
	SpDCCols<IT,NT> TransposeConst() const;		//!< Const version, doesn't touch the existing object
	SpDCCols<IT,NT> * TransposeConstPtr() const;

	void RowSplit(int numsplits)
	{
		BooleanRowSplit(*this, numsplits);	// only works with boolean arrays
	}
    
    void ColSplit(int parts, std::vector< SpDCCols<IT,NT> > & matrices); //!< \attention Destroys calling object (*this)
    void ColSplit(int parts, std::vector< SpDCCols<IT,NT>* > & matrices); //!< \attention Destroys calling object (*this)
    void ColSplit(std::vector<IT> & cutSizes, std::vector< SpDCCols<IT,NT> > & matrices); //!< \attention Destroys calling object (*this)
    void ColSplit(std::vector<IT> & cutSizes, std::vector< SpDCCols<IT,NT>* > & matrices); //!< \attention Destroys calling object (*this)
    void ColConcatenate(std::vector< SpDCCols<IT,NT> > & matrices);    //!< \attention Destroys its parameters (matrices)
    void ColConcatenate(std::vector< SpDCCols<IT,NT>* > & matrices);    //!< \attention Destroys its parameters (matrices)

	void Split(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB); 	//!< \attention Destroys calling object (*this)
	void Merge(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB);	//!< \attention Destroys its parameters (partA & partB)

	void CreateImpl(const std::vector<IT> & essentials);
	void CreateImpl(IT size, IT nRow, IT nCol, std::tuple<IT, IT, NT> * mytuples);
    void CreateImpl(IT * _cp, IT * _jc, IT * _ir, NT * _numx, IT _nz, IT _nzc, IT _m, IT _n);


	Arr<IT,NT> GetArrays() const;
	std::vector<IT> GetEssentials() const;
	const static IT esscount;

	bool isZero() const { return (nnz == 0); }
	IT getnrow() const { return m; }
	IT getncol() const { return n; }
	IT getnnz() const { return nnz; }
	IT getnzc() const { return (nnz == 0) ? 0: dcsc->nzc; }
	int getnsplit() const { return splits; }
	
	std::ofstream& put(std::ofstream & outfile) const;
	std::ifstream& get(std::ifstream & infile);
	void PrintInfo() const;
	void PrintInfo(std::ofstream & out) const;

	template <typename SR> 
	int PlusEq_AtXBt(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);  
	
	template <typename SR>
	int PlusEq_AtXBn(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);
	
	template <typename SR>
	int PlusEq_AnXBt(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);  
	
	template <typename SR>
	int PlusEq_AnXBn(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);
	
    Dcsc<IT, NT> * GetDCSC() const 	// only for single threaded matrices
    {
        return dcsc;
    }
    
    Dcsc<IT, NT> * GetDCSC(int i) const 	// only for split (multithreaded) matrices
    {
        return dcscarr[i];
    }
    
    auto GetInternal() const    { return GetDCSC(); }
    auto GetInternal(int i) const  { return GetDCSC(i); }
	

private:
	void CopyDcsc(Dcsc<IT,NT> * source);
	SpDCCols<IT,NT> ColIndex(const std::vector<IT> & ci) const;	//!< col indexing without multiplication	

	template <typename SR, typename NTR>
	SpDCCols< IT, typename promote_trait<NT,NTR>::T_promote > OrdOutProdMult(const SpDCCols<IT,NTR> & rhs) const;	

	template <typename SR, typename NTR>
	SpDCCols< IT, typename promote_trait<NT,NTR>::T_promote > OrdColByCol(const SpDCCols<IT,NTR> & rhs) const;	
	
	SpDCCols (IT size, IT nRow, IT nCol, const std::vector<IT> & indices, bool isRow);	// Constructor for indexing
	SpDCCols (IT nRow, IT nCol, Dcsc<IT,NT> * mydcsc);			// Constructor for multiplication

	// Anonymous union
	union {
		Dcsc<IT, NT> * dcsc;
		Dcsc<IT, NT> ** dcscarr;
	};

	IT m;
	IT n;
	IT nnz;
	
	int splits;	// for multithreading

	template <class IU, class NU>
	friend class SpDCCols;		// Let other template instantiations (of the same class) access private members
	
	template <class IU, class NU>
	friend class SpTuples;

	// AL: removed this because it appears illegal and causes this compiler warning:
	// warning: dependent nested name specifier 'SpDCCols<IU, NU>::' for friend class declaration is not supported; turning off access control for 'SpDCCols'
	//template <class IU, class NU>
	//friend class SpDCCols<IU, NU>::SpColIter;
	
	template<typename IU>
	friend void BooleanRowSplit(SpDCCols<IU, bool> & A, int numsplits);

	template<typename IU, typename NU1, typename NU2>
	friend SpDCCols<IU, typename promote_trait<NU1,NU2>::T_promote > EWiseMult (const SpDCCols<IU,NU1> & A, const SpDCCols<IU,NU2> & B, bool exclude);

	template<typename N_promote, typename IU, typename NU1, typename NU2, typename _BinaryOperation>
	friend SpDCCols<IU, N_promote > EWiseApply (const SpDCCols<IU,NU1> & A, const SpDCCols<IU,NU2> & B, _BinaryOperation __binary_op, bool notB, const NU2& defaultBVal);

	template <typename RETT, typename IU, typename NU1, typename NU2, typename _BinaryOperation, typename _BinaryPredicate> 
	friend SpDCCols<IU,RETT> EWiseApply (const SpDCCols<IU,NU1> & A, const SpDCCols<IU,NU2> & B, _BinaryOperation __binary_op, _BinaryPredicate do_op, bool allowANulls, bool allowBNulls, const NU1& ANullVal, const NU2& BNullVal, const bool allowIntersect);

	template<class SR, class NUO, class IU, class NU1, class NU2>
	friend SpTuples<IU, NUO> * Tuples_AnXBn 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B, bool clearA, bool clearB);

	template<class SR, class NUO, class IU, class NU1, class NU2>
	friend SpTuples<IU, NUO> * Tuples_AnXBt 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B, bool clearA, bool clearB);

	template<class SR, class NUO, class IU, class NU1, class NU2>
	friend SpTuples<IU, NUO> * Tuples_AtXBn 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B, bool clearA, bool clearB);

	template<class SR, class NUO, class IU, class NU1, class NU2>
	friend SpTuples<IU, NUO> * Tuples_AtXBt 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B, bool clearA, bool clearB);

	template <typename SR, typename IU, typename NU, typename RHS, typename LHS>
	friend void dcsc_gespmv (const SpDCCols<IU, NU> & A, const RHS * x, LHS * y);

	template <typename SR, typename IU, typename NU, typename RHS, typename LHS>
	friend void dcsc_gespmv_threaded (const SpDCCols<IU, NU> & A, const RHS * x, LHS * y);
    
    template <typename SR, typename IU, typename NU, typename RHS, typename LHS>
    friend void dcsc_gespmv_threaded_nosplit (const SpDCCols<IU, NU> & A, const RHS * x, LHS * y);
    
    template <typename SR, typename IU, typename NUM, typename DER, typename IVT, typename OVT>
    friend int generic_gespmv_threaded (const SpMat<IU,NUM,DER> & A, const int32_t * indx, const IVT * numx, int32_t nnzx,
                                        int32_t * & sendindbuf, OVT * & sendnumbuf, int * & sdispls, int p_c);
};



/****************************************************************************/
/********************* PUBLIC CONSTRUCTORS/DESTRUCTORS **********************/
/****************************************************************************/

template <class IT, class NT>
const IT SpDCCols<IT,NT>::esscount = static_cast<IT>(4);


template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols():dcsc(NULL), m(0), n(0), nnz(0), splits(0){
}

// Allocate all the space necessary
template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols(IT size, IT nRow, IT nCol, IT nzc)
:m(nRow), n(nCol), nnz(size), splits(0)
{
	if(nnz > 0)
		dcsc = new Dcsc<IT,NT>(nnz, nzc);
	else
		dcsc = NULL; 
}

template <class IT, class NT>
SpDCCols<IT,NT>::~SpDCCols()
{
	if(nnz > 0)
	{
		if(dcsc != NULL) 
		{	
			if(splits > 0)
			{
				for(int i=0; i<splits; ++i)
					delete dcscarr[i];
				delete [] dcscarr;
			}
			else
			{
				delete dcsc;	
			}
		}
	}
}

// Copy constructor (constructs a new object. i.e. this is NEVER called on an existing object)
// Derived's copy constructor can safely call Base's default constructor as base has no data members 
template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols(const SpDCCols<IT,NT> & rhs)
: m(rhs.m), n(rhs.n), nnz(rhs.nnz), splits(rhs.splits)
{
	if(splits > 0)
	{
		for(int i=0; i<splits; ++i)
			CopyDcsc(rhs.dcscarr[i]);
	}
	else
	{
		CopyDcsc(rhs.dcsc);
	}
}

/** 
 * Constructor for converting SpTuples matrix -> SpDCCols (may use a private memory heap)
 * @param[in] 	rhs if transpose=true, 
 *	\n		then rhs is assumed to be a row sorted SpTuples object 
 *	\n		else rhs is assumed to be a column sorted SpTuples object
 **/
template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols(const SpTuples<IT, NT> & rhs, bool transpose)
: m(rhs.m), n(rhs.n), nnz(rhs.nnz), splits(0)
{	 
	
	if(nnz == 0)	// m by n matrix of complete zeros
	{
		if(transpose) std::swap(m,n);
		dcsc = NULL;	
	} 
	else
	{
		if(transpose)
		{
      std::swap(m,n);
			IT localnzc = 1;
			for(IT i=1; i< rhs.nnz; ++i)
			{
				if(rhs.rowindex(i) != rhs.rowindex(i-1))
				{
					++localnzc;
	 			}
	 		}
			dcsc = new Dcsc<IT,NT>(rhs.nnz,localnzc);	
			dcsc->jc[0]  = rhs.rowindex(0); 
			dcsc->cp[0] = 0;

			for(IT i=0; i<rhs.nnz; ++i)
	 		{
				dcsc->ir[i]  = rhs.colindex(i);		// copy rhs.jc to ir since this transpose=true
				dcsc->numx[i] = rhs.numvalue(i);
			}

			IT jspos  = 1;		
			for(IT i=1; i<rhs.nnz; ++i)
			{
				if(rhs.rowindex(i) != dcsc->jc[jspos-1])
				{
					dcsc->jc[jspos] = rhs.rowindex(i);	// copy rhs.ir to jc since this transpose=true
					dcsc->cp[jspos++] = i;
				}
			}		
			dcsc->cp[jspos] = rhs.nnz;
	 	}
		else
		{
			IT localnzc = 1;
			for(IT i=1; i<rhs.nnz; ++i)
			{
				if(rhs.colindex(i) != rhs.colindex(i-1))
				{
					++localnzc;
				}
			}
			dcsc = new Dcsc<IT,NT>(rhs.nnz,localnzc);	
			dcsc->jc[0]  = rhs.colindex(0); 
			dcsc->cp[0] = 0;

			for(IT i=0; i<rhs.nnz; ++i)
			{
				dcsc->ir[i]  = rhs.rowindex(i);		// copy rhs.ir to ir since this transpose=false
				dcsc->numx[i] = rhs.numvalue(i);
			}

			IT jspos = 1;		
			for(IT i=1; i<rhs.nnz; ++i)
			{
				if(rhs.colindex(i) != dcsc->jc[jspos-1])
				{
					dcsc->jc[jspos] = rhs.colindex(i);	// copy rhs.jc to jc since this transpose=true
					dcsc->cp[jspos++] = i;
				}
			}		
			dcsc->cp[jspos] = rhs.nnz;
		}
	} 
}




/**
 * Multithreaded Constructor for converting tuples matrix -> SpDCCols
 * @param[in] 	rhs if transpose=true,
 *	\n		then tuples is assumed to be a row sorted list of tuple objects
 *	\n		else tuples is assumed to be a column sorted list of tuple objects
 **/


template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols(IT nRow, IT nCol, IT nTuples, const std::tuple<IT, IT, NT>*  tuples, bool transpose)
: m(nRow), n(nCol), nnz(nTuples), splits(0)
{
    
    if(nnz == 0)	// m by n matrix of complete zeros
    {
        dcsc = NULL;
    }
    else
    {
        int totThreads=1;
#ifdef _OPENMP
#pragma omp parallel
        {
            totThreads = omp_get_num_threads();
        }
#endif
        
        std::vector <IT> tstart(totThreads);
        std::vector <IT> tend(totThreads);
        std::vector <IT> tdisp(totThreads+1);
        
        // extra memory, but replaces an O(nnz) loop by an O(nzc) loop
        IT* temp_jc = new IT[nTuples];
        IT* temp_cp = new IT[nTuples];

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            int threadID = 0;
#ifdef _OPENMP
            threadID = omp_get_thread_num();
#endif
            IT start = threadID * (nTuples / totThreads);
            IT end = (threadID + 1) * (nTuples / totThreads);
            if(threadID == (totThreads-1)) end = nTuples;
            IT curpos=start;
            if(end>start) // no work for the current thread
            {
                temp_jc[start] = std::get<1>(tuples[start]);
                temp_cp[start] = start;
                for (IT i = start+1; i < end; ++i)
                {
                    if(std::get<1>(tuples[i]) != temp_jc[curpos] )
                    {
                        temp_jc[++curpos] = std::get<1>(tuples[i]);
                        temp_cp[curpos] = i;
                    }
                }
            }
           
            tstart[threadID] = start;
            if(end>start) tend[threadID] = curpos+1;
            else tend[threadID] = end; // start=end
        }

        
        // serial part
        for(int t=totThreads-1; t>0; --t)
        {
            if(tend[t] > tstart[t] && tend[t-1] > tstart[t-1])
            {
                if(temp_jc[tstart[t]] == temp_jc[tend[t-1]-1])
                {
                    tstart[t] ++;
                }
            }
        }
        
        tdisp[0] = 0;
        for(int t=0; t<totThreads; ++t)
        {
            tdisp[t+1] = tdisp[t] + tend[t] - tstart[t];
        }

        IT localnzc = tdisp[totThreads];
        dcsc = new Dcsc<IT,NT>(nTuples,localnzc);
    
#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            int threadID = 0;
#ifdef _OPENMP
            threadID = omp_get_thread_num();
#endif
            std::copy(temp_jc + tstart[threadID],  temp_jc + tend[threadID], dcsc->jc + tdisp[threadID]);
            std::copy(temp_cp + tstart[threadID],  temp_cp + tend[threadID], dcsc->cp + tdisp[threadID]);
        }
        dcsc->cp[localnzc] = nTuples;

        delete [] temp_jc;
        delete [] temp_cp;
        
#ifdef _OPENMP
#pragma omp parallel for schedule (static)
#endif
        for(IT i=0; i<nTuples; ++i)
        {
            dcsc->ir[i]  = std::get<0>(tuples[i]);
            dcsc->numx[i] = std::get<2>(tuples[i]);
        }
     }
    
    if(transpose) Transpose(); // this is not efficient, think to improve later. We included this parameter anyway to make this constructor different from another constracttor when the fourth argument is passed as 0.
}



/*
template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols(IT nRow, IT nCol, IT nTuples, const tuple<IT, IT, NT>*  tuples)
: m(nRow), n(nCol), nnz(nTuples), splits(0)
{
    
    if(nnz == 0)	// m by n matrix of complete zeros
    {
        dcsc = NULL;
    }
    else
    {
        IT localnzc = 1;
#pragma omp parallel for schedule (static) default(shared) reduction(+:localnzc)
        for(IT i=1; i<nTuples; ++i) // not scaling well, try my own version
        {
            if(std::get<1>(tuples[i]) != std::get<1>(tuples[i-1]))
            {
                ++localnzc;
            }
        }
        
        dcsc = new Dcsc<IT,NT>(nTuples,localnzc);
        dcsc->jc[0]  = std::get<1>(tuples[0]);
        dcsc->cp[0] = 0;
        
#pragma omp parallel for schedule (static)
        for(IT i=0; i<nTuples; ++i)
        {
            dcsc->ir[i]  = std::get<0>(tuples[i]);
            dcsc->numx[i] = std::get<2>(tuples[i]);
        }
        
        IT jspos = 1;
        for(IT i=1; i<nTuples; ++i) // now this loop
        {
            if(std::get<1>(tuples[i]) != dcsc->jc[jspos-1])
            {
                dcsc->jc[jspos] = std::get<1>(tuples[i]);
                dcsc->cp[jspos++] = i;
            }
        }
        dcsc->cp[jspos] = nTuples;
    }
}
*/

/****************************************************************************/
/************************** PUBLIC OPERATORS ********************************/
/****************************************************************************/

/**
 * The assignment operator operates on an existing object
 * The assignment operator is the only operator that is not inherited.
 * But there is no need to call base's assigment operator as it has no data members
 */
template <class IT, class NT>
SpDCCols<IT,NT> & SpDCCols<IT,NT>::operator=(const SpDCCols<IT,NT> & rhs)
{
	// this pointer stores the address of the class instance
	// check for self assignment using address comparison
	if(this != &rhs)		
	{
		if(dcsc != NULL && nnz > 0)
		{
			delete dcsc;
		}
		if(rhs.dcsc != NULL)	
		{
			dcsc = new Dcsc<IT,NT>(*(rhs.dcsc));
			nnz = rhs.nnz;
		}
		else
		{
			dcsc = NULL;
			nnz = 0;
		}
		
		m = rhs.m; 
		n = rhs.n;
		splits = rhs.splits;
	}
	return *this;
}

template <class IT, class NT>
SpDCCols<IT,NT> & SpDCCols<IT,NT>::operator+= (const SpDCCols<IT,NT> & rhs)
{
	// this pointer stores the address of the class instance
	// check for self assignment using address comparison
	if(this != &rhs)		
	{
		if(m == rhs.m && n == rhs.n)
		{
			if(rhs.nnz == 0)
			{
				return *this;
			}
			else if(nnz == 0)
			{
				dcsc = new Dcsc<IT,NT>(*(rhs.dcsc));
				nnz = dcsc->nz;
			}
			else
			{
				(*dcsc) += (*(rhs.dcsc));
				nnz = dcsc->nz;
			}		
		}
		else
		{
			std::cout<< "Not addable: " << m  << "!=" << rhs.m << " or " << n << "!=" << rhs.n <<std::endl;		
		}
	}
	else
	{
		std::cout<< "Missing feature (A+A): Use multiply with 2 instead !"<<std::endl;	
	}
	return *this;
}

template <class IT, class NT>
template <typename _UnaryOperation, typename GlobalIT>
SpDCCols<IT,NT>* SpDCCols<IT,NT>::PruneI(_UnaryOperation __unary_op, bool inPlace, GlobalIT rowOffset, GlobalIT colOffset)
{
	if(nnz > 0)
	{
		Dcsc<IT,NT>* ret = dcsc->PruneI (__unary_op, inPlace, rowOffset, colOffset);
		if (inPlace)
		{
			nnz = dcsc->nz;
	
			if(nnz == 0) 
			{	
				delete dcsc;
				dcsc = NULL;
			}
			return NULL;
		}
		else
		{
			// wrap the new pruned Dcsc into a new SpDCCols
			SpDCCols<IT,NT>* retcols = new SpDCCols<IT, NT>();
			retcols->dcsc = ret;
			retcols->nnz = retcols->dcsc->nz;
			retcols->n = n;
			retcols->m = m;
			return retcols;
		}
	}
	else
	{
		if (inPlace)
		{
			return NULL;
		}
		else
		{
			SpDCCols<IT,NT>* retcols = new SpDCCols<IT, NT>();
			retcols->dcsc = NULL;
			retcols->nnz = 0;
			retcols->n = n;
			retcols->m = m;
			return retcols;
		}
	}
}

template <class IT, class NT>
template <typename _UnaryOperation>
SpDCCols<IT,NT>* SpDCCols<IT,NT>::Prune(_UnaryOperation __unary_op, bool inPlace)
{
	if(nnz > 0)
	{
		Dcsc<IT,NT>* ret = dcsc->Prune (__unary_op, inPlace);
		if (inPlace)
		{
			nnz = dcsc->nz;
	
			if(nnz == 0) 
			{	
				delete dcsc;
				dcsc = NULL;
			}
			return NULL;
		}
		else
		{
			// wrap the new pruned Dcsc into a new SpDCCols
			SpDCCols<IT,NT>* retcols = new SpDCCols<IT, NT>();
			retcols->dcsc = ret;
			retcols->nnz = retcols->dcsc->nz;
			retcols->n = n;
			retcols->m = m;
			return retcols;
		}
	}
	else
	{
		if (inPlace)
		{
			return NULL;
		}
		else
		{
			SpDCCols<IT,NT>* retcols = new SpDCCols<IT, NT>();
			retcols->dcsc = NULL;
			retcols->nnz = 0;
			retcols->n = n;
			retcols->m = m;
			return retcols;
		}
	}
}



template <class IT, class NT>
template <typename _BinaryOperation>
SpDCCols<IT,NT>* SpDCCols<IT,NT>::PruneColumn(NT* pvals, _BinaryOperation __binary_op, bool inPlace)
{
    if(nnz > 0)
    {
        Dcsc<IT,NT>* ret = dcsc->PruneColumn (pvals, __binary_op, inPlace);
        if (inPlace)
        {
            nnz = dcsc->nz;
            
            if(nnz == 0)
            {
                delete dcsc;
                dcsc = NULL;
            }
            return NULL;
        }
        else
        {
            // wrap the new pruned Dcsc into a new SpDCCols
            SpDCCols<IT,NT>* retcols = new SpDCCols<IT, NT>();
            retcols->dcsc = ret;
            retcols->nnz = retcols->dcsc->nz;
            retcols->n = n;
            retcols->m = m;
            return retcols;
        }
    }
    else
    {
        if (inPlace)
        {
            return NULL;
        }
        else
        {
            SpDCCols<IT,NT>* retcols = new SpDCCols<IT, NT>();
            retcols->dcsc = NULL;
            retcols->nnz = 0;
            retcols->n = n;
            retcols->m = m;
            return retcols;
        }
    }
}

template <class IT, class NT>
void SpDCCols<IT,NT>::PruneColumnByIndex(const std::vector<IT>& ci)
{
    if (nnz > 0)
    {
        dcsc->PruneColumnByIndex(ci);
        nnz = dcsc->nz;
    }
}

template <class IT, class NT>
template <typename _BinaryOperation>
SpDCCols<IT,NT>* SpDCCols<IT,NT>::PruneColumn(IT* pinds, NT* pvals, _BinaryOperation __binary_op, bool inPlace)
{
    if(nnz > 0)
    {
        Dcsc<IT,NT>* ret = dcsc->PruneColumn (pinds, pvals, __binary_op, inPlace);
        if (inPlace)
        {
            nnz = dcsc->nz;

            if(nnz == 0)
            {
                delete dcsc;
                dcsc = NULL;
            }
            return NULL;
        }
        else
        {
            // wrap the new pruned Dcsc into a new SpDCCols
            SpDCCols<IT,NT>* retcols = new SpDCCols<IT, NT>();
            retcols->dcsc = ret;
            retcols->nnz = retcols->dcsc->nz;
            retcols->n = n;
            retcols->m = m;
            return retcols;
        }
    }
    else
    {
        if (inPlace)
        {
            return NULL;
        }
        else
        {
            SpDCCols<IT,NT>* retcols = new SpDCCols<IT, NT>();
            retcols->dcsc = NULL;
            retcols->nnz = 0;
            retcols->n = n;
            retcols->m = m;
            return retcols;
        }
    }
}


template <class IT, class NT>
void SpDCCols<IT,NT>::SetDifference (const SpDCCols<IT,NT> & rhs)
{
	if(this != &rhs)		
	{
		if(m == rhs.m && n == rhs.n)
		{
			if(rhs.nnz == 0)
			{
				return;
			}
			else if (rhs.nnz != 0 && nnz != 0)
			{
				dcsc->SetDifference (*(rhs.dcsc));
				nnz = dcsc->nz;
				if(nnz == 0 )
					dcsc = NULL;
			}		
		}
		else
		{
			std::cout<< "Matrices do not conform for A - B !"<<std::endl;		
		}
	}
	else
	{
		std::cout<< "Missing feature (A .* A): Use Square_EWise() instead !"<<std::endl;	
	}
}

// Aydin (June 2021): Make the exclude case of this call SetDifference above instead
template <class IT, class NT>
void SpDCCols<IT,NT>::EWiseMult (const SpDCCols<IT,NT> & rhs, bool exclude)
{
	if(this != &rhs)		
	{
		if(m == rhs.m && n == rhs.n)
		{
			if(rhs.nnz == 0)
			{
				if(exclude)	// then we don't exclude anything
				{
					return;
				}
				else	// A .* zeros() is zeros()
				{
					*this = SpDCCols<IT,NT>(0,m,n,0);	// completely reset the matrix
				}
			}
			else if (rhs.nnz != 0 && nnz != 0)
			{
				dcsc->EWiseMult (*(rhs.dcsc), exclude);
				nnz = dcsc->nz;
				if(nnz == 0 )
					dcsc = NULL;
			}		
		}
		else
		{
			std::cout<< "Matrices do not conform for A .* op(B) !"<<std::endl;		
		}
	}
	else
	{
		std::cout<< "Missing feature (A .* A): Use Square_EWise() instead !"<<std::endl;	
	}
}

/**
 * @Pre {scaler should NOT contain any zero entries}
 */
template <class IT, class NT>
void SpDCCols<IT,NT>::EWiseScale(NT ** scaler, IT m_scaler, IT n_scaler)
{
	if(m == m_scaler && n == n_scaler)
	{
		if(nnz > 0)
			dcsc->EWiseScale (scaler);
	}
	else
	{
		std::cout<< "Matrices do not conform for EWiseScale !"<<std::endl;		
	}
}	


/****************************************************************************/
/********************* PUBLIC MEMBER FUNCTIONS ******************************/
/****************************************************************************/

template <class IT, class NT>
void SpDCCols<IT,NT>::CreateImpl(IT * _cp, IT * _jc, IT * _ir, NT * _numx, IT _nz, IT _nzc, IT _m, IT _n)
{
    m = _m;
    n = _n;
    nnz =  _nz;
    
    if(nnz > 0)
        dcsc = new Dcsc<IT,NT>(_cp, _jc, _ir, _numx, _nz, _nzc, false);	// memory not owned by DCSC
    else
        dcsc = NULL; 
}

template <class IT, class NT>
void SpDCCols<IT,NT>::CreateImpl(const std::vector<IT> & essentials)
{
	assert(essentials.size() == esscount);
	nnz = essentials[0];
	m = essentials[1];
	n = essentials[2];

	if(nnz > 0)
		dcsc = new Dcsc<IT,NT>(nnz,essentials[3]);
	else
		dcsc = NULL; 
}

template <class IT, class NT>
void SpDCCols<IT,NT>::CreateImpl(IT size, IT nRow, IT nCol, std::tuple<IT, IT, NT> * mytuples)
{
	SpTuples<IT,NT> tuples(size, nRow, nCol, mytuples);        
	tuples.SortColBased();
	
#ifdef DEBUG
  std::pair<IT,IT> rlim = tuples.RowLimits(); 
  std::pair<IT,IT> clim = tuples.ColLimits();

  std::ofstream oput;
  std::stringstream ss;
  std::string rank;
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	ss << myrank;
	ss >> rank;
  std::string ofilename = "Read";
	ofilename += rank;
	oput.open(ofilename.c_str(), std::ios_base::app );
	oput << "Creating of dimensions " << nRow << "-by-" << nCol << " of size: " << size << 
			" with row range (" << rlim.first  << "," << rlim.second << ") and column range (" << clim.first  << "," << clim.second << ")" << std::endl;
	if(tuples.getnnz() > 0)
	{ 
		IT minfr = joker::get<0>(tuples.front());
		IT minto = joker::get<1>(tuples.front());
		IT maxfr = joker::get<0>(tuples.back());
		IT maxto = joker::get<1>(tuples.back());

		oput << "Min: " << minfr << ", " << minto << "; Max: " << maxfr << ", " << maxto << std::endl;
	}
	oput.close();
#endif

	SpDCCols<IT,NT> object(tuples, false);	
	*this = object;
}


template <class IT, class NT>
std::vector<IT> SpDCCols<IT,NT>::GetEssentials() const
{
	std::vector<IT> essentials(esscount);
	essentials[0] = nnz;
	essentials[1] = m;
	essentials[2] = n;
	essentials[3] = (nnz > 0) ? dcsc->nzc : 0;
	return essentials;
}

template <class IT, class NT>
template <typename NNT>
SpDCCols<IT,NT>::operator SpDCCols<IT,NNT> () const
{
	Dcsc<IT,NNT> * convert;
	if(nnz > 0)
		convert = new Dcsc<IT,NNT>(*dcsc);
	else
		convert = NULL;

	return SpDCCols<IT,NNT>(m, n, convert);
}


template <class IT, class NT>
template <typename NIT, typename NNT>
SpDCCols<IT,NT>::operator SpDCCols<NIT,NNT> () const
{
	Dcsc<NIT,NNT> * convert;
	if(nnz > 0)
		convert = new Dcsc<NIT,NNT>(*dcsc);
	else
		convert = NULL;

	return SpDCCols<NIT,NNT>(m, n, convert);
}


template <class IT, class NT>
Arr<IT,NT> SpDCCols<IT,NT>::GetArrays() const
{
	Arr<IT,NT> arr(3,1);

	if(nnz > 0)
	{
		arr.indarrs[0] = LocArr<IT,IT>(dcsc->cp, dcsc->nzc+1);
		arr.indarrs[1] = LocArr<IT,IT>(dcsc->jc, dcsc->nzc);
		arr.indarrs[2] = LocArr<IT,IT>(dcsc->ir, dcsc->nz);
		arr.numarrs[0] = LocArr<NT,IT>(dcsc->numx, dcsc->nz);
	}
	else
	{
		arr.indarrs[0] = LocArr<IT,IT>(NULL, 0);
		arr.indarrs[1] = LocArr<IT,IT>(NULL, 0);
		arr.indarrs[2] = LocArr<IT,IT>(NULL, 0);
		arr.numarrs[0] = LocArr<NT,IT>(NULL, 0);
	
	}
	return arr;
}

/**
  * O(nnz log(nnz)) time Transpose function
  * \remarks Performs a lexicographical sort
  * \remarks Mutator function (replaces the calling object with its transpose)
  */
template <class IT, class NT>
void SpDCCols<IT,NT>::Transpose()
{
	if(nnz > 0)
	{
		SpTuples<IT,NT> Atuples(*this);
		Atuples.SortRowBased();

		// destruction of (*this) is handled by the assignment operator
		*this = SpDCCols<IT,NT>(Atuples,true);
	}
	else
	{
		*this = SpDCCols<IT,NT>(0, n, m, 0);
	}
}


/**
  * O(nnz log(nnz)) time Transpose function
  * \remarks Performs a lexicographical sort
  * \remarks Const function (doesn't mutate the calling object)
  */
template <class IT, class NT>
SpDCCols<IT,NT> SpDCCols<IT,NT>::TransposeConst() const
{
	SpTuples<IT,NT> Atuples(*this);
	Atuples.SortRowBased();

	return SpDCCols<IT,NT>(Atuples,true);
}

/**
 * O(nnz log(nnz)) time Transpose function
 * \remarks Performs a lexicographical sort
 * \remarks Const function (doesn't mutate the calling object)
 */
template <class IT, class NT>
SpDCCols<IT,NT> * SpDCCols<IT,NT>::TransposeConstPtr() const
{
	SpTuples<IT,NT> Atuples(*this);
	Atuples.SortRowBased();
	
	return new SpDCCols<IT,NT>(Atuples,true);
}

/** 
  * Splits the matrix into two parts, simply by cutting along the columns
  * Simple algorithm that doesn't intend to split perfectly, but it should do a pretty good job
  * Practically destructs the calling object also (frees most of its memory)
  * \todo {special case of ColSplit, to be deprecated...}
  */
template <class IT, class NT>
void SpDCCols<IT,NT>::Split(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB) 
{
	IT cut = n/2;
	if(cut == 0)
	{
		std::cout<< "Matrix is too small to be splitted" << std::endl;
		return;
	}

	Dcsc<IT,NT> *Adcsc = NULL;
	Dcsc<IT,NT> *Bdcsc = NULL;

	if(nnz != 0)
	{
		dcsc->Split(Adcsc, Bdcsc, cut);
	}

	partA = SpDCCols<IT,NT> (m, cut, Adcsc);
	partB = SpDCCols<IT,NT> (m, n-cut, Bdcsc);
	
	// handle destruction through assignment operator
	*this = SpDCCols<IT, NT>();		
}

/**
 * Splits the matrix into "parts", simply by cutting along the columns
 * Simple algorithm that doesn't intend to split perfectly, but it should do a pretty good job
 * Practically destructs the calling object also (frees most of its memory)
 */
template <class IT, class NT>
void SpDCCols<IT,NT>::ColSplit(int parts, std::vector< SpDCCols<IT,NT> > & matrices)
{
    if(parts < 2)
    {
        matrices.emplace_back(*this);
    }
    else
    {
        std::vector<IT> cuts(parts-1);
        for(int i=0; i< (parts-1); ++i)
        {
            cuts[i] = (i+1) * (n/parts);
        }
        if(n < parts)
        {
            std::cout<< "Matrix is too small to be splitted" << std::endl;
            return;
        }
        std::vector< Dcsc<IT,NT> * > dcscs(parts, NULL);
        
        if(nnz != 0)
        {
            dcsc->ColSplit(dcscs, cuts);
        }
        
        for(int i=0; i< (parts-1); ++i)
        {
            SpDCCols<IT,NT> matrix = SpDCCols<IT,NT>(m, (n/parts), dcscs[i]);
            matrices.emplace_back(matrix);
        }
        SpDCCols<IT,NT> matrix = SpDCCols<IT,NT>(m, n-cuts[parts-2], dcscs[parts-1]);
        matrices.emplace_back(matrix);
    }
    *this = SpDCCols<IT, NT>();		    // handle destruction through assignment operator
}

/**
 * Splits the matrix into "parts", simply by cutting along the columns
 * Simple algorithm that doesn't intend to split perfectly, but it should do a pretty good job
 * Practically destructs the calling object also (frees most of its memory)
 */
template <class IT, class NT>
void SpDCCols<IT,NT>::ColSplit(int parts, std::vector< SpDCCols<IT,NT>* > & matrices)
{
    if(parts < 2)
    {
        matrices.emplace_back(new SpDCCols<IT,NT>(*this));
    }
    else
    {
        std::vector<IT> cuts(parts-1);
        for(int i=0; i< (parts-1); ++i)
        {
            cuts[i] = (i+1) * (n/parts);
        }
        if(n < parts)
        {
            std::cout<< "Matrix is too small to be splitted" << std::endl;
            return;
        }
        std::vector< Dcsc<IT,NT> * > dcscs(parts, NULL);
        
        if(nnz != 0)
        {
            dcsc->ColSplit(dcscs, cuts);
        }
        
        for(int i=0; i< (parts-1); ++i)
        {
            SpDCCols<IT,NT>* matrix = new SpDCCols<IT,NT>(m, (n/parts), dcscs[i]);
            matrices.emplace_back(matrix);
        }
        SpDCCols<IT,NT>* matrix = new SpDCCols<IT,NT>(m, n-cuts[parts-2], dcscs[parts-1]);
        matrices.emplace_back(matrix);
    }
    *this = SpDCCols<IT, NT>();		    // handle destruction through assignment operator
}

/**
 * Splits the matrix into "parts", simply by cutting along the columns
 * Simple algorithm that doesn't intend to split perfectly, but it should do a pretty good job
 * Practically destructs the calling object also (frees most of its memory)
 */
template <class IT, class NT>
void SpDCCols<IT,NT>::ColSplit(std::vector<IT> & cutSizes, std::vector< SpDCCols<IT,NT> > & matrices)
{
    IT totn = 0;
    int parts = cutSizes.size();
    for(int i = 0; i < parts; i++) totn += cutSizes[i];
    if(parts < 2){
        matrices.emplace_back(*this);
    }
    else if(totn != n){
        std::cout << "Cut sizes are not appropriate" << std::endl;
        return;
    }
    else{
        std::vector<IT> cuts(parts-1);
        cuts[0] = cutSizes[0];
        for(int i = 1; i < parts-1; i++){
            cuts[i] = cuts[i-1] + cutSizes[i];
        }
        std::vector< Dcsc<IT,NT> * > dcscs(parts, NULL);
        
        if(nnz != 0){
            dcsc->ColSplit(dcscs, cuts);
        }
        
        for(int i=0; i< parts; ++i){
            SpDCCols<IT,NT> matrix = SpDCCols<IT,NT>(m, cutSizes[i], dcscs[i]);
            matrices.emplace_back(matrix);
        }
    }
    *this = SpDCCols<IT, NT>();
}

/**
 * [Overloaded function. To be used in case of vector of SpDCCols pointer.]
 * Splits the matrix into "parts", simply by cutting along the columns
 * Simple algorithm that doesn't intend to split perfectly, but it should do a pretty good job
 * Practically destructs the calling object also (frees most of its memory)
 */
template <class IT, class NT>
void SpDCCols<IT,NT>::ColSplit(std::vector<IT> & cutSizes, std::vector< SpDCCols<IT,NT>* > & matrices)
{
    IT totn = 0;
    int parts = cutSizes.size();
    for(int i = 0; i < parts; i++) totn += cutSizes[i];
    if(parts < 2){
        matrices.emplace_back(new SpDCCols<IT,NT>(*this));
    }
    else if(totn != n){
        std::cout << "Cut sizes are not appropriate" << std::endl;
        return;
    }
    else{
        std::vector<IT> cuts(parts-1);
        cuts[0] = cutSizes[0];
        for(int i = 1; i < parts-1; i++){
            cuts[i] = cuts[i-1] + cutSizes[i];
        }
        std::vector< Dcsc<IT,NT> * > dcscs(parts, NULL);
        
        if(nnz != 0){
            dcsc->ColSplit(dcscs, cuts);
        }
        
        for(int i=0; i< parts; ++i){
            SpDCCols<IT,NT>* matrix = new SpDCCols<IT,NT>(m, cutSizes[i], dcscs[i]);
            matrices.emplace_back(matrix);
        }
    }
    *this = SpDCCols<IT, NT>();
}

/**
 * Concatenates (merges) multiple matrices (cut along the columns) into 1 piece
 * ColSplit() method should have been executed on the object beforehand
 */
template <class IT, class NT>
void SpDCCols<IT,NT>::ColConcatenate(std::vector< SpDCCols<IT,NT> > & matrices)
{
    std::vector< SpDCCols<IT,NT> * > nonempties;
    std::vector< Dcsc<IT,NT> * > dcscs;
    std::vector< IT > offsets;
    IT runningoffset = 0;

    for(size_t i=0; i< matrices.size(); ++i)
    {
        if(matrices[i].nnz != 0)
        {
            nonempties.push_back(&(matrices[i]));
            dcscs.push_back(matrices[i].dcsc);
            offsets.push_back(runningoffset);
        }
        runningoffset += matrices[i].n;
    }
    
    if(nonempties.size() < 1)
    {
#ifdef DEBUG
        std::cout << "Nothing to ColConcatenate" << std::endl;
#endif
        n = runningoffset;
    }/*
    else if(nonempties.size() < 2)
    {
        *this =  *(nonempties[0]);
        n = runningoffset; 
    }*/
    else // nonempties.size() > 1
    {
        Dcsc<IT,NT> * Cdcsc = new Dcsc<IT,NT>();
        Cdcsc->ColConcatenate(dcscs, offsets);
        *this = SpDCCols<IT,NT> (nonempties[0]->m, runningoffset, Cdcsc);
    }
    
    // destruct parameters
    for(size_t i=0; i< matrices.size(); ++i)
    {
        matrices[i] = SpDCCols<IT,NT>();
    }
}

/**
 * Concatenates (merges) multiple matrices (cut along the columns) into 1 piece
 * ColSplit() method should have been executed on the object beforehand
 */
template <class IT, class NT>
void SpDCCols<IT,NT>::ColConcatenate(std::vector< SpDCCols<IT,NT>* > & matrices)
{
    std::vector< SpDCCols<IT,NT> * > nonempties;
    std::vector< Dcsc<IT,NT> * > dcscs;
    std::vector< IT > offsets;
    IT runningoffset = 0;

    for(size_t i=0; i< matrices.size(); ++i)
    {
        if(matrices[i]->nnz != 0)
        {
            nonempties.push_back(matrices[i]);
            dcscs.push_back(matrices[i]->dcsc);
            offsets.push_back(runningoffset);
        }
        runningoffset += matrices[i]->n;
    }
    
    if(nonempties.size() < 1)
    {
#ifdef DEBUG
        std::cout << "Nothing to ColConcatenate" << std::endl;
#endif
        n = runningoffset;
    }/*
    else if(nonempties.size() < 2)
    {
        *this =  *(nonempties[0]);
        n = runningoffset; 
    }*/
    else // nonempties.size() > 1
    {
        Dcsc<IT,NT> * Cdcsc = new Dcsc<IT,NT>();
        Cdcsc->ColConcatenate(dcscs, offsets);
        *this = SpDCCols<IT,NT> (nonempties[0]->m, runningoffset, Cdcsc);
    }
    
    // destruct parameters
    for(size_t i=0; i< matrices.size(); ++i)
    {
        delete matrices[i];
    }
}


/** 
  * Merges two matrices (cut along the columns) into 1 piece
  * Split method should have been executed on the object beforehand
 **/
template <class IT, class NT>
void SpDCCols<IT,NT>::Merge(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB) 
{
	assert( partA.m == partB.m );

	Dcsc<IT,NT> * Cdcsc = new Dcsc<IT,NT>();

	if(partA.nnz == 0 && partB.nnz == 0)
	{
		Cdcsc = NULL;
	}
	else if(partA.nnz == 0)
	{
		Cdcsc = new Dcsc<IT,NT>(*(partB.dcsc));
    std::transform(Cdcsc->jc, Cdcsc->jc + Cdcsc->nzc, Cdcsc->jc, std::bind(std::plus<IT>(),std::placeholders::_1, partA.n));
	}
	else if(partB.nnz == 0)
	{
		Cdcsc = new Dcsc<IT,NT>(*(partA.dcsc));
	}
	else
	{
		Cdcsc->Merge(partA.dcsc, partB.dcsc, partA.n);
	}
	*this = SpDCCols<IT,NT> (partA.m, partA.n + partB.n, Cdcsc);

	partA = SpDCCols<IT, NT>();	
	partB = SpDCCols<IT, NT>();
}

/**
 * C += A*B' (Using OuterProduct Algorithm)
 * This version is currently limited to multiplication of matrices with the same precision 
 * (e.g. it can't multiply double-precision matrices with booleans)
 * The multiplication is on the specified semiring (passed as parameter)
 */
template <class IT, class NT>
template <class SR>
int SpDCCols<IT,NT>::PlusEq_AnXBt(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B)
{
	if(A.isZero() || B.isZero())
	{
		return -1;	// no need to do anything
	}
	Isect<IT> *isect1, *isect2, *itr1, *itr2, *cols, *rows;
	SpHelper::SpIntersect(*(A.dcsc), *(B.dcsc), cols, rows, isect1, isect2, itr1, itr2);
	
	IT kisect = static_cast<IT>(itr1-isect1);		// size of the intersection ((itr1-isect1) == (itr2-isect2))
	if(kisect == 0)
	{
		DeleteAll(isect1, isect2, cols, rows);
		return -1;
	}
	
	StackEntry< NT, std::pair<IT,IT> > * multstack;
	IT cnz = SpHelper::SpCartesian< SR > (*(A.dcsc), *(B.dcsc), kisect, isect1, isect2, multstack);  
	DeleteAll(isect1, isect2, cols, rows);

	IT mdim = A.m;	
	IT ndim = B.m;		// since B has already been transposed
	if(isZero())
	{
		dcsc = new Dcsc<IT,NT>(multstack, mdim, ndim, cnz);
	}
	else
	{
		dcsc->AddAndAssign(multstack, mdim, ndim, cnz);
	}
	nnz = dcsc->nz;

	delete [] multstack;
	return 1;	
}

/**
 * C += A*B (Using ColByCol Algorithm)
 * This version is currently limited to multiplication of matrices with the same precision 
 * (e.g. it can't multiply double-precision matrices with booleans)
 * The multiplication is on the specified semiring (passed as parameter)
 */
template <class IT, class NT>
template <typename SR>
int SpDCCols<IT,NT>::PlusEq_AnXBn(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B)
{
	if(A.isZero() || B.isZero())
	{
		return -1;	// no need to do anything
	}
	StackEntry< NT, std::pair<IT,IT> > * multstack;
	int cnz = SpHelper::SpColByCol< SR > (*(A.dcsc), *(B.dcsc), A.n, multstack);  
	
	IT mdim = A.m;	
	IT ndim = B.n;
	if(isZero())
	{
		dcsc = new Dcsc<IT,NT>(multstack, mdim, ndim, cnz);
	}
	else
	{
		dcsc->AddAndAssign(multstack, mdim, ndim, cnz);
	}
	nnz = dcsc->nz;
	
	delete [] multstack;
	return 1;	
}


template <class IT, class NT>
template <typename SR>
int SpDCCols<IT,NT>::PlusEq_AtXBn(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B)
{
	std::cout << "PlusEq_AtXBn function has not been implemented yet !" << std::endl;
	return 0;
}

template <class IT, class NT>
template <typename SR>
int SpDCCols<IT,NT>::PlusEq_AtXBt(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B)
{
	std::cout << "PlusEq_AtXBt function has not been implemented yet !" << std::endl;
	return 0;
}


template <class IT, class NT>
SpDCCols<IT,NT> SpDCCols<IT,NT>::operator() (IT ri, IT ci) const
{
	IT * itr = std::find(dcsc->jc, dcsc->jc + dcsc->nzc, ci);
	if(itr != dcsc->jc + dcsc->nzc)
	{
		IT irbeg = dcsc->cp[itr - dcsc->jc];
		IT irend = dcsc->cp[itr - dcsc->jc + 1];

		IT * ele = std::find(dcsc->ir + irbeg, dcsc->ir + irend, ri);
		if(ele != dcsc->ir + irend)	
		{	
			SpDCCols<IT,NT> SingEleMat(1, 1, 1, 1);	// 1-by-1 matrix with 1 nonzero 
			*(SingEleMat.dcsc->numx) = dcsc->numx[ele - dcsc->ir];
			*(SingEleMat.dcsc->ir) = *ele; 
			*(SingEleMat.dcsc->jc) = *itr;
			(SingEleMat.dcsc->cp)[0] = 0;
			(SingEleMat.dcsc->cp)[1] = 1;
            return SingEleMat;
		}
		else
		{
			return SpDCCols<IT,NT>();  // 0-by-0 empty matrix
		}
	}
	else
	{
		return SpDCCols<IT,NT>();	 // 0-by-0 empty matrix		
	}
}

/** 
 * The almighty indexing polyalgorithm 
 * Calls different subroutines depending the sparseness of ri/ci
 */
template <class IT, class NT>
SpDCCols<IT,NT> SpDCCols<IT,NT>::operator() (const std::vector<IT> & ri, const std::vector<IT> & ci) const
{
	typedef PlusTimesSRing<NT,NT> PT;	

	IT rsize = ri.size();
	IT csize = ci.size();

	if(rsize == 0 && csize == 0)
	{
		// return an m x n matrix of complete zeros
		// since we don't know whether columns or rows are indexed
		return SpDCCols<IT,NT> (0, m, n, 0);		
	}
	else if(rsize == 0)
	{
		return ColIndex(ci);
	}
	else if(csize == 0)
	{
		SpDCCols<IT,NT> LeftMatrix(rsize, rsize, this->m, ri, true);
		return LeftMatrix.OrdColByCol< PT >(*this);
	}
	else	// this handles the (rsize=1 && csize=1) case as well
	{
		SpDCCols<IT,NT> LeftMatrix(rsize, rsize, this->m, ri, true);
		SpDCCols<IT,NT> RightMatrix(csize, this->n, csize, ci, false);
		return LeftMatrix.OrdColByCol< PT >( OrdColByCol< PT >(RightMatrix) );
	}
}

template <class IT, class NT>
std::ofstream & SpDCCols<IT,NT>::put(std::ofstream & outfile) const 
{
	if(nnz == 0)
	{
		outfile << "Matrix doesn't have any nonzeros" <<std::endl;
		return outfile;
	}
	SpTuples<IT,NT> tuples(*this); 
	outfile << tuples << std::endl;
	return outfile;
}


template <class IT, class NT>
std::ifstream & SpDCCols<IT,NT>::get(std::ifstream & infile)
{
	std::cout << "Getting... SpDCCols" << std::endl;
	IT m, n, nnz;
	infile >> m >> n >> nnz;
	SpTuples<IT,NT> tuples(nnz, m, n);        
	infile >> tuples;
	tuples.SortColBased();
        
	SpDCCols<IT,NT> object(tuples, false);
	*this = object;
	return infile;
}


template<class IT, class NT>
void SpDCCols<IT,NT>::PrintInfo(std::ofstream &  out) const
{
	out << "m: " << m ;
	out << ", n: " << n ;
	out << ", nnz: "<< nnz ;

	if(splits > 0)
	{
		out << ", local splits: " << splits << std::endl;
	}
	else
	{
		if(dcsc != NULL)
		{
			out << ", nzc: "<< dcsc->nzc << std::endl;
		}
		else
		{
			out <<", nzc: "<< 0 << std::endl;
		}
	}
}

template<class IT, class NT>
void SpDCCols<IT,NT>::PrintInfo() const
{
	std::cout << "m: " << m ;
	std::cout << ", n: " << n ;
	std::cout << ", nnz: "<< nnz ;

	if(splits > 0)
	{
		std::cout << ", local splits: " << splits << std::endl;
	}
	else
	{
		if(dcsc != NULL)
		{
			std::cout << ", nzc: "<< dcsc->nzc << std::endl;
		}
		else
		{
			std::cout <<", nzc: "<< 0 << std::endl;
		}

		if(m < PRINT_LIMIT && n < PRINT_LIMIT)	// small enough to print
		{
			std::string ** A = SpHelper::allocate2D<std::string>(m,n);
			for(IT i=0; i< m; ++i)
				for(IT j=0; j<n; ++j)
					A[i][j] = "-";
			if(dcsc != NULL)
			{
				for(IT i=0; i< dcsc->nzc; ++i)
				{
					for(IT j = dcsc->cp[i]; j<dcsc->cp[i+1]; ++j)
					{
						IT colid = dcsc->jc[i];
						IT rowid = dcsc->ir[j];
						A[rowid][colid] = std::to_string(dcsc->numx[j]);
					}
				}
			} 
			for(IT i=0; i< m; ++i)
			{
				for(IT j=0; j<n; ++j)
				{
					std::cout << A[i][j];
					std::cout << "\t";
				}
				std::cout << std::endl;
			}
			SpHelper::deallocate2D(A,m);
		}
	}
}


/****************************************************************************/
/********************* PRIVATE CONSTRUCTORS/DESTRUCTORS *********************/
/****************************************************************************/

//! Construct SpDCCols from Dcsc
template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols(IT nRow, IT nCol, Dcsc<IT,NT> * mydcsc)
:dcsc(mydcsc), m(nRow), n(nCol), splits(0)
{
	if (mydcsc == NULL) 
		nnz = 0;
	else
		nnz = mydcsc->nz;
}

//! Create a logical matrix from (row/column) indices array, used for indexing only
template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols (IT size, IT nRow, IT nCol, const std::vector<IT> & indices, bool isRow)
:m(nRow), n(nCol), nnz(size), splits(0)
{
	if(size > 0)
		dcsc = new Dcsc<IT,NT>(size,indices,isRow);
	else
		dcsc = NULL; 
}


/****************************************************************************/
/************************* PRIVATE MEMBER FUNCTIONS *************************/
/****************************************************************************/

template <class IT, class NT>
inline void SpDCCols<IT,NT>::CopyDcsc(Dcsc<IT,NT> * source)
{
	// source dcsc will be NULL if number of nonzeros is zero 
	if(source != NULL)	
		dcsc = new Dcsc<IT,NT>(*source);
	else
		dcsc = NULL;
}

/**
 * \return An indexed SpDCCols object without using multiplication 
 * \pre ci is sorted and is not completely empty.
 * \remarks it is OK for some indices ci[i] to be empty in the indexed SpDCCols matrix 
 *	[i.e. in the output, nzc does not need to be equal to n]
 */
template <class IT, class NT>
SpDCCols<IT,NT> SpDCCols<IT,NT>::ColIndex(const std::vector<IT> & ci) const
{
	IT csize = ci.size();
	if(nnz == 0)	// nothing to index
	{
		return SpDCCols<IT,NT>(0, m, csize, 0);	
	}
	else if(ci.empty())
	{
		return SpDCCols<IT,NT>(0, m,0, 0);
	}

	// First pass for estimation
	IT estsize = 0;
	IT estnzc = 0;
	for(IT i=0, j=0;  i< dcsc->nzc && j < csize;)
	{
		if((dcsc->jc)[i] < ci[j])
		{
			++i;
		}
		else if ((dcsc->jc)[i] > ci[j])
		{
			++j;
		}
		else
		{
			estsize +=  (dcsc->cp)[i+1] - (dcsc->cp)[i];
			++estnzc;
			++i;
			++j;
		}
	}
	
	SpDCCols<IT,NT> SubA(estsize, m, csize, estnzc);	
	if(estnzc == 0)
	{
		return SubA;		// no need to run the second pass
	}
	SubA.dcsc->cp[0] = 0;
	IT cnzc = 0;
	IT cnz = 0;
	for(IT i=0, j=0;  i < dcsc->nzc && j < csize;)
	{
		if((dcsc->jc)[i] < ci[j])
		{
			++i;
		}
		else if ((dcsc->jc)[i] > ci[j])		// an empty column for the output
		{
			++j;
		}
		else
		{
			IT columncount = (dcsc->cp)[i+1] - (dcsc->cp)[i];
			SubA.dcsc->jc[cnzc++] = j;
			SubA.dcsc->cp[cnzc] = SubA.dcsc->cp[cnzc-1] + columncount;
			std::copy(dcsc->ir + dcsc->cp[i], dcsc->ir + dcsc->cp[i+1], SubA.dcsc->ir + cnz);
			std::copy(dcsc->numx + dcsc->cp[i], dcsc->numx + dcsc->cp[i+1], SubA.dcsc->numx + cnz);
			cnz += columncount;
			++i;
			++j;
		}
	}
	return SubA;
}

template <class IT, class NT>
template <typename SR, typename NTR>
SpDCCols< IT, typename promote_trait<NT,NTR>::T_promote > SpDCCols<IT,NT>::OrdOutProdMult(const SpDCCols<IT,NTR> & rhs) const
{
	typedef typename promote_trait<NT,NTR>::T_promote T_promote;  

	if(isZero() || rhs.isZero())
	{
		return SpDCCols< IT, T_promote > (0, m, rhs.n, 0);		// return an empty matrix	
	}
	SpDCCols<IT,NTR> Btrans = rhs.TransposeConst();

	Isect<IT> *isect1, *isect2, *itr1, *itr2, *cols, *rows;
	SpHelper::SpIntersect(*dcsc, *(Btrans.dcsc), cols, rows, isect1, isect2, itr1, itr2);
	
	IT kisect = static_cast<IT>(itr1-isect1);		// size of the intersection ((itr1-isect1) == (itr2-isect2))
	if(kisect == 0)
	{
		DeleteAll(isect1, isect2, cols, rows);
		return SpDCCols< IT, T_promote > (0, m, rhs.n, 0);	
	}
	StackEntry< T_promote, std::pair<IT,IT> > * multstack;
	IT cnz = SpHelper::SpCartesian< SR > (*dcsc, *(Btrans.dcsc), kisect, isect1, isect2, multstack);  
	DeleteAll(isect1, isect2, cols, rows);

	Dcsc<IT, T_promote> * mydcsc = NULL;
	if(cnz > 0)
	{
		mydcsc = new Dcsc< IT,T_promote > (multstack, m, rhs.n, cnz);
		delete [] multstack;
	}
	return SpDCCols< IT,T_promote > (m, rhs.n, mydcsc);	
}


template <class IT, class NT>
template <typename SR, typename NTR>
SpDCCols< IT, typename promote_trait<NT,NTR>::T_promote > SpDCCols<IT,NT>::OrdColByCol(const SpDCCols<IT,NTR> & rhs) const
{
	typedef typename promote_trait<NT,NTR>::T_promote T_promote;  

	if(isZero() || rhs.isZero())
	{
		return SpDCCols<IT, T_promote> (0, m, rhs.n, 0);		// return an empty matrix	
	}
	StackEntry< T_promote, std::pair<IT,IT> > * multstack;
	IT cnz = SpHelper::SpColByCol< SR > (*dcsc, *(rhs.dcsc), n, multstack);  
	
	Dcsc<IT,T_promote > * mydcsc = NULL;
	if(cnz > 0)
	{
		mydcsc = new Dcsc< IT,T_promote > (multstack, m, rhs.n, cnz);
		delete [] multstack;
	}
	return SpDCCols< IT,T_promote > (m, rhs.n, mydcsc);	
}

template<class IT, class NT>
void SpDCCols<IT, NT>::WriteMM(std::string filename, bool onebased)
{
    std::vector<IT> rows, cols;
    std::vector<NT> vals;
    SpHelper::RemoveExistingFile(filename);
    for(typename SpDCCols<IT, NT>::SpColIter colit = begcol(); colit != endcol(); ++colit)
    {
        IT gcol = colit.colid();
        for(typename SpDCCols<IT, NT>::SpColIter::NzIter nzit = begnz(colit); nzit != endnz(colit); ++nzit)
        {
            IT grow = nzit.rowid();
            NT val = nzit.value();
            rows.push_back(grow);
            cols.push_back(gcol);
            vals.push_back(val);
        }
    }
    IT locrow = getnrow();
    IT loccol = getncol();
    std::ofstream os(filename, std::ios_base::binary);
    // write option is not useful to change precision. Go to line 602, filed_conv.hpp to change it manually.
}




template<class IT, class NT>
void SpDCCols<IT, NT>::ProjectRow(std::vector<int> & reducevector){
    reducevector = std::vector<int>(getnrow(),(int)0);
    for(typename SpDCCols<IT, NT>::SpColIter colit = begcol(); colit != endcol(); ++colit)
    {
        IT gcol = colit.colid();
        for(typename SpDCCols<IT, NT>::SpColIter::NzIter nzit = begnz(colit); nzit != endnz(colit); ++nzit)
        {
            IT grow = nzit.rowid();
            NT val = nzit.value();
            reducevector[grow] += 1;
        }
    }
}

}


#endif
