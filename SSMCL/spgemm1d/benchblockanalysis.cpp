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

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mpi.h>
#include <new>
#include <numeric>
#include <omp.h>
#include <regex>
#include <stdint.h>
#include <streambuf>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <tuple>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>
#include <sstream>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid1D.h"
#include "CombBLAS/MPIType.h"
#include "CombBLAS/Operations.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/SPA.h"
#include "CombBLAS/SpCCols.h"
#include "CombBLAS/SpDCCols.h"
#include "CombBLAS/SpDefs.h"
#include "CombBLAS/SpHelper.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/SpParMat.h"
#include "CombBLAS/SpParMat1D.h"
#include "CombBLAS/SpTuples.h"
#include "CombBLAS/csc.h"
#include "CombBLAS/GraphPartitioner.h"
#include "CombBLAS/dcsc.h"
#include "CombBLAS/mtSpGEMM.h"
#include "Tommy/tommytypes.h"
#include "CombBLAS/GraphPartitioner.h"
#include <fstream>
#include <parallel/numeric>
#include <parallel/algorithm>


using namespace std;
using namespace combblas;


template <class IT, class NT>
class Dist 
{ 
public: 
	typedef SpDCCols < IT, NT > DCCols;
	typedef SpParMat < IT, NT, DCCols > MPI_DCCols;
    typedef SpParMat1D < IT, NT, DCCols > MPI_DCCols1D;
};

#define EPS 0.0001

#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif

typedef int64_t IT;
typedef double NT;
typedef SpDCCols < int64_t, double > DER;
typedef SpParMat1D<IT,NT,DER> Sp1D;
typedef SpParMat<IT,NT,DER> Sp2D;
typedef PlusTimesSRing<double, double> PTFF;

void MatrixEqual(Sp2D & A2D, Sp2D & B2D, std::string description){
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(A2D == B2D){
        if(myrank == 0) std::cerr << "1D " << description << " is correct!" << std::endl;
    }else{
        if(myrank == 0) std::cerr << "1D "<< description <<  "is wrong!" << std::endl;
    }
}

void MatrixEqual(Sp1D & A1D, Sp2D & A2D, std::string description){
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    Sp2D A1D2D(A1D,A1D.getblocksizevec());
    if(A1D == A2D){
        if(myrank == 0) std::cerr << "1D " << description << " is correct!" << std::endl;
    }else{
        if(myrank == 0) std::cerr << "1D " << description << " is wrong!" << std::endl;
    }
}


int main(int argc, char* argv[])
{
    int nprocs, myrank, provided, numThreads;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    assert(provided == MPI_THREAD_MULTIPLE);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    #pragma omp parallel
    {
        #pragma omp critical
        numThreads = omp_get_num_threads();
    }

    if(myrank == 0) {
        printf("starting spgemm1drdma binary!\n");
        printf("Total MPI processes %d number of threads are %d \n", nprocs ,numThreads);
        fflush(stdout);
    }
    SpHelper::initdatasetmap();
    SpgemmOpts opts;
    OptParser::parse(argc, argv, &opts);
    if(myrank == 0) opts.PrintOpts();
    std::cerr << "---- starting benchmarking -----" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
