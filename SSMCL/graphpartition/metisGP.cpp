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
#include <parmetis.h>
#include <metis.h>
#include <mpi.h>
using namespace combblas;
using namespace std;

typedef int64_t IT;
typedef double NT;
typedef SpDCCols < int64_t, double > DER;
typedef SpParMat1D<IT,NT,DER> Sp1D;
typedef SpParMat<IT,NT,DER> Sp2D;
typedef PlusTimesSRing<double, double> PTFF;

//!!!!!!! IT type must be LONG!!!!!!!!!!!!

int main(int argc, char ** argv){
    MPI_Init(NULL,NULL);
    SpHelper::initdatasetmap();
    SpgemmOpts opts;
    OptParser::parse(argc, argv, &opts);
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(nprocs != 1) {
        if(myrank ==0)printf("launching with 1 mpi process!\n");
        exit(0);
    }
    for(auto name : opts.dataset){
        shared_ptr<CommGrid1D> commgrid1d = make_shared<CommGrid1D>(MPI_COMM_WORLD);
        std::string mtxfilename = opts.fullfilepath[name];
        Sp1D A(commgrid1d);
        A.ParallelReadMM(mtxfilename);
        SpCCols<IT, NT> spccols(*A.seqptr());
        Csc<IT,NT> * csc = spccols.GetCSC();
        IT wgtflag = 0;
        IT numflag = 0;
        IT ncon = 1;
        IT nparts = opts.nparts;
        IT edgecut;
        // vector<IT> vwgt(csc->n,1.0);
        // vector<IT> adjwgt(csc->nz,1.0);
        vector<double> tpwgts(ncon * nparts, 1. / nparts);
        vector<double> ubvec(ncon,1.05);
        vector<IT> options(3,0);
        vector<IT> results(csc->n,0);
        vector<IT> vsize(csc->n, 1);
        vector<IT> vwgt(csc->n, 1);
        for(IT i=0; i<csc->n; i++){
            if(opts.permute == 3){ 
                vwgt[i] += csc->jc[i+1] - csc->jc[i]; 
            }else if(opts.permute == 4) {
                IT tmp = csc->jc[i+1] - csc->jc[i];
                vwgt[i] += tmp * tmp;
            }
        }
        MPI_Comm comm;
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        double t1 = MPI_Wtime();
        int ret = METIS_PartGraphKway(
            &csc->n,
            &ncon,
            csc->jc,
            csc->ir,
            vwgt.data(), 
            NULL, 
            NULL, 
            &nparts, 
            NULL,
            NULL, 
            NULL, 
            &edgecut, 
            results.data());
        double t2 = MPI_Wtime();
        
        if(ret == METIS_OK){
            if(myrank == 0) std::cerr << "we run partitioner in " << t2-t1 << " s!!" << std::endl;
        }else{
            if(myrank == 0) std::cerr << "not ok !!" << std::endl;
        }
        ofstream out("results.txt");
        for(IT i=0; i < results.size(); i++){
            char tmp[100];
            // sprintf(tmp, "%ld %ld\n", i, results[i]);
            sprintf(tmp, "%ld\n",results[i]);
            out << tmp;
        }
        out.close();
    }

    MPI_Finalize();
    return 0;
}