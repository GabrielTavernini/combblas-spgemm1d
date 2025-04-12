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

#include <cstdint>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <stdint.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SPA.h"
#include "CombBLAS/SpDCCols.h"
#include "CombBLAS/SpHelper.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/SpTuples.h"
#include <fstream>

using namespace std;
using namespace combblas;

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




int main(int argc, char* argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    {
        double vm_usage, resident_set;
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        Sp2D Readingmatrix(fullWorld);
        float lb;
        long nnz;

        string mmfilename(argv[1]);
        string blocksizevecfilename(argv[2]);
        Readingmatrix.ParallelReadMM(mmfilename.c_str(),true,maximum<double>());
        typedef PlusTimesSRing<double, double> PTFF;
        Sp2D A2D(Readingmatrix);
        Sp2D B2D(Readingmatrix);
        vector<IT> BlocksizeVec = SpHelper::ReadIntegersFromFile(blocksizevecfilename);
        assert(BlocksizeVec.size() == nprocs);
        vector<int> BlocksizePrefix(nprocs, 0);
        partial_sum(BlocksizeVec.begin(), BlocksizeVec.end()-1, BlocksizePrefix.begin()+1);
        
        Sp1D A1D(A2D, BlocksizeVec);
        Sp1D B1D(B2D, BlocksizeVec);
        Dcsc<IT,NT> *dcscA = A1D.seqptr()->GetInternal();
        Dcsc<IT,NT> *dcscB = B1D.seqptr()->GetInternal();
        vector<IT> cpVec(dcscA->cp, dcscA->cp+dcscA->nzc+1);
        vector<vector<IT>> GlobalcpVec;
        SpParHelper::AllgatherVector(GlobalcpVec, cpVec);

        IT nrows = A2D.getnrow();

        /* Project columns to a vector */
        vector<bool> projB(nrows, false);
        for(IT i=0; i<dcscB->nzc; ++i)
        {
            for(IT j=dcscB->cp[i]; j < dcscB->cp[i+1]; ++j)
            {
                IT rowid = dcscB->ir[j];
                projB[rowid] = true;
            }
        }
        int tval = 0;
        for(auto x : projB) tval += (int)x;
        vector<IT> RemoteAIdx;
        for(int i=0; i<projB.size(); i++){
            if(projB[i]) RemoteAIdx.push_back(i);
        }

        MPI_PassiveWindow<IT> jcwindows(dcscA->jc, (int)dcscA->nzc+1);
        MPI_PassiveWindow<IT> irwindows(dcscA->ir, (int)dcscA->nz);
        /* Get Block CscA : 1. get tuples of B in blocks */
        
        vector<vector<std::tuple<IT,IT,NT>>> BlockTuples(nprocs, vector<std::tuple<IT,IT,NT>>());
        for(IT i=0; i<dcscB->nzc; ++i)
        {
            IT colid = dcscB->jc[i];
            for(IT j=dcscB->cp[i]; j < dcscB->cp[i+1]; ++j)
            {
                IT rowid = dcscB->ir[j];
                auto blockidx = BlockIndex(BlocksizePrefix, rowid, colid);
                BlockTuples[blockidx.first].push_back(make_tuple(rowid, colid, dcscB->numx[j]));
            }
        }
        vector<Dcsc<IT,NT>> BlockDcscB;
        int LocalCols = (int)B1D.getseq()->getncol();
        for(int i=0; i<nprocs; i++){
            SpTuples<IT,NT> sptuples((int)BlockTuples[i].size(), BlocksizeVec[i], LocalCols,
                                     BlockTuples[i].data(),true);
            sptuples.tuples_deleted = true;
        }

        // auto cscA = GetCsc(A1D.getseq());
        // auto cscB = GetCsc(B1D.getseq());
        // MPI_PassiveWindow<int64_t> colptrwindows(cscA->jc, cscA->n+1);
        // MPI_PassiveWindow<int64_t> rowidwindows(cscA->ir, cscA->nz);
        // MPI_PassiveWindow<double> valuewindows(cscA->num, cscA->nz);

        // /*construct C*/
        // Csc<IT,NT> cscC;
        // cscC.n = cscB->n;
        // cscC.jc = new IT[cscC.n + 1];
        // for(IT i=0; i<cscC.n+1; i++) cscC.jc[i] = 0;
        // IT *row_nz = new IT[cscA->n];
        // // symbolic phase
        // spa_symbolic_kernel(*cscA, *cscB, row_nz);
        // IT sumrownz = 0;
        // for(IT i=0; i<cscA->n; i++){
        //     sumrownz += row_nz[i];
        // }
        // for(int i=1; i<cscC.n+1; i++){
        //     cscC.jc[i] = cscC.jc[i-1] + row_nz[i-1];
        // }
        // cscC.nz = cscC.jc[cscC.n];
        // cscC.ir = new IT[cscC.nz];
        // cscC.num = new NT[cscC.nz];
        // // compute phase 
        // int startcol = 0;
        // int endcol = cscB->n;
        // SPA<true, IT, NT> spa(cscB->n);
        // auto addop = [](double x, double y){return x+y;};
        // for(int j=startcol; j<endcol; j++){
        //     for(int kB=cscB->jc[j]; kB<cscB->jc[j+1]; kB++){
        //         int colA = cscB->ir[kB];
        //         for(int kA=cscA->jc[colA]; kA < cscA->jc[colA+1]; kA++){
        //             NT innerprod = cscA->num[kA] * cscB->num[kB];
        //             spa.Insert(cscA->ir[kA], innerprod, addop);
        //         }
        //     }
        //     spa.OutputReset(cscC.ir + cscC.jc[j], cscC.num + cscC.jc[j]);
        // }
        // auto cscC2d = GetCsc(C2D.seqptr());
        // bool wrongres(false);
        // for(IT i=0; i<cscC.n+1; i++){
        //     if(cscC.jc[i] != cscC2d->jc[i]){
        //         printf("wrong results at i %lu Cjc %lu C2Djc %lu \n", i, cscC.jc[i], cscC2d->jc[i]);
        //         wrongres = true;
        //         break;
        //     }
        // }
        // if(!wrongres) printf("we have correct results!\n");
    }
    
    MPI_Finalize();
    return 0;
}
