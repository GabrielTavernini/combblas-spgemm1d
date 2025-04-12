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
    // {
    //     shared_ptr<CommGrid> World2D = make_shared<CommGrid>(MPI_COMM_WORLD,0,0);
    //     shared_ptr<CommGrid1D> World1D = make_shared<CommGrid1D>(MPI_COMM_WORLD);
    //     // for each dataset
    //     for(auto mtxfile : opts.dataset){
    //         if(myrank==0)std::cerr << "dataset: "<< mtxfile << std::endl;
    //         int niter = opts.niter;
    //         if(opts.AA){
    //             // do A squaring, <int64_t, double>
    //             typedef PlusTimesSRing<double, double> PTFF;
    //             typedef SpDCCols<int64_t, double> DER;
    //             //////////// 2D 
    //             Sp2D Cref(World2D); 
    //             std::vector<double> AA2Dtime;
    //             double perm2dtime (0.0);
    //             double genperm2dtime (0.0);
    //             if(opts.check || opts.run2d){
    //                 Sp2D A(World2D);
    //                 A.ParallelReadMM(mtxfile, true, maximum<double>());
    //                 if(opts.permute2d == 1){ // random permutation for 2D
    //                     genperm2dtime = MPI_Wtime();
    //                     FullyDistVec<int64_t,int64_t> perm;    // get a different permutation
    //                     perm.iota(A.getnrow(), 0);
    //                     perm.RandPerm();
    //                     genperm2dtime = MPI_Wtime() - genperm2dtime;
    //                     perm2dtime = MPI_Wtime();
    //                     A(perm, perm, true);    // in-place permute to save memory
    //                     perm2dtime = MPI_Wtime() - perm2dtime;
    //                 }
    //                 Sp2D B(A);
    //                 Cref = Mult_AnXBn_Synch<PTFF,double,DER>(A,B); // warm up
    //                 if(opts.run2d){
    //                     for(int iter=0; iter < niter; iter++){
    //                         double t1 = MPI_Wtime();
    //                         Sp2D Ctmp = Mult_AnXBn_Synch<PTFF, double, DER>(A,B);
    //                         t1 = MPI_Wtime() - t1;
    //                         AA2Dtime.push_back(t1);
    //                     }
    //                 }
    //             }
    //             //////////// 1D 
    //             if(opts.run1d){
    //                 std::vector<double> AA1Dtime;
    //                 double perm1dtime(0.0);
    //                 double genperm1dtime(0.0);
    //                 Sp1D A(World1D);
    //                 A.ParallelReadMM(mtxfile);
    //                 if(opts.permute1d == 1){
    //                     // random perm, bad for SpGEMM 1D
    //                     genperm1dtime = MPI_Wtime();
    //                     FullyDistVec<int64_t,int64_t> perm;    // get a different permutation
    //                     perm.iota(A.getnrow(), 0);
    //                     perm.RandPerm();
    //                     genperm1dtime = MPI_Wtime() - genperm1dtime;
    //                     perm1dtime = MPI_Wtime();
    //                     A(perm,perm); // inplace perm 1d
    //                     perm1dtime = MPI_Wtime() - perm1dtime;
    //                 }else if(opts.permute1d == 2){
    //                     // graph partition, this permutation requires the sparse matrix to be symmetric!!

    //                 }
    //                 Sp1D B(A);
    //                 Sp1D Ctmp = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, double, DER>(A,B); // warm up
    //                 Sp2D C1Dres = Sp2D(Ctmp, Ctmp.getblocksizevec());
    //                 for(int iter=0; iter < niter; iter++){
    //                     double t1 = MPI_Wtime();
    //                     Sp1D Ctmp = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, double, DER>(A,B);
    //                     t1 = MPI_Wtime() - t1;
    //                     AA1Dtime.push_back(t1);
    //                 }
    //                 if(opts.check){
    //                     if(Cref == C1Dres){
    //                         if(myrank==0)std::cerr<<"SpGEMM 1D result is correct!"<<std::endl;
    //                     }else{
    //                         if(myrank==0)std::cerr<<"SpGEMM 1D result is wong!"<<std::endl;
    //                     }
    //                 }
    //             }

    //             //////////// 3D 
    //             if(opts.run3d){

    //             }
    //         }

    //         if(opts.Rop){
                
    //         }
    //     }
    // }
//     {
//         MPI_Timer timer;
//         string mmname(argv[1]);
//         string executetype(argv[2]);
//         if(executetype  == "fetchall"){
//             // should launch in multi process.
//             /* Test RDMA version CbC */
//             shared_ptr<CommGrid1D> world = make_shared<CommGrid1D>(MPI_COMM_WORLD);
//             Sp1D A1D(world);
//             A1D.ParallelReadMM(mmname);
//             A1D.PrintInfo(0);
            
//             Sp1D B1D(A1D);
//             // 2D benchmark
//             // Sp2D A2D(A1D,A1D.getblocksizevec());
//             // Sp2D B2D(A1D, A1D.getblocksizevec());
//             // timer.start("SpGEMM2D");
//             // Sp2D C2D = Mult_AnXBn_Synch<PTFF, NT, DER>(A2D, B2D);
//             // timer.stop("SpGEMM2D");
//             // Sp1D A1D(A2D);
//             // Sp1D B1D(B2D);
//             // // Fetch all 
//             // timer.start("1DRDMA");
//             Sp1D C1D = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF,NT,DER,IT,NT,NT,DER,DER>(A1D, B1D);
//             // timer.stop("1DRDMA");
//             // MatrixEqual(C1D,C2D, "fetchall");
//         }
//         if(executetype == "spgemm1dBD"){
//             shared_ptr<CommGrid> fullWorld;
//             fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
//             Sp2D Readingmatrix(fullWorld);
//             timer.start("readingmatrix");
//             Readingmatrix.ParallelReadMM(mmname.c_str(),true,maximum<double>());
//             timer.stop("readingmatrix");
//             Sp2D A2D(Readingmatrix);
//             Sp2D B2D(Readingmatrix);
//             timer.start("SpGEMM2D");
//             Sp2D C2D = 
//             Mult_AnXBn_Synch<PTFF, double, SpDCCols<int64_t, double>, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >
//             (A2D, B2D);
//             timer.stop("SpGEMM2D");
//             SpParMat1D<IT,NT,DER> A1D(Readingmatrix);
//             SpParMat1D<IT,NT,DER> B1D(Readingmatrix);
//             timer.start("spgemm1dBD");
//             SpParMat1D<IT,NT,DER> C1D = Mult_AnXBn_1D_BD<PTFF,NT,DER,IT,NT,NT,DER,DER>(A1D, B1D);
//             timer.stop("spgemm1dBD");
//             if(C1D.seqptr() != nullptr){
//                 Sp2D C1D2D(C1D,C1D.getblocksizevec());
//                 if(C1D2D == C2D){
//                     printf("correct results!\n");
//                 }else{
//                     printf("wrong results!\n");
//                 }
//             }
//         }

//         if(executetype == "spgemm1dBDOP"){
//             shared_ptr<CommGrid> fullWorld;
//             fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
//             Sp2D Readingmatrix(fullWorld);
//             timer.start("readingmatrix");
//             Readingmatrix.ParallelReadMM(mmname.c_str(),true,maximum<double>());
//             timer.stop("readingmatrix");
//             Sp2D A2D(Readingmatrix);
//             Sp2D B2D(Readingmatrix);
//             timer.start("SpGEMM2D");
//             Sp2D C2D = 
//             Mult_AnXBn_Synch<PTFF, double, SpDCCols<int64_t, double>, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >
//             (A2D, B2D);
//             timer.stop("SpGEMM2D");
//             SpParMat1D<IT,NT,DER> A1D(Readingmatrix);
//             SpParMat1D<IT,NT,DER> B1D(Readingmatrix);
//             timer.start("spgemm1dBDOP");
//             SpParMat1D<IT,NT,DER> C1D = Mult_AnXBn_1D_OP<PTFF,NT,DER,IT,NT,NT,DER,DER>(A1D, B1D);
//             timer.stop("spgemm1dBDOP");
//             if(C1D.seqptr() != nullptr){
//                 Sp2D C1D2D(C1D,C1D.getblocksizevec());
//                 if(C1D2D == C2D){
//                     printf("correct results!\n");
//                 }else{
//                     printf("wrong results!\n");
//                 }
//             }
//         }

//         if(executetype == "permutespgemm1dBD"){
//             shared_ptr<CommGrid> fullWorld;
//             fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
//             Sp2D Readingmatrix(fullWorld);
//             timer.start("readingmatrix");
//             Readingmatrix.ParallelReadMM(mmname.c_str(),true,maximum<double>());
//             timer.stop("readingmatrix");
//             FullyDistVec<IT, IT> permute;
//             string blocksizefile(argv[3]);
//             vector<IT> blocksizevec = SpHelper::ReadIntegersFromFile(blocksizefile);
//             string distvecfile(argv[4]);
//             permute.ParallelRead(distvecfile, true, maximum<double>());
//             Readingmatrix(permute, permute, true);
//             Sp2D A2Dshuffled(Readingmatrix);
//             Sp2D B2Dshuffled(Readingmatrix);
//             // timer.start("SpGEMM2D");
//             // Sp2D C2D = 
//             // Mult_AnXBn_Synch<PTFF, double, SpDCCols<int64_t, double>, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >
//             // (A2Dshuffled, B2Dshuffled);
//             // timer.stop("SpGEMM2D");
//             Sp1D A1D(A2Dshuffled, blocksizevec, blocksizevec);
//             Sp1D B1D(B2Dshuffled, blocksizevec, blocksizevec);
//             // B1D.StatisticInfo();
//             timer.start("permutespgemm1dBD");
//             Sp1D C1D = Mult_AnXBn_1D_BD<PTFF,NT,DER,IT,NT,NT,DER,DER>(A1D, B1D);
//             timer.stop("permutespgemm1dBD");
//             // if(C1D.seqptr() != nullptr){
//             //     Sp2D C1D2D(C1D,C1D.getblocksizevec());
//             //     if(C1D2D == C2D){
//             //         printf("correct results!\n");
//             //     }else{
//             //         printf("wrong results!\n");
//             //     }
//             // }
//         }

//         if(executetype == "permutespgemm1dBDOP"){
//             shared_ptr<CommGrid> fullWorld;
//             fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
//             Sp2D Readingmatrix(fullWorld);
//             timer.start("readingmatrix");
//             Readingmatrix.ParallelReadMM(mmname.c_str(),true,maximum<double>());
//             timer.stop("readingmatrix");
//             FullyDistVec<IT, IT> permute;
//             string blocksizefile(argv[3]);
//             vector<IT> blocksizevec = SpHelper::ReadIntegersFromFile(blocksizefile);
//             string distvecfile(argv[4]);
//             permute.ParallelRead(distvecfile, true, maximum<double>());
//             Readingmatrix(permute, permute, true);
//             Sp2D A2Dshuffled(Readingmatrix);
//             Sp2D B2Dshuffled(Readingmatrix);
//             // timer.start("SpGEMM2D");
//             // Sp2D C2D = 
//             // Mult_AnXBn_Synch<PTFF, double, SpDCCols<int64_t, double>, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >
//             // (A2Dshuffled, B2Dshuffled);
//             // timer.stop("SpGEMM2D");
//             Sp1D A1D(A2Dshuffled, blocksizevec, blocksizevec);
//             Sp1D B1D(B2Dshuffled, blocksizevec, blocksizevec);
//             // B1D.StatisticInfo();
//             timer.start("permutespgemm1dBDOP");
//             Sp1D C1D = Mult_AnXBn_1D_OP<PTFF,NT,DER,IT,NT,NT,DER,DER>(A1D, B1D);
//             timer.stop("permutespgemm1dBDOP");
//             // if(C1D.seqptr() != nullptr){
//             //     Sp2D C1D2D(C1D,C1D.getblocksizevec());
//             //     if(C1D2D == C2D){
//             //         printf("correct results!\n");
//             //     }else{
//             //         printf("wrong results!\n");
//             //     }
//             // }
//         }

//         if(executetype == "permutefetchall"){
//             shared_ptr<CommGrid> fullWorld;
//             fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
//             Sp2D Readingmatrix(fullWorld);
//             timer.start("readingmatrix");
//             Readingmatrix.ParallelReadMM(mmname.c_str(),true,maximum<double>());
//             timer.stop("readingmatrix");
//             FullyDistVec<IT, IT> permute;
//             string blocksizefile(argv[3]);
//             vector<IT> blocksizevec = SpHelper::ReadIntegersFromFile(blocksizefile);
//             string distvecfile(argv[4]);
//             permute.ParallelRead(distvecfile, true, maximum<double>());
//             Readingmatrix(permute, permute, true);
            
//             Sp2D A2Dshuffled(Readingmatrix);
//             Sp2D B2Dshuffled(Readingmatrix);


//             // GraphPartitioner gp(mmname);
//             // timer.start("SpGEMM2D");
//             // Sp2D C2D = 
//             // Mult_AnXBn_Synch<PTFF, double, SpDCCols<int64_t, double>, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >
//             // (A2Dshuffled, B2Dshuffled);
//             // timer.stop("SpGEMM2D");
//             Sp1D A1D(A2Dshuffled, blocksizevec, blocksizevec);
//             Sp1D B1D(B2Dshuffled, blocksizevec, blocksizevec);
//             B1D.PrintInfo();
//             timer.start("1DRDMA");
//             Sp1D C1D = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF,NT,DER,IT,NT,NT,DER,DER>(A1D, B1D);
//             timer.stop("1DRDMA");
// #ifdef CHECK_CORRECTNESS
//             // Sp2D C1D2D(C1D,C1D.getblocksizevec());
//             // if(C1D2D == C2D){
//             //     printf("correct results!\n");
//             // }else{
//             //     printf("wrong results!\n");
//             // }
// #endif
//         }

//         if(executetype == "spgemm2d"){
//             shared_ptr<CommGrid> fullWorld;
//             fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
//             Sp2D Readingmatrix(fullWorld);
//             timer.start("readingmatrix");
//             Readingmatrix.ParallelReadMM(mmname.c_str(),true,maximum<double>());
//             timer.stop("readingmatrix");
//             Sp2D A2D(Readingmatrix);
//             Sp2D B2D(Readingmatrix);
//             timer.start("SpGEMM2D");
//             Sp2D C2D = 
//             Mult_AnXBn_Synch<PTFF, double, SpDCCols<int64_t, double>, int64_t, double, double, SpDCCols<int64_t, double>, SpDCCols<int64_t, double> >
//             (A2D, B2D);
//             timer.stop("SpGEMM2D");
//         }

//         if(executetype == "symm"){
//             shared_ptr<CommGrid> fullWorld;
//             fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
//             Sp2D Readingmatrix(fullWorld);
//             Readingmatrix.ParallelReadMM(mmname.c_str(),true,maximum<double>());
//             Sp2D RT = Readingmatrix;
//             RT.Transpose();
//             if(!(Readingmatrix == RT)){
//                 SpParHelper::Print("Symmatricizing an unsymmetric input matrix.\n");
//                 Readingmatrix += RT;
//             }else{
//                 SpParHelper::Print("input matrix is symmetrical.\n");
//             }
//             Readingmatrix.ParallelWriteMM(mmname,true);
//         }

//     }
    MPI_Finalize();
    return 0;
}
