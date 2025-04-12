#include <cstdint>
#include <cstdio>
#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <stdint.h>
#include <cmath>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/SpDCCols.h"
#include "CombBLAS/SpHelper.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/SpParMat.h"
#include "Glue.h"
#include "CCGrid.h"
#include "Reductions.h"
#include "Multiplier.h"
#include "SplitMatDist.h"
#include "RestrictionOp.h"
#include <CombBLAS/SpParMat1DFriends.h>

using namespace std;
using namespace combblas;

double comm_bcast;
double comm_reduce;
double comp_summa;
double comp_reduce;
double comp_result;
double comp_reduce_layer;
double comp_split;
double comp_trans;
double comm_split;

double prep1d;
double comm1d;
double comp1d;

#define ITERS 1

typedef int64_t IT;
typedef double NT;
typedef SpDCCols<IT, NT> DER;
typedef SpParMat<IT, NT, DER> Sp2D;
typedef SpParMat1D<IT, NT, DER> Sp1D;
typedef PlusTimesSRing<double, double> PTFF;


vector<IT> blocksize = {};

template<class IU, class NU>
void CBC_AA(SpDCCols<IU, NU> * A, shared_ptr<CommGrid> layerGrid, int permutation = 0){
    
    int myrank = layerGrid->GetRank();
    if(myrank == 0) std::cerr << "=============A square============" << std::endl;
    int nprocs = layerGrid->GetSize();
    // squaring A * A
    Sp2D A2dmat(new DER(*A),layerGrid);
    int numThreads;
    #pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
    string perm = "NoPerm";
    if(permutation == 1) perm = "RandPerm";
    else if(permutation == 2) perm = "GPPerm";
    string suffix = "_" + perm+ "_" + to_string(nprocs) +"MPI" +to_string(numThreads) + "Threads.log";
    Sp2D B2dmat(A2dmat);
    MPI_Timer timer;
    Sp2D C2dmat;
    if(permutation == 2){
        combblas::perfcnt2d.Reset();
        timer.start("AA_2D");
        C2dmat = Mult_AnXBn_Synch<PTFF, NT, DER>(A2dmat, B2dmat);
        timer.stop("AA_2D");
        combblas::perfcnt2d.doublemap["TotTime(s)"] = timer.records["AA_2D"];
        combblas::perfcnt2d.OutputRecords("AA_2D_Detailed" + suffix);
    }

    Sp1D A1dmat(A2dmat,blocksize,blocksize);
    Sp1D B1dmat(B2dmat,blocksize,blocksize);
    combblas::perfcnt1d.Reset();
    timer.start("AA_1D_CBC");
    Sp1D C1dmat = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, NT, DER>(A1dmat,B1dmat);
    timer.stop("AA_1D_CBC");
    combblas::perfcnt1d.doublemap["TotTime(s)"] = timer.records["AA_1D_CBC"];
    combblas::perfcnt1d.OutputRecords("AA_1D_Detailed" + suffix);
    Sp2D tmpC1d(C1dmat,C1dmat.getblocksizevec());
    int64_t nnzA = A1dmat.getnnz();
    int64_t nnzB = B1dmat.getnnz();
    int64_t nnzC = C1dmat.getnnz();
    if(myrank == 0){
        std::cerr << std::left << std::setw(10) << "nnz(A)=" << nnzA << std::endl;
        std::cerr << std::left << std::setw(10) << "nnz(AA)=" << nnzC << std::endl;
    }
    if(permutation == 2){
        if(tmpC1d == C2dmat){
            if(myrank == 0) printf("AA pass!\n");
        }else{
            if(myrank == 0) printf("AA fail!\n");
        }
    }

}

template<class IU, class NU>
void CBC_RTA(SpDCCols<IU, NU> * A, SpDCCols<IU, NU> *R, SpDCCols<IU, NU> *RT, shared_ptr<CommGrid> layerGrid, int permutation=0){
    
    int myrank = layerGrid->GetRank();
    if(myrank == 0) std::cerr << "=============RT * A============" << std::endl;
    int nprocs = layerGrid->GetSize();
    // Left multi in AMG: RTA
    Sp2D A2dmat(new DER(*RT),layerGrid);
    Sp2D B2dmat(new DER(*A), layerGrid);
    int numThreads;
    #pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
    string perm = "NoPerm";
    if(permutation == 1) perm = "RandPerm";
    else if(permutation == 2) perm = "GPPerm";
    string suffix = "_" + perm+ "_" + to_string(nprocs) +"MPI" +to_string(numThreads) + "Threads.log";
    MPI_Timer timer;
    combblas::perfcnt2d.Reset();
    Sp2D dummymat = Mult_AnXBn_Synch<PTFF, NT, DER>(A2dmat, B2dmat);
    timer.start("RTA_2D");
    Sp2D C2dmat = Mult_AnXBn_Synch<PTFF, NT, DER>(A2dmat, B2dmat);
    timer.stop("RTA_2D");
    combblas::perfcnt2d.doublemap["TotTime(s)"] = timer.records["RTA_2D"];
    combblas::perfcnt2d.OutputRecords("RTA_2D_Detailed"+suffix);
    Sp1D A1dmat(A2dmat,blocksize);
    Sp1D B1dmat(B2dmat,blocksize,blocksize);
    combblas::perfcnt1d.Reset();
    timer.start("CBC_RTA_1D");
    Sp1D C1dmat = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, NT, DER>(A1dmat,B1dmat);
    timer.stop("CBC_RTA_1D");
    combblas::perfcnt1d.doublemap["TotTime(s)"] = timer.records["CBC_RTA_1D"];
    combblas::perfcnt1d.OutputRecords("CBC_RTA_1D_Detailed"+suffix);
    Sp2D tmpC1d(C1dmat,C1dmat.getblocksizevec());
    int64_t nnzA = A1dmat.getnnz();
    int64_t nnzB = B1dmat.getnnz();
    int64_t nnzC = C1dmat.getnnz();
    if(myrank == 0){
        std::cerr << std::left << std::setw(10) << "nnz(RT)=" << nnzA << std::endl;
        std::cerr << std::left << std::setw(10) << "nnz(A)=" << nnzB << std::endl;
        std::cerr << std::left << std::setw(10) << "nnz(RTA)=" << nnzC << std::endl;
    }
    if(tmpC1d == C2dmat){
        if(myrank == 0) printf("RTA pass!\n");
    }else{
        if(myrank == 0) printf("RTA fail!\n");
    }
}


template<class IU, class NU>
void CBC_RTAR(SpDCCols<IU, NU> * A, SpDCCols<IU, NU> *R, SpDCCols<IU, NU> *RT, shared_ptr<CommGrid> layerGrid,int permutation=0){
    
    int myrank = layerGrid->GetRank();
    if(myrank == 0)std::cerr << "=============RTA * R============" << std::endl;
    int nprocs = layerGrid->GetSize();
    Sp2D tmpA(new DER(*RT),layerGrid);
    Sp2D tmpB(new DER(*A), layerGrid);
    int numThreads;
    #pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
    string perm = "NoPerm";
    if(permutation == 1) perm = "RandPerm";
    else if(permutation == 2) perm = "GPPerm";
    string suffix = "_" + perm+ "_" + to_string(nprocs) +"MPI" +to_string(numThreads) + "Threads.log";
    MPI_Timer timer;
    Sp2D RTA = Mult_AnXBn_Synch<PTFF, NT, DER>(tmpA, tmpB);
    // Right multi in AMG: RTA
    Sp2D A2dmat(RTA);
    Sp2D B2dmat(new DER(*R),layerGrid);
    combblas::perfcnt2d.Reset();
    timer.start("RTAR_2D");
    Sp2D C2dmat = Mult_AnXBn_Synch<PTFF, NT, DER>(A2dmat, B2dmat);
    timer.stop("RTAR_2D");
    combblas::perfcnt2d.doublemap["TotTime(s)"] = timer.records["RTAR_2D"];
    combblas::perfcnt2d.OutputRecords("RTAR_2D_Detailed" + suffix);
    Sp1D A1dmat(A2dmat,blocksize,{});
    Sp1D B1dmat(B2dmat,{},blocksize);
    combblas::perfcnt1d.Reset();
    timer.start("RTAR_1D");
    Sp1D C1dmat = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, NT, DER>(A1dmat,B1dmat);
    timer.stop("RTAR_1D");
    combblas::perfcnt1d.doublemap["TotTime(s)"] = timer.records["RTAR_1D"];
    combblas::perfcnt1d.OutputRecords("RTAR_1D_Detailed" + suffix);
    Sp2D tmpC1d(C1dmat,C1dmat.getblocksizevec());
    int64_t nnzA = A1dmat.getnnz();
    int64_t nnzB = B1dmat.getnnz();
    int64_t nnzC = C1dmat.getnnz();
    if(myrank == 0){
        std::cerr << std::left << std::setw(10) << "nnz(RTA)=" << nnzA << std::endl;
        std::cerr << std::left << std::setw(10) << "nnz(R)=" << nnzB << std::endl;
        std::cerr << std::left << std::setw(10) << "nnz(RTAR)=" << nnzC << std::endl;
    }
    if(tmpC1d == C2dmat){
        if(myrank == 0) printf("cbc-RTAR pass!\n");
    }else{
        if(myrank == 0) printf("cbc-RTAR fail!\n");
    }
}

template<class IU, class NU>
void CBC_RTAR_Transpose(SpDCCols<IU, NU> * A, SpDCCols<IU, NU> *R, 
SpDCCols<IU, NU> *RT, shared_ptr<CommGrid> layerGrid, int permutation=0){
    int myrank = layerGrid->GetRank();
    int nprocs = layerGrid->GetSize();
    Sp2D tmpA(new DER(*RT),layerGrid);
    Sp2D tmpB(new DER(*A), layerGrid);
    int numThreads;
    #pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
    string perm = "NoPerm";
    if(permutation) perm = "RandPerm";
    string suffix = "_" + perm+ "_" + to_string(nprocs) +"MPI" +to_string(numThreads) + "Threads.log";
    MPI_Timer timer;
    Sp2D RTA = Mult_AnXBn_Synch<PTFF, NT, DER>(tmpA, tmpB);
    // Right multi w/ transpose in AMG: RTA
    Sp2D A2dmat(RTA);
    Sp2D B2dmat(new DER(*R),layerGrid);
    // combblas::perfcnt2d.Reset();
    // timer.start("RTAR_2D");
    Sp2D C2dmat = Mult_AnXBn_Synch<PTFF, NT, DER>(A2dmat, B2dmat);
    // timer.stop("RTAR_2D");
    // combblas::perfcnt2d.OutputRecords("CBC_RTAR_2D_Detailed.log");
    Sp2D AT2dmat(A2dmat); AT2dmat.Transpose();
    Sp1D A1dmat(AT2dmat);
    Sp2D BT2dmat(new DER(*RT),layerGrid);
    Sp1D B1dmat(BT2dmat);
    combblas::perfcnt1d.Reset();
    // switch the position of RTA and R, move the smaller matrix R.
    timer.start("CBC_RTAR_Transpose1D");
    Sp1D C1dmat = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, NT, DER>(B1dmat,A1dmat);
    timer.stop("CBC_RTAR_Transpose1D");
    combblas::perfcnt1d.doublemap["TotTime(s)"] = timer.records["CBC_RTAR_Transpose1D"];
    combblas::perfcnt1d.OutputRecords("CBC_RTAR_Transpose_1D_Detailed" + suffix);
    Sp2D tmpC1d(C1dmat,C1dmat.getblocksizevec());
    tmpC1d.Transpose();
    int64_t nnzA = A1dmat.getnnz();
    int64_t nnzB = B1dmat.getnnz();
    int64_t nnzC = C1dmat.getnnz();
    if(myrank == 0){
        std::cerr << std::left << std::setw(10) << "nnz(RTA)=" << nnzA << std::endl;
        std::cerr << std::left << std::setw(10) << "nnz(R)=" << nnzB << std::endl;
        std::cerr << std::left << std::setw(10) << "nnz(RTAR)=" << nnzC << std::endl;
    }
    if(tmpC1d == C2dmat){
        if(myrank == 0) printf("cbc-RTAR-Transpose1D pass!\n");
    }else{
        if(myrank == 0) printf("cbc-RTAR-Transpose1D fail!\n");
    }
}

template<class IU, class NU>
void CBC_RTAR_OP(SpDCCols<IU, NU> * A, SpDCCols<IU, NU> *R, SpDCCols<IU, NU> *RT, shared_ptr<CommGrid> layerGrid, int permutation=0){
    int myrank = layerGrid->GetRank();
    if(myrank == 0) std::cerr << "=============RTA * R Out Product============" << std::endl;
    int nprocs = layerGrid->GetSize();
    Sp2D tmpA(new DER(*RT),layerGrid);
    Sp2D tmpB(new DER(*A), layerGrid);
    int numThreads;
    #pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
    string perm = "NoPerm";
    if(permutation == 1) perm = "RandPerm";
    else if(permutation == 2) perm = "GPPerm";
    string suffix = "_" + perm+ "_" + to_string(nprocs) +"MPI" +to_string(numThreads) + "Threads.log";
    MPI_Timer timer;
    Sp2D RTA = Mult_AnXBn_Synch<PTFF, NT, DER>(tmpA, tmpB);
    // Right multi in AMG: RTA
    Sp2D A2dmat(RTA);
    Sp2D B2dmat(new DER(*R),layerGrid);
    // combblas::perfcnt2d.Reset();
    // timer.start("RTAR_2D");
    Sp2D C2dmat = Mult_AnXBn_Synch<PTFF, NT, DER>(A2dmat, B2dmat);
    // timer.stop("RTAR_2D");
    // combblas::perfcnt2d.OutputRecords("CBC_RTAR_2D_Detailed.log");
    Sp1D A1dmat(A2dmat,blocksize);
    Sp1D B1dmat(B2dmat,{},blocksize);
    IU Rm = B2dmat.getnrow();
    IU Rn = B2dmat.getncol();
    combblas::perfcnt1dop.Reset();
    timer.start("RTAR_1DOP");
    Sp1D C1dmat = Mult_AnXBn_1D_OP<PTFF, NT, DER>(A1dmat,B1dmat);
    timer.stop("RTAR_1DOP");
    combblas::perfcnt1dop.doublemap["TotTime(s)"] = timer.records["RTAR_1DOP"];
    combblas::perfcnt1dop.OutputRecords("RTAR_1DOP_Detailed" + suffix);
    Sp2D tmpC1d(C1dmat,C1dmat.getblocksizevec());
    int64_t nnzA = A1dmat.getnnz();
    int64_t nnzB = B1dmat.getnnz();
    int64_t nnzC = C1dmat.getnnz();
    if(myrank == 0){
        std::cerr << std::left << std::setw(10) << "nnz(RTA)=" << nnzA << std::endl;
        std::cerr << std::left << std::setw(10) << "nnz(R)=" << nnzB << std::endl;
        std::cerr << std::left << std::setw(10) << "nnz(RTAR)=" << nnzC << std::endl;
    }
    if(tmpC1d == C2dmat){
        if(myrank == 0) printf("cbc-RTAR-1DOP outproduct pass!\n");
    }else{
        if(myrank == 0) printf("cbc-RTAR-1DOP outproduct fail!\n");
    }
}
SpgemmOpts combblas::opts;
int main(int argc, char *argv[])
{
    int provided;
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    
    
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    if(argc < 6)
    {
        if(myrank == 0)
        {
            printf("Usage (random): ./mpipspgemm <GridRows> <GridCols> <Layers> <Type> <Scale> <EDGEFACTOR> \n");
            printf("Usage (input): ./mpipspgemm <GridRows> <GridCols> <Layers> <Type=input> <matA> \n"); //TODO:<Scale>  not meaningful here. Need to remove it.  Still there because current scripts execute without error.
            printf("Example: ./RestrictionOp 4 4 2 ER 19 16 \n");
            printf("Example: ./RestrictionOp 4 4 2 Input matA.mtx\n");
            printf("Type ER: Erdos-Renyi\n");
            printf("Type SSCA: R-MAT with SSCA benchmark parameters\n");
            printf("Type G500: R-MAT with Graph500 benchmark parameters\n");
        }
        return -1;
    }
    
    
    unsigned GRROWS = (unsigned) atoi(argv[1]);
    unsigned GRCOLS = (unsigned) atoi(argv[2]);
    unsigned C_FACTOR = (unsigned) atoi(argv[3]);
    CCGrid CMG(C_FACTOR, GRCOLS);
    int nthreads;
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    if(myrank == 0)std::cerr << "MPI Size: " << nprocs << " Threads Numbers: " << nthreads << std::endl;
    if(GRROWS != GRCOLS)
    {
        SpParHelper::Print("This version of the Combinatorial BLAS only works on a square logical processor grid\n");
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int layer_length = GRROWS*GRCOLS;
    if(layer_length * C_FACTOR != nprocs)
    {
        SpParHelper::Print("The product of <GridRows> <GridCols> <Replicas> does not match the number of processes\n");
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int permute = 0;
    {
        
        string type;
        shared_ptr<CommGrid> layerGrid;
        layerGrid.reset( new CommGrid(CMG.layerWorld, 0, 0) );
        FullyDistVec<int64_t, int64_t> p(layerGrid); // permutation vector defined on layers
        SpDCCols<int64_t, double> *A;
        
        if(string(argv[4]) == string("input")) // input option
        {
            string fileA(argv[5]);
            double t01 = MPI_Wtime();
            permute = stoi(argv[6]);
            if(permute == 0){
                if(myrank == 0) printf("no permutation!\n");
            }else if(permute == 1){
                if(myrank == 0) printf("random permutation!\n");
            }else if(permute == 2){
                if(myrank == 0) printf("using graph partitioner results!\n");
                string permfile(fileA + "." + to_string(nprocs) + ".perm");
                p.ParallelRead(permfile, true, maximum<double>());
                string bsfile(fileA+"."+to_string(nprocs)+".bsvec");
                blocksize = SpHelper::ReadIntegersFromFile(bsfile);
            }
            A = ReadMat<double>(fileA, CMG, permute, p);
            if(myrank == 0) std::cerr << "Reading matrix: " << fileA << std::endl;
        }
        else
        {
            unsigned scale = (unsigned) atoi(argv[5]);
            unsigned EDGEFACTOR = (unsigned) atoi(argv[6]);
            double initiator[4];
            if(string(argv[4]) == string("ER"))
            {
                initiator[0] = .25;
                initiator[1] = .25;
                initiator[2] = .25;
                initiator[3] = .25;
                if(myrank == 0) std::cerr << "Generating ER matrix, scale: " << scale << ", EDGEFACTOR: "<< EDGEFACTOR << std::endl;
                
            }
            else if(string(argv[4]) == string("G500"))
            {
                initiator[0] = .57;
                initiator[1] = .19;
                initiator[2] = .19;
                initiator[3] = .05;
                EDGEFACTOR  = 16;
                if(myrank == 0) std::cerr << "Generating G500 matrix, scale: " << scale << ", EDGEFACTOR: "<< EDGEFACTOR << std::endl;
            }
            else if(string(argv[4]) == string("SSCA"))
            {
                initiator[0] = .6;
                initiator[1] = .4/3;
                initiator[2] = .4/3;
                initiator[3] = .4/3;
                EDGEFACTOR  = 8;
                if(myrank == 0) std::cerr << "Generating SSCA matrix, scale: " << scale << ", EDGEFACTOR: "<< EDGEFACTOR << std::endl;
            }
            else {
                if(myrank == 0)
                    printf("The initiator parameter - %s - is not recognized.\n", argv[5]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if(myrank==0)std::cerr << "initiator params are:";
            for(int i=0; i<4; i++){
                if(myrank ==0 )  std::cerr << initiator[i] << ", ";
            }
            if(myrank ==0) std::cerr << std::endl;
            double t01 = MPI_Wtime();
            A = GenMat<int64_t,double>(CMG, scale, EDGEFACTOR, initiator, true);
            if(myrank == 0) cout << "RMATs Generated : time " << MPI_Wtime() - t01 << endl;
        }
        SpDCCols<int64_t, double>* R;
        SpDCCols<int64_t, double>* RT;
        if(myrank == 0) cout << "Computing restriction matrix \n";
        double t01 = MPI_Wtime();
        RestrictionOp( CMG, A, R, RT);
        if(myrank == 0) cout << "Restriction Op computed : time " << MPI_Wtime() - t01 << endl;
        /*register performance counter metrics.*/
        combblas::perfcnt1d.GetRankInfo();
        combblas::perfcnt1d.RegisterDouble("prepT(ms)",2,1000.); // report time in ms.
        combblas::perfcnt1d.RegisterDouble("commT(ms)",2,1000.);
        combblas::perfcnt1d.RegisterDouble("compT(ms)",2,1000.);
        combblas::perfcnt1d.RegisterInteger("Alocalnnz");
        combblas::perfcnt1d.RegisterDouble("Alocalmem(MB)",2,1.0);
        combblas::perfcnt1d.RegisterInteger("Aneednnz");
        combblas::perfcnt1d.RegisterDouble("Aneedmem(MB)",2,1.0);
        combblas::perfcnt1d.RegisterInteger("Bnnz    ");
        combblas::perfcnt1d.RegisterDouble("Bmem(MB)",2,1.0);
        combblas::perfcnt1d.RegisterInteger("Cnnz    ");
        combblas::perfcnt1d.RegisterDouble("Cmem(MB)",2,1.0);
        combblas::perfcnt1d.RegisterDouble("TotTime(s)",2,1.0); // report total time in s

        combblas::perfcnt2d.GetRankInfo();
        combblas::perfcnt2d.RegisterDouble("bcastA(ms)",2,1000.);
        combblas::perfcnt2d.RegisterDouble("bcastB(ms)",2,1000.);
        combblas::perfcnt2d.RegisterDouble("compT(ms)",2,1000.);
        combblas::perfcnt2d.RegisterDouble("merge(ms)",2,1000.);
        combblas::perfcnt2d.RegisterInteger("Alocalnnz",2,1.);
        combblas::perfcnt2d.RegisterInteger("Blocalnnz",2,1.);
        combblas::perfcnt2d.RegisterInteger("Clocalnnz",2,1.);
        combblas::perfcnt2d.RegisterDouble("TotTime(s)",2,1.0); // report total time in s

        combblas::perfcnt1dop.GetRankInfo();
        combblas::perfcnt1dop.RegisterDouble("sendB(ms)",2,1000.); // report time in ms.
        combblas::perfcnt1dop.RegisterDouble("sendC(ms)",2,1000.);
        combblas::perfcnt1dop.RegisterDouble("mergeC(ms)",2,1000.);
        combblas::perfcnt1dop.RegisterDouble("compT(ms)",2,1000.);
        combblas::perfcnt1dop.RegisterInteger("Alocalnnz");
        combblas::perfcnt1dop.RegisterDouble("Alocalmem(MB)",2,1.0);
        combblas::perfcnt1dop.RegisterInteger("Bnnz    ");
        combblas::perfcnt1dop.RegisterDouble("Bmem(MB)",2,1.0);
        combblas::perfcnt1dop.RegisterInteger("Cnnz    ");
        combblas::perfcnt1dop.RegisterDouble("Cmem(MB)",2,1.0);
        combblas::perfcnt1dop.RegisterDouble("TotTime(s)",2,1.0); // report total time in s

        CBC_AA(A, layerGrid,permute);
        CBC_RTA(A, R, RT, layerGrid,permute);
        CBC_RTAR(A, R, RT, layerGrid,permute);
        CBC_RTAR_OP(A, R, RT, layerGrid,permute);
    }
    
    MPI_Finalize();
    return 0;
}


