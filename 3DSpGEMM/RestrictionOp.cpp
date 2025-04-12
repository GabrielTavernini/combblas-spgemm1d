#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <mpi.h>
#include <numeric>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <stdint.h>
#include <cmath>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/FullyDistVec.h"
#include "CombBLAS/SpDCCols.h"
#include "CombBLAS/SpHelper.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/SpParMat1D.h"
#include "CombBLAS/SpParMat1DFriends.h"
#include "Glue.h"
#include "CCGrid.h"
#include "Reductions.h"
#include "Multiplier.h"
#include "SplitMatDist.h"
#include "RestrictionOp.h"
#include "parmetis.h"

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

typedef int64_t IT;
typedef double NT;
typedef SpDCCols < int64_t, double > DER;
typedef SpParMat1D<IT,NT,DER> Sp1D;
typedef SpParMat<IT,NT,DER> Sp2D;
typedef SpParMat3D<IT,NT,DER> Sp3D;
typedef PlusTimesSRing<double, double> PTFF;
template <class ITT, class NTT>
class Dist 
{ 
public: 
    typedef SpDCCols   < ITT, NTT >         DCCols;
    typedef SpParMat   < ITT, NTT, DCCols > MPI_DCCols;
    typedef SpParMat1D < ITT, NTT, DCCols > MPI_DCCols1D;
};


template<class IT, class NT,class DER>
void MatrixEqual(SpParMat<IT,NT,DER> & A2D, SpParMat<IT,NT,DER> & B2D, std::string description){
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(A2D == B2D){
        if(myrank == 0) std::cerr << description << " is correct!" << std::endl;
    }else{
        if(myrank == 0) std::cerr<< description <<  "is wrong!" << std::endl;
    }
}

template<class IT, class NT,class DER>
void MatrixEqual(SpParMat1D<IT,NT,DER> & A1D, SpParMat<IT,NT,DER> & A2D, std::string description){
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    SpParMat<IT,NT,DER> A1D2D(A1D,A1D.getblocksizevec());
    if(A1D2D == A2D){
        if(myrank == 0) std::cerr << description << " is correct!" << std::endl;
    }else{
        if(myrank == 0) std::cerr << description << " is wrong!" << std::endl;
    }
}

void Benchmark3D(Sp3D & A, Sp3D & B, const SpgemmOpts & opts){
    int myrank; MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    Sp3D Cref = Mult_AnXBn_SUMMA3D<PTFF,double,DER>(A,B); // warm up
    for(int iter=0; iter < opts.niter; iter++){
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        Sp3D Ctmp = Mult_AnXBn_SUMMA3D<PTFF, double, DER>(A,B);
        t1 = MPI_Wtime() - t1;
        perfcnt3d.doublemap["TotTime(s)"] = t1;
    }
}

void Benchmark2D(Sp2D & A, Sp2D & B, const SpgemmOpts & opts){
    int myrank; MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    Sp2D Cref = Mult_AnXBn_Synch<PTFF,double,DER>(A,B); // warm up
    for(int iter=0; iter < opts.niter; iter++){
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        Sp2D Ctmp = Mult_AnXBn_Synch<PTFF, double, DER>(A,B);
        t1 = MPI_Wtime() - t1;
        perfcnt2d.doublemap["TotTime(s)"] = t1;
    }
}

void Benchmark1D(Sp1D & A, Sp1D & B, const SpgemmOpts & opts){
    
    int myrank; MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    for(int iter=0; iter < opts.niter; iter++){
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        Sp1D Ctmp = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, double, DER>(A,B);
        t1 = MPI_Wtime() - t1;
        perfcnt1d.doublemap["TotTime(s)"] = t1;
    }
}

bool RMATGEN(std::string mtxname){
    return mtxname == "ER" || mtxname == "G500" || mtxname == "SSCA";
}


void METISPartition(const Sp1D & A, std::vector<int64_t> & results, int permtype, int64_t nparts, int gptype=0){
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    SpCCols<IT, NT> spccols(*A.seqptr()); // convert to SoCCols
    Csc<IT,NT> * csc = spccols.GetCSC(); // Get Csc
    IT wgtflag = 2; // 0 no wgt, 1. wgt on edges, 2. wgt on vertices 3. wgt on both
    IT numflag = 0;
    IT ncon = 1;
    IT edgecut;
    int myrank;
    int nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    // perm = 1: random perm, perm = 2: metis partition normal perm=3: gp wgt on vertices nnz
    // perm = 4: gp wgt on vertices (nnz*nnz+1)
    vector<IT> vwgt(csc->n,1.0);
    if(permtype >= 3){
        wgtflag = 2; // Weights on the vertices only (adjwgt is NULL).
        vector<IT> tmpvwgt(csc->n,0.0);
        for(typename SpCCols<IT, NT>::SpColIter colit = spccols.begcol(); colit != spccols.endcol(); ++colit)
        {
            IT gcol = colit.colid() + A.getblocksizeprefix()[myrank];
            for(typename SpCCols<IT, NT>::SpColIter::NzIter nzit = spccols.begnz(colit); 
                nzit != spccols.endnz(colit); ++nzit)
            {
                IT grow = nzit.rowid();
                NT val = nzit.value();
                tmpvwgt[colit.colid()]++;
            }
        }
        if(permtype == 4){
            for(IT i=0; i<csc->n; i++) tmpvwgt[i] = tmpvwgt[i] * tmpvwgt[i];
        }
        for(IT i=0; i<csc->n; i++) vwgt[i] += tmpvwgt[i];
    }
    vector<double> tpwgts(ncon * nparts, 1. / nparts);
    vector<double> ubvec(ncon,1.05);
    vector<IT> options(3,0);
    // vector<IT> results(csc->n,0);
    results = vector<IT>(csc->n);
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    vector<int64_t> recv(nprocs,0);
    IT localnvtxs = csc->n;
    MPI_Allgather(
        &localnvtxs, 1, MPI_LONG_LONG, 
        recv.data(), 1, MPI_LONG_LONG, comm );
    IT nvtxs = 0;
    for(int i=0; i<nprocs; i++) nvtxs += recv[i];
    vector<int64_t> vtxdist(nprocs+1,0);
    std::partial_sum(recv.begin(), recv.end(), vtxdist.begin()+1);
    t1 = MPI_Wtime() - t1;
    perfcnt1d.doublemap["gpprep(ms)"] = t1;
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    int ret = -1;
    if(gptype == 0){
        ret = ParMETIS_V3_PartKway(
        vtxdist.data(), 
        csc->jc,
        csc->ir,
        vwgt.data(),
        NULL,
        &wgtflag,
        &numflag,
        &ncon,
        &nparts,
        tpwgts.data(),
        ubvec.data(),
        options.data(),
        &edgecut,
        results.data(),
        &comm
        );
    }else if(gptype == 1){ // metis sequential version
        // std::vector<IT> Cnts;
        // SpParHelper::GatherVector(Cnts, &csc->n, 1, 0, comm);
        // std::vector<IT> localorg;
        // SpHelper::DePrefixSum(csc->jc, csc->n+1, localorg);
        // std::vector<IT> globaljc;
        // std::vector<int> gjcsize;
        // SpParHelper::GatherVectorv(globaljc, gjcsize, localorg.data(), localorg.size(), 0, comm);
        // std::vector<IT> gjc = SpHelper::prefixsum(globaljc,true);
        // std::vector<IT> gir;
        // std::vector<int> girsize;
        // SpParHelper::GatherVectorv(gir,girsize, csc->ir, csc->nz, 0, comm);
        // IT totsize = gjc.back();
        // std::vector<IT> totvwgt; std::vector<int> totvwgtsize;
        // SpParHelper::GatherVectorv(totvwgt, totvwgtsize, vwgt.data(), vwgt.size(), 0, comm); 
        // std::vector<IT> vsize(totsize, 1);
        // std::vector<IT> globalresults(totsize);
        // ret = METIS_PartGraphKway(
        //     &totsize,
        //     &ncon,
        //     gjc.data(),
        //     gir.data(),
        //     totvwgt.data(), 
        //     vsize.data(), 
        //     NULL, 
        //     &nparts, 
        //     tpwgts.data(),
        //     ubvec.data(), 
        //     options.data(), 
        //     &edgecut, 
        //     globalresults.data());
    }
    
    if(ret != METIS_OK){
        if(myrank==0)std::cerr<<"Error on Partition!"<<std::endl;
        exit(0);
    }
    t1 = MPI_Wtime() - t1;
    perfcnt1d.doublemap["metis(ms)"] = t1;
}

SpgemmOpts combblas::opts;

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init(NULL,NULL);
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    SpHelper::initdatasetmap();
    OptParser::parse(argc, argv, &opts);
    if(myrank == 0) opts.PrintOpts();

    /*register performance counter metrics.*/
    combblas::perfcnt1d.GetRankInfo();
    combblas::perfcnt1d.RegisterDouble("Rseq(ms)",2,1000.); // report time in ms.
    combblas::perfcnt1d.RegisterDouble("gpprep(ms)",2,1000.); // report time in ms.
    combblas::perfcnt1d.RegisterDouble("metis(ms)",2,1000.); // report time in ms.
    combblas::perfcnt1d.RegisterDouble("permute(ms)",2,1000.); // report time in ms.
    combblas::perfcnt1d.RegisterDouble("prepT(ms)",2,1000.); // report time in ms.
    combblas::perfcnt1d.RegisterDouble("commT(ms)",2,1000.);
    combblas::perfcnt1d.RegisterDouble("compT(ms)",2,1000.);
    combblas::perfcnt1d.RegisterDouble("otherT(ms)",2,1000.);
    combblas::perfcnt1d.RegisterInteger("Alocalnnz");
    combblas::perfcnt1d.RegisterDouble("Alocalmem(MB)",2,1.0);
    combblas::perfcnt1d.RegisterInteger("Aneednnz");
    combblas::perfcnt1d.RegisterDouble("Aneedmem(MB)",2,1.0);
    combblas::perfcnt1d.RegisterInteger("Bnnz    ");
    combblas::perfcnt1d.RegisterDouble("Bmem(MB)",2,1.0);
    combblas::perfcnt1d.RegisterInteger("Cnnz    ");
    combblas::perfcnt1d.RegisterDouble("Cmem(MB)",2,1.0);
    combblas::perfcnt1d.RegisterDouble("TotTime(s)",2,1.0); // report total time in s
    combblas::perfcnt1d.RegisterDouble("gtot(s)",2,1.0);
    combblas::perfcnt1d.RegisterDouble("stat(ms)",2,1000.);
    // perfcnt1d.doublemap["gtot(s)"] = gtot;
    // perfcnt1d.doublemap["stat(ms)"] = stattime * 1e3;

    combblas::perfcnt2d.GetRankInfo();
    combblas::perfcnt2d.RegisterDouble("Rseq(ms)",2,1000.);
    combblas::perfcnt2d.RegisterDouble("permute(ms)",2,1000.);
    combblas::perfcnt2d.RegisterDouble("bcastA(ms)",2,1000.);
    combblas::perfcnt2d.RegisterDouble("bcastB(ms)",2,1000.);
    combblas::perfcnt2d.RegisterDouble("compT(ms)",2,1000.);
    combblas::perfcnt2d.RegisterDouble("merge(ms)",2,1000.);
    combblas::perfcnt2d.RegisterInteger("Alocalnnz",2,1.);
    combblas::perfcnt2d.RegisterInteger("Blocalnnz",2,1.);
    combblas::perfcnt2d.RegisterInteger("Clocalnnz",2,1.);
    combblas::perfcnt2d.RegisterDouble("TotTime(s)",2,1.0); // report total time in s

    combblas::perfcnt1dop.GetRankInfo();
    combblas::perfcnt1dop.RegisterDouble("TransR(ms)",2,1000.); // 
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
    combblas::perfcnt1dop.RegisterDouble("TransRes(ms)",2,1000.); // 
    combblas::perfcnt1dop.RegisterDouble("TotTime(s)",2,1.0); // report total time in s

    combblas::perfcnt3d.GetRankInfo();
    combblas::perfcnt3d.RegisterDouble("splitmat(ms)",2,1000.); // report time in ms.
    combblas::perfcnt3d.RegisterDouble("bcastA(ms)",2,1000.);
    combblas::perfcnt3d.RegisterDouble("bcastB(ms)",2,1000.);
    combblas::perfcnt3d.RegisterDouble("LComp(ms)",2,1000.); // local compute
    combblas::perfcnt3d.RegisterDouble("mergL(ms)",2,1000.); // merge layer 
    combblas::perfcnt3d.RegisterDouble("SUMMA(ms)",2,1000.); // SUMMA = merge layer + localcompute + bcastA + bcastB
    combblas::perfcnt3d.RegisterDouble("A2AinRe(ms)",2,1000.);// All2All in reduction
    combblas::perfcnt3d.RegisterDouble("Redu(ms)",2,1000.); // reduction
    combblas::perfcnt3d.RegisterDouble("mergeF(ms)",2,1000.); // merge fiber 
    combblas::perfcnt3d.RegisterDouble("TotTime(s)",2,1.); // total time

    combblas::perfcntBC.GetRankInfo();
    combblas::perfcntBC.RegisterInteger("Rounds"); //
    combblas::perfcntBC.RegisterInteger("BFSLoops"); //
    combblas::perfcntBC.RegisterDouble("BFS1D(s)",2,1.); // bfs time
    combblas::perfcntBC.RegisterDouble("BFS2D(s)",2,1.); // bfs time
    combblas::perfcntBC.RegisterDouble("BFS3D(s)",2,1.); // bfs time
    combblas::perfcntBC.RegisterDouble("Tally1D(s)",2,1.); //tally time
    combblas::perfcntBC.RegisterDouble("Tally2D(s)",2,1.); //tally time
    combblas::perfcntBC.RegisterDouble("Tally3D(s)",2,1.); //tally time
    combblas::perfcntBC.RegisterDouble("otherT(s)",2,1.); // other time

    MPI_Timer timer;
    
{
    shared_ptr<CommGrid> World2D = make_shared<CommGrid>(MPI_COMM_WORLD, 0, 0); // 2d Grid
    shared_ptr<CommGrid1D> World1D = make_shared<CommGrid1D>(MPI_COMM_WORLD); // 1D grid
    unsigned GRROWS, GRCOLS, C_FACTOR;
    int npsqrt = sqrt(nprocs);
    GRROWS = GRCOLS = npsqrt;
    C_FACTOR = 1;
    CCGrid CMG(C_FACTOR, GRCOLS); // actually it a 2d grid but we use this class in Rop
    int nthreads;
    #pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
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
    int totalnodesize = SpParHelper::GetTotalNodeSize();
    for(std::string mtxfile : opts.dataset){
        std::string testrecordprefix;
        testrecordprefix += "Dataset" + mtxfile + "_" + to_string(totalnodesize) + "Node_" + to_string(nprocs)+"mpi_"+to_string(nthreads)+"omp_";
        std::string fullfile = opts.fullfilepath[mtxfile];
        Sp2D Read2D(World2D);
        if(myrank==0) std::cerr << "reading mtx file " << fullfile << std::endl;
        timer.start("GetTestMatrix");
        if(!RMATGEN(fullfile)){
            Read2D.ParallelReadMM(fullfile, true, maximum<double>());
        }else{
            SpDCCols<int64_t, double> *A2Dder;
            if(mtxfile == "ER"){
                testrecordprefix += "scale"+to_string(opts.scale_ER) + "_" + "EF"+to_string(opts.EDGEFACTOR_ER);
                A2Dder = GenMat<int64_t,double>(
                    CMG, opts.scale_ER, opts.EDGEFACTOR_ER, 
                    opts.initiator_ER, false); // don't permute we handle permute
            }else if(mtxfile == "G500"){
                testrecordprefix += "scale"+to_string(opts.scale_G500) + "_" + "EF"+to_string(opts.EDGEFACTOR_G500);
                A2Dder = GenMat<int64_t,double>(
                    CMG, opts.scale_G500, opts.EDGEFACTOR_G500, 
                    opts.initiator_G500, false); // don't permute we handle permute
            }else if(mtxfile == "SSCA"){
                testrecordprefix += "scale"+to_string(opts.scale_SSCA) + "_" + "EF"+to_string(opts.EDGEFACTOR_SSCA);
                A2Dder = GenMat<int64_t,double>(
                    CMG, opts.scale_SSCA, opts.EDGEFACTOR_SSCA, 
                    opts.initiator_SSCA, false); // don't permute we handle permute
            }
            Read2D = Sp2D(new DER(*A2Dder),World2D);
        }
        timer.stop("GetTestMatrix");
        // make sure input is square matrix and symmetric matrix since metis only support graph.
        IT Anrows, Ancols;
        bool matsymm(true), matsquare(true);
        Anrows = Read2D.getnrow();
        Ancols = Read2D.getncol();
        Read2D.PrintInfo();
        if(Ancols != Anrows) matsquare = false;
        Sp2D tmpAT(Read2D);
        tmpAT.Transpose();
        if(! (tmpAT == Read2D) ) matsymm = false;
        if(myrank==0)std::cerr<<"matrix Square ? " << matsquare << " Symmetric ? " << matsymm << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        Sp2D A2Dperm(Read2D); // get a backup of origin dataset
        Sp1D Read1D(Read2D); // get origin data in 1D
        Sp1D A1Dperm(Read1D); // get origin data in 1D
        // now decide permutation type for 2D, 
        FullyDistVec<int64_t,int64_t> perm(World2D);    // get a different permutation
        if(opts.permute == 0){
            testrecordprefix += "NoPerm_";
        }else if(opts.permute == 1){
            testrecordprefix += "RandomPerm_";
            // gen perm
            if(myrank==0) std::cerr << "appying random permutation on matrix" << std::endl;
            timer.start("GenRandomSeq");
            perm.iota(A2Dperm.getnrow(), 0);
            perm.RandPerm();
            timer.stop("GenRandomSeq");
            
            perfcnt2d.doublemap["Rseq(ms)"] = timer.records["GenRandomSeq"];
            perfcnt1d.doublemap["Rseq(ms)"] = timer.records["GenRandomSeq"];

            timer.start("Permutation2D");
            A2Dperm(perm, perm, true);    // 2d perm
            timer.stop("Permutation2D");
            perfcnt2d.doublemap["permute(ms)"] = timer.records["Permutation2D"];
            timer.start("Permutation1D");
            A1Dperm(perm,perm);           // 1D perm
            timer.stop("Permutation1D");
            perfcnt1d.doublemap["permute(ms)"] = timer.records["Permutation1D"];
        }else if(opts.permute >= 2){
            // actually if permute >= 2, performance of 2d/3d is bad. so no need to benchmark it.
            // gene perm vector using METIS
            if(!matsquare){
                if(myrank==0)std::cerr<<"metis partition only supprt sqare matrix" << std::endl;
                exit(0);
            }
            if(!matsymm){
                if(myrank==0)std::cerr<<"metis partition only supprt symmetric matrix" << std::endl;
                exit(0);
            }
            testrecordprefix += "METIS";
            if(opts.permute==2) testrecordprefix += "novwgt_";
            else if(opts.permute==3) testrecordprefix += "vwgtnnz_";
            else if(opts.permute==4) testrecordprefix += "vwgtflops_";
            else if(opts.permute==5) testrecordprefix += "vwgtfile_";
            if(myrank==0) {
                std::cerr << "appying metis permutation on matrix, ";
                if(opts.permute==2) std::cerr << "no vertex weight ";
                else if(opts.permute==3) std::cerr << "vertex weight:nnz ";
                else if(opts.permute==4) std::cerr << "vertex weight:flops ";
                else if(opts.permute==5) std::cerr << "vertex weight:file " << opts.permfile;
            }
            MPI_Barrier(comm);
            
            //generate a permutation but do not change A
            std::vector<int64_t> fullresults;
            if(opts.permute >= 2 && opts.permute < 5){
                std::vector<int64_t> results;
                METISPartition(A1Dperm, results, opts.permute, opts.nparts);
                SpParHelper::AllgatherVector(fullresults, results);
            }else if(opts.permute == 5){
                // provide a file, everyone read it.
                if(opts.permfile == ""){
                    opts.permfile = "/global/cfs/cdirs/m1982/yuxihong/spgemm1ddataset/"+ mtxfile + "_flops.graph.part_parmetis." + std::to_string(nprocs);
                }
                std::ifstream inputFile(opts.permfile);
                if(!inputFile.is_open()){
                    if(myrank==0)std::cerr<<"something is wrong with input partition file "<<std::endl;
                    exit(0);
                }
                std::string line;
                while (std::getline(inputFile, line)) { // Read each line from the file
                    IT number;
                    std::istringstream iss(line);
                    if (iss >> number) { // Check if the line contains an integer
                        fullresults.push_back(number);
                    }
                }
            }
            MPI_Barrier(comm);
            std::unordered_map<int, std::vector<int64_t>> idxmap;
            for(IT i=0; i<fullresults.size(); i++){
                idxmap[fullresults[i]].push_back(i);
            }
            std::vector<IT> permvec;
            IT mystart = A1Dperm.getblocksizeprefix()[myrank];
            IT myend = myrank == nprocs-1 ? Ancols : A1Dperm.getblocksizeprefix()[myrank+1];
            IT idx = 0;
            for(int i=0; i<opts.nparts; i++){
                for(auto x : idxmap[i]){
                    if(mystart <= idx && idx < myend){
                        permvec.push_back(x);
                    }
                    idx++;
                }
            }
            perm = FullyDistVec<int64_t, int64_t>(permvec, World2D);
            double tmp;
            MPI_Barrier(comm);
            tmp = MPI_Wtime();
            A2Dperm(perm, perm, true);    // 2d perm
            // A2Dperm.ParallelWriteMM("A2Dperm.mtx",true);
            tmp = MPI_Wtime() - tmp;
            perfcnt2d.doublemap["permute(ms)"] = tmp;
            MPI_Barrier(comm);
            tmp = MPI_Wtime();
            std::vector<IT> blocksize(nprocs);
            IT mypartsize = idxmap[myrank].size();
            MPI_Allgather(
                &mypartsize, 1, MPI_LONG_LONG, 
                blocksize.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
            A1Dperm(perm,perm,blocksize,blocksize);           // 1D perm
            tmp = MPI_Wtime() - tmp;
            perfcnt1d.doublemap["permute(ms)"] = tmp;
            MatrixEqual(A1Dperm, A2Dperm, "permutation");
        }
        A1Dperm.BlockwiseNNZanalysis(testrecordprefix + "BlockAnalyze.txt");
        // preprocessing ends
        bool skip2d = opts.permute>=2 || (!opts.run2d);
        if(opts.AA){
            Sp2D CopyA2D(A2Dperm);
            MPI_Barrier(comm);
            if(!skip2d) {
                if(myrank==0)std::cerr << "Run 2d AA test:" << std::endl;
                std::string file2d = testrecordprefix + "AA2d.log";
                Sp2D CopyB2D(A2Dperm);
                timer.start("AA2D");
                Sp2D Ctmp = Mult_AnXBn_Synch<PTFF, double, DER>(CopyA2D,CopyB2D);
                timer.stop("AA2D");
                perfcnt2d.doublemap["TotTime(s)"] = timer.records["AA2D"];
                perfcnt2d.OutputRecords(file2d);
                if(myrank==0)std::cerr << "Done!"<<std::endl << std::endl;
            }
            // run 1d
            if(opts.run1d){
                std::string file1d = testrecordprefix + "AA1d_gentype"+std::to_string(combblas::opts.gentype[0])+".log";
                if(myrank==0)std::cerr<<"Run 1d AA test:"<< std::endl;
                Sp1D A1D(A1Dperm);
                Sp1D B1D(A1Dperm);
                // timer.start("AA1D");
                MPI_Barrier(comm);
                double tmpstart = MPI_Wtime();
                Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, double, DER>(A1D,B1D,false,false,opts.gentype[0]); // warm up
                double tmpend = MPI_Wtime();
                // timer.stop("AA1D");
                // perfcnt1d.doublemap["TotTime(s)"] = timer.records["AA1D"];
                perfcnt1d.doublemap["TotTime(s)"] = tmpend - tmpstart;
                perfcnt1d.OutputRecords(file1d);
                // run algorithm again and generate communication analysis file.
                opts.comm_ana = true;
                opts.testprefix = testrecordprefix + "AA1d_gentype"+std::to_string(combblas::opts.gentype[0]);
                Sp1D Ctmp = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, double, DER>(A1D,B1D,false,false,opts.gentype[0]); // warm up
                opts.comm_ana = false;
                opts.testprefix = "";
                Sp2D C1Dres = Sp2D(Ctmp, Ctmp.getblocksizevec());
                
                // if(myrank==0)std::cerr<<"Done!";
                if(opts.check){
                    Sp2D B2D(A2Dperm);
                    Sp2D C2Dref = Mult_AnXBn_Synch<PTFF, double, DER>(A2Dperm, B2D);
                    if(C2Dref == C1Dres){
                        if(myrank==0)std::cerr<<"1D AA is correct!";
                    }else{
                        if(myrank==0)std::cerr<<"1D AA is wong!";
                    }
                }
                if(myrank==0)std::cerr<<"Done!"<<std::endl << std::endl;
            }
            // run 3d
            if(opts.run3d){
                for(int layeri : opts.layer){
                    int pr = nprocs / layeri;
                    pr = sqrt(pr);
                    if(pr * pr * layeri != nprocs){
                        if(myrank==0)std::cerr<<"3dlayerconfig has problem " << pr << "x"<<pr<<"x"<<layeri << "!=" << nprocs<<std::endl;
                    }
                    string testname = "AA3D_Pr"+to_string(pr)+"Pc"+to_string(pr)+"layer"+to_string(layeri);
                    if(myrank==0)std::cerr << "Run " << testname <<std::endl;
                    std::string file3d = testrecordprefix + testname + ".log";
                    timer.start("splitmat"+testname);
                    SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> A3D(A2Dperm, layeri, true, false);
                    SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> B3D(A2Dperm, layeri, false, false);
                    timer.stop("splitmat"+testname);
                    perfcnt3d.doublemap["splitmat(ms)"] = timer.records["splitmat"+testname];
                    MPI_Barrier(MPI_COMM_WORLD);
                    timer.start("3Dkernel"+testname);
                    Sp3D C3D = Mult_AnXBn_SUMMA3D<PTFF, double, DER>(A3D,B3D);
                    timer.stop("3Dkernel"+testname);
                    perfcnt3d.doublemap["TotTime(s)"] = timer.records["3Dkernel"+testname];
                    // if(myrank==0)std::cerr << "Done!";
                    if(opts.check){
                        Sp2D C3d2d = C3D.Convert2D();
                        Sp2D B2D(A2Dperm);
                        Sp2D C2Dref = Mult_AnXBn_Synch<PTFF, double, DER>(A2Dperm, B2D);
                        if(C2Dref == C3d2d){
                            if(myrank==0)std::cerr<<"3D AA is correct!";
                        }else{
                            if(myrank==0)std::cerr<<"3D AA is wong!";
                        }
                    }
                    if(myrank==0)std::cerr<< "Done!" <<std::endl << std::endl;
                    perfcnt3d.OutputRecords(file3d);
                }
            }
        }
        if(opts.Rop){
            SpDCCols<int64_t, double> splitA, splitB, splitR, splitRT;
            SpDCCols<int64_t, double> *R, *RT;
            SpDCCols<int64_t, double> *A2Dder = A2Dperm.seqptr();
            timer.start("RestrictionOp");
            RestrictionOp( CMG, A2Dder, R, RT,true); // only R is used, here R is not random permuted.
            timer.stop("RestructionOp");
            // // maybe we need to permute R metis partition R?
            Sp2D R2D(new DER(*R),World2D);
            Sp2D RT2D(R2D); 
            R2D.PrintInfo();
            RT2D.Transpose();
            if(!skip2d) {
                // RT x A
                if(myrank==0)std::cerr << "Run 2d RT x A test:" << std::endl;
                std::string file2d = testrecordprefix + "RTA2d.log";
                timer.start("RTA2D");
                Sp2D RTA2D = Mult_AnXBn_Synch<PTFF, double, DER>(RT2D,A2Dperm);
                timer.stop("RTA2D");
                perfcnt2d.doublemap["TotTime(s)"] = timer.records["RTA2D"];
                perfcnt2d.OutputRecords(file2d);
                if(myrank==0)std::cerr<<"Done!" << std::endl;
                // RTA x R
                if(myrank==0)std::cerr << "Run 2d RTA x R test:"<<std::endl;
                file2d = testrecordprefix + "RTAR2d.log";
                timer.start("RTAR2D");
                Sp2D RTAR2D = Mult_AnXBn_Synch<PTFF, double, DER>(RTA2D,R2D);
                timer.stop("RTAR2D");
                perfcnt2d.doublemap["TotTime(s)"] = timer.records["RTAR2D"];
                perfcnt2d.OutputRecords(file2d);
                if(myrank==0)std::cerr << "Done!" << std::endl << std::endl;
            }
            if(opts.run1d){
                if(myrank==0)std::cerr<<"Run 1d RT x A test!"<<std::endl;
                std::string file1d = testrecordprefix + "RTA1d_gentype"+std::to_string(combblas::opts.gentype[0])+".log";
                Sp1D RT1D(RT2D);
                Sp1D R1D(R2D);
                timer.start("RTA1D");
                Sp1D RTA1D = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, double, DER>(RT1D,A1Dperm,false,false,opts.gentype[0]);
                timer.stop("RTA1D");
                perfcnt1d.doublemap["TotTime(s)"] = timer.records["RTA1D"];
                perfcnt1d.OutputRecords(file1d);
                opts.comm_ana = true;
                opts.testprefix = testrecordprefix + "RTA1d_gentype" + std::to_string(combblas::opts.gentype[0]);
                Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, double, DER>(RT1D,A1Dperm,false,false,opts.gentype[0]);
                opts.comm_ana = false;
                opts.testprefix = "";
                if(opts.check){
                    Sp2D RTA2D = Mult_AnXBn_Synch<PTFF, double, DER>(RT2D,A2Dperm);
                    Sp2D C2D1D(RTA1D,RTA1D.getblocksizevec());
                    if(RTA2D == C2D1D){
                        if(myrank==0)std::cerr<<"1D RTA is correct!";
                    }else{
                        if(myrank==0)std::cerr<<"1D RTA is wong!";
                    }
                }
                if(myrank==0)std::cerr<<"Done!"<<std::endl << std::endl;

                // RTA x R
                if(myrank==0)std::cerr<<"Run RTAR 1D test!"<<std::endl;
                file1d = testrecordprefix + "RTAR1d.log";
                timer.start("RTAR1D");
                Sp1D RTAR1D = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, double, DER>(RTA1D, R1D,false,false,opts.gentype[0]);
                timer.stop("RTAR1D");
                perfcnt1d.doublemap["TotTime(s)"] = timer.records["RTAR1D"];
                perfcnt1d.OutputRecords(file1d);
                opts.comm_ana = true;
                opts.testprefix = testrecordprefix + "RTAR1d";
                Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, double, DER>(RTA1D, R1D,false,false,opts.gentype[0]);
                opts.comm_ana = false;
                opts.testprefix = "";
                if(opts.check){
                    Sp2D RTA2D = Mult_AnXBn_Synch<PTFF, double, DER>(RT2D,A2Dperm);
                    Sp2D RTAR2D = Mult_AnXBn_Synch<PTFF, double, DER>(RTA2D,R2D);
                    Sp2D C2D1D(RTAR1D,RTAR1D.getblocksizevec());
                    if(RTAR2D == C2D1D){
                        if(myrank==0)std::cerr<<"1D RTAR is correct!";
                    }else{
                        if(myrank==0)std::cerr<<"1D RTAR is wong!";
                    }
                }
                if(myrank==0)std::cerr<<"Done!"<<std::endl << std::endl;
                // RTA x R OP
                if(myrank==0)std::cerr<<"Run RTAR 1D OP test!"<<std::endl;
                file1d = testrecordprefix + "RTAR1DOP.log";
                timer.start("RTA1DOP-TransA");
                RTA1D.Transpose();
                timer.stop("RTA1DOP-TransA");
                perfcnt1dop.doublemap["TransR(ms)"] = timer.records["RTA1D-Transpose"];
                timer.start("RTAR1DOP");
                Sp1D RTAROP1D = Mult_AnXBn_1D_OP<PTFF, double, DER>(RT1D, RTA1D);
                timer.stop("RTAR1DOP");
                perfcnt1dop.doublemap["TotTime(s)"] = timer.records["RTAR1DOP"];
                timer.start("RTAR1DOP-TransC");
                RTAROP1D.Transpose();
                timer.stop("RTAR1DOP-TransC");
                perfcnt1dop.doublemap["TransRes(ms)"] = timer.records["RTAR1DOP-TransC"];
                perfcnt1dop.OutputRecords(file1d);
                if(opts.check){
                    Sp2D RTA2D = Mult_AnXBn_Synch<PTFF, double, DER>(RT2D,A2Dperm);
                    Sp2D RTAR2D = Mult_AnXBn_Synch<PTFF, double, DER>(RTA2D,R2D);
                    Sp2D C1D2D(RTAROP1D, RTAROP1D.getblocksizevec());
                    if(RTAR2D == C1D2D){
                        if(myrank==0)std::cerr<<"1D OP RTAR is correct!";
                    }else{
                        if(myrank==0)std::cerr<<"1D OP RTAR is wong!";
                    }
                }
                if(myrank==0)std::cerr<<"Done!" << std::endl << std::endl;
            }
            if(opts.run3d){
                for(int layeri : opts.layer){
                    int pr = nprocs / layeri;
                    pr = sqrt(pr);
                    if(pr * pr * layeri != nprocs){
                        if(myrank==0)std::cerr<<"3dlayerconfig has problem " << pr << "x"<<pr<<"x"<<layeri << "!=" << nprocs<<std::endl;
                    }
                    string testname = "RTA3D_Pr"+to_string(pr)+"Pc"+to_string(pr)+"layer"+to_string(layeri);
                    if(myrank==0)std::cerr << "Run " << testname <<std::endl;
                    std::string file3d = testrecordprefix + testname + ".log";
                    timer.start("splitmat"+testname);
                    SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> RT3D(RT2D, layeri, true, false);
                    SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> A3D(A2Dperm, layeri, false, false);
                    timer.stop("splitmat"+testname);
                    perfcnt3d.doublemap["splitmat(ms)"] = timer.records["splitmat"+testname];
                    timer.start("RTA3D");
                    Sp3D RTA3D = Mult_AnXBn_SUMMA3D<PTFF, double, DER>(RT3D,A3D);
                    timer.stop("RTA3D");
                    Sp2D RTA3D2D = RTA3D.Convert2D();
                    perfcnt3d.doublemap["TotTime(s)"] = timer.records["RTA3D"];
                    perfcnt3d.OutputRecords(file3d);
                    if(opts.check){
                        Sp2D RTA2D = Mult_AnXBn_Synch<PTFF, double, DER>(RT2D,A2Dperm);
                        if(RTA2D == RTA3D2D){
                            if(myrank==0) std::cerr << "3D RTA correct";
                        }else{
                            if(myrank==0) std::cerr << "3D RTA wrong";
                        }
                    }
                    if(myrank==0)std::cerr<<"Done!" << std::endl << std::endl;
                    // testname = "RTAR3D_Pr"+to_string(pr)+"Pc"+to_string(pr)+"layer"+to_string(layeri);
                    // if(myrank==0)std::cerr << "Run " << testname <<std::endl;
                    // file3d = testrecordprefix + testname + ".log";
                    // timer.start("splitmat"+testname);
                    // Sp3D R3D(R2D, layeri, false, false);
                    // Sp3D RTA3D_xxx(RTA3D2D, layeri, true, false);
                    // timer.stop("splitmat"+testname);
                    // perfcnt3d.doublemap["splitmat(ms)"] = timer.records["splitmat"+testname];
                    // timer.start("RTAR3D");
                    // Sp3D RTAR3D = Mult_AnXBn_SUMMA3D<PTFF, double, DER>(RTA3D_xxx,R3D);
                    // timer.stop("RTAR3D");
                    // perfcnt3d.doublemap["TotTime(s)"] = timer.records["RTAR3D"];
                    // perfcnt3d.OutputRecords(file3d);
                    // if(opts.check){
                    //     Sp2D RTAR3D2D = RTAR3D.Convert2D();
                    //     Sp2D RTA2D = Mult_AnXBn_Synch<PTFF, double, DER>(RT2D,A2Dperm);
                    //     Sp2D RTAR2D = Mult_AnXBn_Synch<PTFF, double, DER>(RTA2D,R2D);
                    //     if(RTAR2D == RTAR3D2D){
                    //         if(myrank==0) std::cerr << "3D RTAR correct";
                    //     }else{
                    //         if(myrank==0) std::cerr << "3D RTAR wrong";
                    //     }
                    // }
                    // if(myrank==0)std::cerr<<"Done!" << std::endl << std::endl;
                }
                for(int layeri : opts.layer){
                    int pr = nprocs / layeri;
                    pr = sqrt(pr);
                    if(pr * pr * layeri != nprocs){
                        if(myrank==0)std::cerr<<"3dlayerconfig has problem " << pr << "x"<<pr<<"x"<<layeri << "!=" << nprocs<<std::endl;
                    }
                    SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> RT3D(RT2D, layeri, true, false);
                    SpParMat3D<int64_t, double, SpDCCols < int64_t, double >> A3D(A2Dperm, layeri, false, false);
                    Sp3D RTA3D = Mult_AnXBn_SUMMA3D<PTFF, double, DER>(RT3D,A3D);
                    Sp2D RTA3D2D = RTA3D.Convert2D();
                    std::string testname = "RTAR3D_Pr"+to_string(pr)+"Pc"+to_string(pr)+"layer"+to_string(layeri);
                    if(myrank==0)std::cerr << "Run " << testname <<std::endl;
                    std::string file3d = testrecordprefix + testname + ".log";
                    timer.start("splitmat"+testname);
                    Sp3D R3D(R2D, layeri, false, false);
                    Sp3D RTA3D_xxx(RTA3D2D, layeri, true, false);
                    timer.stop("splitmat"+testname);
                    perfcnt3d.doublemap["splitmat(ms)"] = timer.records["splitmat"+testname];
                    timer.start("RTAR3D");
                    Sp3D RTAR3D = Mult_AnXBn_SUMMA3D<PTFF, double, DER>(RTA3D_xxx,R3D);
                    timer.stop("RTAR3D");
                    perfcnt3d.doublemap["TotTime(s)"] = timer.records["RTAR3D"];
                    perfcnt3d.OutputRecords(file3d);
                    if(opts.check){
                        Sp2D RTAR3D2D = RTAR3D.Convert2D();
                        Sp2D RTA2D = Mult_AnXBn_Synch<PTFF, double, DER>(RT2D,A2Dperm);
                        Sp2D RTAR2D = Mult_AnXBn_Synch<PTFF, double, DER>(RTA2D,R2D);
                        if(RTAR2D == RTAR3D2D){
                            if(myrank==0) std::cerr << "3D RTAR correct";
                        }else{
                            if(myrank==0) std::cerr << "3D RTAR wrong";
                        }
                    }
                    if(myrank==0)std::cerr<<"Done!" << std::endl << std::endl;
                }
            }
        }
        if(opts.BC){ // run betweeness centrality
            if(myrank==0)std::cerr << "Run BC test:" << std::endl;
            std::string file2d = testrecordprefix + "BC.log";
            double total_start = MPI_Wtime();
            std::string outfilename = mtxfile + "_bcout.txt";
            typedef PlusTimesSRing<bool, int> PTBOOLINT;
            typedef PlusTimesSRing<bool, double> PTBOOLDOUBLE;
            typedef PlusTimesSRing<int, double> PTINTDOUBLE;
            typedef PlusTimesSRing<double, int> PTDOUBLEINT;
            typedef PlusTimesSRing<int, int> PTINTINT;
            typedef PlusTimesSRing<double, double> PTDOUBLEDOUBLE;
            vector<int> blocksizevec;
            for(auto x : A1Dperm.getblocksizevec()) blocksizevec.push_back((int)x);
            vector<int> rowblocksizevec;
            for(auto x : A1Dperm.getrowblocksizevec()) rowblocksizevec.push_back((int)x);

            // A1Dperm.Apply(std::bind([](double &a, double &b){return 1.0;},std::placeholders::_1, 1.0));
            A2Dperm.Apply(std::bind([](double &a, double &b){return 1.0;},std::placeholders::_1, 1.0));
            // Dist<IT, int>::MPI_DCCols A2dint = A1Dperm;
            Dist<int, int>::MPI_DCCols A2dint(A2Dperm);
            A2dint.PrintInfo();
            Dist<int, int>::MPI_DCCols A2dintT(A2dint);
            A2dintT.Transpose();
            Dist<int,int>::MPI_DCCols1D A1dint(A2dint,blocksizevec,rowblocksizevec);
            Dist<int,int>::MPI_DCCols1D A1dintT(A2dintT,blocksizevec,rowblocksizevec);
            IT mdim = A1Dperm.getnrow();
            IT ndim = A1Dperm.getncol();
            if(mdim != ndim){
                if(myrank==0)std::cerr<<"matrix is not graph!"<<std::endl;
                exit(0);
            }
            int K4Approx = opts.AproxK;
            int batchSize = opts.batchSize;

            int nPasses = (int) pow(2.0, K4Approx);
            if(nPasses > mdim){
                if(myrank==0)std::cerr<<"nPasses:"<<nPasses<<", larger than mdim: "<< mdim << ", reducing npass ... "<<std::endl;
                while(nPasses > mdim) nPasses/=2;
                if(myrank==0)std::cerr<<"new nPasses:"<<nPasses << std::endl;
            }
            int numBatches = (int) ceil( static_cast<float>(nPasses)/ static_cast<float>(batchSize));

            // get the number of batch vertices for submatrix
            int subBatchSize = batchSize / (A2dintT.getcommgrid())->GetGridCols();
            int nBatchSize = subBatchSize * (A2dintT.getcommgrid())->GetGridCols();
            nPasses = numBatches * nBatchSize;	// update the number of starting vertices	
            if(batchSize % (A2dintT.getcommgrid())->GetGridCols() > 0 && myrank == 0)
            {
                cout << "*** Batchsize is not evenly divisible by the grid dimension ***" << endl;
                cout << "*** Processing "<< nPasses <<" vertices instead"<< endl;
            }
            ostringstream tinfo;
            tinfo << "Batch processing will occur " << numBatches << " times, each processing " << nBatchSize << " vertices (overall)" << endl;
            SpParHelper::Print(tinfo.str());
            vector<int> candidates;
            // Only consider non-isolated vertices
            int vertices = 0;
            int vrtxid = 0; 
            int nlocpass = numBatches * subBatchSize;
            while(vertices < nlocpass)
            {
                vector<int> single;
                vector<int> empty;
                single.push_back(vrtxid); // will return ERROR if vrtxid > N (the column dimension) 
                int locnnz = ((A2dintT.seq())(empty,single)).getnnz();
                int totnnz;
                MPI_Allreduce( &locnnz, &totnnz, 1, MPI_INT, MPI_SUM, (A2dintT.getcommgrid())->GetColWorld());
                if(totnnz > 0)
                {
                    candidates.push_back(vrtxid);
                    ++vertices;
                }
                ++vrtxid;
            }
            vector<int> batch(subBatchSize);
            FullyDistVec<int, double> bc(A2dintT.getcommgrid(), A2dint.getnrow(), 0.0);
            if(myrank==0)std::cerr<<"numbatches:"<<numBatches<<",mdim:" << mdim << ",batchsize:" <<batchSize <<std::endl;
            // some statistics data.
            double tmp_start, tmp_end;
            std::vector<double> roundtime; // everyround time: other time + bfs time + tally time 
            std::vector<double> roundothertime; // everyround time: preparation time + bfs time + tally time 
            // std::vector<double> bfsroundtime; // everyround time: preparation time + bfs time + tally time 
            // detaileed
            // std::vector<double> bfstimeother; // each loop, bfs time, SpGEMM is not included!!!
            std::vector<double> bfsSpgemm3Dtime; // each loop, bfs time  
            std::vector<double> bfsSpgemm2Dtime; // each loop, bfs time 
            std::vector<double> bfsSpgemm1Dtime; // each loop, bfs time 
            std::vector<double> bfsSpgemm3DtimePerRound; // each loop, bfs time  
            std::vector<double> bfsSpgemm2DtimePerRound; // each loop, bfs time 
            std::vector<double> bfsSpgemm1DtimePerRound; // each loop, bfs time 
            std::vector<int> bfsLoopPerRound;
            // std::vector<double> tallyroundtime; // each loop, tally time 
            // detailed
            // std::vector<double> tallytimeother; // each loop, tally time 
            std::vector<double> tallySpgemm3Dtime; // each loop, bfs time 
            std::vector<double> tallySpgemm2Dtime; // each loop, bfs time 
            std::vector<double> tallySpgemm1Dtime; // each loop, bfs time 
            std::vector<double> tallySpgemm3DtimePerRound; // each loop, bfs time 
            std::vector<double> tallySpgemm2DtimePerRound; // each loop, bfs time 
            std::vector<double> tallySpgemm1DtimePerRound; // each loop, bfs time 
            // roundtime = roundothertime + bfstime (bfsspgemm + bfsother) + tallytime(tallyspemm+tallyother)
            roundothertime = std::vector<double>(numBatches,0.0);
            roundtime = std::vector<double>(numBatches,0.0);
            bfsSpgemm1Dtime = std::vector<double>(numBatches,0);
            bfsSpgemm2Dtime = std::vector<double>(numBatches,0);
            bfsSpgemm3Dtime = std::vector<double>(numBatches,0);
            tallySpgemm1Dtime = std::vector<double>(numBatches,0);
            tallySpgemm2Dtime = std::vector<double>(numBatches,0);
            tallySpgemm3Dtime = std::vector<double>(numBatches,0);
            bfsLoopPerRound = std::vector<int>(numBatches, 0);
            MPI_Timer timer;
            bool skip2d = opts.permute>=2 || (!opts.run2d);
            for(int i=0; i< numBatches; ++i)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                tmp_start = MPI_Wtime();
                for(int j=0; j< subBatchSize; ++j)
                {
                    batch[j] = candidates[i*subBatchSize + j];
                }
                Dist<int,int>::MPI_DCCols fringe = A2dintT.SubsRefCol(batch);
                Dist<int,int>::DCCols * nsploc = new Dist<int,int>::DCCols();
                tuple<int,int,int> * mytuples = NULL;
                if(A2dintT.getcommgrid()->GetRankInProcRow() == A2dintT.getcommgrid()->GetRankInProcCol())
                {
                    mytuples = new tuple<int,int,int>[subBatchSize];
                    for(int k =0; k<subBatchSize; ++k)
                    {
                        mytuples[k] = make_tuple(batch[k], k, 1);
                    }
                    nsploc->Create( subBatchSize, A2dintT.getlocalrows(), subBatchSize, mytuples);
                }else{  
                    nsploc->Create( 0, A2dintT.getlocalrows(), subBatchSize, mytuples);
                }
                Dist<int,int>::MPI_DCCols  nsp(nsploc, A2dintT.getcommgrid());
                vector < Dist<int,int>::MPI_DCCols * > bfs; // internally keeps track of depth
                tmp_end = MPI_Wtime();
                roundothertime[i] += tmp_end - tmp_start;
                int loop = 0;
                while( fringe.getnnz() > 0 )
                {
                    MPI_Barrier(MPI_COMM_WORLD);
                    /* before spgemm */
                    tmp_start = MPI_Wtime();
                    nsp += fringe;
                    Dist<int,int>::MPI_DCCols * level = new Dist<int,int>::MPI_DCCols( fringe ); 
                    bfs.push_back(level);
                    tmp_end = MPI_Wtime();
                    roundothertime[i] += tmp_end - tmp_start;
                    // bfstimeother.push_back(tmp_end - tmp_start);
                    IT fringennz = 0;
                    // /* SpGEMM 2d */
                    double tmp2dtime = 0;
                    if(opts.run2d){
                    Dist<int,int>::MPI_DCCols fringeCopy(fringe);
                    timer.start("bfs-spgemm2d");
                    fringe = Mult_AnXBn_Synch<PTINTINT, int, SpDCCols<int, int>>(A2dintT, fringe);
                    timer.stop("bfs-spgemm2d",false,false);
                    tmp2dtime = timer.records["bfs-spgemm2d"];
                    bfsSpgemm2Dtime[i] += (tmp2dtime);
                    bfsSpgemm2DtimePerRound.push_back(tmp2dtime);
                    }
                    /* SpGEMM 1d */
                    double tmp1dtime = 0;
                    if(opts.run1d){
                    // Dist<int,int>::MPI_DCCols fringeT(fringeCopy);
                    Dist<int,int>::MPI_DCCols fringeT(fringe);
                    fringeT.Transpose();
                    Dist<int,int>::MPI_DCCols1D fringeT1D(fringeT,blocksizevec);
                    timer.start("bfs-spgemm1d");
                    Dist<int,int>::MPI_DCCols1D fringe1dnew = 
                    Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTINTINT, int, SpDCCols<int, int>>(fringeT1D,A1dint);
                    timer.stop("bfs-spgemm1d",false,false);
                    fringe1dnew.Transpose();
                    // if(opts.check){
                    //     MatrixEqual(fringe1dnew,fringe,"bfsspgemm");
                    // }
                    tmp1dtime = timer.records["bfs-spgemm1d"];
                    bfsSpgemm1Dtime[i] += (tmp1dtime);
                    bfsSpgemm1DtimePerRound.push_back(tmp1dtime);
                    fringe = Dist<int,int>::MPI_DCCols(fringe1dnew,fringe1dnew.getblocksizevec());
                    }
                    /* SpGEMM 3d */
                    double tmp3dtime = 1e9;
                    if(opts.run3d){
                    for(int layerindex=0; layerindex<opts.layer.size(); layerindex++){
                        int layeri = opts.layer[layerindex];
                        int pr = nprocs / layeri;
                        pr = sqrt(pr);
                        if(pr * pr * layeri != nprocs){
                            if(myrank==0)std::cerr<<"3dlayerconfig has problem " << pr << "x"<<pr<<"x"<<layeri << "!=" << nprocs<<std::endl;
                        }
                        SpParMat3D<int, int, SpDCCols < int, int >> A3D(A2dintT, layeri, true, false);
                        SpParMat3D<int, int, SpDCCols < int, int >> B3D(fringe, layeri, false, false);
                        timer.start("bfs-spgemm3d");
                        SpParMat3D<int, int, SpDCCols < int, int >> C3D = Mult_AnXBn_SUMMA3D<PTINTINT, int, SpDCCols<int, int>>(A3D,B3D);
                        timer.stop("bfs-spgemm3d",false,false);
                        double ttmp =timer.records["bfs-spgemm3d"];
                        tmp3dtime = tmp3dtime < ttmp ? tmp3dtime : ttmp;
                        // if(opts.check){
                        //     Dist<int,int>::MPI_DCCols C2D = C3D.Convert2D();
                        //     MatrixEqual(C2D, fringe,"bfsspgemm3D");
                        // } 
                        if(layerindex == opts.layer.size()-1) fringe = C3D.Convert2D(); // last iteration
                    }
                    bfsSpgemm3Dtime[i] += (tmp3dtime);
                    bfsSpgemm3DtimePerRound.push_back(tmp3dtime);
                    }
                    /*Ewisemulti*/ 
                    tmp_start = MPI_Wtime();
                    fringe = EWiseMult(fringe, nsp, true);
                    tmp_end = MPI_Wtime();
                    // bfstimeother.back() += (tmp_end - tmp_start);
                    roundothertime[i] += tmp_end - tmp_start;
                    loop++;
                    fringennz = fringe.getnnz();
                    if(myrank==0){
                        std::string tmp = "-----BFS Loop: " + std::to_string(loop);
                        std::cerr<<std::setw(18)<<tmp<<", "
                        <<"1D:"<<std::setw(7)<<tmp1dtime<<", "
                        <<"2D:"<<std::setw(7)<<tmp2dtime<<", "
                        <<"3D:"<<std::setw(7)<<tmp3dtime<<", current fringe nnz:" << fringennz << std::endl;
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
                bfsLoopPerRound[i] = bfs.size();
                tmp_start = MPI_Wtime();
                Dist<int, double>::MPI_DCCols nspInv = nsp;
                nspInv.Apply(std::bind(divides<double>(),1,std::placeholders::_1));
                DenseParMat<int, double> bcu(1.0, A2dintT.getcommgrid(), fringe.getlocalrows(), fringe.getlocalcols() );
                tmp_end = MPI_Wtime();
                roundothertime[i] += tmp_end - tmp_start;
                loop = 0;
                if(myrank==0)std::cerr<<"finish bfs step, bfs size"<<bfs.size()<<std::endl;
                for(int j = bfs.size()-1; j > 0; --j)
                {
                    IT productnnz = 0;
                    // if(myrank == 0)std::cerr<<"tally loop"<<loop<<std::endl;
                    MPI_Barrier(MPI_COMM_WORLD);
                    tmp_start = MPI_Wtime();
                    Dist<int,double>::MPI_DCCols w = EWiseMult( *bfs[j], nspInv, false);
                    w.EWiseScale(bcu);
                    tmp_end = MPI_Wtime();
                    // tallytimeother.push_back(tmp_end - tmp_start);
                    roundothertime[i] += tmp_end - tmp_start;
                    Dist<int,double>::MPI_DCCols product;
                    // /* SpGEMM 2d */
                    double tmp2dtime = 0;
                    if(opts.run2d){
                    timer.start("tally-spgemm2d");
                    product = Mult_AnXBn_Synch<PTINTDOUBLE, double, SpDCCols<int, double>>(A2dint,w);
                    timer.stop("tally-spgemm2d",false,false);
                    tmp2dtime = timer.records["tally-spgemm2d"];
                    tallySpgemm2Dtime[i] += tmp2dtime;
                    tallySpgemm2DtimePerRound.push_back(tmp2dtime);
                    }
                    /* SpGEMM 1d */
                    double tmp1dtime = 0;
                    if(opts.run1d){
                    Dist<int,double>::MPI_DCCols wcopy(w);
                    wcopy.Transpose();
                    Dist<int,double>::MPI_DCCols1D wcopy1d(wcopy,blocksizevec);
                    timer.start("tally-spgemm1d");
                    Dist<int,double>::MPI_DCCols1D product1d = 
                    Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTDOUBLEINT, double, SpDCCols<int, double>>(wcopy1d,A1dintT);
                    timer.stop("tally-spgemm1d",false,false);
                    product1d.Transpose();
                    // if(opts.check) MatrixEqual(product1d,product,"tallyspgemm1d");
                    product = Dist<int,double>::MPI_DCCols(product1d,product1d.getblocksizevec());
                    tmp1dtime = timer.records["tally-spgemm1d"];
                    tallySpgemm1Dtime[i] += (tmp1dtime);
                    tallySpgemm1DtimePerRound.push_back(tmp1dtime);
                    }
                    /* SpGEMM 3d */
                    
                    double tmp3dtime = 1e9;
                    if(opts.run3d){
                    for(int layeri : opts.layer){
                        int pr = nprocs / layeri;
                        pr = sqrt(pr);
                        if(pr * pr * layeri != nprocs){
                            if(myrank==0)std::cerr<<"3dlayerconfig has problem " << pr << "x"<<pr<<"x"<<layeri << "!=" << nprocs<<std::endl;
                        }
                        SpParMat3D<int, int, SpDCCols < int, int >> A3D(A2dint, layeri, true, false);
                        SpParMat3D<int, double, SpDCCols < int, double >> B3D(w, layeri, false, false);
                        timer.start("tally-spgemm3d");
                        SpParMat3D<int, double, SpDCCols < int, double >> C3D = Mult_AnXBn_SUMMA3D<PTINTDOUBLE, double, SpDCCols<int, double>>(A3D,B3D);
                        timer.stop("tally-spgemm3d",false,false);
                        double ttmp =timer.records["tally-spgemm3d"];
                        tmp3dtime = tmp3dtime < ttmp ? tmp3dtime : ttmp; 
                        product = C3D.Convert2D();
                        // if(opts.check) {
                        //     Dist<int,double>::MPI_DCCols C2D = C3D.Convert2D();
                        //     MatrixEqual(C2D, product,"tallyspgemm3D");
                        // }
                    }
                    
                    tallySpgemm3Dtime[i] += (tmp3dtime);
                    tallySpgemm3DtimePerRound.push_back(tmp3dtime);
                    }

                    productnnz = product.getnnz();
                    tmp_start = MPI_Wtime();
                    product = EWiseMult(product, *bfs[j-1], false);
                    product = EWiseMult(product, nsp, false);
                    bcu += product;
                    tmp_end = MPI_Wtime();
                    // tallytimeother.back()+=(tmp_end - tmp_start);
                    roundothertime[i] += tmp_end - tmp_start;

                    loop++;
                    if(myrank==0){
                        std::string tmp = "-----Tally Loop: " + std::to_string(loop);
                        std::cerr<<std::setw(18)<<tmp<<", "
                        <<"1D:"<<std::setw(7)<<tmp1dtime<<", "
                        <<"2D:"<<std::setw(7)<<tmp2dtime<<", "
                        <<"3D:"<<std::setw(7)<<tmp3dtime<<", cur product nnz:"<<productnnz<<std::endl;
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
                tmp_start = MPI_Wtime();
                for(int j=0; j < bfs.size(); ++j)
                {
                    delete bfs[j];
                }
                bc += FullyDistVec<int, double>(bcu.Reduce(Row, plus<double>(), 0.0));	// pack along rows
                tmp_end = MPI_Wtime();
                roundothertime[i] += tmp_end - tmp_start;
                if(myrank==0){
                    std::cerr<<"Round:"<<i<<", "
                    <<"BFS 1D:"<<std::setw(5)<<bfsSpgemm1Dtime[i]<<", "
                    <<"2D:"<<std::setw(5)<<bfsSpgemm2Dtime[i]<<", "
                    <<"3D:"<<std::setw(5)<<bfsSpgemm3Dtime[i]<<", "
                    <<"Tally 1D:"<<std::setw(5)<<tallySpgemm1Dtime[i]<<", "
                    <<"2D:"<<std::setw(5)<<tallySpgemm2Dtime[i]<<", "
                    <<"3D:"<<std::setw(5)<<tallySpgemm3Dtime[i]<<", "
                    <<"Others:"<<std::setw(5)<<roundothertime[i]<<std::endl<<std::endl;
                }
                if(i==10) break; 
            }
            
            double roundother = 0.0;
            for(auto x : roundothertime) roundother +=x;
            perfcntBC.doublemap["otherT(s)"] = roundother;
            perfcntBC.integermap["Rounds"] = numBatches;
            int bfsloops = 0;
            for(auto x : bfsLoopPerRound) bfsloops+=x;
            perfcntBC.integermap["BFSLoops"] = bfsloops;
            double tmpbfstime = 0.0;
            for(auto x : bfsSpgemm1Dtime) tmpbfstime += x;
            perfcntBC.doublemap["BFS1D(s)"] = tmpbfstime;
            tmpbfstime = 0.0;
            for(auto x : bfsSpgemm2Dtime) tmpbfstime += x;
            perfcntBC.doublemap["BFS2D(s)"] = tmpbfstime;
            tmpbfstime = 0.0;
            for(auto x : bfsSpgemm3Dtime) tmpbfstime += x;
            perfcntBC.doublemap["BFS3D(s)"] = tmpbfstime;
            tmpbfstime = 0.0;
            for(auto x : tallySpgemm1Dtime) tmpbfstime += x;
            perfcntBC.doublemap["Tally1D(s)"] = tmpbfstime;
            tmpbfstime = 0.0;
            for(auto x : tallySpgemm2Dtime) tmpbfstime += x;
            perfcntBC.doublemap["Tally2D(s)"] = tmpbfstime;
            tmpbfstime = 0.0;
            for(auto x : tallySpgemm3Dtime) tmpbfstime += x;
            perfcntBC.doublemap["Tally3D(s)"] = tmpbfstime;
            
            perfcntBC.OutputRecords(file2d);
            bc.Apply(std::bind(minus<double>(),std::placeholders::_1,nPasses)); // Subtrack nPasses from all the bc scores (because bcu was initialized to all 1's)
            double total_end=MPI_Wtime();
            double TEPS = (nPasses * static_cast<float>(A2dint.getnnz())) / (total_end-total_start);
            std::ofstream output(outfilename);
            bc.SaveGathered(output, 0);
            output.close();
        }
    }
}
    MPI_Finalize();
    return 0;
}


