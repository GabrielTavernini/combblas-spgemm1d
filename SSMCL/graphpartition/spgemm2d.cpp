#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <string>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid.h"
#include "CombBLAS/FullyDistVec.h"
#include "CombBLAS/Operations.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/Semirings.h"
#include "CombBLAS/SpParHelper.h"

using namespace std;
using namespace combblas;
typedef int64_t IT;
typedef double NT;
typedef SpDCCols < int64_t, double > DER;
typedef SpParMat1D<IT,NT,DER> Sp1D;
typedef SpParMat<IT,NT,DER> Sp2D;
typedef vector<IT> VI;

SpgemmOpts combblas::opts;

int main(int argc, char** argv){
    MPI_Init(NULL, NULL);
    int argidx = 1;
    string mtxname(argv[argidx++]);
    string spgemmtype(argv[argidx++]);
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    typedef PlusTimesSRing<double, double> PTFF;
    {
        if(spgemmtype == "2D"){
            shared_ptr<CommGrid> world = make_shared<CommGrid>(MPI_COMM_WORLD,0,0);
            Sp2D A(world);
            A.ParallelReadMM(mtxname,true,maximum<double>());
            Sp2D B(A);
            MPI_Timer timer;
            timer.start("SpGEMM2D");
            Sp2D C = Mult_AnXBn_Synch<PTFF, double, DER>(A,B);
            timer.stop("SpGEMM2D");
        }
        if(spgemmtype == "1D"){
            shared_ptr<CommGrid1D> world = make_shared<CommGrid1D>(MPI_COMM_WORLD);
            Sp1D A(world);
            A.ParallelReadMM(mtxname);
            Sp1D B(A);
            MPI_Timer timer;
            timer.start("SpGEMM1D");
            Sp1D C = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTFF, double, DER>(A,B);
            timer.stop("SpGEMM2D");
        }
    }
    MPI_Finalize();
    return 0;
}