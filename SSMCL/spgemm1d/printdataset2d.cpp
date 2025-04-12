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

int main(int argc, char** argv){
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
    int argidx = 1;
    string mtxname = string(argv[argidx++]);
    {
        shared_ptr<CommGrid> world = make_shared<CommGrid>(MPI_COMM_WORLD,0,0);
        Sp2D A(world);
        A.ParallelReadMM(mtxname,true,maximum<double>());
        A.PrintInfo();
    }
    MPI_Finalize();
    return 0;
}