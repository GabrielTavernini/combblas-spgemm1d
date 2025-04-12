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
    MPI_Init(NULL, NULL);
    int argidx = 1;
    string mtx1(argv[argidx++]);
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    typedef PlusTimesSRing<double, double> PTFF;
    auto maxfun = maximum<double>();
    {
        MPI_Comm comm;
        MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        shared_ptr<CommGrid> world = make_shared<CommGrid>(MPI_COMM_WORLD,0,0);
        Sp2D A(world);
        A.ParallelReadMM(mtx1, true, maxfun);
        A.PrintInfo();
    }
    MPI_Finalize();
    return 0;
}