// Spgemm 1d Matrix class test correctness cpp 

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
#include <parallel/numeric>
#include <parallel/algorithm>
#include <fstream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>
#include <sstream>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid.h"
#include "CombBLAS/CommGrid1D.h"
#include "CombBLAS/MPIType.h"
#include "CombBLAS/Operations.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/SPA.h"
#include "CombBLAS/Semirings.h"
#include "CombBLAS/SpCCols.h"
#include "CombBLAS/SpDCCols.h"
#include "CombBLAS/SpDefs.h"
#include "CombBLAS/SpHelper.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/SpParMat.h"
#include "CombBLAS/SpParMat1D.h"
#include "CombBLAS/SpTuples.h"
#include "CombBLAS/csc.h"
#include "CombBLAS/dcsc.h"
#include "CombBLAS/mtSpGEMM.h"
#include "Tommy/tommytypes.h"

using namespace combblas;

typedef int64_t IT;
typedef double NT;
typedef SpDCCols<IT, NT> DER; 
typedef SpParMat1D<IT, NT, DER> Sp1D;
typedef SpParMat<IT, NT, DER> Sp2D;
typedef PlusTimesSRing<double,double> PTFF;

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


int main(int argc, char** argv){
    int nprocs, myrank;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    int argidx = 1;
    std::string testtype(argv[argidx++]);
    {
        std::shared_ptr<CommGrid> grid2d   = std::make_shared<CommGrid>(MPI_COMM_WORLD,0,0);
        std::shared_ptr<CommGrid1D> grid1d = std::make_shared<CommGrid1D>(MPI_COMM_WORLD);
        Sp2D A(grid2d);
        SpHelper::initdatasetmap();
        string prefix = SpHelper::GetHostPrefix();
        
        for(auto dataset : ckcorrectdlist){
            A.ParallelReadMM(prefix + dataset + ".mtx", true, maximum<double>());
            A.PrintInfo();
            if( A.IsSquareMatrix() ){
                Sp2D B(A);
                MPI_Timer timer;
                timer.start("spgemm2d");
                Sp2D C = Mult_AnXBn_Synch<PTFF, double, DER>(A,B);
                timer.stop("spgemm2d");
            }
        }
    }
        
    MPI_Finalize();
    return 0;
}