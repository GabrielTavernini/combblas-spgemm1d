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
#include "CombBLAS/dcsc.h"

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
    string mtx2(argv[argidx++]);
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
        Sp2D B(world);
        B.ParallelReadMM(mtx2, true, maxfun);
        bool eq = (A==B);
        A.PrintInfo();
        B.PrintInfo();
        MPI_Barrier(comm);
        if(myrank == 0) 
        {
            std::cerr << "mtx1: " << mtx1 << std::endl << "mtx2: " << mtx2 << std::endl;
            if(eq){std::cerr << "they are equal!" << std::endl;}
            else{std::cerr << "they are not equal!" << std::endl;}
        }
        Dcsc<IT, NT> * ptr2d = A.seqptr()->GetDCSC();
        Dcsc<IT, NT> * ptr1d = B.seqptr()->GetDCSC();
        for(int i=0; i<ptr2d->nzc; i++){
            if(ptr2d->cp[i]!=ptr1d->cp[i]){
                std::cerr<<"wrong cp"<<i<<"," << ptr2d->cp[i] << ", "<< ptr1d->cp[i] <<std::endl;
                break;
            }
        }
    }
    MPI_Finalize();
    return 0;
}