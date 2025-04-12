#include <cstddef>
#include <cstdint>
#include <cstdio>
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
#include "CombBLAS/SpParMat.h"

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
    {
        Sp2D A;
        A.ParallelReadMM(argv[1], true, maximum<double>());
        Sp2D AT(A);
        AT.Transpose();
        if( !(A == AT) ){
            printf("not symm, write!!\n");
            A += AT;
            string out(argv[1]);
            out = "Symm_" + out;
            A.ParallelWriteMM(out,true);
        }else{
            printf("graph is symm!!\n");
        }
        
    }
    MPI_Finalize();
    return 0;
}