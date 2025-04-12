// #include <cstdint>
// #include <memory>
// #include <mpi.h>
// #include <numeric>
// #include <string>
// #include <sys/time.h>
// #include <iostream>
// #include <functional>
// #include <algorithm>
// #include <vector>
// #include <sstream>
// #include "CombBLAS/CombBLAS.h"
// #include "CombBLAS/CommGrid1D.h"
// #include "CombBLAS/FullyDistVec.h"
// #include "CombBLAS/Operations.h"
// #include "CombBLAS/SpHelper.h"
// #include <iostream>
// #include <fstream>

// using namespace std;
// using namespace combblas;


// typedef SpDCCols<int64_t,double> DER;
// typedef int64_t IT;
// typedef double NT;
// typedef SpParMat<IT, NT, DER>  Sp2D;
// typedef SpParMat1D<IT, NT, DER>  Sp1D;


// int main(int argc, char* argv[])
// {
//     int nprocs, myrank;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
//     MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
//     SpHelper::initdatasetmap();
//     SpgemmOpts opts;
//     OptParser::parse(argc, argv, &opts);
//     for(auto mtxname : opts.dataset){
//         std::string fullpath = opts.fullfilepath[mtxname];
//         shared_ptr<CommGrid1D> world = make_shared<CommGrid1D>(MPI_COMM_WORLD);
//         Sp1D A1D(world);
//         MPI_Timer timer;
//         timer.start("parallelreading" + fullpath);
//         A1D.ParallelReadMM(fullpath);
//         timer.stop("parallelreading"+fullpath);
//         A1D.BlockwiseNNZanalysis(mtxname + "")
//     }
//     MPI_Finalize();
//     return 0;
// }

