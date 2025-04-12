#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid1D.h"
#include "CombBLAS/Operations.h"
#include "CombBLAS/SpParMat1D.h"
#include <memory>
#include <mpi.h>

using namespace combblas;
int main(int argc, char** argv){
    int nprocs, myrank, numThreads;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    #pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
    if(myrank == 0) printf("Total ranks %d number of threads are %d \n", nprocs, numThreads);
    {
        typedef int64_t IT;
        typedef double NT;
        typedef SpDCCols < int64_t, double > DER;
        typedef SpParMat1D<IT,NT,DER> Sp1D;
        typedef SpParMat<IT,NT,DER> Sp2D;
        double vm_usage, resident_set;
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        Sp2D Readingmatrix(fullWorld);
        float lb;
        long nnz;
        string mmname(argv[1]);
        Readingmatrix.ParallelReadMM(mmname.c_str(),true,maximum<double>());
        shared_ptr<CommGrid1D> grid1d;
        grid1d.reset(new CommGrid1D(MPI_COMM_WORLD));
        Sp1D Rmatrix1D(grid1d);
        Rmatrix1D.ParallelReadMM(mmname.c_str(),"", true);
        Sp2D R1D2D(Rmatrix1D,Rmatrix1D.getblocksizevec());
        if(R1D2D == Readingmatrix){
            printf("reading correct!\n");
        }else{
            printf("reading wrong!\n");
        }
        Rmatrix1D.ParallelWriteMM("writemtx.mtx", true);
        Sp2D Readingmatrix2(fullWorld);
        Readingmatrix2.ParallelReadMM(mmname.c_str(),true, maximum<double>());
        if(Readingmatrix2 == Readingmatrix){
            printf("write correct!\n");
        }else{
            printf("write wrong!\n");
        }
    }
    MPI_Finalize();
    return 0;
}