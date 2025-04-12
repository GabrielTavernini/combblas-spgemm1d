#include "CombBLAS/SpDefs.h"
#include "CombBLAS/SpParHelper.h"
#include <cstdint>
#include <mpi.h>
#include <string>
#include <vector>
#include <cstdint>
#include <limits.h>
using namespace std;
using namespace combblas;

int main(int argc, char ** argv){
    MPI_Init(NULL, NULL);
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    double *windowptr;
    int64_t totaldatainmb = stol(string(argv[1]));
    printf("myrank %d totalsize %lu \n", myrank, totaldatainmb);
    double perrankdata = (double)totaldatainmb * 1e6 / (double)nprocs;
    int64_t perranksize = perrankdata / sizeof(double);
    
    assert(perranksize * nprocs < INT_MAX );
    windowptr = new double[perranksize];
    double *localbuffer = new double[perranksize * nprocs];
    MPI_PassiveWindow<double> * wins = new MPI_PassiveWindow<double>(windowptr, perranksize);
    MPI_Timer timer;
    timer.start("fetchall");
    for(int i=0; i<nprocs-1; i++){
        int targetrank = (myrank + i + 1) % nprocs;
        wins->Lock(targetrank);
        wins->Get(localbuffer + targetrank * perranksize, 0, perranksize, targetrank);
        wins->Unlock(targetrank);
    }
    timer.stop("fetchall",true);

    // Bcast
    timer.start("allgather");
    PMPI_Allgather(windowptr, perranksize, MPI_DOUBLE, localbuffer, perranksize, MPI_DOUBLE, MPI_COMM_WORLD);
    timer.stop("allgather");
    delete wins;
    delete [] windowptr;
    delete [] localbuffer;

    MPI_Finalize();
}