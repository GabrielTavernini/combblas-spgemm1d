#include <CombBLAS/CombBLAS.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <random>
#include <stdio.h>
#include <string>

typedef int IT;
typedef double  NT;
using namespace std;
using namespace combblas;


// template<class IT, class NT>
// void FetchANeeded(SpDCCols<IT, NT> * A, SpDCCols<IT, NT> * B)
// {

// }

int main(int argc, char ** argv){
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    IT mylocalnnz = rand();
    vector<IT> datasize;
    for(int i=20; i<25; i++){
    // for(int i=2; i<5; i++){
        datasize.push_back(1 << i); // m
    }
    // everyone fetch from 1 processes
    // compare with MPI_Bcast 
    for(int i=0; i<datasize.size(); i++){
        // double *data = new double[datasize[i]];
        double *data = NULL, *recvdata = NULL;
        if(myrank == 0){
            data = new double[datasize[i] * nprocs];
            recvdata = new double[datasize[i]];
            for(int j=0; j<datasize[i]*nprocs; j++) data[j] =  (double)j / datasize[i];
        }else{
            recvdata = new double[datasize[i]];
        }
        MPI_Timer timer;
        // timer.start("bcast");
        // MPI_Bcast(data, datasize[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // timer.stop("bcast");

        timer.start("scatter");
        MPI_Scatter(data, datasize[i], MPI_DOUBLE,recvdata, datasize[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        timer.stop("scatter");

        timer.start("oneside");
        int winsize = myrank == 0 ? datasize[i]*nprocs : 0;
        MPI_PassiveWindow<NT> *sendwins = new MPI_PassiveWindow<NT>(data, winsize);
        // double *recvbuffer = new double[datasize[i]];
        sendwins->Lock(0);
        sendwins->Get(recvdata, datasize[i]*myrank, datasize[i], 0);
        sendwins->Unlock(0);
        timer.stop("oneside");
        string res;
        for(int j=0; j<nprocs; j++) res += to_string(recvdata[j]) + " ";
        std::cerr << "myrank: "<< myrank << " res " << res << std::endl;
        delete sendwins;

        delete [] recvdata;
        if(myrank==0)delete [] data;
    }


    // for(int i=0; i<datasize.size(); i++){
    //     int *sendcounts = new int[nprocs];
    //     int *senddisp = new int[nprocs];
    //     int *recvcounts = new int[nprocs];
    //     int *recvdisp = new int[nprocs];
    //     IT totsends = 0;
    //     IT totrecvs = 0;
    //     srand(myrank);
    //     for(int j=0; j<nprocs; j++) sendcounts[j] = datasize[i] + rand() % 10;
    //     for(int j=0; j<nprocs; j++) senddisp[j] = 0;
    //     for(int j=0; j<nprocs; j++) recvdisp[j] = 0;
    //     MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    //     // string sendcountstr;
    //     // for(int j=0; j<nprocs; j++) sendcountstr += to_string(sendcounts[j]) + " ";
    //     // std::cerr << "myrank:" << myrank << ",sendcnt: " << sendcountstr << std::endl;
    //     // MPI_Barrier(MPI_COMM_WORLD);
    //     // string recvcountstr;
    //     // for(int j=0; j<nprocs; j++) recvcountstr += to_string(recvcounts[j]) + "  ";
    //     // std::cerr << "myrank" << myrank << "recvcnt " << recvcountstr << std::endl;

    //     std::partial_sum(sendcounts, sendcounts + nprocs-1, senddisp+1);
    //     std::partial_sum(recvcounts, recvcounts + nprocs-1, recvdisp+1);
    //     totsends = std::accumulate(sendcounts,sendcounts+nprocs,(IT)0);
    //     totrecvs = std::accumulate(recvcounts,recvcounts+nprocs,(IT)0);
    //     NT *sendarr = new NT[totsends];
    //     NT *recvarr = new NT[totrecvs];
    //     // if(myrank == 0) printf("total send size %.4f \n", totsends * 8 * 1e-6);
    //     #pragma omp simd
    //     for(size_t j=0; j<totsends; j++) sendarr[j] = (double)rand() / RAND_MAX;
        
    //     // MPI_Timer timer;
    //     // timer.start("alltoallv");
    //     // MPI_Alltoallv(sendarr, sendcounts, senddisp, MPI_DOUBLE, recvarr, recvcounts, recvdisp, MPI_DOUBLE, MPI_COMM_WORLD);
    //     // timer.stop("alltoallv");


    //     MPI_PassiveWindow<NT> *sendwins = new MPI_PassiveWindow<NT>(sendarr, totsends);
    //     // IT * tmpGsendcnt = new IT[nprocs*nprocs];
    //     IT * Gsenddisp = new IT[nprocs*nprocs];
    //     for(int j=0; j<nprocs*nprocs; j++) Gsenddisp[j] = 0;
    //     MPI_Allgather(senddisp, nprocs, MPI_INT, Gsenddisp, nprocs, MPI_INT, MPI_COMM_WORLD);
        

    //     // string rank0gsent = "Rank0 got allgather vector \n";
    //     // for(int j=0; j<nprocs; j++) {
    //     //     for(int k=0; k<nprocs; k++){
    //     //         rank0gsent += to_string(tmpGsendcnt[j*nprocs+k]) + " ";
    //     //     }
    //     //     rank0gsent += "\n";
    //     // }
    //     // if(myrank == 0) std::cerr << rank0gsent << std::endl;
    //     int targetrank = ( myrank + 1 ) % nprocs;
    //     int round = 0;
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     while (round < nprocs)
    //     {
    //         sendwins->Lock(targetrank);
    //         // printf("myrank %d tmpGsendcnt %d size %d \n", myrank, tmpGsendcnt[targetrank*nprocs+myrank], recvcounts[targetrank]);
    //         sendwins->Get(recvarr + recvdisp[targetrank], Gsenddisp[targetrank*nprocs+myrank], recvcounts[targetrank], targetrank);
    //         sendwins->Unlock(targetrank);
    //         targetrank = ( targetrank + 1 ) % nprocs;
    //         round++;
    //         break;
    //     }
        
    //     delete [] sendcounts;
    //     delete [] senddisp;
    //     delete [] recvcounts;
    //     delete [] recvdisp;
    //     delete [] sendarr;
    //     delete [] recvarr;
    //     delete sendwins;
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     // break;
    // }
    
    MPI_Finalize();
    return 0;
}