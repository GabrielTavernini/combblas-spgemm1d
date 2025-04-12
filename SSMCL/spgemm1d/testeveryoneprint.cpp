#include <cstddef>
#include <cstdio>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <numeric>


void everyoneprint(std::string s){
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    std::string printstr(s);
    int printstrlen = printstr.size();
    std::vector<int> printlencnt(nprocs,0);
    std::vector<int> printlendisp(nprocs,0);
    int totstrlen = 0;
    MPI_Reduce(&printstrlen , &totstrlen, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    char totstrchar[totstrlen];
    MPI_Allgather(&printstrlen, 1, MPI_INT, printlencnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::partial_sum(printlencnt.begin(), printlencnt.end()-1, printlendisp.begin()+1);
    MPI_Gatherv(
        printstr.data(), printstrlen, MPI_CHAR, 
        totstrchar, printlencnt.data(), printlendisp.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
    if(myrank==0){
        std::cerr << totstrchar;
    }
}

int main(){
    MPI_Init(NULL, NULL);
    int myrank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); 
    // std::string mystr;
    // mystr = "myrank" + std::to_string(myrank) + "\n";
    char tmpss[2000];
    sprintf(tmpss, "hellothisisrand %d \n", myrank);
    everyoneprint(std::string(tmpss));
    MPI_Finalize();
    return 0;
}