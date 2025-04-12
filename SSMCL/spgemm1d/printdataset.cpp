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

SpgemmOpts combblas::opts;

int main(int argc, char** argv){
    int nprocs, myrank, numThreads;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
#pragma omp parallel
    {
        #pragma omp critical
        numThreads = omp_get_num_threads();
    }
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    SpHelper::initdatasetmap();
    OptParser::parse(argc, argv, &opts);
    int argidx = 1;
    string mtxname = opts.dataset[0];
    string fullname = opts.fullfilepath[mtxname];
    if(myrank == 0) {
        std::cerr <<"mpi, omp: "<< nprocs  << "," << numThreads<<std::endl;
        std::cerr<<"reading "<< mtxname << ", fullpath: "<< fullname<<std::endl;
    }
    {
        shared_ptr<CommGrid1D> world = make_shared<CommGrid1D>(MPI_COMM_WORLD);
        shared_ptr<CommGrid> world2d = make_shared<CommGrid>(MPI_COMM_WORLD,0,0);
        Sp1D A(world);
        MPI_Timer timer;
        timer.start("readingdataset");
        A.ParallelReadMM(fullname);
        timer.stop("readingdataset");
        IT Anrows, Ancols;
        Anrows = A.getnrow();
        Ancols = A.getncol();
        if(opts.permute == 1){
            // random permute 
            FullyDistVec<int64_t,int64_t> perm(world2d);
            perm.iota(A.getnrow(), 0);
            A(perm,perm); // 1d perm
        }else if(opts.permute == 5){
            std::vector<int64_t> fullresults;
            if(opts.permfile == ""){
                opts.permfile = "/global/cfs/cdirs/m1982/yuxihong/spgemm1ddataset/"+ mtxname + "_flops.graph.part_parmetis." + std::to_string(nprocs);
            }
            std::ifstream inputFile(opts.permfile);
            if(!inputFile.is_open()){
                if(myrank==0)std::cerr<<"something is wrong with input partition file "<<std::endl;
                exit(0);
            }
            std::string line;
            while (std::getline(inputFile, line)) { // Read each line from the file
                IT number;
                std::istringstream iss(line);
                if (iss >> number) { // Check if the line contains an integer
                    fullresults.push_back(number);
                }
            }
            MPI_Barrier(comm);
            std::unordered_map<int, std::vector<int64_t>> idxmap;
            for(IT i=0; i<fullresults.size(); i++){
                idxmap[fullresults[i]].push_back(i);
            }
            std::vector<IT> permvec;
            IT mystart = A.getblocksizeprefix()[myrank];
            IT myend = myrank == nprocs-1 ? Ancols : A.getblocksizeprefix()[myrank+1];
            IT idx = 0;
            for(int i=0; i<opts.nparts; i++){
                for(auto x : idxmap[i]){
                    if(mystart <= idx && idx < myend){
                        permvec.push_back(x);
                    }
                    idx++;
                }
            }
            FullyDistVec<int64_t,int64_t> perm(permvec, world2d);
            std::vector<IT> blocksize(nprocs);
            IT mypartsize = idxmap[myrank].size();
            MPI_Allgather(
                &mypartsize, 1, MPI_LONG_LONG, 
                blocksize.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
            A(perm,perm,blocksize,blocksize);           // 1D perm
        }
        A.PrintInfo();
    }
    MPI_Finalize();
    return 0;
}