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
#include "CombBLAS/FullyDistVec.h"
#include "CombBLAS/Operations.h"
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
    string datapath(argv[1]);
    string datasetname(argv[2]);
    string mtxfilename(argv[3]);
    string respath(argv[4]);
    string graphpartitioner(argv[5]);
    string gpoutput(argv[6]);
    int nparts = stoi(argv[7]);
    string fullmtx = datapath + "/" + mtxfilename;
    string gppath = respath + "/" + graphpartitioner + "/" + gpoutput;
    SpHelper::CheckFileExists(fullmtx,true);
    SpHelper::CheckFileExists(gppath,true);
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if(myrank == 0){
        printf("=================gp2combblas analysis!=================\n");
        printf("matrix market path: %s \n", fullmtx.c_str());
        printf("graph partition path: %s \n", gppath.c_str());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_IO_Reader ioreader(gppath);
    vector<int> lgroup;
    int lmaxgid(0), gmaxgid, lmingid(0), gmingid(0);
    IT gentries(0),lentries(ioreader.lines.size());
    for(auto x : ioreader.lines) lgroup.push_back(stoi(x));
    for(auto x : lgroup) {
        lmaxgid = max(lmaxgid, x);
        lmingid = min(lmingid, x);
    }
    MPI_Allreduce(&lmaxgid, &gmaxgid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&lmingid, &gmingid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&lentries, &gentries, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    bool correctinput(true);
    if(nparts != gmaxgid+1){
        if(myrank == 0) printf("worldsize %d doesn't equal group size %d!", nparts, gmaxgid+1);
        correctinput = false;
    }
    if(gmingid != 0){
        if(myrank == 0) printf("gmin gid is not 0!\n");
        correctinput = false;
    }
    if(correctinput)
    {
        Sp2D A;
        A.ParallelReadMM(fullmtx, true, maximum<double>());
        IT nrows = A.getnrow();
        IT ncols = A.getncol();
        IT nnz = A.getnnz();
        if( myrank == 0 ) printf("global entries %lu  ncols %lu \n",gentries, ncols);
        vector<IT> blocksizevec(nparts,0);
        int localsize = lgroup.size();
        vector<int> sizecnt(nprocs,0);
        vector<int> sizecntdisp(nprocs,0);
        vector<int> group(ncols,0);
        MPI_Allgather(&localsize, 1, MPI_INT, 
        sizecnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
        partial_sum(sizecnt.begin(),sizecnt.end()-1,sizecntdisp.begin()+1);
        MPI_Allgatherv(lgroup.data(), lgroup.size(), MPI_INT, 
        group.data(), sizecnt.data(), sizecntdisp.data(), MPI_INT, 
        MPI_COMM_WORLD);
        vector<vector<int>> groupidx(nparts, vector<int>());
        for(int i=0; i<group.size(); i++) {
            groupidx[group[i]].push_back(i);
            assert(i >= 0 && i < ncols);
        }
        for(int gi=0; gi<groupidx.size(); gi++){
            blocksizevec[gi] = groupidx[gi].size();
        }
        // if(myrank == 0){
        //     for(int gi=0; gi<groupidx.size(); gi++){
        //         printf("%lu ",blocksizevec[gi]);
        //     }
        //     printf("\n");
        //     fflush(stdout);
        // }
        vector<IT> distinputarr;
        if(myrank == 0){
            for(int gi=0; gi<groupidx.size(); gi++){
                for(auto x : groupidx[gi]) distinputarr.push_back(x);
            }
            string outblock = "";
            for(auto x : blocksizevec) outblock += to_string(x) + " ";
            auto ofile = ofstream(respath + "/" + graphpartitioner + "/" + gpoutput + ".bs");
            ofile << outblock;
            ofile.close();
        }
        FullyDistVec<IT, IT> permute(distinputarr, A.getcommgrid());
        IT losize = permute.GetLocVec().size();
        IT glsize;
        MPI_Allreduce(&losize, &glsize, 1, MPI_LONG_LONG, 
        MPI_SUM, MPI_COMM_WORLD);
        assert(glsize == ncols);
        permute.ParallelWrite(respath + "/" +  graphpartitioner + "/" + gpoutput + ".distvec", true,true,true);
        MPI_Timer timer;
        timer.start("permute");
        A(permute,permute,true); // shuffle
        timer.stop("permute");
        timer.start("blockana");
        A.BlockwiseNNZanalysis(blocksizevec, respath + "/" + graphpartitioner + "/" + gpoutput + ".nnzbin");
        timer.stop("blockana");
    }
    MPI_Finalize();
    return 0;
}