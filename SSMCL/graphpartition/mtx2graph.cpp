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
#include "CombBLAS/SpHelper.h"
#include "CombBLAS/SpParMat.h"

using namespace std;
using namespace combblas;
typedef int64_t IT;
typedef double NT;
typedef SpDCCols  <IT,NT    > DER;
typedef SpParMat1D<IT,NT,DER> Sp1D;
typedef SpParMat  <IT,NT,DER> Sp2D;
typedef vector<IT> VI;

int main(int argc, char** argv){
    MPI_Init(NULL, NULL);
    SpHelper::initdatasetmap();
    SpgemmOpts opts;
    OptParser::parse(argc, argv, &opts);
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    for(auto name : opts.dataset){
        string mtxfilename = opts.fullfilepath[name];
        // can be flops|columnnnz|none
        std::string vwgt;
        if(opts.permute == 2){
            vwgt = "none";
        }else if(opts.permute == 3){
            vwgt = "nnz";
        }else if(opts.permute == 4){
            vwgt = "flops";
        }
        string outputname = name + "_" + vwgt + ".graph"; // fixed
        shared_ptr<CommGrid1D> fullworld1d = make_shared<CommGrid1D>(MPI_COMM_WORLD);
        Sp1D A1D(fullworld1d);
        A1D.ParallelReadMM(mtxfilename);
        SpCCols<IT, NT> spcsc(*A1D.seqptr());
        Csc<IT, NT> *cscptr = spcsc.GetCSC();
        IT localedges = 0;
        IT edges = 0;
        string localstr = "";
        IT colprefix = A1D.getblocksizeprefix()[myrank];
        for(IT jlocal=0; jlocal < cscptr->n; jlocal++)
        {
            IT jglobal = jlocal + colprefix;
            IT rowstart = cscptr->jc[jlocal];
            IT rowend   = cscptr->jc[jlocal+1];
            if     (vwgt == "flops")    localstr += " " + to_string( (rowend - rowstart) * (rowend - rowstart) + 1);
            else if(vwgt == "nnz"  )    localstr += " " + to_string(rowend - rowstart + 1 );
            IT vtxedgecnt = 0;
            for(IT rowidx = rowstart; rowidx < rowend; rowidx++)
            {
                IT rowid = cscptr->ir[rowidx];
                if(rowid != jglobal){
                    localstr += " " + to_string(rowid+1);
                    // numx : potential use
                    NT numx = cscptr->num[rowid];
                    localedges++;
                    vtxedgecnt++;
                }
            }
            localstr += " \n";
        }
        IT nvtx = A1D.getncol();
        MPI_Allreduce(&localedges, &edges, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        edges /= 2;
        if(myrank == 0) {
            if(vwgt == "none") localstr = to_string(nvtx) + " " + to_string(edges) + " 000\n" + localstr;
            else localstr = to_string(nvtx) + " " + to_string(edges) + " 010\n" + localstr;
        }
        /* Calculate offset */
        IT localcharsize = localstr.size();
        vector<IT> gcharsize(nprocs,0);
        MPI_Allgather(
            &localcharsize, 1, MPI_LONG_LONG, 
            gcharsize.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
        vector<IT> prefixsum(nprocs,0);
        partial_sum(gcharsize.begin(),gcharsize.end()-1,prefixsum.begin()+1);
        if(myrank == 0){
            SpHelper::RemoveExistingFile(outputname);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        /* Write to FS */
        MPI_File file;
        MPI_File_open(MPI_COMM_WORLD, outputname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        MPI_Offset offset = prefixsum[myrank];
        MPI_File_write_at(file, offset, localstr.c_str(), localstr.size(), MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&file);
    }
    MPI_Finalize();
    return 0;
}