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
    string executetype(argv[1]);
    if(executetype == "mcldataset"){
        string datapath(argv[2]); // mtx filepath prefix
        string datasetname(argv[3]);
        string mtxfilename(argv[4]);
        string respath(argv[5]);
        string graphpartitioner(argv[6]);
        // can be flops|columnnnz|none
        string vwchoice(argv[7]);
        string fullmtx = datapath + "/" + mtxfilename;
        // string outputname = respath + "/" + graphpartitioner + "/" + "gp_" + datasetname + "_" + vwchoice + ".graph";
        string outputname = "gp_" + datasetname + "_" + vwchoice + ".graph";
        int myrank, nprocs;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        Sp2D A;
        A.ParallelReadMM(fullmtx, true, maximum<double>());
        IT nrows = A.getnrow();
        IT ncols = A.getncol();
        IT nnz = A.getnnz();
        // Sp1D A1D(A);
        // IT blocksize = A1D.getblocksizevec()[myrank];
        IT blocksize = ( ncols + nprocs - 1 ) / nprocs;
        vector<IT> splitcols(nprocs,blocksize);
        splitcols[nprocs-1] = ncols - (nprocs-1)*blocksize;
        Sp1D A1D(A,splitcols);
        DER * spSeq = A1D.seqptr();
        /* Get Vertex Weight */
        vector<IT> VertexWeight(ncols,0);
        if(vwchoice == "flops"){
            vector<IT> Localcolnnz(ncols, 0);
            IT startidx = myrank * blocksize;
            IT endidx = (myrank + 1)*blocksize;
            endidx = min(endidx, ncols);
            
            if(endidx < ncols){
                printf("wrong!!!\n");
            }
            for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
            {
                IT gcol = colit.colid() + blocksize * myrank;
                for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
                {
                    IT grow = nzit.rowid();
                    NT val = nzit.value();
                    Localcolnnz[gcol]++;
                }
            }
            for(IT i=startidx; i<endidx; i++) Localcolnnz[i] = Localcolnnz[i]*Localcolnnz[i] + 1;
            MPI_Allreduce(Localcolnnz.data(), VertexWeight.data(), ncols, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        }else if(vwchoice == "columnnnz"){
            vector<IT> Localcolnnz(ncols, 0);
            IT startidx = myrank * blocksize;
            IT endidx = (myrank + 1)*blocksize;
            endidx = min(endidx, ncols);
            if(endidx < ncols){
                printf("wrong!!!\n");
            }
            for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
            {
                IT gcol = colit.colid() + blocksize * myrank;
                for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
                {
                    IT grow = nzit.rowid();
                    NT val = nzit.value();
                    Localcolnnz[gcol]++;
                }
            }
            for(IT i=startidx; i<endidx; i++) Localcolnnz[i] = Localcolnnz[i] + 1;
            MPI_Allreduce(Localcolnnz.data(), VertexWeight.data(), ncols, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        }
        /* Generate map */
        unordered_map<IT, string> map;
        for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
        {
            IT gcol = colit.colid() + blocksize * myrank;
            if(vwchoice != "none") map[gcol] = to_string(VertexWeight[gcol]) + " ";
            else map[gcol] = "";
            for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
            {
                IT grow = nzit.rowid();
                NT val = nzit.value();
                map[gcol] += to_string(grow+1) + " " + to_string(int(val*1.0e4) ) + " ";
            }
        }
        /* Generate localstr */
        string localstr = "";
        if(myrank == 0){
            if(vwchoice != "none") localstr += to_string(ncols) + " " + to_string(nnz/2) + " 011\n";
            else localstr += to_string(ncols) + " " + to_string(nnz/2) + " 001\n";
        }
        for(IT i=myrank * blocksize; i<(myrank+1)*(blocksize); i++){
            if(i>=ncols) break;
            if(map.find(i) != map.end()){
                localstr += map[i] + "\n";
            }else{
                localstr += " \n";
            }
        }
        /* Calculate offset */
        IT localcharsize = localstr.size();
        vector<IT> gcharsize(nprocs,0);
        MPI_Allgather(&localcharsize, 1, MPI_LONG_LONG, gcharsize.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
        vector<IT> prefixsum(nprocs,0);
        partial_sum(gcharsize.begin(),gcharsize.end()-1,prefixsum.begin()+1);
        /* Write to FS */
        MPI_File file;
        MPI_File_open(MPI_COMM_WORLD, outputname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        MPI_Offset offset = prefixsum[myrank];
        MPI_File_write_at(file, offset, localstr.c_str(), localstr.size(), MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&file);
    }
    if(executetype == "general"){
        string mtxfilename(argv[2]);
        // can be flops|columnnnz|none
        string vwchoice(argv[3]);
        string outputname = mtxfilename + ".graph";
        int myrank, nprocs;
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        Sp2D A;
        A.ParallelReadMM(mtxfilename, true, maximum<double>());
        IT nrows = A.getnrow();
        IT ncols = A.getncol();
        IT nnz = A.getnnz();
        Sp1D A1D(A);
        IT localedges = 0;
        IT edges = 0;
        // IT blocksize = A1D.getblocksizevec()[myrank];
        // IT blocksize = ( ncols + nprocs - 1 ) / nprocs;
        // vector<IT> splitcols(nprocs,blocksize);
        // splitcols[nprocs-1] = ncols - (nprocs-1)*blocksize;
        // Sp1D A1D(A,splitcols);
        DER * spSeq = A1D.seqptr();
        /* Get Vertex Weight */
        vector<IT> VertexWeight(ncols,0);
        vector<IT> Localcolnnz(ncols, 0);
        IT startidx = A1D.getblocksizeprefix()[myrank];
        IT endidx = myrank == nprocs-1 ? ncols : A1D.getblocksizeprefix()[myrank+1];
        for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
        {
            IT gcol = colit.colid() + startidx;
            for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
            {
                IT grow = nzit.rowid();
                NT val = nzit.value();
                if(grow != gcol) localedges++;
                Localcolnnz[gcol]++;
            }
        }
        for(IT i=startidx; i<endidx; i++){
            if(vwchoice == "columnnnz")  Localcolnnz[i] = Localcolnnz[i] + 1;
            else if(vwchoice == "flops") Localcolnnz[i] = Localcolnnz[i]*Localcolnnz[i] + 1;
        }
        MPI_Allreduce(Localcolnnz.data(), VertexWeight.data(), ncols, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&localedges, &edges, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        /* Generate map */
        unordered_map<IT, string> map;
        for(typename DER::SpColIter colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit)
        {
            IT gcol = colit.colid() + startidx;
            if(vwchoice != "none") map[gcol] = to_string(VertexWeight[gcol]) + " ";
            else map[gcol] = "";
            for(typename DER::SpColIter::NzIter nzit = spSeq->begnz(colit); nzit != spSeq->endnz(colit); ++nzit)
            {
                IT grow = nzit.rowid();
                NT val = nzit.value();
                map[gcol] += to_string(grow+1) + " " + to_string(int(val*1.0e4) ) + " ";
            }
        }
        /* Generate localstr */
        string localstr = "";
        if(myrank == 0){
            if(vwchoice != "none") localstr += to_string(ncols) + " " + to_string(edges/2) + " 011\n";
            else localstr += to_string(ncols) + " " + to_string(edges/2) + " 001\n";
        }
        IT linelocal = 0;
        IT gline = 0;
        for(IT i=startidx; i<endidx; i++){
            if(i>=ncols) break;
            if(map.find(i) != map.end()){
                localstr += map[i] + "\n";
            }else{
                localstr += " \n";
            }
            linelocal++;
        }
        MPI_Allreduce(&linelocal, &gline, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(myrank==0)printf("line size %ld \n",gline);
        /* Calculate offset */
        IT localcharsize = localstr.size();
        vector<IT> gcharsize(nprocs,0);
        MPI_Allgather(&localcharsize, 1, MPI_LONG_LONG, gcharsize.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
        vector<IT> prefixsum(nprocs,0);
        partial_sum(gcharsize.begin(),gcharsize.end()-1,prefixsum.begin()+1);
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