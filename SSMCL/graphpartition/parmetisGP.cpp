/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mpi.h>
#include <new>
#include <numeric>
#include <omp.h>
#include <regex>
#include <stdint.h>
#include <streambuf>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <tuple>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>
#include <sstream>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid1D.h"
#include "CombBLAS/MPIType.h"
#include "CombBLAS/Operations.h"
#include "CombBLAS/ParFriends.h"
#include "CombBLAS/SPA.h"
#include "CombBLAS/SpCCols.h"
#include "CombBLAS/SpDCCols.h"
#include "CombBLAS/SpDefs.h"
#include "CombBLAS/SpHelper.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/SpParMat.h"
#include "CombBLAS/SpParMat1D.h"
#include "CombBLAS/SpTuples.h"
#include "CombBLAS/csc.h"
#include "CombBLAS/GraphPartitioner.h"
#include "CombBLAS/dcsc.h"
#include "CombBLAS/mtSpGEMM.h"
#include "Tommy/tommytypes.h"
#include "CombBLAS/GraphPartitioner.h"
#include <fstream>
#include <parallel/numeric>
#include <parallel/algorithm>
extern "C" {
    #include <parmetis.h>
    #include <parmetislib.h>
    #include <metis.h>
    #include <GKlib.h>
    #include <defs.h>
}
#define MAXLINE			1280000
#include <mpi.h>

using namespace combblas;
using namespace std;

typedef int64_t IT;
typedef double NT;
typedef SpDCCols < int64_t, double > DER;
typedef SpParMat1D<IT,NT,DER> Sp1D;
typedef SpParMat<IT,NT,DER> Sp2D;
typedef PlusTimesSRing<double, double> PTFF;
#   pragma GCC diagnostic ignored "-Wwrite-strings"

/*************************************************************************
* This function writes out a partition vector
**************************************************************************/
void WritePVector(char *gname, idx_t *vtxdist, idx_t *part, MPI_Comm comm)
{
  idx_t i, j, k, l, rnvtxs, npes, mype, penum;
  FILE *fpin;
  idx_t *rpart;
  char partfile[256];
  MPI_Status status;

  gkMPI_Comm_size(comm, &npes);
  gkMPI_Comm_rank(comm, &mype);

  if (mype == 0) {
    sprintf(partfile, "%s.part", gname);
    if ((fpin = fopen(partfile, "w")) == NULL) 
      errexit("Failed to open file %s", partfile);

    for (i=0; i<vtxdist[1]; i++)
      fprintf(fpin, "%ld\n", part[i]);

    for (penum=1; penum<npes; penum++) {
      rnvtxs = vtxdist[penum+1]-vtxdist[penum];
      rpart = imalloc(rnvtxs, "rpart");
      MPI_Recv((void *)rpart, rnvtxs, IDX_T, penum, 1, comm, &status);

      for (i=0; i<rnvtxs; i++)
        fprintf(fpin, "%ld\n", rpart[i]);

      gk_free((void **)&rpart, LTERM);
    }
    fclose(fpin);
  }
  else
    MPI_Send((void *)part, vtxdist[mype+1]-vtxdist[mype], IDX_T, 0, 1, comm); 

}

/*************************************************************************
* This function reads the CSR matrix
**************************************************************************/
void ParallelReadGraph(graph_t *graph, char *filename, MPI_Comm comm)
{
  idx_t i, k, l, pe;
  idx_t npes, mype, ier;
  idx_t gnvtxs, nvtxs, your_nvtxs, your_nedges, gnedges;
  idx_t maxnvtxs = -1, maxnedges = -1;
  idx_t readew = -1, readvw = -1, dummy, edge;
  idx_t *vtxdist, *xadj, *adjncy, *vwgt, *adjwgt;
  idx_t *your_xadj, *your_adjncy, *your_vwgt, *your_adjwgt, graphinfo[4];
  idx_t fmt, ncon, nobj;
  MPI_Status stat;
  char *line = NULL, *oldstr, *newstr;
  FILE *fpin = NULL;

  gkMPI_Comm_size(comm, &npes);
  gkMPI_Comm_rank(comm, &mype);

  vtxdist = graph->vtxdist = ismalloc(npes+1, 0, "ReadGraph: vtxdist");

  if (mype == npes-1) {
    ier = 0;
    fpin = fopen(filename, "r");

    if (fpin == NULL) {
      printf("COULD NOT OPEN FILE '%s' FOR SOME REASON!\n", filename);
      ier++;
    }

    gkMPI_Bcast(&ier, 1, IDX_T, npes-1, comm);
    if (ier > 0){
      MPI_Finalize();
      exit(0);
    }

    line = gk_cmalloc(MAXLINE+1, "line");

    while (fgets(line, MAXLINE, fpin) && line[0] == '%');

    fmt = ncon = nobj = 0;
    sscanf(line, 
    "%ld %ld %ld %ld %ld", 
        &gnvtxs, &gnedges, &fmt, &ncon, &nobj);
    gnedges *=2;
    readew = (fmt%10 > 0);
    readvw = ((fmt/10)%10 > 0);
    graph->ncon = ncon = (ncon == 0 ? 1 : ncon);
    graph->nobj = nobj = (nobj == 0 ? 1 : nobj);

    /*printf("Nvtxs: %"PRIDX", Nedges: %"PRIDX", Ncon: %"PRIDX"\n", gnvtxs, gnedges, ncon); */

    graphinfo[0] = ncon;
    graphinfo[1] = nobj;
    graphinfo[2] = readvw;
    graphinfo[3] = readew;
    gkMPI_Bcast((void *)graphinfo, 4, IDX_T, npes-1, comm);

    /* Construct vtxdist and send it to all the processors */
    vtxdist[0] = 0;
    for (i=0,k=gnvtxs; i<npes; i++) {
      l = k/(npes-i);
      vtxdist[i+1] = vtxdist[i]+l;
      k -= l;
    }

    gkMPI_Bcast((void *)vtxdist, npes+1, IDX_T, npes-1, comm);
  }
  else {
    gkMPI_Bcast(&ier, 1, IDX_T, npes-1, comm);
    if (ier > 0){
      MPI_Finalize();
      exit(0);
    }

    gkMPI_Bcast((void *)graphinfo, 4, IDX_T, npes-1, comm);
    graph->ncon = ncon = graphinfo[0];
    graph->nobj = nobj = graphinfo[1];
    readvw = graphinfo[2];
    readew = graphinfo[3];

    gkMPI_Bcast((void *)vtxdist, npes+1, IDX_T, npes-1, comm);
  }

  if ((ncon > 1 && !readvw) || (nobj > 1 && !readew)) {
    printf("fmt and ncon/nobj are inconsistant.  Exiting...\n");
    gkMPI_Finalize();
    exit(-1);
  }


  graph->gnvtxs = vtxdist[npes];
  nvtxs = graph->nvtxs = vtxdist[mype+1]-vtxdist[mype];
  xadj  = graph->xadj  = imalloc(graph->nvtxs+1, "ParallelReadGraph: xadj");
  vwgt  = graph->vwgt  = imalloc(graph->nvtxs*ncon, "ParallelReadGraph: vwgt");

  /*******************************************/
  /* Go through first time and generate xadj */
  /*******************************************/
  if (mype == npes-1) {
    maxnvtxs = vtxdist[1];
    for (i=1; i<npes; i++) 
      maxnvtxs = (maxnvtxs < vtxdist[i+1]-vtxdist[i] ? vtxdist[i+1]-vtxdist[i] : maxnvtxs);

    your_xadj = imalloc(maxnvtxs+1, "your_xadj");
    your_vwgt = ismalloc(maxnvtxs*ncon, 1, "your_vwgt");

    maxnedges = 0;
    for (pe=0; pe<npes; pe++) {
      your_nvtxs = vtxdist[pe+1]-vtxdist[pe];

      for (i=0; i<your_nvtxs; i++) {
        your_nedges = 0;

        while (fgets(line, MAXLINE, fpin) && line[0] == '%'); /* skip lines with '#' */
        oldstr = line;
        newstr = NULL;

        if (readvw) {
          for (l=0; l<ncon; l++) {
            your_vwgt[i*ncon+l] = strtoidx(oldstr, &newstr, 10);
            oldstr = newstr;
          }
        }

        for (;;) {
          edge = strtoidx(oldstr, &newstr, 10) -1;
          oldstr = newstr;

          if (edge < 0)
            break;

          if (readew) {
            for (l=0; l<nobj; l++) {
              dummy  = strtoidx(oldstr, &newstr, 10);
              oldstr = newstr;
            }
          }
          your_nedges++;
        }
        your_xadj[i] = your_nedges;
      }

      MAKECSR(i, your_nvtxs, your_xadj);
      maxnedges = (maxnedges < your_xadj[your_nvtxs] ? your_xadj[your_nvtxs] : maxnedges);

      if (pe < npes-1) {
        gkMPI_Send((void *)your_xadj, your_nvtxs+1, IDX_T, pe, 0, comm);
        gkMPI_Send((void *)your_vwgt, your_nvtxs*ncon, IDX_T, pe, 1, comm);
      }
      else {
        for (i=0; i<your_nvtxs+1; i++)
          xadj[i] = your_xadj[i];
        for (i=0; i<your_nvtxs*ncon; i++)
          vwgt[i] = your_vwgt[i];
      }
    }
    fclose(fpin);
    gk_free((void **)&your_xadj, &your_vwgt, LTERM);
  }
  else {
    gkMPI_Recv((void *)xadj, nvtxs+1, IDX_T, npes-1, 0, comm, &stat);
    gkMPI_Recv((void *)vwgt, nvtxs*ncon, IDX_T, npes-1, 1, comm, &stat);
  }

  graph->nedges = xadj[nvtxs];
  adjncy = graph->adjncy = imalloc(xadj[nvtxs], "ParallelReadGraph: adjncy");
  adjwgt = graph->adjwgt = imalloc(xadj[nvtxs]*nobj, "ParallelReadGraph: adjwgt");

  /***********************************************/
  /* Now go through again and record adjncy data */
  /***********************************************/
  if (mype == npes-1) {
    ier = 0;
    fpin = fopen(filename, "r");

    if (fpin == NULL){
      printf("COULD NOT OPEN FILE '%s' FOR SOME REASON!\n", filename);
      ier++;
    }

    gkMPI_Bcast(&ier, 1, IDX_T, npes-1, comm);
    if (ier > 0){
      gkMPI_Finalize();
      exit(0);
    }

    /* get first line again */
    while (fgets(line, MAXLINE, fpin) && line[0] == '%');

    your_adjncy = imalloc(maxnedges, "your_adjncy");
    your_adjwgt = ismalloc(maxnedges*nobj, 1, "your_adjwgt");

    for (pe=0; pe<npes; pe++) {
      your_nvtxs  = vtxdist[pe+1]-vtxdist[pe];
      your_nedges = 0;

      for (i=0; i<your_nvtxs; i++) {
        while (fgets(line, MAXLINE, fpin) && line[0] == '%');
        oldstr = line;
        newstr = NULL;

        if (readvw) {
          for (l=0; l<ncon; l++) {
            dummy  = strtoidx(oldstr, &newstr, 10);
            oldstr = newstr;
          }
        }

        for (;;) {
          edge   = strtoidx(oldstr, &newstr, 10) -1;
          oldstr = newstr;

          if (edge < 0)
            break;

          your_adjncy[your_nedges] = edge;
          if (readew) {
            for (l=0; l<nobj; l++) {
              your_adjwgt[your_nedges*nobj+l] = strtoidx(oldstr, &newstr, 10);
              oldstr = newstr;
            }
          }
          your_nedges++;
        }
      }
      if (pe < npes-1) {
        gkMPI_Send((void *)your_adjncy, your_nedges, IDX_T, pe, 0, comm);
        gkMPI_Send((void *)your_adjwgt, your_nedges*nobj, IDX_T, pe, 1, comm);
      }
      else {
        for (i=0; i<your_nedges; i++)
          adjncy[i] = your_adjncy[i];
        for (i=0; i<your_nedges*nobj; i++)
          adjwgt[i] = your_adjwgt[i];
      }
    }
    fclose(fpin);
    gk_free((void **)&your_adjncy, &your_adjwgt, &line, LTERM);
  }
  else {
    gkMPI_Bcast(&ier, 1, IDX_T, npes-1, comm);
    if (ier > 0){
      gkMPI_Finalize();
      exit(0);
    }

    gkMPI_Recv((void *)adjncy, xadj[nvtxs], IDX_T, npes-1, 0, comm, &stat);
    gkMPI_Recv((void *)adjwgt, xadj[nvtxs]*nobj, IDX_T, npes-1, 1, comm, &stat);
  }

}


//!!!!!!! IT type must be LONG!!!!!!!!!!!!
template<typename T>
void writeVectorToFile(const std::vector<T>& vec, const std::string& filename) {
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    string localstr = "";
    // for(size_t i=0; i<vec.size(); i++) localstr += "rank:" +to_string(myrank) +"," + to_string(i) + ":" + to_string(vec[i]) + "\n";
    for(size_t i=0; i<vec.size(); i++) localstr += to_string(vec[i]) + "\n";
    // for(int i=0; i<3; i++)localstr += "rank:" + to_string(myrank) + "\n";
    // exchange the file size and decide the prefix
    IT localcharsize = localstr.size();
    vector<IT> gcharsize(nprocs,0);
    MPI_Allgather(
        &localcharsize, 1, MPI_LONG_LONG, 
        gcharsize.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
    vector<IT> prefixsum(nprocs,0);
    partial_sum(gcharsize.begin(),gcharsize.end()-1,prefixsum.begin()+1);
    SpHelper::RemoveExistingFile(filename);
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    MPI_Offset offset = prefixsum[myrank];
    MPI_File_write_at(file, offset, localstr.c_str(), localstr.size(), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&file);
}


int main(int argc, char ** argv){
    int nprocs, myrank;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    int argindex = 1;
    string mtxname = string(argv[argindex++]);
    {
        // string graphname (mtxname + ".graph");
        // auto maxop = maximum<double>();
        // shared_ptr<CommGrid1D> grid1d = make_shared<CommGrid1D>(MPI_COMM_WORLD);
        // shared_ptr<CommGrid> grid2d = make_shared<CommGrid>(MPI_COMM_WORLD,0,0);
        // MPI_Comm comm;
        // MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        // Sp1D A1D(grid1d);
        // A1D.ParallelReadMM(mtxname);
        // Sp2D A2D(grid2d);
        // A2D.ParallelReadMM(mtxname, true, maximum<double>());
        // Sp2D AT2D(A2D);
        // AT2D.Transpose();
        // printf("AT2D transpose size %ld %ld \n", AT2D.getnrow(),AT2D.getncol());
        // Sp1D A1D(A2D);
        // A1D.Transpose();
        // Sp2D A1DT(A1D,A1D.getblocksizevec());
        // if(AT2D == A1DT){
        //     printf("Transpose is correct!\n");
        // }else{
        //     printf("Transpose is not correct!\n");
        // }
        // Sp2D pyAT(fullWorld);
        // pyAT.ParallelReadMM("AT.mtx", true, maxop);
        // if(pyAT == AT2D){
        //     printf("At2D is correct!\n");
        // }
        
        // if(AT2D == A2D){
        //     if(myrank==0)printf("matrix is symm!!!\n");
        // }else{
        //     A2D += AT2D;
        //     if(myrank==0)printf("matrix is not symm before, do symm!!!\n");
        // }
        // Sp1D A1D(A2D);
        // MPI_Comm comm;
        // MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        // /*write to file and read it using metis api*/ 
        // A1D.ParallelWriteMetisGraph(mtxname + ".graph","none");
        // graph_t metisgraph;
        // ParallelReadGraph(&metisgraph, (char*)graphname.c_str(), comm);
        // IT nvtxs = metisgraph.nvtxs;
        // IT ncon = metisgraph.ncon;
        // IT localnvtxs = metisgraph.vtxdist[myrank+1] - metisgraph.vtxdist[myrank];

        // ////////////////////////////////////////////////////////////////////////////////////////////////
        // // In METIS, 
        // // The CSR format is a widely-used scheme for storing sparse graphs. Here, the adjacency
        // // structure of a graph is represented by two arrays, xadj and adjncy. 
        // // Weights on the vertices and edges (if any) are represented by using two additional arrays, vwgt and adjwgt.
        // // consider a graph with n vertices and m edges, 
        // // In the CSR format, this graph can be described using arrays of the following sizes:
        // // xadj[n + 1], vwgt[n], adjncy[2m], and adjwgt[2m]
        // // 
        // // !!! converting to Csc in CombBLAS, n is csc->n, 2m is csc->nz,  xadj is just jc (on each processor) 
        // // METIS only accepts undirected graph and it will store both (u,v) and (v,u).
        // // so adjncy size is 2m. It's also the same in CombBLAS. 
        // // adjncy is just ir ptr.
        // // vwgt and adjwgt can be simple 1. but pay attend to the size of array.
        // ////////////////////////////////////////////////////////////////////////////////////////////////
        
        // SpDCCols<IT, NT> *seqptr = A1D.seqptr();
        // vector<int64_t> recv(nprocs,0);
        // MPI_Allgather(
        //     &localnvtxs, 1, MPI_LONG_LONG, 
        //     recv.data(), 1, MPI_LONG_LONG, comm );
        // for(int i=0; i<nprocs; i++) nvtxs += recv[i];
        // vector<int64_t> vtxdist(nprocs+1,0);
        // std::partial_sum(recv.begin(), recv.end(), vtxdist.begin()+1);
        // if(myrank ==0){
        //     std::cerr << "vtsdist :";
        //     for(int i=0; i<nprocs+1; i++){
        //         std::cerr << vtxdist[i] << ", ";
        //     }
        //     std::cerr << std::endl;

        //     std::cerr << "metis vtsdist :";
        //     for(int i=0; i<nprocs+1; i++){
        //         std::cerr << metisgraph.vtxdist[i] << ", ";
        //     }
        //     std::cerr << std::endl;
        // }



        // std::vector< std::tuple<IT,IT, NT> > Tuples;
        // IT localedges = 0;
        // for(typename DER::SpColIter colit = seqptr->begcol(); colit != seqptr->endcol(); ++colit)
        // {
        //     IT gcol = colit.colid() + A1D.getblocksizeprefix()[myrank];
        //     for(typename DER::SpColIter::NzIter nzit = seqptr->begnz(colit); nzit != seqptr->endnz(colit); ++nzit)
        //     {
        //         IT grow = nzit.rowid();
        //         NT val = nzit.value();
        //         if(grow != gcol){
        //             Tuples.push_back(make_tuple(grow,colit.colid(),val));
        //             localedges++;
        //         }
        //     }
        // }
        // IT nrows = A1D.getnrow();
        // IT localcol = seqptr->getncol();
        // SpTuples<IT, NT>sptuples(Tuples.size(), nrows, localcol, Tuples.data(), true);
        // sptuples.tuples_deleted = true;
        // SpCCols<IT, NT> spccols(sptuples,false);
        // Csc<IT, NT> *cscptr = spccols.GetCSC();
        
        // if(myrank==0) std::cerr << "check xadj" << std::endl;
        // MPI_Barrier(MPI_COMM_WORLD);
        // bool correct = true;
        // string metisstr = "";
        // printf("myrank %d my local col %ld metis nvtx %ld gnvtx %ld \n", 
        // myrank, seqptr->getncol(),
        // metisgraph.nvtxs, metisgraph.gnvtxs);
        
        // for(int i=0; i<cscptr->n+1; i++){
        //     if(metisgraph.xadj[i] != cscptr->jc[i]){
        //         std::cerr << "myrank " << myrank << ", xadj err " << i << ", " << 
        //         metisgraph.xadj[i] << ", " << cscptr->jc[i] << std::endl;
        //         correct = false;
        //     }
        //     if(!correct)break;
        // }
        // if(correct) printf("jc is correct!\n");


        // for(int i=0; i<cscptr->nz; i++){
        //     if(metisgraph.adjncy[i] != cscptr->ir[i]){
        //         std::cerr << "myrank " << myrank << ", adjncy err " << i << ", " << 
        //         metisgraph.adjncy[i] << ", " << cscptr->ir[i] << std::endl;
        //         correct = false;
        //     }
        //     if(!correct)break;
        // }
        // if(correct) printf("ir is correct!\n");
        // IT nparts = 10;
        // IT wgtflag = 3;
        // IT numflag = 0;
        // IT edgecut;
        // vector<double> tpwgts(ncon * nparts, 1. / (double)nparts);
        // vector<double> ubvec(ncon,1.05);
        // vector<IT> options(3);
        // options[0] = 1;
        // options[PMV3_OPTION_DBGLVL] = 1;
        // options[PMV3_OPTION_SEED] = 1024;
        // vector<IT> results(localnvtxs,0);
        // MPI_Barrier(MPI_COMM_WORLD);

        // wgtflag = 0;
        // int ret = ParMETIS_V3_PartKway(
        //     vtxdist.data(), 
        //     cscptr->jc,
        //     cscptr->ir,
        //     NULL,
        //     NULL,
        //     &wgtflag,
        //     &numflag,
        //     &ncon,
        //     &nparts,
        //     tpwgts.data(),
        //     ubvec.data(),
        //     options.data(),
        //     &edgecut,
        //     results.data(),
        //     &comm
        // );

        // int ret = ParMETIS_V3_PartKway(
        //     metisgraph.vtxdist, 
        //     metisgraph.xadj,
        //     metisgraph.adjncy,
        //     metisgraph.vwgt,
        //     metisgraph.adjwgt,
        //     &wgtflag,
        //     &numflag,
        //     &ncon,
        //     &nparts,
        //     tpwgts.data(),
        //     ubvec.data(),
        //     options.data(),
        //     &edgecut,
        //     results.data(),
        //     &comm
        // );

    }

    MPI_Finalize();
    return 0;
}