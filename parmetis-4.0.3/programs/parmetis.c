/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * main.c
 *
 * This is the entry point of the ILUT
 *
 * Started 10/19/95
 * George
 *
 * $Id: parmetis.c,v 1.5 2003/07/30 21:18:54 karypis Exp $
 *
 */

#include <parmetisbin.h>
#include <omp.h>

/*************************************************************************
* Let the game begin
**************************************************************************/
int main(int argc, char *argv[])
{
  idx_t i, j, npes, mype, optype, nparts, adptf, options[10];
  idx_t *part=NULL, *sizes=NULL;
  graph_t graph;
  real_t ipc2redist, *xyz=NULL, *tpwgts=NULL, ubvec[MAXNCON];
  MPI_Comm comm;
  idx_t numflag=0, wgtflag=0, ndims, edgecut;
  char xyzfilename[8192];

  MPI_Init(&argc, &argv);
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  gkMPI_Comm_size(comm, &npes);
  gkMPI_Comm_rank(comm, &mype);
  int numThreads = 0;
#pragma omp parallel
    {
        #pragma omp critical
        numThreads = omp_get_num_threads();
    }
  if(mype==0){
    printf("Total MPI processes %ld number of threads are %d \n", npes ,numThreads);
    fflush(stdout);
  }

  optype     = atoi(argv[2]);
  int nnparts = 8; // 16,32,64,128,256,512,1024,4096,
  int npartsarray[8] = {16,32,64,128,256,512,1024,4096};
  nparts = -1;
  if (argc == 4) {
    npartsarray[0]     = atoi(argv[3]);
    nnparts = 1;
  }
  
//   adptf      = atoi(argv[4]);
//   ipc2redist = atof(argv[5]);
  adptf = 0;
  ipc2redist = 0;
  options[0] = 1;
  options[PMV3_OPTION_DBGLVL] = 0;
  options[PMV3_OPTION_SEED] = 0;
//   options[PMV3_OPTION_DBGLVL] = atoi(argv[6]);
//   options[PMV3_OPTION_SEED]   = atoi(argv[7]);

  if (mype == 0) {
    printf("reading file: %s\n", argv[1]);
    printf("nparts %ld \n", nparts);
    fflush(stdout);
  }
  double t1 = MPI_Wtime();
  ParallelReadGraph(&graph, argv[1], comm);
  double t2 = MPI_Wtime();

  if(mype == 0){
    printf("finish reading file %s, it takes %.3f time\n",argv[1],t2-t1);
    fflush(stdout);
  }

    for(int nni=0; nni < nnparts; nni++)
{
    nparts = npartsarray[nni];
    rset(graph.ncon, 1.05, ubvec);
    tpwgts = rmalloc(nparts*graph.ncon, "tpwgts");
    rset(nparts*graph.ncon, 1.0/(real_t)nparts, tpwgts);

    if (optype >= 20) { 
    sprintf(xyzfilename, "%s.xyz", argv[1]);
    xyz = ReadTestCoordinates(&graph, xyzfilename, &ndims, comm);
    }

    if (mype == 0) 
    printf("finished reading file: %s\n", argv[1]);

    part  = ismalloc(graph.nvtxs, mype%nparts, "main: part");
    sizes = imalloc(2*npes, "main: sizes");

    switch (optype) {
    case 1: 
        wgtflag = 3;
        double tmptime = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        tmptime = MPI_Wtime();
        ParMETIS_V3_PartKway(graph.vtxdist, graph.xadj, graph.adjncy, graph.vwgt, 
            graph.adjwgt, &wgtflag, &numflag, &graph.ncon, &nparts, tpwgts, ubvec, 
            options, &edgecut, part, &comm);
        tmptime = MPI_Wtime() - tmptime;
        double maxtime;
        MPI_Allreduce(&tmptime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(mype==0){
            printf("%s parmetis partition into %ld parts takes time: %.4f\n",argv[1],nparts,maxtime);
            fflush(stdout);
        }
        WritePVector(argv[1], graph.vtxdist, part, &nparts, MPI_COMM_WORLD); 
        break;
    case 2:
        wgtflag = 3;
        options[PMV3_OPTION_PSR] = PARMETIS_PSR_COUPLED;
        ParMETIS_V3_RefineKway(graph.vtxdist, graph.xadj, graph.adjncy, graph.vwgt, 
            graph.adjwgt, &wgtflag, &numflag, &graph.ncon, &nparts, tpwgts, ubvec, 
            options, &edgecut, part, &comm);
        WritePVector(argv[1], graph.vtxdist, part, &nparts, MPI_COMM_WORLD); 
        break;
    case 3:
        options[PMV3_OPTION_PSR] = PARMETIS_PSR_COUPLED;
        graph.vwgt = ismalloc(graph.nvtxs, 1, "main: vwgt");
        if (npes > 1) {
        AdaptGraph(&graph, adptf, comm);
        }
        else {
        wgtflag = 3;
        ParMETIS_V3_PartKway(graph.vtxdist, graph.xadj, graph.adjncy, graph.vwgt, 
            graph.adjwgt, &wgtflag, &numflag, &graph.ncon, &nparts, tpwgts, 
            ubvec, options, &edgecut, part, &comm);

        printf("Initial partitioning with edgecut of %"PRIDX"\n", edgecut);
        for (i=0; i<graph.ncon; i++) {
            for (j=0; j<graph.nvtxs; j++) {
            if (part[j] == i)
                graph.vwgt[j*graph.ncon+i] = adptf; 
            else
                graph.vwgt[j*graph.ncon+i] = 1; 
            }
        }
        }

        wgtflag = 3;
        ParMETIS_V3_AdaptiveRepart(graph.vtxdist, graph.xadj, graph.adjncy, graph.vwgt, 
            NULL, graph.adjwgt, &wgtflag, &numflag, &graph.ncon, &nparts, tpwgts, ubvec, 
        &ipc2redist, options, &edgecut, part, &comm);
        break;
    case 4: 
        ParMETIS_V3_NodeND(graph.vtxdist, graph.xadj, graph.adjncy, &numflag, options, 
            part, sizes, &comm);
        /* WriteOVector(argv[1], graph.vtxdist, part, comm);   */
        break;

    case 5: 
        ParMETIS_SerialNodeND(graph.vtxdist, graph.xadj, graph.adjncy, &numflag, options, 
            part, sizes, &comm);
        /* WriteOVector(argv[1], graph.vtxdist, part, comm);  */ 
        printf("%"PRIDX" %"PRIDX" %"PRIDX" %"PRIDX" %"PRIDX" %"PRIDX" %"PRIDX"\n", sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], sizes[6]);
        break;
    case 11: 
        /* TestAdaptiveMETIS(graph.vtxdist, graph.xadj, graph.adjncy, part, options, adptf, comm); */
        break;
    case 20: 
        wgtflag = 3;
        ParMETIS_V3_PartGeomKway(graph.vtxdist, graph.xadj, graph.adjncy, graph.vwgt, 
            graph.adjwgt, &wgtflag, &numflag, &ndims, xyz, &graph.ncon, &nparts, 
            tpwgts, ubvec, options, &edgecut, part, &comm);
        break;
    case 21: 
        ParMETIS_V3_PartGeom(graph.vtxdist, &ndims, xyz, part, &comm);
        break;
    }
}

    gk_free((void **)&part, &sizes, &tpwgts, &graph.vtxdist, &graph.xadj, &graph.adjncy, 
            &graph.vwgt, &graph.adjwgt, &xyz, LTERM);

  MPI_Comm_free(&comm);

  MPI_Finalize();

  return 0;
}


/*************************************************************************
* This function changes the numbering to be from 1 instead of 0
**************************************************************************/
void ChangeToFortranNumbering(idx_t *vtxdist, idx_t *xadj, idx_t *adjncy, idx_t mype, idx_t npes)
{
  idx_t i, nvtxs, nedges;

  nvtxs = vtxdist[mype+1]-vtxdist[mype];
  nedges = xadj[nvtxs];

  for (i=0; i<npes+1; i++)
    vtxdist[i]++;
  for (i=0; i<nvtxs+1; i++)
    xadj[i]++;
  for (i=0; i<nedges; i++)
    adjncy[i]++;

  return;
}
