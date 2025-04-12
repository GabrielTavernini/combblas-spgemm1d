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

// These macros should be defined before stdint.h is included
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#include <stdint.h>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>  // Required for stringstreams
#include <ctime>
#include <cmath>
// #include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpDCCols.h"
#include "CombBLAS/SpParMat1D.h"
#include <CombBLAS/SpParMat1DFriends.h>

using namespace combblas;
using namespace std;

// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"

template <class NT>
class Dist 
{ 
public: 
	typedef SpDCCols < int, NT > DCCols;
	typedef SpParMat < int, NT, DCCols > MPI_DCCols;
	typedef SpParMat1D < int, NT, DCCols > MPI_DCCols1D;
};

typedef int IT;
typedef int NT;

SpgemmOpts combblas::opts;
int main(int argc, char* argv[])
{
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
   
    typedef PlusTimesSRing<bool, int> PTBOOLINT;
    typedef PlusTimesSRing<int, bool> PTINTBOOL;
    typedef PlusTimesSRing<int, int> PTINTINT;
    typedef PlusTimesSRing<bool, double> PTBOOLDOUBLE;
    typedef PlusTimesSRing<int, double> PTINTDOUBLE;
    typedef PlusTimesSRing<double, int> PTDOUBLEINT;

    /*register performance counter metrics.*/
    combblas::perfcnt1d.GetRankInfo();
    combblas::perfcnt1d.RegisterDouble("prepT(ms)",2,1000.); // report time in ms.
    combblas::perfcnt1d.RegisterDouble("commT(ms)",2,1000.);
    combblas::perfcnt1d.RegisterDouble("compT(ms)",2,1000.);
    combblas::perfcnt1d.RegisterInteger("Alocalnnz");
    combblas::perfcnt1d.RegisterDouble("Alocalmem(MB)",2,1.0);
    combblas::perfcnt1d.RegisterInteger("Aneednnz");
    combblas::perfcnt1d.RegisterDouble("Aneedmem(MB)",2,1.0);
    combblas::perfcnt1d.RegisterInteger("Bnnz    ");
    combblas::perfcnt1d.RegisterDouble("Bmem(MB)",2,1.0);
    combblas::perfcnt1d.RegisterInteger("Cnnz    ");
    combblas::perfcnt1d.RegisterDouble("Cmem(MB)",2,1.0);
    combblas::perfcnt2d.GetRankInfo();
    combblas::perfcnt2d.RegisterDouble("bcastA(ms)",2,1000.);
    combblas::perfcnt2d.RegisterDouble("bcastB(ms)",2,1000.);
    combblas::perfcnt2d.RegisterDouble("compT(ms)",2,1000.);
    combblas::perfcnt2d.RegisterDouble("merge(ms)",2,1000.);

    // arg index
    int argpos = 1;
    {

        string graphtype, gemmtype, inputfile;
        int permute;
        // reading matrix 
        Dist<NT>::MPI_DCCols *Aread;
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        // graph input choices
        graphtype = string(argv[argpos++]);
        if(graphtype == "input"){
            inputfile = string(argv[argpos++]);
            gemmtype = string(argv[argpos++]);
            permute = atoi(argv[argpos++]);
            if(myrank ==0) printf("input %s gemmtype %s permuted %d \n", inputfile.c_str(), gemmtype.c_str(), permute);
            Aread->ParallelReadMM(inputfile, true, maximum<double>());
            // if(permute){
            //     FullyDistVec<IT,IT> perm;    // get a different permutation
            //     perm.iota(Aread->getnrow(), 0);
            //     perm.RandPerm();
            //     // Aread->operator(perm,perm,true);
            //     (*Aread)(perm, perm, true);    // in-place permute to save memory
            // }
        }else {
            unsigned scale = (unsigned) atoi(argv[argpos++]);
            unsigned EDGEFACTOR = (unsigned) atoi(argv[argpos++]);
            int permute = atoi(argv[argpos++]);
            gemmtype = string(argv[argpos++]);
            double initiator[4];
            if(graphtype == string("ER"))
            {
                initiator[0] = .25;
                initiator[1] = .25;
                initiator[2] = .25;
                initiator[3] = .25;
            }
            else if(graphtype == string("G500"))
            {
                initiator[0] = .57;
                initiator[1] = .19;
                initiator[2] = .19;
                initiator[3] = .05;
                EDGEFACTOR  = 16;
            }
            else if(graphtype == string("SSCA"))
            {
                initiator[0] = .6;
                initiator[1] = .4/3;
                initiator[2] = .4/3;
                initiator[3] = .4/3;
                EDGEFACTOR  = 8;
            }
            else {
                if(myrank == 0) printf("The initiator parameter - %s - is not recognized.\n", argv[5]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            MPI_Comm comm = MPI_COMM_WORLD;
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(comm);
            ostringstream minfo;
            int nprocs = DEL->commGrid->GetSize();
            minfo << "Started Generation of scale "<< scale << endl;
            minfo << "Using " << nprocs << " MPI processes" << endl;
            SpParHelper::Print(minfo.str());
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, permute, false );
            // don't generate packed edges, that function uses MPI_COMM_WORLD which can not be used in a single layer!
            SpParHelper::Print("Generated renamed edge lists\n");
            ostringstream tinfo;
            SpParHelper::Print(tinfo.str());
            Aread = new Dist<NT>::MPI_DCCols(*DEL, false);
            delete DEL;
        }
        
        Dist<NT>::MPI_DCCols A(new Dist<NT>::DCCols(*Aread->seqptr()),fullWorld);	// construct object
        Dist<NT>::MPI_DCCols AT(A);
        A.Transpose();
        if(A == AT){
            if(myrank == 0) printf("the graph is symmetric!!\n");
        }else{
            if(myrank == 0) printf("the graph is not symmetric!!\n");
        }
        delete Aread;
        

        // reading input 
        // npass = 2**K4Approx 
        // numBatches = npass / batchsize 
        // subbatch = batchsize / sqrt(process)(4/8)

        // say batchsize = 8*2**P = 2**(3+P) (P=0,1,2,3,...)
        // subbatch = 2**(3+P) / 2**3 = 2**P

        // examples, K4Approx = 10, batchsize = 8 => subbatch = 2 

        // Between Centrality parameters
        int K4Approx = atoi(argv[argpos++]);
        int batchSize = atoi(argv[argpos++]);
        // output filename
        std::ofstream output(argv[argpos++]); 
        // BC related
        int nPasses = (int) pow(2.0, K4Approx);
        int numBatches = (int) ceil( static_cast<float>(nPasses)/ static_cast<float>(batchSize));

        // get the number of batch vertices for submatrix
        int subBatchSize = batchSize / (AT.getcommgrid())->GetGridCols();
        int nBatchSize = subBatchSize * (AT.getcommgrid())->GetGridCols();
        nPasses = numBatches * nBatchSize;// update the number of starting vertices

        if(batchSize % (AT.getcommgrid())->GetGridCols() > 0 && myrank == 0)
        {
            cout << "*** Batchsize is not evenly divisible by the grid dimension ***" << endl;
            cout << "*** Processing "<< nPasses <<" vertices instead"<< endl;
        }

        // if(myrank == 0) printf("k4Approx %d -> npasses %d, batchsize %d numBatches %d \n", K4Approx, nPasses,batchSize, numBatches);
        ostringstream tinfo;
        tinfo << "Batch processing will occur " << numBatches << " times, each processing " << nBatchSize << " vertices (overall)" << endl;
        SpParHelper::Print(tinfo.str());

        vector<int> candidates;
        // Only consider non-isolated vertices
        int vertices = 0;
        int vrtxid = 0; 
        int nlocpass = numBatches * subBatchSize;
        while(vertices < nlocpass)
        {
            vector<int> single;
            vector<int> empty;
            single.push_back(vrtxid); // will return ERROR if vrtxid > N (the column dimension) 
            int locnnz = ((AT.seq())(empty,single)).getnnz();
            int totnnz;
            MPI_Allreduce( &locnnz, &totnnz, 1, MPI_INT, 
            MPI_SUM, (AT.getcommgrid())->GetColWorld());
            if(totnnz > 0)
            {
                candidates.push_back(vrtxid);
                ++vertices;
            }
            ++vrtxid;
        }
        
        SpParHelper::Print("Candidates chosen, precomputation finished\n");
        double t1 = MPI_Wtime();
        vector<int> batch(subBatchSize);
        FullyDistVec<int, double> bc(AT.getcommgrid(), A.getnrow(), 0.0);

        for(int i=0; i< numBatches; ++i)
        {
            for(int j=0; j< subBatchSize; ++j)
            {
                batch[j] = candidates[i*subBatchSize + j];
            }
            
            Dist<int>::MPI_DCCols fringe = AT.SubsRefCol(batch); // fringe col size is batch * nprocs_in_col

            // Create nsp by setting (r,i)=1 for the ith root vertex with label r
            // Inially only the diagonal processors have any nonzeros (because we chose roots so)
            Dist<int>::DCCols * nsploc = new Dist<int>::DCCols();
            tuple<int, int, int> * mytuples = NULL;	
            if(AT.getcommgrid()->GetRankInProcRow() == AT.getcommgrid()->GetRankInProcCol())
            {
                mytuples = new tuple<int, int, int>[subBatchSize];
                for(int k =0; k<subBatchSize; ++k)
                {
                    mytuples[k] = make_tuple(batch[k], k, 1);
                }
                nsploc->Create( subBatchSize, AT.getlocalrows(), subBatchSize, mytuples);
            }
            else
            {  
                nsploc->Create( 0, AT.getlocalrows(), subBatchSize, mytuples);
            }
        
            Dist<int>::MPI_DCCols  nsp(nsploc, AT.getcommgrid());
            vector < Dist<int>::MPI_DCCols * > bfs;// internally keeps track of depth
            int nrows = fringe.getnrow();
            int ncols = fringe.getncol();
            int nnz = fringe.getnnz();
            if(myrank == 0) printf("fringe size nrow %d ncol %d nnz %d\n", nrows, ncols, nnz);
            SpParHelper::Print("Exploring via BFS...\n");
            MPI_Timer timer;
            int loopcnt = 0;

            while( fringe.getnnz() > 0 )
            {
                nsp += fringe;
                Dist<int>::MPI_DCCols * level = new Dist<int>::MPI_DCCols( fringe ); 
                bfs.push_back(level);
                // perfcnt2d.Reset();
                timer.start("SpGEMM2D");
                // Dist<int>::MPI_DCCols fringeres = PSpGEMM<PTBOOLINT>(AT, fringe);
                Dist<int>::MPI_DCCols fringeres = PSpGEMM<PTINTINT>(AT, fringe);
                timer.stop("SpGEMM2D");
                // perfcnt2d.OutputRecords("BC2D_Iteration" + to_string(loopcnt) + ".log");
                // 3D SpGEMM
                {   
                    
                }       
                if(gemmtype == "1D"){
                    /*transpose impl*/ 
                    Dist<int>::MPI_DCCols fringeT = fringe;
                    // timer.start("transposefringe");
                    fringeT.Transpose();
                    // timer.stop("transposefringe");
                    Dist<int>::MPI_DCCols1D A1D(A);
                    Dist<int>::MPI_DCCols1D fringeT1D(fringeT);
                    timer.start("SpGEMM1D_Transimpl");
                    fringeT1D = PSpGEMM<PTINTINT>(fringeT1D,A1D);
                    timer.stop("SpGEMM1D_Transimpl");
                    Dist<int>::MPI_DCCols f1d2d(fringeT1D,fringeT1D.getblocksizevec());
                    // timer.start("transpose1dres");
                    f1d2d.Transpose();
                    // timer.stop("transpose1dres");
                    string filename  = "BC1DTranspose_Iteration" + to_string(loopcnt) + ".log";
                    string description = "";


                    if(f1d2d == fringeres){
                        if(myrank == 0) printf("1d exploring is correct!\n");
                    }else{
                        if(myrank == 0) printf("1d exploring is wrong!\n");
                    }
                    
                }
                fringe = fringeres;
                fringe = EWiseMult(fringe, nsp, true);
                loopcnt++;
                // break;
            }
            if(myrank == 0) printf("loopcnt is %d bfs size is %ld \n", loopcnt, bfs.size());
            // break;
            // Apply the unary function 1/x to every element in the matrix
            // 1/x works because no explicit zeros are stored in the sparse matrix nsp
            Dist<double>::MPI_DCCols nspInv = nsp;
            // nspInv.Apply(bind1st(divides<double>(), 1));
            nspInv.Apply(std::bind(divides<double>(),1, std::placeholders::_1));
            // create a dense matrix with all 1's 
            DenseParMat<int, double> bcu(1.0, AT.getcommgrid(), fringe.getlocalrows(), fringe.getlocalcols() );
            SpParHelper::Print("Tallying...\n");
            // char printchar[1000];
            // sprintf(printchar, "bfs size %ld \n", bfs.size());
            // SpParHelper::Print(printchar);
            // BC update for all vertices except the sources
            for(int j = bfs.size()-1; j > 0; --j)
            {
                Dist<double>::MPI_DCCols w = EWiseMult( *bfs[j], nspInv, false);
                w.EWiseScale(bcu);
                timer.start("SpGEMM2D_tallying");
                Dist<double>::MPI_DCCols product = PSpGEMM<PTINTDOUBLE>(A,w);
                timer.stop("SpGEMM2D_tallying");
                if(gemmtype == "1D"){

                    /*transpose impl*/ 
                    Dist<double>::MPI_DCCols wT = w;
                    // timer.start("transposew");
                    wT.Transpose();
                    // timer.stop("transposew");
                    Dist<double>::MPI_DCCols1D wT1D(wT);
                    Dist<int>::MPI_DCCols1D AT1D(AT);
                    // Dist<int>::MPI_DCCols1D fringeT1D(fringeT);
                    timer.start("SpGEMM1D_Transimpl");
                    Dist<double>::MPI_DCCols1D product1D
                    = Mult_AnXBn_1D_CbC_RDMA_FetchAll<PTDOUBLEINT, double, SpDCCols<int, double>>(wT1D,AT1D);
                    timer.stop("SpGEMM1D_Transimpl");
                    Dist<double>::MPI_DCCols p1d2d(product1D,product1D.getblocksizevec());
                    // timer.start("transpose1dres");
                    p1d2d.Transpose();
                    // timer.stop("transpose1dres");
                    if(p1d2d == product){
                        if(myrank == 0) printf("1d tallying is correct!\n");
                    }else{
                        if(myrank == 0) printf("1d tallying is wrong!\n");
                    }
                }
                product = EWiseMult(product, *bfs[j-1], false);
                product = EWiseMult(product, nsp, false);
                bcu += product;
            }
            for(int j=0; j < bfs.size(); ++j)
            {
                delete bfs[j];
            }
            SpParHelper::Print("Adding bc contributions...\n");
            bc += FullyDistVec<int, double>(bcu.Reduce(Row, plus<double>(), 0.0));	// pack along rows
        }
        
        bc.Apply(std::bind(minus<double>(),std::placeholders::_1,nPasses));
        double t2=MPI_Wtime();
        double TEPS = (nPasses * static_cast<float>(A.getnnz())) / (t2-t1);
        if( myrank == 0)
        {
            cout<<"Computation finished"<<endl;	
            fprintf(stdout, "%.6lf seconds elapsed for %d starting vertices\n", t2-t1, nPasses);
            fprintf(stdout, "TEPS score is: %.6lf\n", TEPS);
        }
        bc.SaveGathered(output, 0);
        output.close();
    }

    // make sure the destructors for all objects are called before MPI::Finalize()
    MPI_Finalize();
    return 0;
}
