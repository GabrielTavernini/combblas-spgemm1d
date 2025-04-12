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
#include "CombBLAS/CommGrid.h"
#include "CombBLAS/Operations.h"
#include "CombBLAS/SpHelper.h"
#include <functional>
#include <memory>
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
#include "CombBLAS/SpParMat1D.h"
#include <CombBLAS/SpParMat1DFriends.h>

using namespace combblas;
using namespace std;

// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
typedef int IT;
template <class NT>
class Dist 
{ 
public: 
    typedef SpDCCols   < IT, NT >         DCCols;
    typedef SpParMat   < IT, NT, DCCols > MPI_DCCols;
    typedef SpParMat1D < IT, NT, DCCols > MPI_DCCols1D;
};


typedef int NT;
typedef PlusTimesSRing<bool, int> PTBOOLINT;
typedef PlusTimesSRing<bool, double> PTBOOLDOUBLE;
typedef PlusTimesSRing<int, double> PTINTDOUBLE;
typedef PlusTimesSRing<int, int> PTINTINT;
typedef PlusTimesSRing<double, double> PTDOUBLEDOUBLE;



void BC_Main(const int K4Approx, const int batchSize, SpgemmOpts & opts){
    double total_start=MPI_Wtime();
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    // some statistics data.
    std::vector<double> roundtime; // everyround time: preparation time + bfs time + tally time 
    std::vector<double> bfscount; // everyround, the size of bfs loop 
    std::vector<double> bfstimeother; // each loop, bfs time, SpGEMM is not included!!!
    std::vector<double> bfSpgemm3Dtime; // each loop, bfs time  
    std::vector<double> bfSpgemm2Dtime; // each loop, bfs time 
    std::vector<double> bfSpgemm1Dtime; // each loop, bfs time 

    std::vector<double> tallycount; // everyround, the size of tally loop 
    std::vector<double> tallytimeother; // each loop, tally time 
    std::vector<double> tallySpgemm3Dtime; // each loop, bfs time 
    std::vector<double> tallySpgemm2Dtime; // each loop, bfs time 
    std::vector<double> tallySpgemm1Dtime; // each loop, bfs time 


    // Between Centrality parameters
    std::string dname = opts.dataset[0]; 
    std::string outfilename = dname + "_bcout.txt";
    std::string fullpath = opts.fullfilepath[dname];
    shared_ptr<CommGrid> fullworld = make_shared<CommGrid>(MPI_COMM_WORLD,0,0);

    /*no matter whether the input is integer, we first read it using double, then convert it to int.*/ 
    Dist<double>::MPI_DCCols Adouble(fullworld); 
    Adouble.ParallelReadMM(opts.fullfilepath[opts.dataset[0]], true, maximum<double>());
    // convert value of A to 1.0
    Adouble.Apply(std::bind([](double &a, double &b){return 1.0;},std::placeholders::_1, 1.0)); 
    // convert the double class to int class
    Dist<int>::MPI_DCCols A(Adouble);
    A.PrintInfo();
    Dist<int>::MPI_DCCols AT(A);
    AT.Transpose();
    IT mdim = A.getnrow();
    IT ndim = A.getncol();
    if(mdim != ndim){
        if(myrank==0)std::cerr<<"matrix is not graph!"<<std::endl;
        exit(0);
    }
    // reading matrix 
    // BC related
    int nPasses = (int) pow(2.0, K4Approx);
    if(nPasses > mdim){
        if(myrank==0)std::cerr<<"nPasses:"<<nPasses<<", larger than mdim: "<< mdim << ", reducing npass ... "<<std::endl;
        while(nPasses > mdim) nPasses/=2;
        if(myrank==0)std::cerr<<"new nPasses:"<<nPasses << std::endl;
    }
    int numBatches = (int) ceil( static_cast<float>(nPasses)/ static_cast<float>(batchSize));

    // get the number of batch vertices for submatrix
    int subBatchSize = batchSize / (AT.getcommgrid())->GetGridCols();
    int nBatchSize = subBatchSize * (AT.getcommgrid())->GetGridCols();
    nPasses = numBatches * nBatchSize;	// update the number of starting vertices	

    if(batchSize % (AT.getcommgrid())->GetGridCols() > 0 && myrank == 0)
    {
        cout << "*** Batchsize is not evenly divisible by the grid dimension ***" << endl;
        cout << "*** Processing "<< nPasses <<" vertices instead"<< endl;
    }
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
        vector<IT> single;
        vector<IT> empty;
        single.push_back(vrtxid); // will return ERROR if vrtxid > N (the column dimension) 
        int locnnz = ((AT.seq())(empty,single)).getnnz();
        int totnnz;
        MPI_Allreduce( &locnnz, &totnnz, 1, MPI_INT, MPI_SUM, (AT.getcommgrid())->GetColWorld());
        if(totnnz > 0)
        {
            candidates.push_back(vrtxid);
            ++vertices;
        }
        ++vrtxid;
    }
    
    SpParHelper::Print("Candidates chosen, precomputation finished\n");
    vector<IT> batch(subBatchSize);
    FullyDistVec<IT, double> bc(AT.getcommgrid(), A.getnrow(), 0.0);
    if(myrank==0)std::cerr<<"numbatches:"<<numBatches<<"," << mdim << "," <<batchSize <<std::endl;

    for(int i=0; i< numBatches; ++i)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double round_start = MPI_Wtime();


        // perforance per batch should be relatively the same.
        for(int j=0; j< subBatchSize; ++j)
        {
            batch[j] = candidates[i*subBatchSize + j];
        }
        
        Dist<NT>::MPI_DCCols fringe = AT.SubsRefCol(batch);

        // Create nsp by setting (r,i)=1 for the ith root vertex with label r
        // Inially only the diagonal processors have any nonzeros (because we chose roots so)
        Dist<NT>::DCCols * nsploc = new Dist<NT>::DCCols();
        tuple<IT, IT, NT> * mytuples = NULL;	
        if(AT.getcommgrid()->GetRankInProcRow() == AT.getcommgrid()->GetRankInProcCol())
        {
            mytuples = new tuple<IT, IT, NT>[subBatchSize];
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
    
        Dist<NT>::MPI_DCCols  nsp(nsploc, AT.getcommgrid());

        double bfs_start = MPI_Wtime();
        vector < Dist<NT>::MPI_DCCols * > bfs; // internally keeps track of depth
        // SpParHelper::Print("Exploring via BFS...\n");
        IT round = 0;
        while( fringe.getnnz() > 0 )
        {
            double tt0 = MPI_Wtime();
            nsp += fringe;
            Dist<NT>::MPI_DCCols * level = new Dist<NT>::MPI_DCCols( fringe ); 
            bfs.push_back(level);
            // fringe = PSpGEMM<PTBOOLINT>(AT, fringe);
            double spgemmt0 = MPI_Wtime();
            fringe = PSpGEMM<PTINTINT>(AT, fringe);
            double spgemmt1 = MPI_Wtime();
            fringe = EWiseMult(fringe, nsp, true);
            double tt1 = MPI_Wtime();
        }
        double bfs_end = MPI_Wtime();
        
        // Apply the unary function 1/x to every element in the matrix
        // 1/x works because no explicit zeros are stored in the sparse matrix nsp
        Dist<double>::MPI_DCCols nspInv = nsp;
        nspInv.Apply(std::bind(divides<double>(),1,std::placeholders::_1));

        // create a dense matrix with all 1's 
        DenseParMat<IT, double> bcu(1.0, AT.getcommgrid(), fringe.getlocalrows(), fringe.getlocalcols() );
        
        // SpParHelper::Print("Tallying...\n");
        // BC update for all vertices except the sources
        double tally_start = MPI_Wtime();
        round = 0;
        for(int j = bfs.size()-1; j > 0; --j)
        {
            double tt0 = MPI_Wtime();
            Dist<double>::MPI_DCCols w = EWiseMult( *bfs[j], nspInv, false);
            w.EWiseScale(bcu);
            double spgemmt0 = MPI_Wtime();
            Dist<double>::MPI_DCCols product = PSpGEMM<PTINTDOUBLE>(A,w);
            double spgemmt1 = MPI_Wtime();
            product = EWiseMult(product, *bfs[j-1], false);
            product = EWiseMult(product, nsp, false);
            bcu += product;
            double tt1 = MPI_Wtime();
        }
        for(int j=0; j < bfs.size(); ++j)
        {
            delete bfs[j];
        }
        double tally_end = MPI_Wtime();
        // if(myrank==0)std::cerr<<"tally time:"<<tally_end - tally_start <<std::endl;
        // SpParHelper::Print("Adding bc contributions...\n");
        bc += FullyDistVec<int, double>(bcu.Reduce(Row, plus<double>(), 0.0));	// pack along rows
        double round_end = MPI_Wtime();
    }
    bc.Apply(std::bind(minus<double>(),std::placeholders::_1,nPasses)); // Subtrack nPasses from all the bc scores (because bcu was initialized to all 1's)
    double total_end=MPI_Wtime();
    double TEPS = (nPasses * static_cast<float>(A.getnnz())) / (total_end-total_start);
    std::ofstream output(outfilename);
    bc.SaveGathered(output, 0);
    output.close();
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    SpHelper::initdatasetmap();
    SpgemmOpts opts;
    OptParser::parse(argc, argv, &opts);
    if(opts.AproxK == -1 || opts.batchSize == -1){
        if(myrank==0)std::cerr<<"forget to set approxk and batchsize!"<<std::endl;
    }
    BC_Main(opts.AproxK, opts.batchSize, opts);
    MPI_Finalize();
    return 0;
}
