#ifndef _SP_PAR_MAT_1D_FRIENDS_H_
#define _SP_PAR_MAT_1D_FRIENDS_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <sstream>
#include <tuple>
#include <vector>
#include <iterator>

#include "CombBLAS/Compare.h"
#include "CombBLAS/SpParMat1D.h"
#include "SpMat.h"
#include "SpTuples.h"
#include "SpDCCols.h"
#include "CommGrid.h"
#include "CommGrid1D.h"

#include "MPIType.h"
#include "LocArr.h"
#include "SpDefs.h"
#include "Deleter.h"
#include "SpHelper.h"
#include "SpParHelper.h"
#include "FullyDistVec.h"
#include "Friends.h"
#include "Operations.h"
#include "DistEdgeList.h"
#include "mtSpGEMM.h"
#include "MultiwayMerge.h"
#include "CombBLAS.h"

#include "FullyDistVec1D.h"

namespace combblas
{



// Generate RDMA Calls configs based on project vector of each process and remote cp and remote jc array of A.
// this function will fetch one column per RDMA Call.
template<typename IT>
inline void GenerateRDMACallAndMemOffset_FetchCbC(
    const std::vector<IT> & blocksizeprefixA,
    const std::vector<std::vector<IT>> & remoteAcp, 
    const std::vector<std::vector<IT>> & remoteAjc,
    const std::vector<int> & projrow, 
    IT & totalnnz, 
    IT & totalremotennz,
    IT & Aneedednzccnt,
    std::vector<IT> & Aneededcp, 
    std::vector<IT> & Aneededjc, 
    std::vector<IT> & Aneededoffset,
    std::vector<std::vector<IT>> & AoriginoffsetPerRank,
    std::vector<std::vector<IT>> & AneededoffsetPerRank,
    std::vector<std::vector<IT>> & AneedednnzPerRank)
{
    int nprocs = remoteAjc.size();
    int myrank; MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    /*V1: we fetch column by column, too many MPI_Get Calls!!*/
    for(int p=0; p<nprocs; p++){
        AneededoffsetPerRank[p].push_back(0);
        for(int kr=0; kr < remoteAjc[p].size(); kr++){
            IT colid  = remoteAjc[p][kr];
            if(projrow[kr + blocksizeprefixA[p]] >= 1){
                IT curcolelems = remoteAcp[p][kr+1] - remoteAcp[p][kr];
                totalnnz += curcolelems; // save total non-zero elements
                if(myrank != p) totalremotennz+= curcolelems;
                Aneedednzccnt++; // save total non-zero columns
                Aneededjc.push_back(colid + blocksizeprefixA[p]); //save global column id
                Aneededcp.push_back(curcolelems); //save nz in each column, this is not cp at this stage.
                Aneededoffset[p] += curcolelems; // save nz in each process, this is not offset at this stage.
                AoriginoffsetPerRank[p].push_back(remoteAcp[p][kr]); // save origin offset of the nzc.
                AneededoffsetPerRank[p].push_back(Aneededoffset[p]); // save needed offset of the nzc.
                AneedednnzPerRank[p].push_back(curcolelems); // save elements per columns in per mpi rank.
            }
        }
    }
    /*postprocess Aneededcp to be correct format*/
    std::vector<IT> tmp = Aneededcp;
    Aneededcp.resize(tmp.size()+1);
    Aneededcp[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededcp.begin()+1);
    /*postprocess Aneededoffset to be correct format*/
    tmp = Aneededoffset;
    Aneededoffset.resize(tmp.size()+1);
    Aneededoffset[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededoffset.begin()+1);
}


// Generate RDMA Calls configs based on project vector of each process and remote cp and remote jc array of A.
// this function will fetch column by column, but we will merge continuous column..
template<class IT>
inline void GenerateRDMACallAndMemOffset_MergeColumn(
    const std::vector<IT> & blocksizeprefixA,
    const std::vector<std::vector<IT>> & remoteAcp, 
    const std::vector<std::vector<IT>> & remoteAjc,
    const std::vector<int> & projrow, 
    IT & totalnnz, 
    IT & totalremotennz,
    IT & Aneedednzccnt,
    std::vector<IT> & Aneededcp, 
    std::vector<IT> & Aneededjc, 
    std::vector<IT> & Aneededoffset,
    std::vector<std::vector<IT>> & AoriginoffsetPerRank,
    std::vector<std::vector<IT>> & AneededoffsetPerRank,
    std::vector<std::vector<IT>> & AneedednnzPerRank){
    /*V2: we fetch column by column, but merge call for continuous column, too low hit rate!!*/
    IT continuehit = 0, totalhit = 0;
    int nprocs = remoteAcp.size();
    int myrank; MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    for(int p=0; p<nprocs; p++){
        AneededoffsetPerRank[p].push_back(0);
        IT prevcolid = -1;
        for(int kr=0; kr < remoteAjc[p].size(); kr++){
            IT colid  = remoteAjc[p][kr];
            if(projrow[kr + blocksizeprefixA[p]] >= 1){
                totalhit++;
                IT curcolelems = remoteAcp[p][kr+1] - remoteAcp[p][kr];
                totalnnz += curcolelems; // save total non-zero elements
                if(myrank != p) totalremotennz+= curcolelems;
                Aneedednzccnt++; // save total non-zero columns
                Aneededjc.push_back(colid + blocksizeprefixA[p]); //save global column id
                Aneededcp.push_back(curcolelems); //save nz in each column, this is not cp at this stage.
                Aneededoffset[p] += curcolelems; // save nz in each process, this is not offset at this stage.
                if(prevcolid == -1){ /*first call*/ 
                    AneededoffsetPerRank[p].push_back(Aneededoffset[p]); // save needed offset of the nzc.
                    AoriginoffsetPerRank[p].push_back(remoteAcp[p][kr]); // save origin offset of the nzc.
                    AneedednnzPerRank[p].push_back(curcolelems); // save elements per columns in per mpi rank.
                    prevcolid = colid;
                }else if(prevcolid + 1 == colid){
                    continuehit++;
                    AneededoffsetPerRank[p].back() += curcolelems;
                    AneedednnzPerRank[p].back() += curcolelems;
                    prevcolid = colid;
                }else{ /*the column is not adjacent with previous column*/
                    AneededoffsetPerRank[p].push_back(Aneededoffset[p]); // save needed offset of the nzc.
                    AoriginoffsetPerRank[p].push_back(remoteAcp[p][kr]); // save origin offset of the nzc.
                    AneedednnzPerRank[p].push_back(curcolelems); // save elements per columns in per mpi rank.
                }
                
            }
        }
    }
    // very low hit rate!!!
    // printf("myrank %d totalhit %lu continuehit %lu percent %f \n", myrank, totalhit, continuehit, (double)continuehit / totalhit);
    /*postprocess Aneededcp to be correct format*/
    std::vector<IT> tmp = Aneededcp;
    Aneededcp.resize(tmp.size()+1);
    Aneededcp[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededcp.begin()+1);
    /*postprocess Aneededoffset to be correct format*/
    tmp = Aneededoffset;
    Aneededoffset.resize(tmp.size()+1);
    Aneededoffset[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededoffset.begin()+1);
}

// Generate RDMA Calls configs based on project vector of each process and remote cp and remote jc array of A.
// this function will fetch chunk by chunk. it will split the remote jc into K (1024 e.g.) chunks. 
// if any column in a chunk is required by source process, then it will fetch all columns inside the chunk. 
template<class IT>
inline void GenerateRDMACallAndMemOffset_FetchChunkV3(
    const std::vector<IT> & blocksizeprefixA,
    const std::vector<std::vector<IT>> & remoteAcp, 
    const std::vector<std::vector<IT>> & remoteAjc,
    const std::vector<int> & projrow, 
    IT & totalnnz, 
    IT & totalremotennz,
    IT & Aneedednzccnt,
    std::vector<IT> & Aneededcp, 
    std::vector<IT> & Aneededjc, 
    std::vector<IT> & Aneededoffset,
    std::vector<std::vector<IT>> & AoriginoffsetPerRank,
    std::vector<std::vector<IT>> & AneededoffsetPerRank,
    std::vector<std::vector<IT>> & AneedednnzPerRank){
    int nprocs = remoteAcp.size();
    int myrank; MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    /*V3: we split remote nzc into K parts, once we need data in the part. we fetch the whole part. */
    for(int p=0; p<nprocs; p++){
        IT remotenzc = remoteAjc[p].size();
        if(remotenzc == 0) continue;
        int ChunksK = 2048;
        while(ChunksK > 1 && ChunksK > remotenzc) ChunksK /= 2;
        if(ChunksK == 0)std::cerr<<"chunksK is zero"<<std::endl;
        /*first decide whether we need to fetch the data from remote process*/ 
        IT batchsize = remotenzc / ChunksK;
        std::vector<int> hitbatch(ChunksK,0);
        int needfetch = 0;
        // record whether the batch will be requested.
        for(IT bi=0; bi < ChunksK; bi++){
            IT startidx = bi * batchsize;
            IT endidx = bi == ChunksK-1 ? remotenzc : ( bi + 1 ) * batchsize ;
            IT curbatchdatasize = 0;
            for(IT kr = startidx; kr < endidx; kr++){
                IT colid  = remoteAjc[p][kr];
                if(projrow[colid + blocksizeprefixA[p]] >= 1){
                    hitbatch[bi] = 1;
                    needfetch = 1;
                }
            }
        }
        if(needfetch==0)continue; // we didn't hit any batch, fast return
        // do remote memory offset process
        AneededoffsetPerRank[p].push_back(0);
        IT prevbatchid = -1;
        for(IT bi=0; bi < ChunksK; bi++){
            if(hitbatch[bi]==0)continue;
            IT startidx = bi * batchsize;
            IT endidx = bi == ChunksK-1 ? remotenzc : ( bi + 1 ) * batchsize ;
            if(prevbatchid == -1 || prevbatchid + 1 != bi){
                // we meet the first batch or a new distinct batch, save it as a new call.
                IT thisbatchnnz = 0;
                for(IT kr = startidx; kr < endidx; kr++){
                    IT colid  = remoteAjc[p][kr];
                    IT curcolelems = remoteAcp[p][kr+1] - remoteAcp[p][kr];
                    totalnnz += curcolelems;
                    if(myrank != p) totalremotennz += curcolelems;
                    Aneedednzccnt++;
                    Aneededjc.push_back(colid + blocksizeprefixA[p]);
                    thisbatchnnz += curcolelems;
                    Aneededcp.push_back(curcolelems); //save nz in each column, this is not cp at this stage.
                    Aneededoffset[p] += curcolelems; // save nz in each process, this is not offset at this stage.
                }
                AneedednnzPerRank[p].push_back(thisbatchnnz); // save elements per columns in per mpi rank.
                AoriginoffsetPerRank[p].push_back(remoteAcp[p][startidx]); // save origin offset of the nzc.
                AneededoffsetPerRank[p].push_back(Aneededoffset[p]); // save needed offset of the nzc.
                prevbatchid = bi;
            }else if(prevbatchid + 1 == bi){ // we have two successive batch, merge it with previous call.
                for(IT kr = startidx; kr < endidx; kr++){
                    IT colid  = remoteAjc[p][kr];
                    IT curcolelems = remoteAcp[p][kr+1] - remoteAcp[p][kr];
                    totalnnz += curcolelems;
                    if(myrank != p) totalremotennz+= curcolelems;
                    Aneedednzccnt++;
                    Aneededjc.push_back(colid + blocksizeprefixA[p]);
                    Aneededcp.push_back(curcolelems); //save nz in each column, this is not cp at this stage.
                    Aneededoffset[p] += curcolelems; // save nz in each process, this is not offset at this stage.
                    AneededoffsetPerRank[p].back() += curcolelems; // save needed offset of the nzc.
                    AneedednnzPerRank[p].back() += curcolelems; // save elements per columns in per mpi rank.
                }
                prevbatchid = bi;
            }else{
                printf("just make sure no other cases!!\n");
            }
        }
    }
    /*postprocess Aneededcp to be correct format*/
    std::vector<IT> tmp = Aneededcp;
    Aneededcp.resize(tmp.size()+1);
    Aneededcp[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededcp.begin()+1);
    /*postprocess Aneededoffset to be correct format*/
    tmp = Aneededoffset;
    Aneededoffset.resize(tmp.size()+1);
    Aneededoffset[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededoffset.begin()+1);
}


// Generate RDMA Calls configs based on project vector of each process and remote cp and remote jc array of A.
// this function will fetch chunk by chunk. it will split the remote jc into K (1024 e.g.) chunks. 
// if any column in a chunk is required by source process, then it will fetch all columns inside the chunk. 
template<class IT>
inline void GenerateRDMACallAndMemOffset_FetchChunkV1(
    const std::vector<IT> & blocksizeprefixA,
    const std::vector<std::vector<IT>> & remoteAcp, 
    const std::vector<std::vector<IT>> & remoteAjc,
    const std::vector<int> & projrow, 
    IT & totalnnz, 
    IT & totalremotennz,
    IT & Aneedednzccnt,
    std::vector<IT> & Aneededcp, 
    std::vector<IT> & Aneededjc, 
    std::vector<IT> & Aneededoffset,
    std::vector<std::vector<IT>> & AoriginoffsetPerRank,
    std::vector<std::vector<IT>> & AneededoffsetPerRank,
    std::vector<std::vector<IT>> & AneedednnzPerRank){
    int nprocs = remoteAcp.size();
    int myrank; MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    /*V3: we split remote nzc into K parts, once we need data in the part. we fetch the whole part. */
    for(int p=0; p<nprocs; p++){
        IT remotenzc = remoteAjc[p].size();
        if(remotenzc == 0) continue;
        int ChunksK = 2048;
        while(ChunksK > 1 && ChunksK > remotenzc) ChunksK /= 2;
        if(ChunksK == 0)std::cerr<<"chunksK is zero"<<std::endl;
        /*first decide whether we need to fetch the data from remote process*/ 
        IT batchsize = remotenzc / ChunksK;
        std::vector<int> hitbatch(ChunksK,0);
        int needfetch = 0;
        // record whether the batch will be requested.
        for(IT bi=0; bi < ChunksK; bi++){
            IT startidx = bi * batchsize;
            IT endidx = bi == ChunksK-1 ? remotenzc : ( bi + 1 ) * batchsize ;
            IT curbatchdatasize = 0;
            for(IT kr = startidx; kr < endidx; kr++){
                IT colid  = remoteAjc[p][kr];
                if(projrow[colid + blocksizeprefixA[p]] >= 1){
                    hitbatch[bi] = 1;
                    needfetch = 1;
                }
            }
        }
        if(needfetch==0)continue; // we didn't hit any batch, fast return
        // do remote memory offset process
        AneededoffsetPerRank[p].push_back(0);
        IT prevbatchid = -1;
        for(IT bi=0; bi < ChunksK; bi++){
            if(hitbatch[bi]==0)continue;
            IT startidx = bi * batchsize;
            IT endidx = bi == ChunksK-1 ? remotenzc : ( bi + 1 ) * batchsize ;
            // we meet the first batch or a new distinct batch, save it as a new call.
            for(IT kr = startidx; kr < endidx; kr++){
                IT colid  = remoteAjc[p][kr];
                IT curcolelems = remoteAcp[p][kr+1] - remoteAcp[p][kr];
                totalnnz += curcolelems;
                if(myrank != p) totalremotennz+= curcolelems;
                Aneedednzccnt++;
                Aneededjc.push_back(colid + blocksizeprefixA[p]);
                Aneededcp.push_back(curcolelems); //save nz in each column, this is not cp at this stage.
                Aneededoffset[p] += curcolelems; // save nz in each process, this is not offset at this stage.
                AoriginoffsetPerRank[p].push_back(remoteAcp[p][kr]); // save origin offset of the nzc.
                AneededoffsetPerRank[p].push_back(Aneededoffset[p]); // save needed offset of the nzc.
                AneedednnzPerRank[p].push_back(curcolelems); // save elements per columns in per mpi rank.
            }
        }
    }
    /*postprocess Aneededcp to be correct format*/
    std::vector<IT> tmp = Aneededcp;
    Aneededcp.resize(tmp.size()+1);
    Aneededcp[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededcp.begin()+1);
    /*postprocess Aneededoffset to be correct format*/
    tmp = Aneededoffset;
    Aneededoffset.resize(tmp.size()+1);
    Aneededoffset[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededoffset.begin()+1);
}


// Generate RDMA Calls configs based on project vector of each process and remote cp and remote jc array of A.
// this function will fetch all remote data.
template<typename IT>
inline void GenerateRDMACallAndMemOffset_FetchAll(
    const std::vector<IT> & blocksizeprefixA,
    const std::vector<std::vector<IT>> & remoteAcp, 
    const std::vector<std::vector<IT>> & remoteAjc,
    const std::vector<int> & projrow, 
    IT & totalnnz, 
    IT & totalremotennz,
    IT & Aneedednzccnt,
    std::vector<IT> & Aneededcp, 
    std::vector<IT> & Aneededjc, 
    std::vector<IT> & Aneededoffset,
    std::vector<std::vector<IT>> & AoriginoffsetPerRank,
    std::vector<std::vector<IT>> & AneededoffsetPerRank,
    std::vector<std::vector<IT>> & AneedednnzPerRank)
{
    int nprocs = remoteAjc.size();
    int myrank; MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    for(int p=0; p<nprocs; p++){ // for each process
        Aneededoffset[p] = 0;
        for(int kr=0; kr < remoteAjc[p].size(); kr++){ // for each column of the process
            IT colid  = remoteAjc[p][kr];  // column id
            IT curcolelems = remoteAcp[p][kr+1] - remoteAcp[p][kr]; // nnz in this column
            
            totalnnz += curcolelems; // accumulate total nnz in Aneeded, used when create Aneeded obj
            if(myrank != p) totalremotennz+= curcolelems; // accumulate remote nnz needed to fetch, for stats.
            Aneedednzccnt++; // accumulate total non-zero columns, used when creating Aneeded obj

            Aneededjc.push_back(colid + blocksizeprefixA[p]); //save global column id, used when creating Aneeded obj
            Aneededcp.push_back(curcolelems); //save nnz in each column, this is not cp at this stage. will do prefix.

            // save needed offset of the nzc, must before update Aneededoffset!!!
            AneededoffsetPerRank[p].push_back(Aneededoffset[p]); 
            Aneededoffset[p] += curcolelems; // save nnz in each process, this is not offset at this stage.
            
            // save nnz per request, used as size.
            AneedednnzPerRank[p].push_back(curcolelems); 

            // save origin offset of the nzc, used in remote_disp.
            AoriginoffsetPerRank[p].push_back(remoteAcp[p][kr]); 
        }
    }
    /*postprocess Aneededcp to be correct format*/
    std::vector<IT> tmp = Aneededcp;
    Aneededcp.resize(tmp.size()+1);
    Aneededcp[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededcp.begin()+1);
    /*postprocess Aneededoffset to be correct format*/
    tmp = Aneededoffset;
    Aneededoffset.resize(tmp.size()+1);
    Aneededoffset[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededoffset.begin()+1);
}

// Generate RDMA Calls configs based on project vector of each process and remote cp and remote jc array of A.
// this function will fetch all remote data.
template<typename IT>
inline void GenerateRDMACallAndMemOffset_FetchAll1Call(
    const std::vector<IT> & blocksizeprefixA,
    const std::vector<std::vector<IT>> & remoteAcp, 
    const std::vector<std::vector<IT>> & remoteAjc,
    const std::vector<int> & projrow, 
    IT & totalnnz, 
    IT & totalremotennz,
    IT & Aneedednzccnt,
    std::vector<IT> & Aneededcp, 
    std::vector<IT> & Aneededjc, 
    std::vector<IT> & Aneededoffset,
    std::vector<std::vector<IT>> & AoriginoffsetPerRank,
    std::vector<std::vector<IT>> & AneededoffsetPerRank,
    std::vector<std::vector<IT>> & AneedednnzPerRank)
{
    int nprocs = remoteAjc.size();
    int myrank; MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // since we fetch all, let's just use a single RDMA call
    for(int p=0; p<nprocs; p++){ // for each process
        if(remoteAjc[p].size() == 0) continue;
        for(int kr=0; kr < remoteAjc[p].size(); kr++){ // for each column of the process
            IT colid  = remoteAjc[p][kr];  // column id
            IT curcolelems = remoteAcp[p][kr+1] - remoteAcp[p][kr]; // nnz in this column
            totalnnz += curcolelems; // accumulate total nnz in Aneeded, used when create Aneeded obj
            if(myrank != p) totalremotennz+= curcolelems; // accumulate remote nnz needed to fetch, for stats.
            Aneedednzccnt++; // accumulate total non-zero columns, used when creating Aneeded obj
            Aneededjc.push_back(colid + blocksizeprefixA[p]); //save global column id, used when creating Aneeded obj
            Aneededcp.push_back(curcolelems); //save nnz in each column, this is not cp at this stage. will do prefix.
        }
        // just 0
        AneededoffsetPerRank[p].push_back(0); 
        // just total nnz
        Aneededoffset[p] = remoteAcp[p].back(); 
        // save total nnz
        AneedednnzPerRank[p].push_back(remoteAcp[p].back()); 
        // just the begin
        AoriginoffsetPerRank[p].push_back(0); 
    }
    /*postprocess Aneededcp to be correct format*/
    std::vector<IT> tmp = Aneededcp;
    Aneededcp.resize(tmp.size()+1);
    Aneededcp[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededcp.begin()+1);
    /*postprocess Aneededoffset to be correct format*/
    tmp = Aneededoffset;
    Aneededoffset.resize(tmp.size()+1);
    Aneededoffset[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededoffset.begin()+1);
}



/**
 * @brief A detailed Communication Analysis Function for RDMA CbC Implementation. Write results to file.
 * The Kprocs define how many processes are involved. It can be larger than mpi processes.
 * It's only a simulation of communication of RDMA implementation.
 * @tparam IT 
 * @tparam NT1 
 * @tparam NT2 
 * @tparam DER1 
 * @tparam DER2 
 * @param A 
 * @param B 
 * @param Kprocs 
 * @param outfilename 
 */
template <class IT, class NT1, class NT2, class DER1, class DER2>
inline void CommInfo_RDMACbC(SpParMat1D<IT, NT1, DER1> &A, SpParMat1D<IT, NT2, DER2> &B, 
int Kprocs, std::string outprefix="TMP")
{
    int myrank;
    int nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    IT Ancols = A.getncol();
    IT Anrows = A.getnrow();
    IT Bncols = B.getncol();
    IT Bnrows = B.getnrow();
    if(Ancols != Bnrows){
        if(myrank==0)std::cerr << "CommInfo_RDMACbC input error: dimension not match!" << std::endl;
        exit(0);
    }
    std::vector<IT> blocksizevec, blocksizeprefix; 
    std::vector<IT> rowblocksizevec, rowblocksizeprefix;
    SpParMat1D<IT, NT1, DER1>::fillblocksizevec(blocksizevec, Kprocs, Bncols);
    SpParMat1D<IT, NT1, DER1>::fillblocksizevec(rowblocksizevec, Kprocs, Ancols);
    SpParMat1D<IT, NT1, DER1>::generateblocksizeprefix(blocksizevec,blocksizeprefix);
    SpParMat1D<IT, NT1, DER1>::generateblocksizeprefix(rowblocksizevec,rowblocksizeprefix);


    /*********************************************************************************************
    Part 1: Compute global projection row vector (projrow) for K processes.
    we go through local column id and decide which projrow it should go to. 
    Then calculate the contribution. But multiple processes may share the same projrow when nprocs != Kprocs.
    The Union of these projrows are final correct projrow.
    **********************************************************************************************/
    IT start = 0, end = 0, acc = 0;
    IT mystart = B.getblocksizeprefix()[myrank];
    IT myend = myrank == nprocs - 1 ? Bncols : B.getblocksizeprefix()[myrank+1];
    // for each interval, my contribution (start and end in local dimension).
    std::vector<std::vector<IT>> roundstartend(Kprocs,std::vector<IT>());
    for(int i=0; i<Kprocs; i++){
        start = acc;
        end = blocksizevec[i] + acc;
        // now compute start and end
        //case 1: not in the range 
        if(myend <= start || mystart >= end){
            // skip
        }else{
            IT intervalstart = std::max(mystart, start); // already local
            IT intervalend   = std::min(myend  , end  );
            roundstartend[i] = {intervalstart,intervalend};
        }
        acc += blocksizevec[i];
    }
    std::vector<std::vector<int>> localproj(Kprocs, std::vector<int>(Bnrows, 0));
    for(typename DER2::SpColIter colit = B.seqptr()->begcol(); colit != B.seqptr()->endcol(); ++colit)
    {
        IT col = colit.colid() + B.getblocksizeprefix()[myrank];
        // decide where should this col contribute
        int curidx = -1;
        for(int i=0; i<roundstartend.size(); i++){
            if(roundstartend[i].size() == 0) continue;
            if(roundstartend[i][0] <= col && col < roundstartend[i][1]){
                curidx = i;
                break;
            }
        }
        if(curidx == -1) continue;
        for(typename DER2::SpColIter::NzIter nzit = B.seqptr()->begnz(colit); nzit != B.seqptr()->endnz(colit); ++nzit)
        {
            IT row = nzit.rowid();
            localproj[curidx][row] = 1;
        }
    }
    // Union.
    std::vector<std::vector<int>> gproj(Kprocs, std::vector<int>(Bnrows, 0));
    for(int i=0; i<Kprocs; i++){
        MPI_Allreduce(
            localproj[i].data(), gproj[i].data(), Bnrows, 
            MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    }
    /*********************************************************************************************
    Part 2: Compute remote cp and jc array of A for K processes.
    **********************************************************************************************/
    
    std::vector<std::vector<IT>> remotecp(nprocs), remotejc(nprocs);
    A.ExchangeDcscIndex(remotecp, remotejc);
    // create a map to record the global column id and it's nnz in this column. 
    // this is slow but we are not for performance.
    std::unordered_map<IT, IT> nzcmap;
    for(int pi=0; pi < remotecp.size(); pi++){ // each process
        for(int i=0; i<remotejc[pi].size(); i++)  // each column
        /*absolute column id*/
        nzcmap[remotejc[pi][i] + A.getblocksizeprefix()[pi]] = remotecp[pi][i+1] - remotecp[pi][i];
    }
    std::vector<std::vector<IT>> remotecpK(Kprocs), remotejcK(Kprocs);
    if(nprocs == Kprocs) { // we are the same size, no need to change.
        remotecpK = remotecp;
        remotejcK = remotejc;
    }else{
        std::vector<std::vector<IT>> tmpcp2d(Kprocs);
        for(int i=0; i<nprocs;i++){ // remotecp is of length nprocs
            for(int j=0; j<remotecp[i].size()-1;j++){
                tmpcp2d[i].push_back(remotecp[i][j+1] - remotecp[i][j]); // remove prefixsum
            }
        }
        std::vector<IT> tmpcp;
        SpHelper::flatten(tmpcp, tmpcp2d);
        std::vector<IT> tmpjc; 
        SpHelper::flatten(tmpjc, remotejc);
        int curstopidx = 1;
        for(int i=0; i<tmpjc.size(); i++){
            if( 
                ( curstopidx == blocksizeprefix.size() )  /*case 1: the last block*/ 
                ||
                (curstopidx < blocksizeprefix.size() && tmpjc[i] < blocksizeprefix[curstopidx])
                ){
                remotejcK[curstopidx-1].push_back(tmpjc[i]);
                remotecpK[curstopidx-1].push_back(tmpcp[i]);
            }else{
                while(curstopidx < blocksizeprefix.size() && tmpjc[i] >= blocksizeprefix[curstopidx]){
                    curstopidx++; // move to next suitable bock for this jc.
                }
                remotejcK[curstopidx-1].push_back(tmpjc[i]);
                remotecpK[curstopidx-1].push_back(tmpcp[i]);
            }
        }
        for(int i=0; i<remotecpK.size(); i++){
            std::vector<IT> tmp = SpHelper::prefixsum(remotecpK[i],true);
            remotecpK[i] = tmp;
        }
    }


    //-------------------------------------------STATISTIC-------------------------------------------
    // in real impl, we only need to calculate the memory offset etc for myself.
    // here we compute all memory offset etc for statisic records.
    // rdmacalls_detailed is a 3D vector. 
    // index [i][j][k] means kth RDMA call that ith process request from jth process. 
    // the value means the elements requested by this RDMA call.
    std::vector<std::vector<std::vector<IT>>> rdmacalls_detail(Kprocs);
    for(int i=0; i<Kprocs; i++){
        IT totalnnz = 0, Aneedednzccnt = 0; // input params for Aneeded object.
        IT totalremotennz;
        std::vector<IT> Aneededjc; // memory buffer for Aneeded DCSC jc array.
        std::vector<IT> Aneededcp; // memory buffer for Aneeded DCSC cp array.
        /* some aux array for fetching remote ir and numx */
        std::vector<IT> Aneededoffset(Kprocs, 0); // offset of elements in needed ir/numx array
        // offset of elements in origin ir/numx array in per mpi rank
        std::vector<std::vector<IT>> AoriginoffsetPerRank(Kprocs); 
        // offset of elements in needed ir/numx array in per mpi rank
        std::vector<std::vector<IT>> AneededoffsetPerRank(Kprocs);
        // number of elements in needed ir/numx array per column in per mpi rank
        std::vector<std::vector<IT>> AneedednnzPerRank(Kprocs); 
        GenerateRDMACallAndMemOffset_FetchCbC(
        blocksizeprefix,remotecpK, remotejcK, gproj[i],
        totalnnz,totalremotennz,Aneedednzccnt,Aneededcp, Aneededjc,
        Aneededoffset,AoriginoffsetPerRank,
        AneededoffsetPerRank,AneedednnzPerRank);
        rdmacalls_detail.push_back(AneedednnzPerRank);
    }

    // write to file 
    std::vector<int> gprojflatten;
    SpHelper::flatten(gprojflatten, gproj);
    IT Annz = A.getnnz();
    IT Bnnz = B.getnnz();
    if(myrank == 0){
        // Write a square table, each one has a number.
        // save it to file
        std::stringstream namess;
        namess << "BrowProjCommInfo_" << outprefix << "_Nprocs" << Kprocs << ".bin";
        std::ofstream binfs(namess.str(), std::ios::binary);
        for(IT i=0; i<gprojflatten.size(); i++) binfs << gprojflatten[i];
        binfs.close();

        for(int i=0; i<Kprocs; i++){
            // write redmacalls_detail to seperate binary
            namess.str("");
            namess << "rdmacalls_" << outprefix << "_Nprocs" << Kprocs << "_Rank" << i << ".bin";
            std::ofstream binfs(namess.str(), std::ios::binary);
            for(int j=0; j<nprocs; j++){
                for(auto x : rdmacalls_detail[i][j]) binfs << x;
            }
            binfs.close();
        }
    }
}

extern SpgemmOpts opts;

/**
 * @brief 1D column-by-column SpGEMM algorithm based on RDMA, fetch all required data from remote process in one time.
 * we first generate the needed columns from the A matrix, and request the data frome remote process. 
 * After we have everything in local, we call localSpGEMM directly.
 * - It's possible that blocksizevec of A is different from rowblocksize of B. It's not an issue because we won't use 
 * it. But when we analyze the communication, maybe it's an issue.
 * @tparam SR 
 * @tparam NUO 
 * @tparam UDERO 
 * @tparam IU 
 * @tparam NU1 
 * @tparam NU2 
 * @tparam UDERA 
 * @tparam UDERB 
 * @param A 
 * @param B 
 * @param clearA 
 * @param clearB 
 * @return SpParMat1D<IU, NUO, UDERO> 
 */
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
SpParMat1D<IU, NUO, UDERO> Mult_AnXBn_1D_CbC_RDMA_FetchAll(SpParMat1D<IU,NU1,UDERA> & A, SpParMat1D<IU,NU2,UDERB> & B, bool clearA = false, bool clearB = false, int msggen=5)
{
    /* MPI basic info, timer setup */
    double prep1d = MPI_Wtime();
    double dataneeded(0.0), datafetched(0.0);
    int nprocs, myrank;
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    /* basic information, get local seq data. */
    IU mdim = A.getnrow();
    IU kdim = A.getncol();
    IU Brows = B.getnrow();
    if(kdim != Brows){
        printf("kdim not match!\n");
        SpParMat1D<IU, NUO, UDERO> C(mdim,kdim);
        return C;
    }
    IU ndim = B.getncol();
    UDERA * Ader = A.seqptr();
    UDERB * Bder = B.seqptr();
    std::vector<IU> blocksizeA = A.getblocksizevec();
    std::vector<IU> blocksizeprefixA = A.getblocksizeprefix();
    std::vector<IU> blocksizeB = B.getblocksizevec();
    std::vector<IU> blocksizeprefixB = B.getblocksizeprefix();
    std::vector<IU> rowblocksizeB = B.getrowblocksizevec();
    for(int i=0; i<rowblocksizeB.size();i++) {
        if(rowblocksizeB[i] != blocksizeA[i]){
            if(myrank==0) std::cerr << "B matrix rowblocksize is different from A blocksizevec" << 
            ", " << std::endl << "it's not a problem but please explcitly set it to be equal. " << std::endl;
            exit(0);
        }
    }
    
    /*********************************************************************************************
    Part 1: Generate Aneeded object
    we first get all cp and jc array from all mpi process, as this won't consume too much memory. 
    Then we can calculate the needed columns of A.
    Then we get ir and numx using RDMA API.
    **********************************************************************************************/
    
    IU totalnnz = 0, Aneedednzccnt = 0; // input params for Aneeded object.
    IU totalremotennz = 0;
    std::vector<IU> Aneededjc; // memory buffer for Aneeded DCSC jc array.
    std::vector<IU> Aneededcp; // memory buffer for Aneeded DCSC cp array.
    
    /* some aux array for fetching remote ir and numx */
    std::vector<IU> Aneededoffset(nprocs, 0); // offset of elements in needed ir/numx array
    std::vector<std::vector<IU>> AoriginoffsetPerRank(nprocs,std::vector<IU>()); // offset of elements in origin ir/numx array in per mpi rank
    std::vector<std::vector<IU>> AneededoffsetPerRank(nprocs,std::vector<IU>()); // offset of elements in needed ir/numx array in per mpi rank
    std::vector<std::vector<IU>> AneedednnzPerRank(nprocs,std::vector<IU>()); // number of elements in needed ir/numx array per column in per mpi rank
    
    /* send non-zero column index (cp) and size (jc) in csc to others. This is Expensive Call!!*/ 
    std::vector<std::vector<IU>> remoteAcp(nprocs,std::vector<IU>()), remoteAjc(nprocs,std::vector<IU>()); //! each vector represents one MPI process
    
    A.ExchangeDcscIndex(remoteAcp, remoteAjc); 
    /* project all non-zero elements along columns to a vector, also for rows */
    std::vector<int> projrow;
    Bder->ProjectRow(projrow);
    
    /* check outgoing row */
    /*
    IU outgoingrow = 0;
    for(int i=0; i<nprocs; i++){
        if(i == myrank) continue;
        IU js = blocksizeprefixA[i];
        IU je = i == nprocs-1 ? kdim : blocksizeprefixA[i+1];
        for(IU j= js; j< je; j++){
            if(projrow[j] >= 1){
                outgoingrow++;
            }
        }
    }
    std::cerr<<"myrank"<<myrank<<"outgoingrow"<<outgoingrow<<std::endl;
    */

    // if(myrank==0)std::cerr<<"[debug] in Spgemm1d gentype "<<msggen << std::endl;
    if(msggen==0){
        // get all nnz of A
        GenerateRDMACallAndMemOffset_FetchAll1Call(
            blocksizeprefixA,remoteAcp, remoteAjc,projrow,
            totalnnz,totalremotennz,Aneedednzccnt,Aneededcp, Aneededjc,
            Aneededoffset,AoriginoffsetPerRank,
            AneededoffsetPerRank,AneedednnzPerRank);
    }
    if(msggen==1){ /*looping to calculate memory reuqirement and offsets*/
        GenerateRDMACallAndMemOffset_FetchCbC(
            blocksizeprefixA,remoteAcp, remoteAjc,projrow,
            totalnnz,totalremotennz,Aneedednzccnt,Aneededcp, Aneededjc,
            Aneededoffset,AoriginoffsetPerRank,
            AneededoffsetPerRank,AneedednnzPerRank);
    }

    if(msggen==2){ /*looping to calculate memory reuqirement and offsets*/
        GenerateRDMACallAndMemOffset_MergeColumn(
            blocksizeprefixA,remoteAcp, remoteAjc,projrow,
            totalnnz,totalremotennz,Aneedednzccnt,Aneededcp, Aneededjc,
            Aneededoffset,AoriginoffsetPerRank,
            AneededoffsetPerRank,AneedednnzPerRank);
    }
    if(msggen==3){ /*looping to calculate memory reuqirement and offsets*/
        GenerateRDMACallAndMemOffset_FetchChunkV1(
            blocksizeprefixA,remoteAcp, remoteAjc, projrow,
            totalnnz,totalremotennz,Aneedednzccnt,Aneededcp, Aneededjc,
            Aneededoffset,AoriginoffsetPerRank,
            AneededoffsetPerRank,AneedednnzPerRank);
    }
    if(msggen==5){ /*looping to calculate memory reuqirement and offsets*/
        GenerateRDMACallAndMemOffset_FetchChunkV3(
            blocksizeprefixA,remoteAcp, remoteAjc, projrow,
            totalnnz,totalremotennz,Aneedednzccnt,Aneededcp, Aneededjc,
            Aneededoffset,AoriginoffsetPerRank,
            AneededoffsetPerRank,AneedednnzPerRank);
    }
    projrow.clear(); projrow.shrink_to_fit(); // clean memory
    
    /* construct SpDCCols object, mdim and ndim is the same as global matrix. */ 
    SpDCCols<IU, NU1> Aneeded(totalnnz, mdim, kdim, Aneedednzccnt);
    Dcsc<IU, NU1> * Aneededptr = Aneeded.GetDCSC();
    if(Aneededptr != nullptr){
        /*Fill cp */ 
        std::copy(Aneededcp.begin(), Aneededcp.end(), Aneededptr->cp);
        /*Fill jc */ 
        std::copy(Aneededjc.begin(), Aneededjc.end(), Aneededptr->jc);
        std::vector<IU>().swap(Aneededcp); // clean memory
        std::vector<IU>().swap(Aneededjc); // clean memory
    }
    /* construct window and attach memory buffer */
    Dcsc<IU, NU1> *dcscA = Ader->GetDCSC();
    IU *irptr   = dcscA == nullptr ? nullptr : dcscA->ir;
    NU1 *numptr = dcscA == nullptr ? nullptr : dcscA->numx;
    IU nz       = dcscA == nullptr ? 0       : dcscA->nz;
    

    /*tmp stats goes here, one should comment here for performant run*/ 
    // double Aneedmem = totalremotennz * (sizeof(IU)+sizeof(NU1)) * 1e-6;
    // double gAmem = A.getnnz() * (sizeof(IU)+sizeof(NU1)) * 1e-6 ;
    // gAmem = Aneedmem / gAmem * 100.;
    // int tmprdmacalls=0;
    // for(int i=0; i<nprocs; i++){
    //     if(i != myrank) tmprdmacalls += AneedednnzPerRank[i].size();
    // } 
    // char tmpss[2000];
    // if(myrank!=0)sprintf(tmpss, "Rank %-3d  %6.3f  %9.3f%%   %4d\n", myrank, Aneedmem, gAmem, tmprdmacalls);
    // if(myrank ==0){
    //     char ttss[2000];
    //     sprintf(tmpss, "Rank     Afetched(MB) Percent rdmacalls\n");
    //     sprintf(ttss, "Rank %-3d  %6.3f  %9.3f%%   %4d\n", myrank, Aneedmem, gAmem, tmprdmacalls);
    //     strcat(tmpss, ttss);
    // }
    // SpParHelper::EveryonePrint(std::string(tmpss), comm);
    if(opts.memorylimit > 0 && totalnnz * (sizeof(IU)+sizeof(NU1)) * 1e-9 > 0.8 * opts.memorylimit ){
        std::cerr << "myrank " << myrank << " my requested A is larger than 80\% of memory limit" << 
        opts.memorylimit << " GB." << std::endl;
    }
    prep1d = MPI_Wtime() - prep1d;
    /*********************************************************************************************
    Part 2: RDMA Operation, Fetch Remote A
    **********************************************************************************************/
    
    double comm1d = MPI_Wtime();
    std::vector<double> rdmatime(nprocs,0);
    #ifdef SKIP_COMM
    #else
    MPI_PassiveWindow<IU> *Airwins = new MPI_PassiveWindow<IU>(irptr, nz);
    MPI_PassiveWindow<NU1> *Anumwins = new MPI_PassiveWindow<NU1>(numptr, nz);
    // Airwins->LockAll();
    // Anumwins->LockAll();
    /* Fetch required ir/numx memory buffer from remote process */
    int targetrank = (myrank + 1) % nprocs; // shifting, make sure we request from different rank.
    int round = 0;
    while(round < nprocs){
        IU curnzccnt = AneedednnzPerRank[targetrank].size();
        if(curnzccnt != 0) {
            if(targetrank != myrank){
                double t1 = MPI_Wtime();
                Airwins->Lock(targetrank, MPI_OneSided_LockType::SHARED);
                Anumwins->Lock(targetrank, MPI_OneSided_LockType::SHARED);
                for(int idx=0; idx<curnzccnt; idx++){
                    IU localbuffoffset = Aneededoffset[targetrank] + AneededoffsetPerRank[targetrank][idx];
                    IU size = AneedednnzPerRank[targetrank][idx];
                    IU remotel = remoteAcp[targetrank].back();
                    IU remotedisp = AoriginoffsetPerRank[targetrank][idx];
                    Airwins->Get(
                        Aneededptr->ir + localbuffoffset, remotedisp, size,targetrank);
                    Anumwins->Get(
                        Aneededptr->numx + localbuffoffset,remotedisp, size, targetrank);
                }
                Airwins->Unlock(targetrank);
                Anumwins->Unlock(targetrank);
                t1 = MPI_Wtime() - t1;
                rdmatime[targetrank] = t1;
            }else{
                double t1 = MPI_Wtime();
                for(int idx=0; idx < curnzccnt; idx++){
                    std::copy(
                        dcscA->ir + AoriginoffsetPerRank[targetrank][idx], 
                        dcscA->ir + AoriginoffsetPerRank[targetrank][idx] + AneedednnzPerRank[targetrank][idx],
                        Aneededptr->ir + Aneededoffset[targetrank] + AneededoffsetPerRank[targetrank][idx]
                    );
                    std::copy(
                        dcscA->numx + AoriginoffsetPerRank[targetrank][idx], 
                        dcscA->numx + AoriginoffsetPerRank[targetrank][idx] + AneedednnzPerRank[targetrank][idx],
                        Aneededptr->numx + Aneededoffset[targetrank] + AneededoffsetPerRank[targetrank][idx]
                    );
                }
                t1 = MPI_Wtime() - t1;
                rdmatime[myrank] = t1;
            }
        }
        targetrank = (targetrank + 1) % nprocs; // prepare for next target rank
        round++;
    }
    // Airwins->UnlockAll();
    // Anumwins->UnlockAll();
    delete Airwins;
    delete Anumwins;
    #endif

    if(opts.comm_ana){
        // before delete memory, do records
        // AnnedednnzPerRank is the most detailed data structure.
        std::vector<int> rdmacalls(nprocs,0);
        for(int i=0; i<nprocs;i++){
            rdmacalls[i] = AneedednnzPerRank[i].size();
        }
        std::vector<double> rdmamems(nprocs,0);
        for(int i=0; i<nprocs;i++){
            IU totnnz = 0;
            for(auto xxx : AneedednnzPerRank[i]) totnnz += xxx;
            // std::cerr << "totnnz " << totnnz << std::endl;
            rdmamems[i] = totnnz * ( sizeof(IU) + sizeof(NUO) ) * 1.0e-6 ;
        }
        // write rdma calls
        std::vector<int> rdmatotcalls;
        SpParHelper::AllgatherVector(rdmatotcalls, rdmacalls);
        // write rdma mem
        std::vector<double> rdmatotmems;
        SpParHelper::AllgatherVector(rdmatotmems, rdmamems);
        std::vector<double> rdmatottime;
        SpParHelper::AllgatherVector(rdmatottime, rdmatime);
        if(myrank==0){
            std::cerr << "write comm analysis to file"<<std::endl;
            std::string testprefix = opts.testprefix;
            std::string filename = "rdmacalls" + testprefix +".bin";
            SpHelper::RemoveExistingFile(filename);
            std::ofstream binfs(filename, std::ios::binary);
            binfs.write((char*)rdmatotcalls.data(), sizeof(int)*rdmatotcalls.size());
            binfs.close();
            filename =  "rdmamems" + testprefix + ".bin";
            SpHelper::RemoveExistingFile(filename);
            binfs = std::ofstream(filename, std::ios::binary);
            binfs.write((char*)rdmatotmems.data(), sizeof(double)*rdmatotmems.size());
            binfs.close();
            filename =  "rdmatimes" + testprefix + ".bin";
            SpHelper::RemoveExistingFile(filename);
            binfs = std::ofstream(filename, std::ios::binary);
            binfs.write((char*)rdmatottime.data(), sizeof(double)*rdmatottime.size());
            binfs.close();
        }
        // a very detailed analysis, write each call and size from every process
        // every write two x nprocs file. 1st file is `nprocess` int file .
        // 2nd file is a K double file. where sum of the `nprocs` int is K.
    }

    std::vector<IU>().swap(Aneededoffset); // clean memory
    std::vector<std::vector<IU>>().swap(AneededoffsetPerRank); // clean memory
    std::vector<std::vector<IU>>().swap(AoriginoffsetPerRank); // clean memory
    std::vector<std::vector<IU>>().swap(AneedednnzPerRank); // clean memory
    comm1d = MPI_Wtime() - comm1d;

    #ifdef SKIP_COMP
    SpParMat1D<IU, NUO, UDERO> C(A);
    return C;
    #else
    /******************************************************************
    Part 2: Launch SpGEMM and get C matrix
    *******************************************************************/
    // MPI_Barrier(comm);
    double comp1d = MPI_Wtime();
    SpTuples<IU, NUO> *sptuples = LocalHybridSpGEMM<SR,NUO>(Aneeded, *Bder, false, false);
    comp1d = MPI_Wtime() - comp1d;
    double others = MPI_Wtime();
    UDERO *Cder = new UDERO(*sptuples, false);
    // double o1 = MPI_Wtime();
    // if(myrank==0)std::cerr<<"P1 time:"<<o1-others<<std::endl;
    // SpParMat1D<IU, NUO, UDERO> C(mdim,ndim,B.getblocksizevec(), Cder, A.getrowblocksizevec());
    SpParMat1D<IU, NUO, UDERO> C(mdim,ndim,B.getblocksizevec(), Cder, A.getsharedgrid(), A.getrowblocksizevec());
    // double o2 = MPI_Wtime();
    // if(myrank==0)std::cerr<<"P2 time:"<<o2-o1<<std::endl;
    // delete sptuples;
    // double o3 = MPI_Wtime();
    // if(myrank==0)std::cerr<<"P3 time:"<<o3-o2<<std::endl;
    others = MPI_Wtime() - others;
    // if(myrank==0)std::cerr<<"Total other time:"<<others<<std::endl;
    double statstart = MPI_Wtime();
    perfcnt1d.doublemap["prepT(ms)"] = prep1d;
    perfcnt1d.doublemap["commT(ms)"] = comm1d;
    perfcnt1d.doublemap["compT(ms)"] = comp1d;
    perfcnt1d.doublemap["otherT(ms)"] = others;
    perfcnt1d.integermap["Alocalnnz"] = Ader->getnnz();
    perfcnt1d.doublemap["Alocalmem(MB)"] = Ader->getnnz() * (sizeof(IU) + sizeof(NU1)) * 1e-6;
    perfcnt1d.integermap["Aneednnz"] = Aneeded.getnnz();
    perfcnt1d.doublemap["Aneedmem(MB)"] = Aneeded.getnnz() * (sizeof(IU) + sizeof(NU1)) * 1e-6;
    perfcnt1d.integermap["Bnnz    "] = Bder->getnnz();
    perfcnt1d.doublemap["Bmem(MB)"] = Bder->getnnz() * (sizeof(IU) + sizeof(NU1)) * 1e-6;
    perfcnt1d.integermap["Cnnz    "] = Cder->getnnz();
    perfcnt1d.doublemap["Cmem(MB)"] = Cder->getnnz() * (sizeof(IU) + sizeof(NU1)) * 1e-6;
    double statsends = MPI_Wtime();
    double stattime  = statsends - statstart;
    double tottime = stattime + others + comp1d + prep1d + comm1d;
    double statpct = stattime / tottime;
    double gtot;
    perfcnt1d.doublemap["gtot(s)"] = gtot;
    perfcnt1d.doublemap["stat(ms)"] = stattime * 1e3;
    return C;
    #endif
    
}



/**
 * @brief 1D column-by-column SpGEMM algorithm based on RDMA, fetch all required data from remote process in one time.
 * we first generate the needed columns from the A matrix, and request the data frome remote process. 
 * After we have everything in local, we call localSpGEMM directly.
 * - It's possible that blocksizevec of A is different from rowblocksize of B. It's not an issue because we won't use 
 * it. But when we analyze the communication, maybe it's an issue.
 * @tparam SR 
 * @tparam NUO 
 * @tparam UDERO 
 * @tparam IU 
 * @tparam NU1 
 * @tparam NU2 
 * @tparam UDERA 
 * @tparam UDERB 
 * @param A 
 * @param B 
 * @param clearA 
 * @param clearB 
 * @return SpParMat1D<IU, NUO, UDERO> 
 */
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
SpParMat1D<IU, NUO, UDERO> Mult_AnXBn_1D_CbC_RDMA_FetchAllV2(SpParMat1D<IU,NU1,UDERA> & A, SpParMat1D<IU,NU2,UDERB> & B, bool clearA = false, bool clearB = false, int msggen=5)
{
    /* MPI basic info, timer setup */
    double prep1d = MPI_Wtime();
    double dataneeded(0.0), datafetched(0.0);
    int nprocs, myrank;
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    /* basic information, get local seq data. */
    IU mdim = A.getnrow();
    IU kdim = A.getncol();
    IU Brows = B.getnrow();
    if(kdim != Brows){
        printf("kdim not match!\n");
        SpParMat1D<IU, NUO, UDERO> C(mdim,kdim);
        return C;
    }
    IU ndim = B.getncol();
    UDERA * Ader = A.seqptr();
    UDERB * Bder = B.seqptr();
    std::vector<IU> blocksizeA = A.getblocksizevec();
    std::vector<IU> blocksizeprefixA = A.getblocksizeprefix();
    std::vector<IU> blocksizeB = B.getblocksizevec();
    std::vector<IU> blocksizeprefixB = B.getblocksizeprefix();
    std::vector<IU> rowblocksizeB = B.getrowblocksizevec();
    for(int i=0; i<rowblocksizeB.size();i++) {
        if(rowblocksizeB[i] != blocksizeA[i]){
            if(myrank==0) std::cerr << "B matrix rowblocksize is different from A blocksizevec" << 
            ", " << std::endl << "it's not a problem but please explcitly set it to be equal. " << std::endl;
            exit(0);
        }
    }
    
    /*********************************************************************************************
    Part 1: Generate Aneeded object
    we first get all cp and jc array from all mpi process, as this won't consume too much memory. 
    Then we can calculate the needed columns of A.
    Then we get ir and numx using RDMA API.
    **********************************************************************************************/
    
    IU totalnnz = 0, Aneedednzccnt = 0; // input params for Aneeded object.
    IU totalremotennz = 0;
    std::vector<IU> Aneededjc; // memory buffer for Aneeded DCSC jc array.
    std::vector<IU> Aneededcp; // memory buffer for Aneeded DCSC cp array.
    
    /* some aux array for fetching remote ir and numx */
    std::vector<IU> Aneededoffset(nprocs, 0); // offset of elements in needed ir/numx array
    std::vector<std::vector<IU>> AoriginoffsetPerRank(nprocs,std::vector<IU>()); // offset of elements in origin ir/numx array in per mpi rank
    std::vector<std::vector<IU>> AneededoffsetPerRank(nprocs,std::vector<IU>()); // offset of elements in needed ir/numx array in per mpi rank
    std::vector<std::vector<IU>> AneedednnzPerRank(nprocs,std::vector<IU>()); // number of elements in needed ir/numx array per column in per mpi rank
    
    /* send non-zero column index (cp) and size (jc) in csc to others. This is Expensive Call!!*/ 
    std::vector<std::vector<IU>> remoteAcp(nprocs,std::vector<IU>()), remoteAjc(nprocs,std::vector<IU>()); //! each vector represents one MPI process
    int targetrank;
    int round;
    MPI_Barrier(comm);
    double t1 = MPI_Wtime();
    // A.ExchangeDcscIndex(remoteAcp, remoteAjc); 
    // std::cerr << "mylocal nzc "<< remoteAjc[myrank].size() << std::endl;
    IU localcpsize = 0, localjcsize = 0, localnzc = 0;
    IU * cpptr = nullptr, * jcptr = nullptr;
    if(Ader->getnnz() != 0){
        Dcsc<IU, NU1> * dcscA =  Ader->GetDCSC(); 
        // localcpsize = dcscA->nzc+1;
        // localjcsize = dcscA->nzc;
        localnzc = dcscA->nzc;
        cpptr = dcscA->cp;
        jcptr = dcscA->jc;
    }
    std::vector<IU> globalnzcvec(nprocs);
    MPI_Allgather(
        &localnzc, 1, MPI_LONG_LONG, 
        globalnzcvec.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
    for(int i=0; i<nprocs; i++){
        remoteAcp[i].resize(globalnzcvec[i]+1);
        remoteAjc[i].resize(globalnzcvec[i]);
    }
    MPI_PassiveWindow<IU> *Acpwins = new MPI_PassiveWindow<IU>(cpptr, localnzc+1);
    MPI_PassiveWindow<IU> *Ajcwins = new MPI_PassiveWindow<IU>(jcptr, localnzc);
    Acpwins->LockAll();
    Ajcwins->LockAll();
    targetrank = (myrank+1) % nprocs;
    round = 0;
    while(round < nprocs){
        if(globalnzcvec[targetrank] == 0){
            round++;
            targetrank = (targetrank + 1) % nprocs; // prepare for next target rank
            continue;
        }

        if(targetrank != myrank){
            // Acpwins->Lock(targetrank, MPI_OneSided_LockType::SHARED);
            // Ajcwins->Lock(targetrank, MPI_OneSided_LockType::SHARED);
            Acpwins->Get(remoteAcp[targetrank].data(), 0, globalnzcvec[targetrank]+1, targetrank);
            Ajcwins->Get(remoteAjc[targetrank].data(), 0, globalnzcvec[targetrank], targetrank);
            // Acpwins->Unlock(targetrank);
            // Ajcwins->Unlock(targetrank);
        }else{
            std::copy(cpptr, cpptr + globalnzcvec[targetrank] + 1, remoteAcp[targetrank].begin());
            std::copy(jcptr, jcptr + globalnzcvec[targetrank]    , remoteAjc[targetrank].begin());
        }
        round++;
        targetrank = (targetrank + 1) % nprocs; // prepare for next target rank
    }
    Acpwins->UnlockAll();
    Ajcwins->UnlockAll();
    delete Acpwins;
    delete Ajcwins;
    double t0 = MPI_Wtime();
    std::cerr << "t0-t1" << t0-t1<<std::endl;
    /* project all non-zero elements along columns to a vector, also for rows */
    std::vector<int> projrow;
    Bder->ProjectRow(projrow);
    
    /* check outgoing row */
    /*
    IU outgoingrow = 0;
    for(int i=0; i<nprocs; i++){
        if(i == myrank) continue;
        IU js = blocksizeprefixA[i];
        IU je = i == nprocs-1 ? kdim : blocksizeprefixA[i+1];
        for(IU j= js; j< je; j++){
            if(projrow[j] >= 1){
                outgoingrow++;
            }
        }
    }
    std::cerr<<"myrank"<<myrank<<"outgoingrow"<<outgoingrow<<std::endl;
    */

    // if(myrank==0)std::cerr<<"[debug] in Spgemm1d gentype "<<msggen << std::endl;
    if(msggen==0){
        // get all nnz of A
        GenerateRDMACallAndMemOffset_FetchAll1Call(
            blocksizeprefixA,remoteAcp, remoteAjc,projrow,
            totalnnz,totalremotennz,Aneedednzccnt,Aneededcp, Aneededjc,
            Aneededoffset,AoriginoffsetPerRank,
            AneededoffsetPerRank,AneedednnzPerRank);
    }
    if(msggen==1){ /*looping to calculate memory reuqirement and offsets*/
        GenerateRDMACallAndMemOffset_FetchCbC(
            blocksizeprefixA,remoteAcp, remoteAjc,projrow,
            totalnnz,totalremotennz,Aneedednzccnt,Aneededcp, Aneededjc,
            Aneededoffset,AoriginoffsetPerRank,
            AneededoffsetPerRank,AneedednnzPerRank);
    }

    if(msggen==2){ /*looping to calculate memory reuqirement and offsets*/
        GenerateRDMACallAndMemOffset_MergeColumn(
            blocksizeprefixA,remoteAcp, remoteAjc,projrow,
            totalnnz,totalremotennz,Aneedednzccnt,Aneededcp, Aneededjc,
            Aneededoffset,AoriginoffsetPerRank,
            AneededoffsetPerRank,AneedednnzPerRank);
    }
    if(msggen==3){ /*looping to calculate memory reuqirement and offsets*/
        GenerateRDMACallAndMemOffset_FetchChunkV1(
            blocksizeprefixA,remoteAcp, remoteAjc, projrow,
            totalnnz,totalremotennz,Aneedednzccnt,Aneededcp, Aneededjc,
            Aneededoffset,AoriginoffsetPerRank,
            AneededoffsetPerRank,AneedednnzPerRank);
    }
    if(msggen==5){ /*looping to calculate memory reuqirement and offsets*/
        GenerateRDMACallAndMemOffset_FetchChunkV3(
            blocksizeprefixA,remoteAcp, remoteAjc, projrow,
            totalnnz,totalremotennz,Aneedednzccnt,Aneededcp, Aneededjc,
            Aneededoffset,AoriginoffsetPerRank,
            AneededoffsetPerRank,AneedednnzPerRank);
    }
    projrow.clear(); projrow.shrink_to_fit(); // clean memory
    
    /* construct SpDCCols object, mdim and ndim is the same as global matrix. */ 
    SpDCCols<IU, NU1> Aneeded(totalnnz, mdim, kdim, Aneedednzccnt);
    Dcsc<IU, NU1> * Aneededptr = Aneeded.GetDCSC();
    if(Aneededptr != nullptr){
        /*Fill cp */ 
        std::copy(Aneededcp.begin(), Aneededcp.end(), Aneededptr->cp);
        /*Fill jc */ 
        std::copy(Aneededjc.begin(), Aneededjc.end(), Aneededptr->jc);
        std::vector<IU>().swap(Aneededcp); // clean memory
        std::vector<IU>().swap(Aneededjc); // clean memory
    }
    /* construct window and attach memory buffer */
    Dcsc<IU, NU1> *dcscA = Ader->GetDCSC();
    IU *irptr   = dcscA == nullptr ? nullptr : dcscA->ir;
    NU1 *numptr = dcscA == nullptr ? nullptr : dcscA->numx;
    IU nz       = dcscA == nullptr ? 0       : dcscA->nz;
    

    /*tmp stats goes here, one should comment here for performant run*/ 
    // double Aneedmem = totalremotennz * (sizeof(IU)+sizeof(NU1)) * 1e-6;
    // double gAmem = A.getnnz() * (sizeof(IU)+sizeof(NU1)) * 1e-6 ;
    // gAmem = Aneedmem / gAmem * 100.;
    // int tmprdmacalls=0;
    // for(int i=0; i<nprocs; i++){
    //     if(i != myrank) tmprdmacalls += AneedednnzPerRank[i].size();
    // } 
    // char tmpss[2000];
    // if(myrank!=0)sprintf(tmpss, "Rank %-3d  %6.3f  %9.3f%%   %4d\n", myrank, Aneedmem, gAmem, tmprdmacalls);
    // if(myrank ==0){
    //     char ttss[2000];
    //     sprintf(tmpss, "Rank     Afetched(MB) Percent rdmacalls\n");
    //     sprintf(ttss, "Rank %-3d  %6.3f  %9.3f%%   %4d\n", myrank, Aneedmem, gAmem, tmprdmacalls);
    //     strcat(tmpss, ttss);
    // }
    // SpParHelper::EveryonePrint(std::string(tmpss), comm);
    if(opts.memorylimit > 0 && totalnnz * (sizeof(IU)+sizeof(NU1)) * 1e-9 > 0.8 * opts.memorylimit ){
        std::cerr << "myrank " << myrank << " my requested A is larger than 80\% of memory limit" << 
        opts.memorylimit << " GB." << std::endl;
    }
    prep1d = MPI_Wtime() - prep1d;
    /*********************************************************************************************
    Part 2: RDMA Operation, Fetch Remote A
    **********************************************************************************************/
    
    double comm1d = MPI_Wtime();
    std::vector<double> rdmatime(nprocs,0);
    #ifdef SKIP_COMM
    #else
    MPI_PassiveWindow<IU> *Airwins = new MPI_PassiveWindow<IU>(irptr, nz);
    MPI_PassiveWindow<NU1> *Anumwins = new MPI_PassiveWindow<NU1>(numptr, nz);
    // Airwins->LockAll();
    // Anumwins->LockAll();
    /* Fetch required ir/numx memory buffer from remote process */
    targetrank = (myrank + 1) % nprocs; // shifting, make sure we request from different rank.
    round = 0;
    while(round < nprocs){
        IU curnzccnt = AneedednnzPerRank[targetrank].size();
        if(curnzccnt != 0) {
            if(targetrank != myrank){
                double t1 = MPI_Wtime();
                Airwins->Lock(targetrank, MPI_OneSided_LockType::SHARED);
                Anumwins->Lock(targetrank, MPI_OneSided_LockType::SHARED);
                for(int idx=0; idx<curnzccnt; idx++){
                    IU localbuffoffset = Aneededoffset[targetrank] + AneededoffsetPerRank[targetrank][idx];
                    IU size = AneedednnzPerRank[targetrank][idx];
                    IU remotel = remoteAcp[targetrank].back();
                    IU remotedisp = AoriginoffsetPerRank[targetrank][idx];
                    Airwins->Get(
                        Aneededptr->ir + localbuffoffset, remotedisp, size,targetrank);
                    Anumwins->Get(
                        Aneededptr->numx + localbuffoffset,remotedisp, size, targetrank);
                }
                Airwins->Unlock(targetrank);
                Anumwins->Unlock(targetrank);
                t1 = MPI_Wtime() - t1;
                rdmatime[targetrank] = t1;
            }else{
                double t1 = MPI_Wtime();
                for(int idx=0; idx < curnzccnt; idx++){
                    std::copy(
                        dcscA->ir + AoriginoffsetPerRank[targetrank][idx], 
                        dcscA->ir + AoriginoffsetPerRank[targetrank][idx] + AneedednnzPerRank[targetrank][idx],
                        Aneededptr->ir + Aneededoffset[targetrank] + AneededoffsetPerRank[targetrank][idx]
                    );
                    std::copy(
                        dcscA->numx + AoriginoffsetPerRank[targetrank][idx], 
                        dcscA->numx + AoriginoffsetPerRank[targetrank][idx] + AneedednnzPerRank[targetrank][idx],
                        Aneededptr->numx + Aneededoffset[targetrank] + AneededoffsetPerRank[targetrank][idx]
                    );
                }
                t1 = MPI_Wtime() - t1;
                rdmatime[myrank] = t1;
            }
        }
        targetrank = (targetrank + 1) % nprocs; // prepare for next target rank
        round++;
    }
    // Airwins->UnlockAll();
    // Anumwins->UnlockAll();
    delete Airwins;
    delete Anumwins;
    #endif

    if(opts.comm_ana){
        // before delete memory, do records
        // AnnedednnzPerRank is the most detailed data structure.
        std::vector<int> rdmacalls(nprocs,0);
        for(int i=0; i<nprocs;i++){
            rdmacalls[i] = AneedednnzPerRank[i].size();
        }
        std::vector<double> rdmamems(nprocs,0);
        for(int i=0; i<nprocs;i++){
            IU totnnz = 0;
            for(auto xxx : AneedednnzPerRank[i]) totnnz += xxx;
            // std::cerr << "totnnz " << totnnz << std::endl;
            rdmamems[i] = totnnz * ( sizeof(IU) + sizeof(NUO) ) * 1.0e-6 ;
        }
        // write rdma calls
        std::vector<int> rdmatotcalls;
        SpParHelper::AllgatherVector(rdmatotcalls, rdmacalls);
        // write rdma mem
        std::vector<double> rdmatotmems;
        SpParHelper::AllgatherVector(rdmatotmems, rdmamems);
        std::vector<double> rdmatottime;
        SpParHelper::AllgatherVector(rdmatottime, rdmatime);
        if(myrank==0){
            std::cerr << "write comm analysis to file"<<std::endl;
            std::string testprefix = opts.testprefix;
            std::string filename = "rdmacalls" + testprefix +".bin";
            SpHelper::RemoveExistingFile(filename);
            std::ofstream binfs(filename, std::ios::binary);
            binfs.write((char*)rdmatotcalls.data(), sizeof(int)*rdmatotcalls.size());
            binfs.close();
            filename =  "rdmamems" + testprefix + ".bin";
            SpHelper::RemoveExistingFile(filename);
            binfs = std::ofstream(filename, std::ios::binary);
            binfs.write((char*)rdmatotmems.data(), sizeof(double)*rdmatotmems.size());
            binfs.close();
            filename =  "rdmatimes" + testprefix + ".bin";
            SpHelper::RemoveExistingFile(filename);
            binfs = std::ofstream(filename, std::ios::binary);
            binfs.write((char*)rdmatottime.data(), sizeof(double)*rdmatottime.size());
            binfs.close();
        }
        // a very detailed analysis, write each call and size from every process
        // every write two x nprocs file. 1st file is `nprocess` int file .
        // 2nd file is a K double file. where sum of the `nprocs` int is K.
    }

    std::vector<IU>().swap(Aneededoffset); // clean memory
    std::vector<std::vector<IU>>().swap(AneededoffsetPerRank); // clean memory
    std::vector<std::vector<IU>>().swap(AoriginoffsetPerRank); // clean memory
    std::vector<std::vector<IU>>().swap(AneedednnzPerRank); // clean memory
    comm1d = MPI_Wtime() - comm1d;

    #ifdef SKIP_COMP
    SpParMat1D<IU, NUO, UDERO> C(A);
    return C;
    #else
    /******************************************************************
    Part 2: Launch SpGEMM and get C matrix
    *******************************************************************/
    // MPI_Barrier(comm);
    double comp1d = MPI_Wtime();
    SpTuples<IU, NUO> *sptuples = LocalHybridSpGEMM<SR,NUO>(Aneeded, *Bder, false, false);
    comp1d = MPI_Wtime() - comp1d;
    double others = MPI_Wtime();
    UDERO *Cder = new UDERO(*sptuples, false);
    // double o1 = MPI_Wtime();
    // if(myrank==0)std::cerr<<"P1 time:"<<o1-others<<std::endl;
    // SpParMat1D<IU, NUO, UDERO> C(mdim,ndim,B.getblocksizevec(), Cder, A.getrowblocksizevec());
    SpParMat1D<IU, NUO, UDERO> C(mdim,ndim,B.getblocksizevec(), Cder, A.getsharedgrid(), A.getrowblocksizevec());
    // double o2 = MPI_Wtime();
    // if(myrank==0)std::cerr<<"P2 time:"<<o2-o1<<std::endl;
    // delete sptuples;
    // double o3 = MPI_Wtime();
    // if(myrank==0)std::cerr<<"P3 time:"<<o3-o2<<std::endl;
    others = MPI_Wtime() - others;
    // if(myrank==0)std::cerr<<"Total other time:"<<others<<std::endl;
    double statstart = MPI_Wtime();
    perfcnt1d.doublemap["prepT(ms)"] = prep1d;
    perfcnt1d.doublemap["commT(ms)"] = comm1d;
    perfcnt1d.doublemap["compT(ms)"] = comp1d;
    perfcnt1d.doublemap["otherT(ms)"] = others;
    perfcnt1d.integermap["Alocalnnz"] = Ader->getnnz();
    perfcnt1d.doublemap["Alocalmem(MB)"] = Ader->getnnz() * (sizeof(IU) + sizeof(NU1)) * 1e-6;
    perfcnt1d.integermap["Aneednnz"] = Aneeded.getnnz();
    perfcnt1d.doublemap["Aneedmem(MB)"] = Aneeded.getnnz() * (sizeof(IU) + sizeof(NU1)) * 1e-6;
    perfcnt1d.integermap["Bnnz    "] = Bder->getnnz();
    perfcnt1d.doublemap["Bmem(MB)"] = Bder->getnnz() * (sizeof(IU) + sizeof(NU1)) * 1e-6;
    perfcnt1d.integermap["Cnnz    "] = Cder->getnnz();
    perfcnt1d.doublemap["Cmem(MB)"] = Cder->getnnz() * (sizeof(IU) + sizeof(NU1)) * 1e-6;
    double statsends = MPI_Wtime();
    double stattime  = statsends - statstart;
    double tottime = stattime + others + comp1d + prep1d + comm1d;
    double statpct = stattime / tottime;
    double gtot;
    perfcnt1d.doublemap["gtot(s)"] = gtot;
    perfcnt1d.doublemap["stat(ms)"] = stattime * 1e3;
    return C;
    #endif
    
}

template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
void PSpGEMM(SpParMat1D<IU,NU1,UDER1> & A, SpParMat1D<IU,NU2,UDER2> & B, SpParMat1D<IU,NUO,UDERO> & out, bool clearA = false, bool clearB = false)
{
    out = Mult_AnXBn_1D_CbC_RDMA_FetchAll<SR, NUO, UDERO> (A, B, clearA, clearB );
}

template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
SpParMat1D<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER2,UDER2>::T_promote>
    PSpGEMM(SpParMat1D<IU,NU1,UDER1> & A, SpParMat1D<IU,NU2,UDER2> & B, bool clearA = false, bool clearB = false)
{
    typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
    typedef typename promote_trait<UDER1,UDER2>::T_promote DER_promote;
    return Mult_AnXBn_1D_CbC_RDMA_FetchAll<SR, N_promote, DER_promote> (A, B, clearA, clearB );
}



#define SCALAREQCHECK(a,b,hint) \
if((a) != (b)) { \
    if(myrank == 0){ \
        printf(hint); \
        fflush(stdout); \
    } \
    exit(0); \
}

#define VECTOREQCHECK(a,b,hint) \
for(int vi=0; vi < a.size(); vi++) { \
    if(a[vi] != b[vi]){ \
        if(myrank == 0){ \
            printf(hint); \
            fflush(stdout); \
        } \
    } \
    exit(0); \
}



// Block Diagonal Implementation
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
SpParMat1D<IU, NUO, UDERO> Mult_AnXBn_1D_BD(SpParMat1D<IU,NU1,UDERA> & A, SpParMat1D<IU,NU2,UDERB> & B, bool clearA = false, bool clearB = false )
{

    /*get basic info.*/
    // A and B should all be column split.
    auto grid1d = A.getgrid();
    int nprocs = grid1d->GetSize();
    int myrank = grid1d->GetRank();
    IU mdim = A.getnrow();
    IU kdim = A.getncol(); /*dim check*/ SCALAREQCHECK(kdim, B.getnrow(), "kdim mismatch!\n")
    IU ndim = B.getncol();

    // get all block dimension.
    std::vector<IU> blocksizeA = A.getblocksizevec(); 
    std::vector<IU> blocksizeprefixA = A.getblocksizeprefix(); 
    std::vector<IU> blocksizeB = B.getblocksizevec(); 
    std::vector<IU> blocksizeprefixB = B.getblocksizeprefix();
    std::vector<IU> rowblocksizeA = A.getrowblocksizevec();
    std::vector<IU> rowblocksizeprefixA = A.getrowblocksizeprefix();
    std::vector<IU> rowblocksizeB = B.getrowblocksizevec(); 
    // VECTOREQCHECK(blocksizeA, rowblocksizeB, "bsA and rbsB mismatch!\n")
    std::vector<IU> rowblocksizeprefixB = B.getrowblocksizeprefix();
    
    // tmp pointers
    std::tuple<IU,IU,NU1> *tmpAtuples;
    std::tuple<IU,IU,NU2> *tmpBtuples;
    std::tuple<IU,IU,NUO> *tmpCtuples;
    SpTuples<IU, NU1> *tmpAsptuples;
    SpTuples<IU, NU2> *tmpBsptuples;
    SpTuples<IU, NUO> *tmpCsptuples;

    UDERA *adiagDER, *aoffdiagDER;
    UDERB *bdiagDER, *boffdiagDER;
    UDERO *C_S1D1D2DER, *C_D1S2DER, *C_S1S2DER;

    UDERA * spSeqA = A.seqptr();
    UDERB * spSeqB = B.seqptr();


    MPI_Timer timer;
    timer.start("part1");
    /*Part 1: (S1+D1) * D2, S1+D1 IS original local A data*/
    std::vector<std::tuple<IU,IU,NU2>> bdiag;
    std::vector<std::tuple<IU,IU,NU2>> boffdiag;
    IU bdiagrowstart = rowblocksizeprefixB[myrank];
    IU bdiagrowend = myrank == nprocs - 1 ? kdim : rowblocksizeprefixB[myrank+1];

    //TODO: Let's first do not use MPI Onesided comm. For simplicity.
    // bool boffdiagprojvec[kdim]; // offdiagonal projection of local B matrix.  This represents which row of A we should request.
    // #pragma omp parallel for schedule(static, 64)
    // for(IU i=0; i<kdim; i++) boffdiagprojvec[i]=false;
    std::vector<std::vector<std::tuple<IU,IU,NU2>>> sendoffbtuples(nprocs);
    for(typename UDERB::SpColIter colit = spSeqB->begcol(); colit != spSeqB->endcol(); ++colit)
    {
        IU gcol = colit.colid() + blocksizeprefixB[myrank];
        for(typename UDERB::SpColIter::NzIter nzit = spSeqB->begnz(colit); nzit != spSeqB->endnz(colit); ++nzit)
        {
            IU grow = nzit.rowid();
            if(grow >= bdiagrowstart && grow < bdiagrowend){
                bdiag.push_back(std::make_tuple(grow - bdiagrowstart, colit.colid(), nzit.value()));
            }else{ // offdiag elems
                // boffdiagprojvec[grow] = true; // projection 
                // send offdiag for part 2
                int owner = 0;
                while(owner <= nprocs-1 && rowblocksizeprefixB[owner] <= grow) owner++;
                owner--;
                sendoffbtuples[owner].push_back(std::make_tuple(grow-rowblocksizeprefixB[owner],gcol,nzit.value()));
                boffdiag.push_back(std::make_tuple(grow, colit.colid(),nzit.value()));
            }
        }
    }

    tmpBsptuples = new SpTuples<IU, NU2>(bdiag.size(),rowblocksizeB[myrank],blocksizeB[myrank],bdiag.data(),true);
    tmpBsptuples->tuples_deleted = true;
    bdiagDER = new UDERB(*tmpBsptuples,false);
    delete tmpBsptuples;
    std::vector<std::tuple<IU,IU,NU2>>().swap(bdiag); // clear bdiag memory
    tmpCsptuples = LocalHybridSpGEMM<SR, NUO, IU, NU1, NU2>(*spSeqA, *bdiagDER, false, false);
    C_S1D1D2DER = new UDERO(*tmpCsptuples,false); // output is part of C: (S1+D1) * D2
    delete tmpCsptuples;

    UDERB * BrowsDER;
    IU Browsdatasize;
    std::tuple<IU,IU,NU2> *recvoffbtuples = SpParHelper::ExchangeDataGeneral(sendoffbtuples, MPI_COMM_WORLD, Browsdatasize);
    tmpBsptuples = new SpTuples<IU, NU2>(Browsdatasize, rowblocksizeB[myrank], ndim, recvoffbtuples,true);
    BrowsDER = new UDERB(*tmpBsptuples, false); // cleared recvoffbtuples;
    delete tmpBsptuples;
    timer.stop("part1",true);
    timer.start("part2");
    /**************************************/
    // Part 2: D1 * S2
    /**************************************/
    // split A into diag and offdiag, this is local operations. just double the memory of A.
    std::vector<std::tuple<IU,IU,NU1>> adiag;
    std::vector<std::tuple<IU,IU,NU1>> aoffdiag;
    //!!! Here we distribute the data to a row wise split!
    IU adiagrowstart = rowblocksizeprefixA[myrank];
    IU adiagrowend = myrank == nprocs - 1 ? mdim : rowblocksizeprefixA[myrank+1];
    for(typename UDERA::SpColIter colit = spSeqA->begcol(); colit != spSeqA->endcol(); ++colit)
    {
        IU gcol = colit.colid() + blocksizeprefixA[myrank];
        for(typename UDERA::SpColIter::NzIter nzit = spSeqA->begnz(colit); nzit != spSeqA->endnz(colit); ++nzit)
        {
            IU grow = nzit.rowid();
            if(grow >= adiagrowstart && grow < adiagrowend){
                adiag.push_back(std::make_tuple(grow-adiagrowstart,colit.colid(),nzit.value()));
            }else{
                aoffdiag.push_back(std::make_tuple(grow,colit.colid(),nzit.value()));
            }
        }
    }

    tmpAsptuples = new SpTuples<IU,NU1>(adiag.size(), rowblocksizeA[myrank], blocksizeA[myrank], adiag.data(),true);
    tmpAsptuples->tuples_deleted = true;
    adiagDER = new UDERA(*tmpAsptuples,false);
    std::vector<std::tuple<IU,IU,NU2>>().swap(adiag);
    delete tmpAsptuples;
    UDERO *tmpder;
    
    tmpCsptuples = LocalHybridSpGEMM<SR, NUO>(*adiagDER, *BrowsDER, true, true); 
    tmpder = new UDERO(*tmpCsptuples, false);
    delete tmpCsptuples;
    
    std::vector<std::vector<std::tuple<IU,IU,NUO>>> offdiagrestuples(nprocs); // need to send the results back.
    //!!! Here we distribute the data BACK to a col wise split!
    for(typename UDERB::SpColIter colit = tmpder->begcol(); colit != tmpder->endcol(); ++colit)
    {
        IU gcol = colit.colid();
        for(typename UDERB::SpColIter::NzIter nzit = tmpder->begnz(colit); nzit != tmpder->endnz(colit); ++nzit)
        {
            IU grow = nzit.rowid() + rowblocksizeprefixB[myrank];
            int owner = 0;
            while(owner <= nprocs-1 && blocksizeprefixB[owner] <= gcol) owner++;
            owner--;
            offdiagrestuples[owner].push_back(std::make_tuple(grow, gcol - blocksizeprefixB[owner], nzit.value()));
        }
    }
    IU datasize;
    tmpCtuples = SpParHelper::ExchangeDataGeneral(offdiagrestuples, MPI_COMM_WORLD, datasize);
    tmpCsptuples = new SpTuples<IU, NUO>(datasize, kdim, blocksizeB[myrank], tmpCtuples);
    C_D1S2DER = new UDERO(*tmpCsptuples, false); 
    delete tmpCsptuples;
    timer.stop("part2",true);
    /*Offdiag multiplication*/
    tmpAsptuples = new SpTuples<IU,NU1>(aoffdiag.size(), mdim, blocksizeA[myrank], aoffdiag.data(),true);
    tmpAsptuples->tuples_deleted = true;
    aoffdiagDER = new UDERA(*tmpAsptuples,false);
    delete tmpAsptuples;
    SpParMat1D<IU, NUO,UDERO> Aoffdiag(mdim, kdim, blocksizeA, aoffdiagDER, rowblocksizeA);

    tmpBsptuples = new SpTuples<IU,NU2>(boffdiag.size(), kdim, blocksizeB[myrank], boffdiag.data(),true);
    tmpBsptuples->tuples_deleted = true;
    boffdiagDER = new UDERB(*tmpBsptuples,false);
    delete tmpBsptuples;
    SpParMat1D<IU, NUO,UDERO> Boffdiag(kdim, ndim, blocksizeB, boffdiagDER, rowblocksizeB);
    
    SpParMat1D<IU, NUO, UDERO> offdiag1D = Mult_AnXBn_1D_CbC_RDMA_FetchAll<SR, NUO, UDERO>(Aoffdiag,Boffdiag);
    UDERO * spSeqC = offdiag1D.seqptr();
    timer.start("merge");
    *spSeqC += *C_D1S2DER;
    *spSeqC += *C_S1D1D2DER;
    timer.stop("merge");
    return offdiag1D;
}

// Block Out Product Implementation
template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
SpParMat1D<IU, NUO, UDERO> Mult_AnXBn_1D_OP(SpParMat1D<IU,NU1,UDERA> & A, SpParMat1D<IU,NU2,UDERB> & B, bool clearA = false, bool clearB = false )
{
    /*get basic info.*/
    // A and B should all be column split.
    auto grid1d = A.getgrid();
    int nprocs = grid1d->GetSize();
    int myrank = grid1d->GetRank();
    IU mdim = A.getnrow();
    IU kdim = A.getncol(); /*dim check*/ SCALAREQCHECK(kdim, B.getnrow(), "kdim mismatch!\n")
    IU ndim = B.getncol();
    // get all block dimension.
    std::vector<IU> blocksizeA = A.getblocksizevec(); 
    std::vector<IU> blocksizeprefixA = A.getblocksizeprefix(); 
    std::vector<IU> blocksizeB = B.getblocksizevec(); 
    std::vector<IU> blocksizeprefixB = B.getblocksizeprefix();
    std::vector<IU> rowblocksizeA = A.getrowblocksizevec();
    std::vector<IU> rowblocksizeprefixA = A.getrowblocksizeprefix();
    std::vector<IU> rowblocksizeB = B.getrowblocksizevec(); 
    std::vector<IU> rowblocksizeprefixB = B.getrowblocksizeprefix();
    
    // tmp pointers
    std::tuple<IU,IU,NU1> *tmpstdtuples;
    SpTuples<IU, NU1> *tmpsptuples;
    UDERB *BrowsDER;
    UDERO *CDER;
    UDERA * spSeqA = A.seqptr();
    UDERB * spSeqB = B.seqptr();
    // spSeqB->WriteMM("spSeqDER" + std::to_string(myrank) + ".mtx", true);
    double sendBt, sendCt, mergeCt, compt;
    sendBt = MPI_Wtime();
    std::vector<std::vector<std::tuple<IU,IU,NU2>>> sendbtuples(nprocs);
    for(typename UDERB::SpColIter colit = spSeqB->begcol(); colit != spSeqB->endcol(); ++colit)
    {
        IU gcol = colit.colid() + blocksizeprefixB[myrank];
        for(typename UDERB::SpColIter::NzIter nzit = spSeqB->begnz(colit); nzit != spSeqB->endnz(colit); ++nzit)
        {
            IU grow = nzit.rowid();
            // std::pair<int,int> locingrid = B.BlockIndex(grow, gcol);
            // int owner = locingrid.first;
            int owner = 0;
            while(owner <= nprocs-1 && rowblocksizeprefixB[owner] <= grow) owner++;
            owner--;
            sendbtuples[owner].push_back(std::make_tuple(grow-rowblocksizeprefixB[owner],gcol,nzit.value()));
        }
    }
    // printf("myrank %d send %d to rank 0. \n", myrank, sendbtuples[0].size());
    IU Browsdatasize;
    std::tuple<IU,IU,NU2> *recvoffbtuples = SpParHelper::ExchangeDataGeneral(sendbtuples, MPI_COMM_WORLD, 
    Browsdatasize);
    // if(myrank == 0) printf("rank 0 receive %ld elements in total rowblocksizeB %ld ndim %ld \n", Browsdatasize, 
    // rowblocksizeB[myrank], ndim);
    tmpsptuples = new SpTuples<IU, NU2>(Browsdatasize, rowblocksizeB[myrank], ndim, recvoffbtuples, false);
    BrowsDER = new UDERB(*tmpsptuples, false); // cleared recvoffbtuples;
    delete tmpsptuples;
    sendBt = MPI_Wtime() - sendBt;
    compt = MPI_Wtime();
    // BrowsDER->WriteMM("browsDER" + std::to_string(myrank) + ".mtx", true);
    // spSeqA->WriteMM("ADER" + std::to_string(myrank) + ".mtx", true);
    tmpsptuples = LocalHybridSpGEMM<SR, NUO, IU, NU1, NU2>(*spSeqA, *BrowsDER, false, false);
    compt = MPI_Wtime() - compt;

    sendCt = MPI_Wtime();
    UDERO *tmpder = new UDERO(*tmpsptuples,false); 
    // tmpder->WriteMM("PartC"+std::to_string(myrank) + ".mtx", true);
    delete tmpsptuples;

    std::vector<std::vector<std::tuple<IU,IU,NU2>>> sendctuples(nprocs);
    //!!! Here we distribute the data BACK to a col wise split!
    for(typename UDERB::SpColIter colit = tmpder->begcol(); colit != tmpder->endcol(); ++colit)
    {
        IU gcol = colit.colid();
        for(typename UDERB::SpColIter::NzIter nzit = tmpder->begnz(colit); nzit != tmpder->endnz(colit); ++nzit)
        {
            int owner = 0;
            while(owner <= nprocs-1 && blocksizeprefixB[owner] <= gcol) owner++;
            owner--;
            sendctuples[owner].push_back(std::make_tuple(nzit.rowid(), gcol - blocksizeprefixB[owner], nzit.value()));
        }
    }
    IU datasize;
    tmpstdtuples = SpParHelper::ExchangeDataGeneral(sendctuples, MPI_COMM_WORLD, datasize);
    sendCt = MPI_Wtime() - sendCt;

    mergeCt = MPI_Wtime();
    ColLexiCompare<IU,NUO> collexicogcmp;
    if(!SpHelper::is_sorted(tmpstdtuples, tmpstdtuples+datasize, collexicogcmp))
    __gnu_parallel::sort(tmpstdtuples , tmpstdtuples+datasize, collexicogcmp);
    // merge tuples that have the same index
    int nthreads = 1;
    #pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    std::vector<std::tuple<IU,IU,NUO>> sortedtuples;
    
    if(datasize < nthreads){
        // too small, call seq impl
        IU didx = 0;
        while (didx < datasize){
            IU currow = std::get<0>(tmpstdtuples[didx]);
            IU curcol = std::get<1>(tmpstdtuples[didx]);
            NUO curval = std::get<2>(tmpstdtuples[didx]);
            IU mergeend = didx+1;
            while(mergeend < datasize 
            && std::get<0>(tmpstdtuples[mergeend]) == currow
            && std::get<1>(tmpstdtuples[mergeend]) == curcol ){
                curval += std::get<2>(tmpstdtuples[mergeend]);
                mergeend ++;
            }
            sortedtuples.push_back(std::make_tuple(currow,curcol,curval));
            didx = mergeend;
        }
    }
    else{
        // large data, let's parallel it.
        std::vector<std::vector<std::tuple<IU,IU,NUO>>> sortedtuplesThreads(nthreads);
        #pragma omp parallel 
        {
            int mytid = omp_get_thread_num();
            IU bs = datasize / nthreads;
            IU startidx = bs * mytid;
            IU endidx = bs * (mytid+1);
            if(mytid == nthreads-1) endidx = datasize;
            IU didx = startidx;
            while (didx < endidx){
                IU currow = std::get<0>(tmpstdtuples[didx]);
                IU curcol = std::get<1>(tmpstdtuples[didx]);
                NUO curval = std::get<2>(tmpstdtuples[didx]);
                IU mergeend = didx+1;
                while(mergeend < endidx 
                && std::get<0>(tmpstdtuples[mergeend]) == currow
                && std::get<1>(tmpstdtuples[mergeend]) == curcol ){
                    curval += std::get<2>(tmpstdtuples[mergeend]);
                    mergeend ++;
                }
                sortedtuplesThreads[mytid].push_back(std::make_tuple(currow,curcol,curval));
                didx = mergeend;
            }
        }
        // merge dup items that across different threads.
        std::vector<int> dupsign(nthreads,0);
        for(int i=0; i<nthreads-1; i++){
            // if current vector or next vector is empty, let's skip the round.
            if(sortedtuplesThreads[i].size() == 0 
                || sortedtuplesThreads[i+1].size() == 0 ){
                    continue;
            }
            IU currow = std::get<0>(sortedtuplesThreads[i].back());
            IU curcol = std::get<1>(sortedtuplesThreads[i].back());
            NUO curval = std::get<2>(sortedtuplesThreads[i].back());
            if(std::get<0>(sortedtuplesThreads[i+1].front()) == currow
            && std::get<1>(sortedtuplesThreads[i+1].front()) == curcol){
                curval += std::get<2>(sortedtuplesThreads[i+1].front());
                dupsign[i+1] = 1;
                sortedtuplesThreads[i].back() = std::tuple<IU,IU,NUO>(currow,curcol,curval);
            }
        }
        IU tottuples = 0;
        for(int i=0; i<nthreads; i++){
            tottuples += sortedtuplesThreads[i].size() - dupsign[i];
        }
        sortedtuples.resize(tottuples);
        IU curptr = 0;
        for(int i=0; i<nthreads; i++){
            if(sortedtuplesThreads[i].size() == 0) continue;
            std::copy(sortedtuplesThreads[i].begin() + dupsign[i],sortedtuplesThreads[i].end(),
            sortedtuples.begin()+curptr);
            curptr += sortedtuplesThreads[i].size() - dupsign[i];
        }
    }
    mergeCt = MPI_Wtime() - mergeCt;

    datasize = sortedtuples.size();
    // printf("myrank %d datasize %ld \n",myrank, datasize);
    tmpsptuples = new SpTuples<IU, NUO>(datasize, mdim, blocksizeB[myrank], sortedtuples.data(), true);
    tmpsptuples->tuples_deleted = true;
    UDERO * seq = new UDERO(*tmpsptuples, false);
    delete tmpsptuples;
    // merge results and return 
    SpParMat1D<IU, NUO, UDERO> C(mdim, ndim, B.getblocksizevec(), seq, A.getrowblocksizevec());
    // IU nnzC = C.getnnz();
    // if(myrank == 0)printf("Cnnz is %ld \n", nnzC);
    perfcnt1dop.doublemap["sendB(ms)"] = sendBt;
    perfcnt1dop.doublemap["sendC(ms)"] = sendCt;
    perfcnt1dop.doublemap["mergeC(ms)"] = mergeCt;
    perfcnt1dop.doublemap["compT(ms)"] = compt;
    perfcnt1dop.integermap["Alocalnnz"] = spSeqA->getnnz();
    perfcnt1dop.doublemap["Alocalmem(MB)"] = spSeqA->getnnz() * (sizeof(IU) + sizeof(NUO)) * 1e-6;
    perfcnt1dop.integermap["Bnnz    "] = BrowsDER->getnnz();
    perfcnt1dop.doublemap["Bmem(MB)"] = BrowsDER->getnnz() * (sizeof(IU) + sizeof(NUO)) * 1e-6;
    perfcnt1dop.integermap["Cnnz    "] = seq->getnnz();
    perfcnt1dop.doublemap["Cmem(MB)"] = seq->getnnz() * (sizeof(IU) + sizeof(NUO)) * 1e-6;
    return C;
}

}

#endif
