#ifndef _SP_PAR_MAT_1D_H_
#define _SP_PAR_MAT_1D_H_

#include <cstdint>
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <mpi.h>
#include <tuple>
#include <vector>
#include <iterator>

#include "CombBLAS/Compare.h"
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
SpParMat1D<IU, NUO, UDERO> Mult_AnXBn_1D_CbC_RDMA_FetchAll(SpParMat1D<IU,NU1,UDERA> & A, SpParMat1D<IU,NU2,UDERB> & B, bool clearA = false, bool clearB = false)
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
    IU nnzA = Ader->getnnz();
    IU nnzB = Bder->getnnz();
    /*********************************************************************************************
    Part 1: Generate Aneeded object
    we first get all cp and jc array from all mpi process, as this won't consume too much memory. 
    Then we can calculate the needed columns of A.
    Then we get ir and numx using RDMA API.
    **********************************************************************************************/
    IU totalnnz = 0, Aneedednzccnt = 0; // input params for Aneeded object.
    std::vector<IU> Aneededjc; // memory buffer for Aneeded DCSC jc array.
    std::vector<IU> Aneededcp; // memory buffer for Aneeded DCSC cp array.
    /* some aux array for fetching remote ir and numx */
    std::vector<IU> Aneededoffset(nprocs, 0); // offset of elements in needed ir/numx array
    std::vector<std::vector<IU>> AoriginoffsetPerRank(nprocs); // offset of elements in origin ir/numx array in per mpi rank
    std::vector<std::vector<IU>> AneededoffsetPerRank(nprocs); // offset of elements in needed ir/numx array in per mpi rank
    std::vector<std::vector<IU>> AneedednzcntPerRank(nprocs); // number of elements in needed ir/numx array per column in per mpi rank
    /* send non-zero column index (cp) and size (jc) in csc to others */ 
    std::vector<std::vector<IU>> remoteAcp(nprocs), remoteAjc(nprocs); //! each vector represents one MPI process
    double tmp = MPI_Wtime();
    A.ExchangeDcscIndex(remoteAcp, remoteAjc);
    tmp = MPI_Wtime() - tmp;
    double t_countdisp = MPI_Wtime();
    /* project all non-zero elements along columns to a vector, also for rows */
    std::vector<int> tmprow(Brows,0);
    // IU diagstart = blocksizeprefix[myrank];
    // IU diagstop = myrank == nprocs - 1 ? mdim : blocksizeprefix[myrank+1];
    for(IU i=0; i<Brows; i++) tmprow[i] = false;
    for(typename UDERB::SpColIter colit = Bder->begcol(); colit != Bder->endcol(); ++colit)
    {
        for(typename UDERB::SpColIter::NzIter nzit = Bder->begnz(colit); nzit != Bder->endnz(colit); ++nzit)
        {
            IU grow = nzit.rowid();
            tmprow[grow] = 1;
        }
    }
    // check project vector is correct!
    std::vector<int> globalrow(Brows * nprocs);
    MPI_Allgather(
        tmprow.data(), Brows, MPI_INT, 
        globalrow.data(), Brows, MPI_INT, comm);
    if(myrank == 0){
        // save it to file
        char filechar[1000];
        sprintf(filechar, "BrowProj_M%d_N%d_K%d_Nprocs%d.bin",mdim,ndim,kdim,nprocs);
        FILE * f = fopen(filechar, "wb");
        fwrite(globalrow.data(), sizeof(int), Brows*nprocs, f);
        fclose(f);
    }
    if(0){ /*looping to calculate memory reuqirement and offsets*/
    /*V1: we fetch column by column, too many MPI_Get Calls!!*/
    for(int p=0; p<nprocs; p++){
        AneededoffsetPerRank[p].push_back(0);
        for(int kr=0; kr < remoteAjc[p].size(); kr++){
            IU colid  = remoteAjc[p][kr];
            if(tmprow[kr + blocksizeprefixA[p]] == 1){
                IU curcolelems = remoteAcp[p][kr+1] - remoteAcp[p][kr];
                totalnnz += curcolelems; // save total non-zero elements
                Aneedednzccnt++; // save total non-zero columns
                Aneededjc.push_back(colid + blocksizeprefixA[p]); //save global column id
                Aneededcp.push_back(curcolelems); //save nz in each column, this is not cp at this stage.
                Aneededoffset[p] += curcolelems; // save nz in each process, this is not offset at this stage.
                AoriginoffsetPerRank[p].push_back(remoteAcp[p][kr]); // save origin offset of the nzc.
                AneededoffsetPerRank[p].push_back(Aneededoffset[p]); // save needed offset of the nzc.
                AneedednzcntPerRank[p].push_back(curcolelems); // save elements per columns in per mpi rank.
            }
        }
    }
    /*postprocess Aneededcp to be correct format*/
    std::vector<IU> tmp = Aneededcp;
    Aneededcp.resize(tmp.size()+1);
    Aneededcp[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededcp.begin()+1);
    /*postprocess Aneededoffset to be correct format*/
    tmp = Aneededoffset;
    Aneededoffset.resize(tmp.size()+1);
    Aneededoffset[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededoffset.begin()+1);
    std::vector<IU> ().swap(tmp); // clean tmp
    }

    if(0){ /*looping to calculate memory reuqirement and offsets*/
    /*V2: we fetch column by column, but merge call for continuous column, too low hit rate!!*/
    IU continuehit = 0, totalhit = 0;
    for(int p=0; p<nprocs; p++){
        AneededoffsetPerRank[p].push_back(0);
        IU prevcolid = -1;
        for(int kr=0; kr < remoteAjc[p].size(); kr++){
            IU colid  = remoteAjc[p][kr];
            if(tmprow[kr + blocksizeprefixA[p]]){
                totalhit++;
                IU curcolelems = remoteAcp[p][kr+1] - remoteAcp[p][kr];
                totalnnz += curcolelems; // save total non-zero elements
                Aneedednzccnt++; // save total non-zero columns
                Aneededjc.push_back(colid + blocksizeprefixA[p]); //save global column id
                Aneededcp.push_back(curcolelems); //save nz in each column, this is not cp at this stage.
                Aneededoffset[p] += curcolelems; // save nz in each process, this is not offset at this stage.
                if(prevcolid == -1){ /*first call*/ 
                    AneededoffsetPerRank[p].push_back(Aneededoffset[p]); // save needed offset of the nzc.
                    AoriginoffsetPerRank[p].push_back(remoteAcp[p][kr]); // save origin offset of the nzc.
                    AneedednzcntPerRank[p].push_back(curcolelems); // save elements per columns in per mpi rank.
                    prevcolid = colid;
                }else if(prevcolid + 1 == colid){
                    continuehit++;
                    AneededoffsetPerRank[p].back() += curcolelems;
                    AneedednzcntPerRank[p].back() += curcolelems;
                    prevcolid = colid;
                }else{ /*the column is not adjacent with previous column*/
                    AneededoffsetPerRank[p].push_back(Aneededoffset[p]); // save needed offset of the nzc.
                    AoriginoffsetPerRank[p].push_back(remoteAcp[p][kr]); // save origin offset of the nzc.
                    AneedednzcntPerRank[p].push_back(curcolelems); // save elements per columns in per mpi rank.
                }
                
            }
        }
    }
    // very low hit rate!!!
    // printf("myrank %d totalhit %lu continuehit %lu percent %f \n", myrank, totalhit, continuehit, (double)continuehit / totalhit);
    /*postprocess Aneededcp to be correct format*/
    std::vector<IU> tmp = Aneededcp;
    Aneededcp.resize(tmp.size()+1);
    Aneededcp[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededcp.begin()+1);
    /*postprocess Aneededoffset to be correct format*/
    tmp = Aneededoffset;
    Aneededoffset.resize(tmp.size()+1);
    Aneededoffset[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededoffset.begin()+1);
    std::vector<IU> ().swap(tmp); // clean tmp
    }


    if(1){ /*looping to calculate memory reuqirement and offsets*/
    /*V3: we split remote nzc into K parts, once we need data in the part. we fetch the whole part. */
    for(int p=0; p<nprocs; p++){
        IU remotenzc = remoteAjc[p].size();
        if(remotenzc == 0) continue;
        int ChunksK = 2048;
        while(ChunksK > 1 && ChunksK > remotenzc) ChunksK /= 2;
        /*first decide whether we need to fetch the data from remote process*/ 
        IU batchsize = (remotenzc + ChunksK -1) / ChunksK;
        std::vector<bool> hitbatch(ChunksK,false);
        bool needfetch(false);
        // record whether the batch will be requested.
        for(IU bi=0; bi < ChunksK; bi++){
            IU startidx = bi * batchsize;
            IU endidx = std::min( ( bi + 1 ) * batchsize, remotenzc);
            IU curbatchdatasize = 0;
            for(IU kr = startidx; kr < endidx; kr++){
                IU colid  = remoteAjc[p][kr];
                if(tmprow[colid + blocksizeprefixA[p]]){
                    hitbatch[bi] = true;
                    needfetch = true;
                    dataneeded += (sizeof(IU) + sizeof(NUO)) * (remoteAcp[p][kr+1] - remoteAcp[p][kr]);
                }
            }
        }
        if(!needfetch)continue; // we didn't hit any batch, fast return
        // do remote memory offset process
        AneededoffsetPerRank[p].push_back(0);
        IU prevbatchid = -1;
        for(IU bi=0; bi < ChunksK; bi++){
            if(!hitbatch[bi])continue;
            IU startidx = bi * batchsize;
            IU endidx = std::min( ( bi + 1 ) * batchsize, remotenzc);
            if(prevbatchid == -1 || prevbatchid + 1 != bi){
                // we meet the first batch or a new distinct batch, save it as a new call.
                for(IU kr = startidx; kr < endidx; kr++){
                    IU colid  = remoteAjc[p][kr];
                    IU curcolelems = remoteAcp[p][kr+1] - remoteAcp[p][kr];
                    datafetched += (sizeof(IU) + sizeof(NU1)) * (curcolelems);
                    totalnnz += curcolelems;
                    Aneedednzccnt++;
                    Aneededjc.push_back(colid + blocksizeprefixA[p]);
                    Aneededcp.push_back(curcolelems); //save nz in each column, this is not cp at this stage.
                    Aneededoffset[p] += curcolelems; // save nz in each process, this is not offset at this stage.
                    AoriginoffsetPerRank[p].push_back(remoteAcp[p][kr]); // save origin offset of the nzc.
                    AneededoffsetPerRank[p].push_back(Aneededoffset[p]); // save needed offset of the nzc.
                    AneedednzcntPerRank[p].push_back(curcolelems); // save elements per columns in per mpi rank.
                }
                prevbatchid = bi;
            }else if(prevbatchid + 1 == bi){ // we have two successive batch, merge it with previous call.
                for(IU kr = startidx; kr < endidx; kr++){
                    IU colid  = remoteAjc[p][kr];
                    IU curcolelems = remoteAcp[p][kr+1] - remoteAcp[p][kr];
                    datafetched += (sizeof(IU) + sizeof(NU1)) * (curcolelems);
                    totalnnz += curcolelems;
                    Aneedednzccnt++;
                    Aneededjc.push_back(colid + blocksizeprefixA[p]);
                    Aneededcp.push_back(curcolelems); //save nz in each column, this is not cp at this stage.
                    Aneededoffset[p] += curcolelems; // save nz in each process, this is not offset at this stage.
                    AneededoffsetPerRank[p].back() += curcolelems; // save needed offset of the nzc.
                    AneedednzcntPerRank[p].back() += curcolelems; // save elements per columns in per mpi rank.
                }
                prevbatchid = bi;
            }else{
                printf("just make sure no other cases!!\n");
            }
        }
    }
    /*postprocess Aneededcp to be correct format*/
    std::vector<IU> tmp = Aneededcp;
    Aneededcp.resize(tmp.size()+1);
    Aneededcp[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededcp.begin()+1);
    /*postprocess Aneededoffset to be correct format*/
    tmp = Aneededoffset;
    Aneededoffset.resize(tmp.size()+1);
    Aneededoffset[0] = 0;
    std::partial_sum(tmp.begin(), tmp.end(),Aneededoffset.begin()+1);
    std::vector<IU> ().swap(tmp); // clean tmp
    }

    // delete[] tmprow;
    tmprow = std::vector<int>(); // clean memory
    t_countdisp = MPI_Wtime() - t_countdisp;
    // perfcnt1d.prep1dcalidxtime = t_countdisp;
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

    MPI_PassiveWindow<IU> *Airwins = new MPI_PassiveWindow<IU>(irptr, nz);
    MPI_PassiveWindow<NU1> *Anumwins = new MPI_PassiveWindow<NU1>(numptr, nz);
    /* Fetch required ir/numx memory buffer from remote process */
    int targetrank = (myrank + 1) % nprocs; // shifting, make sure we request from different rank.
    int round = 0;
    prep1d = MPI_Wtime() - prep1d;
    double comm1d = MPI_Wtime();

    // std::vector<int> remotecallcnt(nprocs);
    // std::vector<int> remotefetchsize(nprocs);

    // Airwins->LockAll();
    // Anumwins->LockAll();

    while(round < nprocs){
        IU curnzccnt = AneedednzcntPerRank[targetrank].size();
        // /*debug use*/ 
        // remotecallcnt[targetrank] = curnzccnt;
        // remotefetchsize[targetrank] = 0;
        if(curnzccnt != 0) {
            if(targetrank != myrank){
                Airwins->Lock(targetrank, MPI_OneSided_LockType::SHARED);
                Anumwins->Lock(targetrank, MPI_OneSided_LockType::SHARED);
                for(int idx=0; idx<curnzccnt; idx++){
                    Airwins->Get(Aneededptr->ir + Aneededoffset[targetrank] + AneededoffsetPerRank[targetrank][idx],   
                    AoriginoffsetPerRank[targetrank][idx], AneedednzcntPerRank[targetrank][idx], targetrank);
                    Anumwins->Get(Aneededptr->numx + Aneededoffset[targetrank] + AneededoffsetPerRank[targetrank][idx],
                    AoriginoffsetPerRank[targetrank][idx], AneedednzcntPerRank[targetrank][idx], targetrank);
                    // /*debug use*/ 
                    // remotefetchsize[targetrank]+=AneedednzcntPerRank[targetrank][idx];
                }
                Airwins->Unlock(targetrank);
                Anumwins->Unlock(targetrank);
            }else{
                for(int idx=0; idx < curnzccnt; idx++){
                    std::copy(dcscA->ir + AoriginoffsetPerRank[targetrank][idx], dcscA->ir + AoriginoffsetPerRank[targetrank][idx] + AneedednzcntPerRank[targetrank][idx],
                    Aneededptr->ir + Aneededoffset[targetrank] + AneededoffsetPerRank[targetrank][idx]);
                    std::copy(dcscA->numx + AoriginoffsetPerRank[targetrank][idx], dcscA->numx + AoriginoffsetPerRank[targetrank][idx] + AneedednzcntPerRank[targetrank][idx],
                    Aneededptr->numx + Aneededoffset[targetrank] + AneededoffsetPerRank[targetrank][idx]);
                }
            }
        }
        targetrank = (targetrank + 1) % nprocs; // prepare for next target rank
        round++;
    }
    // Airwins->UnlockAll();
    // Anumwins->UnlockAll();

    delete Airwins;
    delete Anumwins;

    std::vector<IU>().swap(Aneededoffset); // clean memory
    std::vector<std::vector<IU>>().swap(AneededoffsetPerRank); // clean memory
    std::vector<std::vector<IU>>().swap(AoriginoffsetPerRank); // clean memory
    std::vector<std::vector<IU>>().swap(AneedednzcntPerRank); // clean memory
    comm1d = MPI_Wtime() - comm1d;
    // /*debug use*/ 
    // std::vector<int> remotecallcnt_ag(nprocs*nprocs);
    // std::vector<int> remotefetchsize_ag(nprocs*nprocs);
    // MPI_Allgather(remotecallcnt.data(), nprocs, MPI_INT, remotecallcnt_ag.data(), nprocs, MPI_INT, MPI_COMM_WORLD);
    // MPI_Allgather(remotefetchsize.data(), nprocs, MPI_INT, remotefetchsize_ag.data(), nprocs, MPI_INT, MPI_COMM_WORLD);
    // if(myrank == 0){
    //     std::cerr << "remote call stats: \n" ;
    //     for(int i=0; i<nprocs; i++){
    //         std::string rankstr = "Rank " + to_string(i) + ": ";
    //         std::cerr << std::setw(10) << std::left << rankstr;
    //         for(int j=0; j<nprocs; j++){
    //             std::cerr << std::setw(7) << remotecallcnt_ag[i*nprocs + j];
    //         }
    //         std::cerr << std::endl;
    //     }
    //     std::cerr << "remote elems stats: \n" ;
    //     for(int i=0; i<nprocs; i++){
    //         std::string rankstr = "Rank " + to_string(i) + ": ";
    //         std::cerr << std::setw(10) << rankstr;
    //         for(int j=0; j<nprocs; j++){
    //             std::cerr << std::setw(10) << std::left << std::scientific << std::setprecision(2) << remotefetchsize_ag[i*nprocs + j] * 16 * 1e-6;
    //         }
    //         std::cerr << std::endl;
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);

    /******************************************************************
    Part 2: Launch SpGEMM and get C matrix
    *******************************************************************/
    double comp1d = MPI_Wtime();
    SpTuples<IU, NUO> *sptuples = LocalHybridSpGEMM<SR,NUO>(Aneeded, *Bder, false, false);
    comp1d = MPI_Wtime() - comp1d;
    UDERO *Cder = new UDERO(*sptuples, false);
    SpParMat1D<IU, NUO, UDERO> C(mdim,ndim,B.getblocksizevec(), Cder, A.getrowblocksizevec());
    delete sptuples;

    perfcnt1d.doublemap["prepT(ms)"] = prep1d;
    perfcnt1d.doublemap["commT(ms)"] = comm1d;
    perfcnt1d.doublemap["compT(ms)"] = comp1d;
    perfcnt1d.integermap["Alocalnnz"] = Ader->getnnz();
    perfcnt1d.doublemap["Alocalmem(MB)"] = Ader->getnnz() * (sizeof(IU) + sizeof(NU1)) * 1e-6;
    perfcnt1d.integermap["Aneednnz"] = Aneeded.getnnz();
    perfcnt1d.doublemap["Aneedmem(MB)"] = Aneeded.getnnz() * (sizeof(IU) + sizeof(NU1)) * 1e-6;
    perfcnt1d.integermap["Bnnz    "] = Bder->getnnz();
    perfcnt1d.doublemap["Bmem(MB)"] = Bder->getnnz() * (sizeof(IU) + sizeof(NU1)) * 1e-6;
    perfcnt1d.integermap["Cnnz    "] = Cder->getnnz();
    perfcnt1d.doublemap["Cmem(MB)"] = Cder->getnnz() * (sizeof(IU) + sizeof(NU1)) * 1e-6;
    return C;
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
    IU kdim = A.getncol(); 
    /*dim check*/ 
    if(kdim != B.getrow()){
        if(myrank==0)std::cerr<<"kdim mistmatch!!"<<std::endl;
        exit(0);
    }
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
    SpParMat1D<IU, NUO, UDERO> C(seq, std::make_shared<CommGrid1D>(MPI_COMM_WORLD));
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
