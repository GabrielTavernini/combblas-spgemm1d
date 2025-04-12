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



#ifndef _SP_HELPER_H_
#define _SP_HELPER_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include "SpDefs.h"
#include "StackEntry.h"
#include "Isect.h"
#include "HeapEntry.h"
#include "SequenceHeaps/knheap.h"
#include "hash.hpp"
#include <parallel/numeric>
#include <cstdlib>
#include <unistd.h>
#include <limits.h>
namespace combblas {

// for large dataset test 
static std::unordered_map<std::string, std::string> datasetmap;
// for small dataset test and check correctness
static std::vector<std::string> ckcorrectdlist;


struct SpgemmOpts
{
    bool comm_ana = false;
    std::string testprefix = "";
    bool display = false;
    // BC parameters
    int AproxK = -1;
    int batchSize = -1;
    std::string BCoutput;
    double memorylimit = 0.0; // memory limit of each mpi process in GB 

    int niter = 5; // iteration per benchmark 
    bool logtofile = false; // record detailed time to solution to log file.
    bool AA = false;  // Do A square Benchmark
    bool Rop = false; // Restriction Op Benchmark
    bool BC = false;  // Bentweeness Centrality Benchmark
    bool check = false; // Check correctness
    // 1d configuration
    int permute = 0;
    int nparts = 0;
    std::string permfile = "";
    // 0: no permutation, 1: random permutation 2: ParMetis permutation
    bool run1d = false;
    std::vector<int> gentype = {5};
    // 2d configuration
    // 0: no permutation, 1: random permutation
    bool run2d = false;
    
    // 3d configuration
    // prow x pcol x layer = nprocs
    bool run3d = false;
    // std::vector<int> prow;
    // std::vector<int> pcol;
    std::vector<int> layer;

    // ER
    bool runER = false;
    double initiator_ER[4];
    unsigned EDGEFACTOR_ER;
    unsigned scale_ER;
    // G500
    bool runG500 = false;
    double initiator_G500[4];
    unsigned EDGEFACTOR_G500  = 16;
    unsigned scale_G500;
    // SSCA
    bool runSSCA = false;
    double initiator_SSCA[4];
    unsigned EDGEFACTOR_SSCA  = 8;
    unsigned scale_SSCA;
    std::vector<std::string> dataset; // list of dataset for benchmark
    std::unordered_map<std::string, std::string> fullfilepath;
    std::string curdataset;
    void PrintOpts(){
        std::cerr << "Benchmark Dataset:" << std::endl ;
        std::cerr << "    Real Dataset:" << std::endl;
        for(auto x : dataset) std::cerr <<  "           " << x << std::endl;
        if(runER){
            std::cerr << "    ER: scale " << scale_ER << " EdgeFactor " << EDGEFACTOR_ER << std::endl;
        }
        if(runG500){
            std::cerr << "    G500: scale " << scale_G500 << " EdgeFactor " << EDGEFACTOR_G500 << std::endl;
        }
        if(runSSCA){
            std::cerr << "    SSCA: scale " << scale_SSCA << " EdgeFactor " << EDGEFACTOR_SSCA << std::endl;
        }
        std::cerr << "iteration time:" << niter << std::endl;
        std::cerr <<"Benchmark Algorithm:" << std::endl;
        
        if(run1d){
            std::cerr << "    SpGEMM 1D: ";
            std::cerr << "genType: ";
            for(auto x : gentype) std::cerr << x << ", ";
            std::cerr << std::endl;
        }
        if(run2d){
            std::cerr << "    SpGEMM 2D: ";
            std::cerr << std::endl;
        }
        if(run3d){
            std::cerr << "    SpGEMM 3D, layer config: ";
            for(int i=0; i<layer.size();i++) std::cerr << layer[i] << ", "; 
            // std::cerr << "ProcessRow x ProcessCol x processLayer = " << prow << " x " << pcol << " x " << layer;
            std::cerr << std::endl;
        }
        if(permute==0) std::cerr << "No Permutaiton";
        else if(permute == 1) std::cerr << "Random Permutation";
        else if(permute >= 2) std::cerr << "Metis Partition Permutation";
        std::cerr << std::endl;
        std::cerr <<"Benchmark Applications:" << std::endl;
        if(AA) std::cerr << "    A Square" << std::endl;
        if(Rop) std::cerr << "    Restriction OP" << std::endl;
        if(BC) std::cerr << "    Betweeness Centrality" << std::endl;
        std::cerr << "=============================================" << std::endl;
    }
};

// SpgemmOpts opts;

struct OptParser{
typedef std::vector<std::string> VS;
// split the string 
static std::vector<std::string> SplitString(const std::string& input, char delimiter = ',') {
    std::vector<std::string> tokens;
    std::istringstream iss(input);
    std::string token;
    while (std::getline(iss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

static int parse( int argc, char** argv, SpgemmOpts *opts){
    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("--Rop", argv[i]) == 0) {
            opts->Rop = true;
        }
        else if ( strcmp("--AA", argv[i]) == 0 ) {
            opts->AA = true;
        }
        else if ( strcmp("--BC", argv[i]) == 0 && i + 1 < argc) {
            opts->BC = true;
            VS clist = SplitString(std::string(argv[i+1]));
            opts->AproxK = stoi(clist[0]);
            opts->batchSize = stoi(clist[1]);
            i++;
        }
        else if ( strcmp("--dataset", argv[i]) == 0 && i+1 < argc ) {
            // dataset string is a list of dataset joined with ,
            // e.g. vir,euk,arch,nlpkkt200
            // then it will run the test on these dataset
            VS dstrlist = SplitString(std::string(argv[i+1]));
            for(auto d : dstrlist){
                if(datasetmap.find(d) != datasetmap.end()){
                    opts->fullfilepath[d]=datasetmap[d];
                    opts->dataset.push_back(d);
                }else{
                    printf("dataset %s not found in my map!!\n",d.c_str());
                    exit(0);
                }
            }
            i++;
            
        }
        else if ( strcmp("--ER", argv[i]) == 0 && i+1 < argc ) {
            // ER random graph
            // list the scale and EDGEFACTOR behind like 16,10 
            opts->runER = true;
            opts->initiator_ER[0] = .25;
            opts->initiator_ER[1] = .25;
            opts->initiator_ER[2] = .25;
            opts->initiator_ER[3] = .25;
            VS erconfig = SplitString(std::string(argv[i+1]));
            opts->scale_ER = (unsigned) stoi(std::string(erconfig[0]));
            opts->EDGEFACTOR_ER = (unsigned) stoi(std::string(erconfig[1]));
            i++;
            opts->dataset.push_back("ER");
        }
        else if ( strcmp("--G500", argv[i]) == 0 && i+1 < argc ) {
            // ER random graph
            opts->runG500 = true;
            opts->initiator_G500[0] = .57;
            opts->initiator_G500[1] = .19;
            opts->initiator_G500[2] = .19;
            opts->initiator_G500[3] = .05;
            opts->scale_G500 = (unsigned)atoi(argv[i+1]);
            i++;
            opts->dataset.push_back("G500");
        }
        else if ( strcmp("--SSCA", argv[i]) == 0 && i+1 < argc ) {
            // ER random graph
            opts->runSSCA = true;
            opts->initiator_SSCA[0] = .6;
            opts->initiator_SSCA[1] = .4/3;
            opts->initiator_SSCA[2] = .4/3;
            opts->initiator_SSCA[3] = .4/3;
            opts->scale_SSCA = (unsigned) atoi(argv[i+1]);
            i++;
            opts->dataset.push_back("SSCA");
        }
        else if ( strcmp("--metisgp", argv[i]) == 0 && i+1 < argc) {
            VS clist = SplitString(std::string(argv[i+1]));
            std::string tmp(argv[i+1]);
            if(clist[0] == "none"){
                opts->permute = 2;
            }else if(clist[0] == "nnz"){
                opts->permute = 3;
            }else if(clist[0] == "flops"){
                opts->permute = 4;
            }else if(clist[0] == "file"){
                opts->permute = 5;
                if (clist.size() == 3){
                    opts->permfile = clist[2]; // if file, then one should provide permute file.
                }
            }
            opts->nparts = stoi(clist[1]);
            i++;
        }
        else if ( strcmp("--randperm", argv[i]) == 0) {
            opts->permute = 1;
        }
        else if ( strcmp("--1d", argv[i]) == 0 ) {
            opts->run1d = true;
        }
        else if ( strcmp("--gentype", argv[i]) == 0 && i+1<argc ) {
            opts->gentype = {};
            VS clist = SplitString(std::string(argv[i+1]));
            for(auto x : clist){
                opts->gentype.push_back(std::atoi(x.c_str()));
            }
            i++;
        }
        else if ( strcmp("--2d", argv[i]) == 0 ) {
            opts->run2d = true;
        }
        
        else if ( strcmp("--display", argv[i]) == 0 ) {
            // show results in terminal
            opts->display = true;
        }
        else if ( strcmp("--3d", argv[i]) == 0 && i+1 < argc ) {
            opts->run3d = true;
            VS clist = SplitString(std::string(argv[i+1]));
            for(int i=0; i<clist.size();i++){
                opts->layer.push_back(stoi(clist[i]));
            }
            i++;
        }
        else if ( strcmp("--check", argv[i]) == 0 ) {
            opts->check = true; // if check, we will run 2d algorithm
        }
        else if ( strcmp("--niter", argv[i]) == 0 && i+1 < argc ) {
            opts->niter = atoi(argv[i+1]);
            i++;
        }
        else if ( strcmp("--memlimit", argv[i]) == 0 && i+1 < argc ) {
            opts->memorylimit = atof(argv[i+1]);
            i++;
        }
        else if ( strcmp("--logtofile", argv[i]) == 0 ) {
            opts->logtofile = true;
        }
        // else if ( strcmp("-h", argv[i]) == 0 || strcmp("--help", argv[i]) == 0 ) {
        //     exit(0);
        // }
        else {
            fprintf( stderr, "error: unrecognized option %s\n", argv[i] );
            exit(1);
        }
    }
    return 0;
}


};


template <class IT, class NT>
class Dcsc;

template <class IT, class NT>
class Csc;

template <class IT, class NT>
class SpDCCols;

template <class IT, class NT>
class SpCCols;

template <class IT, class NT>
class SpTuples;


class SpHelper
{


public:
/****************************************************************************/
/********************* Extension of STD LIB *********************************/
/****************************************************************************/
template<typename T>
static double calculateMean(const std::vector<T>& data) {
    if(data.size()==0)return 0;
    double sum = 0.0;
    for (T value : data) {
        sum += (double)value;
    }
    return sum / data.size();
}
template<typename T>
static double calculateStandardError(const std::vector<T>& data) {
    if(data.size()==0)return 0;
    double mean = calculateMean(data);
    double sumSquaredDifferences = 0.0;

    for (T value : data) {
        double diff = (double)value - mean;
        sumSquaredDifferences += diff * diff;
    }

    double variance = sumSquaredDifferences / (data.size() - 1);
    return std::sqrt(variance);
}

// convert a prefixsum back to origin array, 
// note that prefixsum must contain the tot sum, the prefixsum array size is n+1.
// otherwise we can't reverse the orgarr.
template<typename T>
static void DePrefixSum(const T * prefixsum, size_t length, std::vector<T> & orgarr){
    orgarr = std::vector<T>(length-1);
    #pragma omp parallel for 
    for(size_t i=1; i<length;i++){
        orgarr[i-1] = prefixsum[i] - prefixsum[i-1];
    }
}

static std::string GetHostPrefix(){
    std::string prefix;
    if(GetHostName() == "chitu"){ // chitu
        prefix = "/home/hongy0a/graphclustering/dataset/";
    }else if(GetHostName().substr(0,5) == "login" || GetHostName().substr(0,3) == "nid"){ // perlmutter 
        prefix = "/global/cfs/cdirs/m1982/yuxihong/spgemm1ddataset/";
    }
    return prefix;
}
static void initdatasetmap(){
    std::string prefix = GetHostPrefix();
    datasetmap["cage15"] = prefix + "cage15.mtx";
    datasetmap["diereal"] = prefix + "dielFilterV3real.mtx";
    datasetmap["dela24"] = prefix + "delaunay_n24.mtx";
    datasetmap["ldoor"] = prefix + "ldoor.mtx";
    datasetmap["hv15r"] = prefix + "HV15R.mtx";
    datasetmap["nlpkkt160"] = prefix + "nlpkkt160.mtx";
    datasetmap["nlpkkt200"] = prefix + "nlpkkt200.mtx";
    datasetmap["mouse_gene"] = prefix + "mouse_gene.mtx";
    datasetmap["queen"] = prefix + "Queen_4147.mtx";
    datasetmap["stokes"] = prefix + "stokes.mtx";
    datasetmap["vir"] = prefix + "vir_vs_vir_30_50length_propermm.mtx";
    datasetmap["euk"] = prefix + "euk_vs_euk_30_50length_propermm.mtx";
    datasetmap["arch"] = prefix + "arch_vs_arch_30_50length_propermm.mtx"; 
    datasetmap["orkut"] = prefix + "com-Orkut.mtx"; // square, non-symm
    datasetmap["isom100-1"] = "/global/cfs/cdirs/m1982/HipMCL/iso_m100/subgraph1/subgraph1_iso_vs_iso_30_70length_ALL.m100.indexed.mtx"; // symm 23 GB
    datasetmap["isom100-3"] = "/global/cfs/cdirs/m1982/HipMCL/iso_m100/subgraph3/subgraph3_iso_vs_iso_30_70length_ALL.m100.indexed.mtx"; // symm 383 GB
    datasetmap["friends"] = "/global/cfs/cdirs/m1982/www/GNN/unused/com-Friendster-generalmm.mtx"; // symm 66 GB
    datasetmap["mo16"] = "/global/cfs/cdirs/m1982/yuxihong/spgemm1ddataset/MOLIERE_2016.mtx"; // symm 83 GB
    datasetmap["agatha"] = "/global/cfs/cdirs/m1982/yuxihong/spgemm1ddataset/AGATHA_2015.mtx"; // symm 98 GB
    datasetmap["metaclust50"] = "/global/cfs/cdirs/m1982/www/HipMCL/Metaclust/Renamed_graph_Metaclust50_MATRIX_DENSE.txt";// symm 909 GB
    ckcorrectdlist = {
        "GD96_d",
        "odepb400",
        "lpi_ex73a",
        "cat_ears_3_1",
        "fs_680_3",
        "c8_mat11_I",
        "sme3Db",
        "lp_80bau3b",
        "G43",
        "1138_bus",
        "ex22",
        "bcsstk01"
    };

}
// A util function to get full path of dataset.
static std::string DatasetFullPath(std::string datasetname){
    if(datasetmap.find(datasetname) != datasetmap.end()) return datasetmap[datasetname];
    else {
        printf("dataset not found!!\n");
        exit(0);
    }
}

static std::string GetHostName() {
    char hostname[1024];
    if (gethostname(hostname, 1024) != 0) {
        std::cerr << "Error getting hostname on Linux\n";
        return "";
    }
    return std::string(hostname);
}

/**
 * @brief Compute prefixsum utility function. Slow sequential version.
 * 
 * @tparam T 
 * @param origin input vector.
 * @param keeplastelem if ture, size of return vector will be origin.size() + 1, last element is the total sum.
 * @return std::vector<T> return prefixsum vector.
 */
template<typename T>
static std::vector<T> prefixsum(std::vector<T> & origin, bool keeplastelem = false){
    size_t len = origin.size();
    if(keeplastelem) len++;
    std::vector<T> results(len,0);
    if(keeplastelem) __gnu_parallel::partial_sum(origin.begin(), origin.end(), results.begin()+1);
    else __gnu_parallel::partial_sum(origin.begin(), origin.end()-1, results.begin()+1);
    return results;
}


// flatten a 2D vector into a 1D vector
template <typename T>
static void flatten(std::vector<T> & flattenvector, const std::vector<std::vector<T>> & inputvector){
    size_t glen = 0;
    for(auto x : inputvector) glen += x.size();
    flattenvector = std::vector<T>(glen);
    size_t idx = 0;
    for(auto vec : inputvector) 
    for(auto x : vec) flattenvector[idx++] = x;
}

template <typename T>
static std::vector<size_t> find_order(const std::vector<T> & values);
    
template <typename IT1, typename NT1, typename IT2, typename NT2>
static void push_to_vectors(
    std::vector<IT1> & rows, std::vector<IT1> & cols, std::vector<NT1> & vals, IT2 ii, IT2 jj, 
    NT2 vv, int symmetric, bool onebased = true);

static void ProcessLinesWithStringKeys(
    std::vector< std::map < std::string, uint64_t> > & allkeys, std::vector<std::string> & lines, int nprocs);

template <typename IT1, typename NT1>
static void ProcessStrLinesNPermute(
    std::vector<IT1> & rows, std::vector<IT1> & cols, std::vector<NT1> & vals, std::vector<std::string> & lines, 
    std::map<std::string, uint64_t> & ultperm);

template <typename IT1, typename NT1>
static void ProcessLines(
    std::vector<IT1> & rows, std::vector<IT1> & cols, std::vector<NT1> & vals, std::vector<std::string> & lines, 
    int symmetric, int type, bool onebased = true);

template <typename T>
static const T * p2a (const std::vector<T> & v)   // pointer to array
{
    if(v.empty()) return NULL;
    else return (&v[0]);
}

template <typename T>
static T * p2a (std::vector<T> & v)   // pointer to array
{
    if(v.empty()) return NULL;
    else return (&v[0]);
}


template<typename _ForwardIterator>
static bool is_sorted(_ForwardIterator __first, _ForwardIterator __last)
{
    if (__first == __last)
        return true;

    _ForwardIterator __next = __first;
    for (++__next; __next != __last; __first = __next, ++__next)
        if (*__next < *__first)
            return false;
    return true;
    }
template<typename _ForwardIterator, typename _StrictWeakOrdering>
    static bool is_sorted(_ForwardIterator __first, _ForwardIterator __last,  _StrictWeakOrdering __comp)
    {
        if (__first == __last)
            return true;

    _ForwardIterator __next = __first;
    for (++__next; __next != __last; __first = __next, ++__next)
        if (__comp(*__next, *__first))
                return false;
        return true;
}
template<typename _ForwardIter, typename T>
static void iota(_ForwardIter __first, _ForwardIter __last, T __val)
{
    while (__first != __last)
            *__first++ = __val++;
}
template<typename In, typename Out, typename UnPred>
static Out copyIf(In first, In last, Out result, UnPred pred) 
{
    for ( ;first != last; ++first)
            if (pred(*first))
                *result++ = *first;
    return(result);
}

template<typename T, typename I1, typename I2>
static T ** allocate2D(I1 m, I2 n)
{
    T ** array = new T*[m];
    for(I1 i = 0; i<m; ++i) 
        array[i] = new T[n];
    return array;
}
template<typename T, typename I>
static void deallocate2D(T ** array, I m)
{
    for(I i = 0; i<m; ++i) 
        delete [] array[i];
    delete [] array;
}

/****************************************************************************/
/********************* IO ***************************************************/
/****************************************************************************/
template<typename T>
static void PrintSquareTables(std::ostream &os, std::vector<std::vector<T>> & data, bool isfloat, int cellwidth = 10){
    int nprocs = data.size();
    for(int i=0; i<data.size(); i++){
        if(data[i].size() != nprocs) {
            printf("PrintSquareTable data is wrong!\n");
            exit(0);
        }
    }
    os << "Rank"; for(int i=0; i<cellwidth-4; i++) os << " ";
    for(int i=0; i<nprocs; i++){
        os << std::left << std::setw(cellwidth) << i;
    }
    os << std::endl;
    for(int i=0; i<(nprocs+1) * cellwidth;i++) os << "-";
    os << std::endl;
    for(int i=0; i<nprocs; i++){
        os << std::setw(cellwidth) << std::left << i ;
        for(int j=0; j<nprocs; j++){
            if(isfloat){
                os << std::setprecision(1) << std::scientific << std::setw(cellwidth) << data[i][j];
            }else{
                os << std::setw(cellwidth) << std::left << data[i][j];
            }
        }
        os << std::endl;
    }
}
static void RemoveExistingFile(std::string filename){
    std::ifstream afile(filename.c_str());
    if (afile.good()) {
        afile.close();
        if (remove(filename.c_str()) == 0) {
            std::cout << filename << " exists " << ", deleted it." << std::endl;
        }
    }
}

static bool CheckFileExists(std::string filename, bool notfoundexit){

    // Create an input stream
    std::ifstream fileStream(filename);
    bool exists;
    // Check if the file stream is open, which means the file exists
    if (fileStream.is_open()) {
        // std::cout << "File exists: " << filename << std::endl;
        exists = true;
    } else {
        // std::cout << "File does not exist: " << filename << std::endl;
        exists = false;
    }
    if(notfoundexit && !exists){
        printf("file %s not exists, exit!\n", filename.c_str());
        exit(0);
    }
    return exists;
}
static void WriteIntegersToFile(const std::vector<int> a, const std::string filename) {
    std::ofstream os(filename);
    if(os.is_open()){
        for(auto x : a) os << x << " ";
    }else{
        std::cerr << "open file err: " << filename << std::endl;
    }
}
static void WriteIntegersToFile(const std::vector<int64_t> a, const std::string filename) {
    std::ofstream os(filename);
    if(os.is_open()){
        for(auto x : a) os << x << " ";
    }else{
        std::cerr << "open file err: " << filename << std::endl;
    }
}
static std::vector<int64_t> ReadIntegersFromFile(const std::string& filename) {
    std::vector<int64_t> result;
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return result; // Return an empty vector if file couldn't be opened
    }
    std::string line;
    std::getline(inputFile, line);
    std::istringstream iss(line);
    int64_t num;
    while (iss >> num) {
        result.push_back(num);
    }
    inputFile.close();
    return result;
}


template <typename SR, typename NT1, typename NT2, typename IT, typename OVT>
static IT Popping(NT1 * numA, NT2 * numB, StackEntry< OVT, std::pair<IT,IT> > * multstack,
        IT & cnz, KNHeap< std::pair<IT,IT> , IT > & sHeap, Isect<IT> * isect1, Isect<IT> * isect2);

template <typename IT, typename NT1, typename NT2>
static void SpIntersect(const Dcsc<IT,NT1> & Adcsc, const Dcsc<IT,NT2> & Bdcsc, Isect<IT>* & cols, Isect<IT>* & rows, 
        Isect<IT>* & isect1, Isect<IT>* & isect2, Isect<IT>* & itr1, Isect<IT>* & itr2);

template <typename SR, typename IT, typename NT1, typename NT2, typename OVT>
static IT SpCartesian(const Dcsc<IT,NT1> & Adcsc, const Dcsc<IT,NT2> & Bdcsc, IT kisect, Isect<IT> * isect1, 
        Isect<IT> * isect2, StackEntry< OVT, std::pair<IT,IT> > * & multstack);

template <typename SR, typename IT, typename NT1, typename NT2, typename OVT>
static IT SpColByCol(const Dcsc<IT,NT1> & Adcsc, const Dcsc<IT,NT2> & Bdcsc, IT nA,	 
        StackEntry< OVT, std::pair<IT,IT> > * & multstack);

template <typename NT, typename IT>
static void ShrinkArray(NT * & array, IT newsize)
{
    NT * narray = new NT[newsize];
    memcpy(narray, array, newsize*sizeof(NT));	// copy only a portion of the old elements

    delete [] array;
    array = narray;
}

template <typename NT, typename IT>
static void DoubleStack(StackEntry<NT, std::pair<IT,IT> > * & multstack, IT & cnzmax, IT add)
{
    StackEntry<NT, std::pair<IT,IT> > * tmpstack = multstack; 		
    multstack = new StackEntry<NT, std::pair<IT,IT> >[2* cnzmax + add];
    memcpy(multstack, tmpstack, sizeof(StackEntry<NT, std::pair<IT,IT> >) * cnzmax);
    
    cnzmax = 2*cnzmax + add;
    delete [] tmpstack;
}



};


template <typename T>
std::vector<size_t> SpHelper::find_order(const std::vector<T> & values)
{
    size_t index = 0;
    std::vector< std::pair<T, size_t> > tosort;
    for(auto & val: values)
    {
        tosort.push_back(std::make_pair(val,index++));
    }
    sort(tosort.begin(), tosort.end());
    std::vector<size_t> permutation;
    for(auto & sorted: tosort)
    {
        permutation.push_back(sorted.second);
    }
    return permutation;
}

template <typename IT1, typename NT1, typename IT2, typename NT2>
void SpHelper::push_to_vectors(
    std::vector<IT1> & rows, std::vector<IT1> & cols, std::vector<NT1> & vals, 
    IT2 ii, IT2 jj, NT2 vv, int symmetric, bool onebased)
{
    if(onebased)
    {
        ii--;  /* adjust from 1-based to 0-based */
        jj--;
    }
    rows.push_back(ii);
    cols.push_back(jj);
    vals.push_back(vv);
    if(symmetric && ii != jj)
    {
        rows.push_back(jj);
        cols.push_back(ii);
        vals.push_back(vv);
    }
}


inline void SpHelper::ProcessLinesWithStringKeys(
    std::vector< std::map < std::string, uint64_t> > & allkeys, std::vector<std::string> & lines, int nprocs)
{
    std::string frstr, tostr;
    uint64_t frhash, tohash;    
    double vv;
    for (auto itr=lines.begin(); itr != lines.end(); ++itr)
    {
        char fr[MAXVERTNAME];
        char to[MAXVERTNAME];
        sscanf(itr->c_str(), "%s %s %lg", fr, to, &vv);
        frstr = std::string(fr);
        tostr = std::string(to);
        MurmurHash3_x64_64(frstr.c_str(),frstr.size(),0,&frhash);
        MurmurHash3_x64_64(tostr.c_str(),tostr.size(),0,&tohash);	

        double range_fr = static_cast<double>(frhash) * static_cast<double>(nprocs);
        double range_to = static_cast<double>(tohash) * static_cast<double>(nprocs);
            size_t owner_fr = range_fr / static_cast<double>(std::numeric_limits<uint64_t>::max());
            size_t owner_to = range_to / static_cast<double>(std::numeric_limits<uint64_t>::max());

        // cout << frstr << " with hash " << frhash << " is going to " << owner_fr << endl;
        // cout << tostr << " with hash " << tohash << " is going to " << owner_to << endl;

        allkeys[owner_fr].insert(std::make_pair(frstr, frhash)); 
        allkeys[owner_to].insert(std::make_pair(tostr, tohash));
    }
    lines.clear();
}

template <typename IT1, typename NT1>
void SpHelper::ProcessStrLinesNPermute(
    std::vector<IT1> & rows, std::vector<IT1> & cols, std::vector<NT1> & vals, std::vector<std::string> & lines, 
    std::map<std::string, uint64_t> & ultperm)
{
    char * fr = new char[MAXVERTNAME];
        char * to = new char[MAXVERTNAME];
        std::string frstr, tostr;
        double vv;
        for (auto itr=lines.begin(); itr != lines.end(); ++itr)
        {
        sscanf(itr->c_str(), "%s %s %lg", fr, to, &vv);
        frstr = std::string(fr);
        tostr = std::string(to);

        rows.emplace_back((IT1) ultperm[frstr]);
        cols.emplace_back((IT1) ultperm[tostr]); 
        vals.emplace_back((NT1) vv);
    }
    delete [] fr;
    delete [] to;
    lines.clear();
}

template <typename IT1, typename NT1>
void SpHelper::ProcessLines(
    std::vector<IT1> & rows, std::vector<IT1> & cols, std::vector<NT1> & vals, std::vector<std::string> & lines, 
    int symmetric, int type, bool onebased)
{
    if(type == 0)   // real
    {
        int64_t ii, jj;
        double vv;
        for (auto itr=lines.begin(); itr != lines.end(); ++itr)
        {
            // string::c_str() -> Returns a pointer to an array that contains a null-terminated sequence of characters (i.e., a C-string)
            sscanf(itr->c_str(), "%ld %ld %lg", &ii, &jj, &vv);
            SpHelper::push_to_vectors(rows, cols, vals, ii, jj, vv, symmetric, onebased);
        }
    }
    else if(type == 1) // integer
    {
        int64_t ii, jj, vv;
        for (auto itr=lines.begin(); itr != lines.end(); ++itr)
        {
            sscanf(itr->c_str(), "%ld %ld %ld", &ii, &jj, &vv);
            SpHelper::push_to_vectors(rows, cols, vals, ii, jj, vv, symmetric, onebased);
        }
    }
    else if(type == 2) // pattern
    {
        int64_t ii, jj;
        for (auto itr=lines.begin(); itr != lines.end(); ++itr)
        {
            sscanf(itr->c_str(), "%ld %ld", &ii, &jj);
            SpHelper::push_to_vectors(rows, cols, vals, ii, jj, 1, symmetric, onebased);
        }
    }
    else
    {
        std::cout << "COMBBLAS: Unrecognized matrix market scalar type" << std::endl;
    }
    lines.clear();
}



/**
 * Pop an element, do the numerical semiring multiplication & insert the result into multstack
 */
template <typename SR, typename NT1, typename NT2, typename IT, typename OVT>
IT SpHelper::Popping(NT1 * numA, NT2 * numB, StackEntry< OVT, std::pair<IT,IT> > * multstack, 
			IT & cnz, KNHeap< std::pair<IT,IT>,IT > & sHeap, Isect<IT> * isect1, Isect<IT> * isect2)
{
	std::pair<IT,IT> key;	
	IT inc;
	sHeap.deleteMin(&key, &inc);

	OVT value = SR::multiply(numA[isect1[inc].current], numB[isect2[inc].current]);
	if (!SR::returnedSAID())
	{
		if(cnz != 0)
		{
			if(multstack[cnz-1].key == key)	// already exists
			{
				multstack[cnz-1].value = SR::add(multstack[cnz-1].value, value);
			}
			else
			{
				multstack[cnz].value = value;
				multstack[cnz].key   = key;
				++cnz;
			}
		}
		else
		{
			multstack[cnz].value = value;
			multstack[cnz].key   = key;
			++cnz;
		}
	}
	return inc;
}

/**
  * Finds the intersecting row indices of Adcsc and col indices of Bdcsc  
  * @param[IT] Bdcsc {the transpose of the dcsc structure of matrix B}
  * @param[IT] Adcsc {the dcsc structure of matrix A}
  **/
template <typename IT, typename NT1, typename NT2>
void SpHelper::SpIntersect(const Dcsc<IT,NT1> & Adcsc, const Dcsc<IT,NT2> & Bdcsc, Isect<IT>* & cols, Isect<IT>* & rows, 
				Isect<IT>* & isect1, Isect<IT>* & isect2, Isect<IT>* & itr1, Isect<IT>* & itr2)
{
	cols = new Isect<IT>[Adcsc.nzc];
	rows = new Isect<IT>[Bdcsc.nzc];
	
	for(IT i=0; i < Adcsc.nzc; ++i)			
	{
		cols[i].index	= Adcsc.jc[i];		// column index
		cols[i].size	= Adcsc.cp[i+1] - Adcsc.cp[i];
		cols[i].start	= Adcsc.cp[i];		// pointer to row indices
		cols[i].current = Adcsc.cp[i];		// pointer to row indices
	}
	for(IT i=0; i < Bdcsc.nzc; ++i)			
	{
		rows[i].index	= Bdcsc.jc[i];		// column index
		rows[i].size	= Bdcsc.cp[i+1] - Bdcsc.cp[i];
		rows[i].start	= Bdcsc.cp[i];		// pointer to row indices
		rows[i].current = Bdcsc.cp[i];		// pointer to row indices
	}

	/* A single set_intersection would only return the elements of one sequence 
	 * But we also want random access to the other array's elements 
	 * Thus we do the intersection twice
	 */
	IT mink = std::min(Adcsc.nzc, Bdcsc.nzc);
	isect1 = new Isect<IT>[mink];	// at most
	isect2 = new Isect<IT>[mink];	// at most
	itr1 = std::set_intersection(cols, cols + Adcsc.nzc, rows, rows + Bdcsc.nzc, isect1);	
	itr2 = std::set_intersection(rows, rows + Bdcsc.nzc, cols, cols + Adcsc.nzc, isect2);	
	// itr1 & itr2 are now pointing to one past the end of output sequences
}

/**
 * Performs cartesian product on the dcsc structures. 
 * Indices to perform the product are given by isect1 and isect2 arrays
 * Returns the "actual" number of elements in the merged stack
 * Bdcsc is "already transposed" (i.e. Bdcsc->ir gives column indices, and Bdcsc->jc gives row indices)
 **/
template <typename SR, typename IT, typename NT1, typename NT2, typename OVT>
IT SpHelper::SpCartesian(const Dcsc<IT,NT1> & Adcsc, const Dcsc<IT,NT2> & Bdcsc, IT kisect, Isect<IT> * isect1, 
		Isect<IT> * isect2, StackEntry< OVT, std::pair<IT,IT> > * & multstack)
{	
	std::pair<IT,IT> supremum(std::numeric_limits<IT>::max(), std::numeric_limits<IT>::max());
	std::pair<IT,IT> infimum (std::numeric_limits<IT>::min(), std::numeric_limits<IT>::min());
 
	KNHeap< std::pair<IT,IT> , IT > sHeapDcsc(supremum, infimum);	

	// Create a sequence heap that will eventually construct DCSC of C
	// The key to sort is pair<col_ind, row_ind> so that output is in column-major order
	for(IT i=0; i< kisect; ++i)
	{
		std::pair<IT,IT> key(Bdcsc.ir[isect2[i].current], Adcsc.ir[isect1[i].current]);
		sHeapDcsc.insert(key, i);
	}

	IT cnz = 0;						
	IT cnzmax = Adcsc.nz + Bdcsc.nz;	// estimate on the size of resulting matrix C
	multstack = new StackEntry< OVT, std::pair<IT,IT> > [cnzmax];	

	bool finished = false;
	while(!finished)		// multiplication loop  (complexity O(flops * log (kisect))
	{
		finished = true;
		if (cnz + kisect > cnzmax)		// double the size of multstack
		{
			DoubleStack(multstack, cnzmax, kisect);
		} 

		// inc: the list to increment its pointer in the k-list merging
		IT inc = Popping< SR >(Adcsc.numx, Bdcsc.numx, multstack, cnz, sHeapDcsc, isect1, isect2);
		isect1[inc].current++;	
		
		if(isect1[inc].current < isect1[inc].size + isect1[inc].start)
		{
			std::pair<IT,IT> key(Bdcsc.ir[isect2[inc].current], Adcsc.ir[isect1[inc].current]);
			sHeapDcsc.insert(key, inc);	// push the same element with a different key [increasekey]
			finished = false;
		}
		// No room to go in isect1[], but there is still room to go in isect2[i]
		else if(isect2[inc].current + 1 < isect2[inc].size + isect2[inc].start)
		{
			isect1[inc].current = isect1[inc].start;	// wrap-around
			isect2[inc].current++;

			std::pair<IT,IT> key(Bdcsc.ir[isect2[inc].current], Adcsc.ir[isect1[inc].current]);
			sHeapDcsc.insert(key, inc);	// push the same element with a different key [increasekey]
			finished = false;
		}
		else // don't push, one of the lists has been deplated
		{
			kisect--;
			if(kisect != 0)
			{
				finished = false;
			}
		}
	}
	return cnz;
}


template <typename SR, typename IT, typename NT1, typename NT2, typename OVT>
IT SpHelper::SpColByCol(const Dcsc<IT,NT1> & Adcsc, const Dcsc<IT,NT2> & Bdcsc, IT nA, 
			StackEntry< OVT, std::pair<IT,IT> > * & multstack)
{
	IT cnz = 0;
	IT cnzmax = Adcsc.nz + Bdcsc.nz;	// estimate on the size of resulting matrix C
	multstack = new StackEntry<OVT, std::pair<IT,IT> >[cnzmax];	 

	float cf  = static_cast<float>(nA+1) / static_cast<float>(Adcsc.nzc);
	IT csize = static_cast<IT>(ceil(cf));   // chunk size
	IT * aux;
	//IT auxsize = Adcsc.ConstructAux(nA, aux);
	Adcsc.ConstructAux(nA, aux);

	for(IT i=0; i< Bdcsc.nzc; ++i)		// for all the columns of B
	{
		IT prevcnz = cnz;
		IT nnzcol = Bdcsc.cp[i+1] - Bdcsc.cp[i];
		HeapEntry<IT, NT1> * wset = new HeapEntry<IT, NT1>[nnzcol]; 
		// heap keys are just row indices (IT) 
		// heap values are <numvalue, runrank>  
		// heap size is nnz(B(:,i)

		// colnums vector keeps column numbers requested from A
		std::vector<IT> colnums(nnzcol);

		// colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
		// colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)
		std::vector< std::pair<IT,IT> > colinds(nnzcol);		
    std::copy(Bdcsc.ir + Bdcsc.cp[i], Bdcsc.ir + Bdcsc.cp[i+1], colnums.begin());
		
		Adcsc.FillColInds(&colnums[0], colnums.size(), colinds, aux, csize);
		IT maxnnz = 0;	// max number of nonzeros in C(:,i)	
		IT hsize = 0;
		
		for(IT j = 0; (unsigned)j < colnums.size(); ++j)		// create the initial heap 
		{
			if(colinds[j].first != colinds[j].second)	// current != end
			{
				wset[hsize++] = HeapEntry< IT,NT1 > (Adcsc.ir[colinds[j].first], j, Adcsc.numx[colinds[j].first]);
				maxnnz += colinds[j].second - colinds[j].first;
			} 
		}	
		std::make_heap(wset, wset+hsize);

		if (cnz + maxnnz > cnzmax)		// double the size of multstack
		{
			SpHelper::DoubleStack(multstack, cnzmax, maxnnz);
		} 

		// No need to keep redefining key and hentry with each iteration of the loop
		while(hsize > 0)
		{
			std::pop_heap(wset, wset + hsize);         // result is stored in wset[hsize-1]
			IT locb = wset[hsize-1].runr;	// relative location of the nonzero in B's current column 

			// type promotion done here: 
			// static T_promote multiply(const T1 & arg1, const T2 & arg2)
			//	return (static_cast<T_promote>(arg1) * static_cast<T_promote>(arg2) );
			OVT mrhs = SR::multiply(wset[hsize-1].num, Bdcsc.numx[Bdcsc.cp[i]+locb]);
			if (!SR::returnedSAID())
			{
				if(cnz != prevcnz && multstack[cnz-1].key.second == wset[hsize-1].key)	// if (cnz == prevcnz) => first nonzero for this column
				{
					multstack[cnz-1].value = SR::add(multstack[cnz-1].value, mrhs);
				}
				else
				{
					multstack[cnz].value = mrhs;
					multstack[cnz++].key = std::make_pair(Bdcsc.jc[i], wset[hsize-1].key);	
					// first entry is the column index, as it is in column-major order
				}
			}
			
			if( (++(colinds[locb].first)) != colinds[locb].second)	// current != end
			{
				// runr stays the same !
				wset[hsize-1].key = Adcsc.ir[colinds[locb].first];
				wset[hsize-1].num = Adcsc.numx[colinds[locb].first];  
				std::push_heap(wset, wset+hsize);
			}
			else
			{
				--hsize;
			}
		}
		delete [] wset;
	}
	delete [] aux;
	return cnz;
}

inline std::pair<int,int> BlockIndex(std::vector<int> BlocksizePrefix, int RowId, int ColId)
{
    int bridx,bcidx,nblocks=BlocksizePrefix.size();
    int owner = 0;
    while(owner <= nblocks-1 && BlocksizePrefix[owner] <= ColId) owner++;
    owner-=1;
    bcidx = owner;
    owner = 0;
    while(owner <= nblocks-1 && BlocksizePrefix[owner] <= RowId) owner++;
    owner-=1;
    bridx = owner;
    return std::make_pair(bridx, bcidx);
}

/**
 * @brief Project non-zero elements of a Csc object along both Column and Row.
 * 
 * @tparam IT index type
 * @tparam NT numeric type
 * @param nrows [in] number of rows
 * @param ncols [in] number of cols
 * @param csc [in] Csc Object
 * @param ProjCol [out] Project non-zero elements in a Column to a bool vector
 * @param ProjRow [out] Project non-zero elements in a Row to a bool vector
 * @param projectcol [in] whether to do projection in Column, default true
 * @param projectrow [in] whether to do projection in Row, default true
Suppose csc has the following format,  

 x x     x
          
 x   x    
 x x x   x

ProjCol will be [true, true, true, false, true]
ProjRow will be [true, false, true, true]

 */
template<class IT, class NT, class DER>
void ProjectCscAlongRowAndCol(IT nrows, IT ncols, DER * seq, bool * ProjCol, bool * ProjRow, 
bool projectcol=true, bool projectrow=true){
    if(projectcol){
        for(typename DER::SpColIter colit = seq->begcol(); colit != seq->endcol(); ++colit)
        {
            IT gcol = colit.colid();
            ProjCol[gcol] = true;
        }
    }

    if(projectrow){
        for(typename DER::SpColIter colit = seq->begcol(); colit != seq->endcol(); ++colit)
        {
            IT gcol = colit.colid();
            for(typename DER::SpColIter::NzIter nzit = seq->begnz(colit); nzit != seq->endnz(colit); ++nzit)
            {
                IT grow = nzit.rowid();
                ProjRow[grow] = true;
            }
        }
    }

}






}

#endif
