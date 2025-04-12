
#include "CombBLAS/csc.h"
#include "CombBLAS/dcsc.h"
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/tuple.hpp"
#include "jlcxx/stl.hpp"
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <string>
#include <vector>
#include <tuple>
#include <cstdint>
#include <fstream>

using namespace std;


// An interface for FMM 
struct triplet_matrix {
    int64_t nrows_ = 0;
    int64_t ncols_ = 0;
    std::vector<int64_t> rows_;
    std::vector<int64_t> cols_;
    std::vector<double> vals_;   // or int64_t, float, std::complex<double>, etc.
    vector<int64_t> ReadMtxGetSize()       {return {nrows_, ncols_};}
    vector<double>  ReadMtxGetValues()    {return vals_;}
    vector<int64_t> ReadMtxGetRowindex()   {return rows_;}
    vector<int64_t> ReadMtxGetColindex()   {return cols_;}
};

triplet_matrix mmread(std::string filename){
    std::ifstream input_stream(filename);
    triplet_matrix mat;
    fast_matrix_market::read_matrix_market_triplet(
                input_stream,
                mat.nrows_, mat.ncols_,
                mat.rows_, mat.cols_, mat.vals_);
    // fmm automatically -1 for coordinates, but julia is 1 based. so add it back.
    #pragma omp simd
    for(int64_t i=0; i<mat.rows_.size(); i++){
        mat.rows_[i]++;
        mat.cols_[i]++;
    }
    return mat;
}

// template class combblas::SpDCCols<int64_t, double>;
// template class combblas::SpMat<int64_t, double,combblas::SpDCCols<int64_t, double>>;


JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    mod.add_type<triplet_matrix>("triplet_matrix")
    .method("ReadMtxGetSize",&triplet_matrix::ReadMtxGetSize)
    .method("ReadMtxGetRowindex",&triplet_matrix::ReadMtxGetRowindex)
    .method("ReadMtxGetColindex",&triplet_matrix::ReadMtxGetColindex)
    .method("ReadMtxGetValues",&triplet_matrix::ReadMtxGetValues)
    ;
    mod.method("mmread", &mmread);
    mod.add_type<combblas::Csc<int64_t, double>>("Csc");
    mod.add_type<combblas::Dcsc<int64_t, double>>("Dcsc");
}