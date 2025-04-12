


#include <fast_matrix_market/fast_matrix_market.hpp>
#include <fstream>
struct triplet_matrix {
    int64_t nrows = 0, ncols = 0;
    std::vector<int64_t> rows, cols;
    std::vector<double> vals;       // or int64_t, float, std::complex<double>, etc.
};
int main(){

    triplet_matrix mat;
    std::ifstream input_stream("/pscratch/sd/y/yuxihong/graphclustering/dataset/1DspGEMM/mouse_gene.mtx");

    fast_matrix_market::read_matrix_market_triplet(
                    input_stream,
                    mat.nrows, mat.ncols,
                    mat.rows, mat.cols, mat.vals);
    for(int i=0; i<10; i++){
        printf("i %d %ld %ld \n", i, mat.rows[i], mat.cols[i]);
    }
    return 0;
}
