#include <iostream>
#include <mkl_spblas.h>
#include <mkl_types.h>
#include <mkl.h>

int main() {
    // Define the dimensions of the matrices
    int m = 1000; // rows of A
    int n = 1000; // columns of B
    int k = 1000; // rows of B and columns of A

    // Define the sparse matrices A and B
    int nnz_A = 10000; // Number of non-zero elements in A
    int nnz_B = 10000; // Number of non-zero elements in B

    double* values_A = new double[nnz_A];
    MKL_INT* row_index_A = new MKL_INT[m + 1];
    MKL_INT* col_index_A = new MKL_INT[nnz_A];

    double* values_B = new double[nnz_B];
    MKL_INT* row_index_B = new MKL_INT[k + 1];
    MKL_INT* col_index_B = new MKL_INT[nnz_B];

    // Fill values of A and B matrices (considering they are already in CSR format)
    // Fill values_A, row_index_A, col_index_A, values_B, row_index_B, col_index_B accordingly

    // Define the CSR matrices
    sparse_matrix_t A, B, C;
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, m, k, row_index_A, row_index_A + 1, col_index_A, values_A);
    mkl_sparse_d_create_csr(&B, SPARSE_INDEX_BASE_ZERO, k, n, row_index_B, row_index_B + 1, col_index_B, values_B);

    // Create a handle for the output matrix C
    mkl_sparse_d_create_csr(&C, SPARSE_INDEX_BASE_ZERO, m, n, nullptr, nullptr, nullptr, nullptr);

    // Perform sparse matrix-matrix multiplication
    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A, B, &C);

    // Deallocate matrices A and B
    mkl_sparse_destroy(A);
    mkl_sparse_destroy(B);

    // Retrieve the result matrix C
    sparse_status_t status;
    double* result_values;
    MKL_INT* result_row_index;
    MKL_INT* result_col_index;
    MKL_INT nrows, ncols;
    // status = mkl_sparse_d_export_csr(C, &SPARSE_INDEX_BASE_ZERO, &nrows, &ncols, &result_row_index, &result_row_index[1], &result_col_index, &result_values);

    // Print the result matrix C
    for (int i = 0; i < nrows; ++i) {
        for (int j = result_row_index[i]; j < result_row_index[i + 1]; ++j) {
            std::cout << "C[" << i << "][" << result_col_index[j] << "] = " << result_values[j] << std::endl;
        }
    }

    // Deallocate the result matrix C
    mkl_sparse_destroy(C);
    delete[] result_values;
    delete[] result_row_index;
    delete[] result_col_index;

    // Deallocate memory for matrices A and B
    delete[] values_A;
    delete[] row_index_A;
    delete[] col_index_A;

    delete[] values_B;
    delete[] row_index_B;
    delete[] col_index_B;

    return 0;
}
