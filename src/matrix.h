#ifndef MATRIX_H
#define MATRIX_H

typedef struct matrix matrix;
struct matrix
{
    int rows;
    int cols;
    double *val;
};

// Shortcut evaluate functions
#define mget(mat, i, j) mat.val[(i - 1) * mat.cols + (j - 1)]
#define mgetp(mat, i, j) mat->val[(i - 1) * mat->cols + (j - 1)]

// Function declarations
matrix new_matrix(const int rows, const int cols);
matrix matrix_add(const matrix *A, const matrix *B);
matrix matrix_sub(const matrix *A, const matrix *B);
matrix matrix_mult(const matrix *A, const matrix *B);
matrix matrix_transpose(const matrix *A);
void delete_matrix(matrix *A);
//
matrix matrix_sum_rows(const matrix *A);
matrix matrix_scalar_mult(const matrix *A, double scalar);
matrix matrix_mult_add_col(const matrix *W, const matrix *A, const matrix *b);

// Computes: result = (A * B^T) * scalar
matrix matrix_mult_transB_scale(const matrix *A, const matrix *B, double scalar);

// Computes: result = (A^T * B)
matrix matrix_multT_B(const matrix *A, const matrix *B);

#endif // MATRIX_H
