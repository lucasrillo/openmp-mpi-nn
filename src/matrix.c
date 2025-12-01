#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "matrix.h"

matrix new_matrix(const int rows, const int cols)
{
    matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    assert(rows > 0);
    assert(cols > 0);
    // Use calloc for zero-initialization (faster than malloc + loop)
    mat.val = (double *)calloc(rows * cols, sizeof(double));
    assert(mat.val != NULL);
    return mat;
}

matrix matrix_add(const matrix *A, const matrix *B)
{
    const int rows = A->rows;
    const int cols = A->cols;
    assert(rows == B->rows);
    assert(cols == B->cols);
    matrix C = new_matrix(rows, cols);
    for (int i = 1; i <= rows; i++)
        for (int j = 1; j <= cols; j++)
        {
            mget(C, i, j) = mgetp(A, i, j) + mgetp(B, i, j);
        }
    return C;
}

matrix matrix_sub(const matrix *A, const matrix *B)
{
    const int rows = A->rows;
    const int cols = A->cols;
    assert(rows == B->rows);
    assert(cols == B->cols);
    matrix C = new_matrix(rows, cols);
    for (int i = 1; i <= rows; i++)
        for (int j = 1; j <= cols; j++)
        {
            mget(C, i, j) = mgetp(A, i, j) - mgetp(B, i, j);
        }
    return C;
}

matrix matrix_mult(const matrix *A, const matrix *B) // Matrix mult v2: cache optimized
{
    const int rowsA = A->rows;
    const int colsA = A->cols;
    const int rowsB = B->rows;
    const int colsB = B->cols;
    assert(colsA == rowsB);
    
    matrix C = new_matrix(rowsA, colsB);
    matrix Btranspose = new_matrix(colsB, rowsB);
    
    // Transpose B for cache-friendly access
    for (int i = 1; i <= rowsB; i++)
        for (int j = 1; j <= colsB; j++)
            mget(Btranspose, j, i) = mgetp(B, i, j);
    
    // Both accesses are row-wise (cache-friendly)
    for (int i = 1; i <= rowsA; i++)
        for (int j = 1; j <= colsB; j++)
        {
            double sum = 0.0;
            for (int k = 1; k <= colsA; k++)
                sum += mgetp(A, i, k) * mget(Btranspose, j, k);
            mget(C, i, j) = sum;
        }
    
    delete_matrix(&Btranspose);
    return C;
}

matrix matrix_transpose(const matrix *A)
{
    const int rows = A->rows;
    const int cols = A->cols;
    matrix At = new_matrix(cols, rows);

    for (int i = 1; i <= rows; i++)
        for (int j = 1; j <= cols; j++)
            mget(At, j, i) = mgetp(A, i, j);

    return At;
}

void delete_matrix(matrix *A)
{
    if (A->val == NULL)
        return;
    free(A->val);
    A->val = NULL;
    A->rows = 0;
    A->cols = 0;
}

matrix matrix_sum_rows(const matrix *A)
{
    matrix v = new_matrix(A->rows, 1);
    for (int i = 1; i <= A->rows; i++)
    {
        double sum = 0.0;
        for (int j = 1; j <= A->cols; j++)
            sum += mgetp(A, i, j);
        mget(v, i, 1) = sum;
    }
    return v;
}

matrix matrix_scalar_mult(const matrix *A, double scalar)
{
    matrix C = new_matrix(A->rows, A->cols);
    for (int i = 1; i <= A->rows; i++)
        for (int j = 1; j <= A->cols; j++)
            mget(C, i, j) = mgetp(A, i, j) * scalar;
    return C;
}

matrix matrix_mult_add_col(const matrix *W, const matrix *A, const matrix *b)
{
    const int rowsW = W->rows;
    const int colsW = W->cols;
    const int rowsA = A->rows;
    const int colsA = A->cols;
    assert(colsW == rowsA);
    assert(rowsW == b->rows);
    assert(b->cols == 1);
    
    matrix Z = new_matrix(rowsW, colsA);
    matrix Atranspose = new_matrix(colsA, rowsA);
    
    // Transpose A for cache-friendly access
    for (int i = 1; i <= rowsA; i++)
        for (int j = 1; j <= colsA; j++)
            mget(Atranspose, j, i) = mgetp(A, i, j);
    
    // Compute W*A + b in one pass
    for (int i = 1; i <= rowsW; i++)
    {
        double bias = mgetp(b, i, 1);
        for (int j = 1; j <= colsA; j++)
        {
            double sum = bias;
            for (int k = 1; k <= colsW; k++)
                sum += mgetp(W, i, k) * mget(Atranspose, j, k);
            mget(Z, i, j) = sum;
        }
    }
    
    delete_matrix(&Atranspose);
    return Z;
}

// Computes: result = (A * B^T) * scalar
// Used in backward pass: dW = (dZ * A^T) / m
matrix matrix_mult_transB_scale(const matrix *A, const matrix *B, double scalar)
{
    const int rowsA = A->rows;
    const int colsA = A->cols;
    const int rowsB = B->rows;
    const int colsB = B->cols;
    assert(colsA == colsB);  // A * B^T requires A.cols == B.cols
    
    matrix C = new_matrix(rowsA, rowsB);
    
    // Both A and B are accessed row-wise (cache-friendly)
    for (int i = 1; i <= rowsA; i++)
        for (int j = 1; j <= rowsB; j++)
        {
            double sum = 0.0;
            for (int k = 1; k <= colsA; k++)
                sum += mgetp(A, i, k) * mgetp(B, j, k);
            mget(C, i, j) = sum * scalar;
        }
    
    return C;
}

// Computes: result = A^T * B
// Used in backward pass: dA_prev = W^T * dZ
matrix matrix_multT_B(const matrix *A, const matrix *B)
{
    const int rowsA = A->rows;
    const int colsA = A->cols;
    const int rowsB = B->rows;
    const int colsB = B->cols;
    assert(rowsA == rowsB);  // A^T * B requires A.rows == B.rows
    
    matrix C = new_matrix(colsA, colsB);
    
    // Reorganize for better cache access
    // C[i,j] = sum_k A[k,i] * B[k,j]
    // Process B column by column with A transposed access
    for (int i = 1; i <= colsA; i++)
        for (int j = 1; j <= colsB; j++)
        {
            double sum = 0.0;
            for (int k = 1; k <= rowsA; k++)
                sum += mgetp(A, k, i) * mgetp(B, k, j);
            mget(C, i, j) = sum;
        }
    
    return C;
}