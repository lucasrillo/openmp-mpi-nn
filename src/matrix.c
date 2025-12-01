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
    mat.val = (double *)malloc(sizeof(double) * rows * cols);
    for (int i = 0; i < (rows * cols); i++)
    {
        mat.val[i] = 0.0;
    }
    return mat;
}

void print_matrix_full(const matrix *mat, char *varname)
{
    assert(mat->rows > 0);
    assert(mat->cols > 0);
    printf("\n %.100s =\n", &varname[1]);
    for (int i = 1; i <= mat->rows; i++)
    {
        printf(" | ");
        for (int j = 1; j <= mat->cols; j++)
        {
            printf("%10.3e", mgetp(mat, i, j));
            if (j < mat->cols)
            {
                printf(", ");
            }
            else
            {
                printf(" ");
            }
        }
        printf("|\n");
    }
    printf("\n");
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
    matrix Btranspose = new_matrix(colsB, rowsB);  // FIX: Swap dimensions
    
    // FIX: Correct transposition
    for (int i = 1; i <= rowsB; i++)
        for (int j = 1; j <= colsB; j++)
        {
            mget(Btranspose, j, i) = mgetp(B, i, j);
        }
    
    // Now both accesses are row-wise (cache-friendly)
    for (int i = 1; i <= rowsA; i++)
        for (int j = 1; j <= colsB; j++)
        {
            double sum = 0.0;
            for (int k = 1; k <= colsA; k++)
            {
                sum += mgetp(A, i, k) * mget(Btranspose, j, k);
            }
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
        {
            mget(At, j, i) = mgetp(A, i, j);
        }

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

matrix matrix_add_col_vector(const matrix *A, const matrix *v)
{
    assert(A->rows == v->rows);
    assert(v->cols == 1);
    matrix C = new_matrix(A->rows, A->cols);
    for (int i = 1; i <= A->rows; i++)
        for (int j = 1; j <= A->cols; j++)
        {
            mget(C, i, j) = mgetp(A, i, j) + mgetp(v, i, 1);
        }
    return C;
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
        {
            mget(C, i, j) = mgetp(A, i, j) * scalar;
        }
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
        {
            mget(Atranspose, j, i) = mgetp(A, i, j);
        }
    
    // Compute W*A + b in one pass
    for (int i = 1; i <= rowsW; i++)
    {
        double bias = mgetp(b, i, 1);
        for (int j = 1; j <= colsA; j++)
        {
            double sum = bias;  // Add bias directly
            for (int k = 1; k <= colsW; k++)
            {
                sum += mgetp(W, i, k) * mget(Atranspose, j, k);
            }
            mget(Z, i, j) = sum;
        }
    }
    
    delete_matrix(&Atranspose);
    return Z;
}