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

matrix matrix_mult(const matrix *A, const matrix *B)
{
    const int rowsA = A->rows;
    const int colsA = A->cols;
    const int rowsB = B->rows;
    const int colsB = B->cols;
    assert(colsA == rowsB);
    matrix C = new_matrix(rowsA, colsB);
    for (int i = 1; i <= rowsA; i++)
        for (int j = 1; j <= colsB; j++)
            for (int k = 1; k <= colsA; k++)
            {
                mget(C, i, j) += mgetp(A, i, k) * mgetp(B, k, j);
            }
    return C;
}

matrix matrix_mult_v2(const matrix *A, const matrix *B)
{
    const int rowsA = A->rows;
    const int colsA = A->cols;
    const int rowsB = B->rows;
    const int colsB = B->cols;
    assert(colsA == rowsB);
    matrix C = new_matrix(rowsA, colsB);
    matrix Btranspose = new_matrix(rowsB, colsB);
    for (int i = 1; i <= rowsA; i++)
        for (int j = 1; j <= colsB; j++)
        {
            mget(Btranspose, i, j) = mgetp(B, j, i);
        }
    for (int i = 1; i <= rowsA; i++)
        for (int j = 1; j <= colsB; j++)
            for (int k = 1; k <= colsA; k++)
            {
                mget(C, i, j) += mgetp(A, i, k) * mget(Btranspose, j, k);
            }
    return C;
}

matrix matrix_dot_mult(const matrix *A, const matrix *B)
{
    const int rows = A->rows;
    const int cols = A->cols;
    assert(rows == B->rows);
    assert(cols == B->cols);
    matrix C = new_matrix(rows, cols);
    for (int i = 1; i <= rows; i++)
        for (int j = 1; j <= cols; j++)
        {
            mget(C, i, j) = mgetp(A, i, j) * mgetp(B, i, j);
        }
    return C;
}

vector new_vector(const int size)
{
    vector vec;
    vec.size = size;
    assert(size > 0);
    vec.val = (double *)malloc(sizeof(double) * size);
    for (int i = 0; i < (size); i++)
    {
        vec.val[i] = 0.0;
    }
    return vec;
}

void print_vector_full(const vector *vec, char *varname)
{
    assert(vec->size > 0);
    printf("\n");
    printf(" %.100s =\n", &varname[1]);
    printf(" | ");
    for (int i = 1; i <= vec->size; i++)
    {
        printf("%10.3e", vgetp(vec, i));
        if (i < vec->size)
        {
            printf(", ");
        }
    }
    printf(" |^T\n\n");
}

vector vector_add(const vector *x, const vector *y)
{
    const int size = x->size;
    assert(size == y->size);
    vector z = new_vector(size);
    for (int i = 1; i <= size; i++)
    {
        vget(z, i) = vgetp(x, i) + vgetp(y, i);
    }
    return z;
}

vector vector_sub(const vector *x, const vector *y)
{
    const int size = x->size;
    assert(size == y->size);
    vector z = new_vector(size);
    for (int i = 1; i <= size; i++)
    {
        vget(z, i) = vgetp(x, i) - vgetp(y, i);
    }
    return z;
}

double vector_dot_mult(const vector *x, const vector *y)
{
    const int size = x->size;
    assert(size == y->size);
    double z = 0.0;
    for (int i = 1; i <= size; i++)
    {
        z += vgetp(x, i) * vgetp(y, i);
    }
    return z;
}

void print_scalar_full(const double *z, char *varname)
{
    printf("\n %.100s =\n", &varname[1]);
    printf(" %10.3e \n\n", *z);
}

vector matrix_vector_mult(const matrix *A, const vector *x)
{
    const int rows = A->rows;
    const int cols = A->cols;
    const int size = x->size;
    assert(cols == size);
    vector Ax = new_vector(rows);
    for (int i = 1; i <= rows; i++)
    {
        double tmp = 0.0;
        for (int j = 1; j <= size; j++)
        {
            tmp += mgetp(A, i, j) * vgetp(x, j);
        }
        vget(Ax, i) = tmp;
    }
    return Ax;
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

vector solve(const matrix *A, const vector *b)
{
    const int rows = A->rows;
    const int cols = A->cols;
    const int size = b->size;
    assert(rows == cols);
    assert(rows == size);
    vector x = new_vector(rows);
    for (int i = 1; i <= (size - 1); i++) // LOOP OVER EACH COLUMN
    {
        // Select largest pivot in current column
        int p = i;
        double maxA = -100.0e0;
        for (int j = i; j <= size; j++)
        {
            double tmp = fabs(mgetp(A, j, i));
            if (tmp > maxA)
            {
                p = j;
                maxA = tmp;
            }
        }
        // See if matrix is singular
        if (maxA <= 1.0e-14)
        {
            printf(" Cannot invert system\n");
            exit(1);
        }
        // Pivot (aka interchange rows)
        if (p != i)
        {
            for (int j = 1; j <= size; j++)
            {
                double tmp1 = mgetp(A, i, j);
                mgetp(A, i, j) = mgetp(A, p, j);
                mgetp(A, p, j) = tmp1;
            }
            double tmp2 = vgetp(b, i);
            vgetp(b, i) = vgetp(b, p);
            vgetp(b, p) = tmp2;
        }
        // Eliminate below diagonal
        for (int j = (i + 1); j <= size; j++)
        {
            double dm = mgetp(A, j, i) / mgetp(A, i, i);
            for (int k = 1; k <= size; k++)
            {
                mgetp(A, j, k) = mgetp(A, j, k) - dm * mgetp(A, i, k);
            }
            vgetp(b, j) = vgetp(b, j) - dm * vgetp(b, i);
        }
    }
    // Backward substitution
    vget(x, size) = vgetp(b, size) / mgetp(A, size, size);
    for (int j = 1; j <= (size - 1); j++)
    {
        double sum = 0.0e0;
        for (int k = (size - j + 1); k <= size; k++)
        {
            sum = sum + mgetp(A, size - j, k) * vget(x, k);
        }
        vget(x, size - j) = (vgetp(b, size - j) - sum) / mgetp(A, size - j, size - j);
    }
    return x;
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

void delete_vector(vector *v)
{
    if (v->val == NULL)
        return;
    free(v->val);
    v->val = NULL;
    v->size = 0;
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
