#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "nn.h"

matrix relu(const matrix *Z)
{
    matrix A = new_matrix(Z->rows, Z->cols);
    for (int i = 1; i <= Z->rows; i++)
        for (int j = 1; j <= Z->cols; j++)
            mget(A, i, j) = fmax(0.0, mgetp(Z, i, j));
    return A;
}

matrix softmax(const matrix *Z)
{
    matrix A = new_matrix(Z->rows, Z->cols);
    for (int j = 1; j <= Z->cols; j++)
    {
        double max_val = mgetp(Z, 1, j);
        for (int i = 2; i <= Z->rows; i++)
            max_val = fmax(max_val, mgetp(Z, i, j));

        double sum = 0.0;
        for (int i = 1; i <= Z->rows; i++)
        {
            mget(A, i, j) = exp(mgetp(Z, i, j) - max_val);
            sum += mget(A, i, j);
        }
        for (int i = 1; i <= Z->rows; i++)
            mget(A, i, j) /= sum;
    }
    return A;
}

matrix relu_backward(const matrix *dA, const matrix *Z_cache)
{
    matrix dZ = new_matrix(dA->rows, dA->cols);
    for (int i = 1; i <= dA->rows; i++)
        for (int j = 1; j <= dA->cols; j++)
            mget(dZ, i, j) = mgetp(Z_cache, i, j) > 0 ? mgetp(dA, i, j) : 0.0;
    return dZ;
}

linear_cache linear_forward(const matrix *A, const matrix *W, const matrix *b)
{
    linear_cache cache;
    cache.A = *A;
    cache.W = *W;
    cache.b = *b;

    matrix WA = matrix_mult(W, A);
    cache.Z = matrix_add_col_vector(&WA, b);
    delete_matrix(&WA);

    return cache;
}

layer_cache linear_activation_forward(const matrix *A_prev, const matrix *W,
                                      const matrix *b, const char *activation)
{
    layer_cache cache;
    cache.linear = linear_forward(A_prev, W, b);

    if (strcmp(activation, "relu") == 0)
        cache.A = relu(&cache.linear.Z);
    else if (strcmp(activation, "softmax") == 0)
        cache.A = softmax(&cache.linear.Z);

    return cache;
}

forward_pass L_model_forward(const matrix *X, const nn_params *params)
{
    forward_pass fwd;
    fwd.caches = (layer_cache *)malloc(sizeof(layer_cache) * params->L);

    matrix A = *X;
    for (int l = 0; l < params->L - 1; l++)
    {
        fwd.caches[l] = linear_activation_forward(&A, &params->W[l], &params->b[l], "relu");
        delete_matrix(&A);
        A = fwd.caches[l].A;
    }

    fwd.caches[params->L - 1] = linear_activation_forward(&A,
                                                          &params->W[params->L - 1],
                                                          &params->b[params->L - 1], "softmax");
    delete_matrix(&A);
    fwd.AL = fwd.caches[params->L - 1].A;

    return fwd;
}

double compute_cost(const matrix *AL, const matrix *Y)
{
    int m = Y->cols;
    double cost = 0.0;

    for (int j = 1; j <= m; j++)
        for (int i = 1; i <= Y->rows; i++)
            if (mgetp(Y, i, j) > 0)
                cost -= mgetp(Y, i, j) * log(mgetp(AL, i, j) + 1e-8);

    return cost / m;
}

linear_grads linear_backward(const matrix *dZ, const linear_cache *cache)
{
    linear_grads grads;
    int m = cache->A.cols;

    matrix A_T = matrix_transpose(&cache->A);
    matrix dW_unnorm = matrix_mult(dZ, &A_T);
    grads.dW = matrix_scalar_mult(&dW_unnorm, 1.0 / m);
    delete_matrix(&A_T);
    delete_matrix(&dW_unnorm);

    matrix db_sum = matrix_sum_rows(dZ);
    grads.db = matrix_scalar_mult(&db_sum, 1.0 / m);
    delete_matrix(&db_sum);

    matrix W_T = matrix_transpose(&cache->W);
    grads.dA_prev = matrix_mult(&W_T, dZ);
    delete_matrix(&W_T);

    return grads;
}

linear_grads linear_activation_backward(const matrix *dA, const layer_cache *cache,
                                        const char *activation)
{
    matrix dZ;
    if (strcmp(activation, "relu") == 0)
        dZ = relu_backward(dA, &cache->linear.Z);
    else
        dZ = *dA;

    linear_grads grads = linear_backward(&dZ, &cache->linear);

    if (strcmp(activation, "relu") == 0)
        delete_matrix(&dZ);

    return grads;
}

nn_grads L_model_backward(const matrix *AL, const matrix *Y,
                          const forward_pass *fwd, int L)
{
    nn_grads grads;
    grads.dW = (matrix *)malloc(sizeof(matrix) * L);
    grads.db = (matrix *)malloc(sizeof(matrix) * L);

    matrix dAL = matrix_sub(AL, Y);

    linear_grads current = linear_activation_backward(&dAL,
                                                      &fwd->caches[L - 1],
                                                      "softmax");
    grads.dW[L - 1] = current.dW;
    grads.db[L - 1] = current.db;

    matrix dA = current.dA_prev;
    delete_matrix(&dAL);

    for (int l = L - 2; l >= 0; l--)
    {
        current = linear_activation_backward(&dA, &fwd->caches[l], "relu");
        grads.dW[l] = current.dW;
        grads.db[l] = current.db;
        delete_matrix(&dA);
        dA = current.dA_prev;
    }
    delete_matrix(&dA);

    return grads;
}