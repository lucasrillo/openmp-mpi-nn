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

    // Process column by column (each column is a sample)
    for (int j = 1; j <= Z->cols; j++)
    {
        // Find max for numerical stability
        double max_val = mgetp(Z, 1, j);
        for (int i = 2; i <= Z->rows; i++)
            max_val = fmax(max_val, mgetp(Z, i, j));

        // Compute exp and sum
        double sum = 0.0;
        for (int i = 1; i <= Z->rows; i++)
        {
            mget(A, i, j) = exp(mgetp(Z, i, j) - max_val);
            sum += mget(A, i, j);
        }
        
        // Normalize
        double inv_sum = 1.0 / sum;
        for (int i = 1; i <= Z->rows; i++)
            mget(A, i, j) *= inv_sum;
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

    cache.Z = matrix_mult_add_col(W, A, b);

    return cache;
}

layer_cache linear_activation_forward(const matrix *A_prev, const matrix *W, const matrix *b, activation_t activation)
{
    layer_cache cache;
    cache.linear = linear_forward(A_prev, W, b);

    switch (activation)
    {
        case ACTIVATION_RELU:
            cache.A = relu(&cache.linear.Z);
            break;
        case ACTIVATION_SOFTMAX:
            cache.A = softmax(&cache.linear.Z);
            break;
    }

    return cache;
}

forward_pass L_model_forward(const matrix *X, const nn_params *params)
{
    forward_pass fwd;
    fwd.caches = (layer_cache *)malloc(sizeof(layer_cache) * params->L);
    const matrix *A_ptr = X;

    for (int l = 0; l < params->L - 1; l++)
    {
        fwd.caches[l] = linear_activation_forward(A_ptr, &params->W[l], &params->b[l], ACTIVATION_RELU);
        A_ptr = &fwd.caches[l].A;
    }
    fwd.caches[params->L - 1] = linear_activation_forward(A_ptr, &params->W[params->L - 1], &params->b[params->L - 1], ACTIVATION_SOFTMAX);
    fwd.AL = fwd.caches[params->L - 1].A;

    return fwd;
}

double compute_cost(const matrix *AL, const matrix *Y)
{
    const int m = Y->cols;
    double cost = 0.0;

    // Cross-entropy loss
    for (int j = 1; j <= m; j++)
        for (int i = 1; i <= Y->rows; i++)
            if (mgetp(Y, i, j) > 0)
                cost -= mgetp(Y, i, j) * log(mgetp(AL, i, j) + 1e-8);

    return cost / m;
}

linear_grads linear_backward(const matrix *dZ, const linear_cache *cache)
{
    linear_grads grads;
    const int m = cache->A.cols;
    const double inv_m = 1.0 / m;

    // Fused operation for dW = (dZ * A^T) / m
    grads.dW = matrix_mult_transB_scale(dZ, &cache->A, inv_m);

    // Compute db = sum(dZ, axis=1) / m
    matrix db_sum = matrix_sum_rows(dZ);
    grads.db = matrix_scalar_mult(&db_sum, inv_m);
    delete_matrix(&db_sum);

    // Fused operation for dA_prev = W^T * dZ
    grads.dA_prev = matrix_multT_B(&cache->W, dZ);

    return grads;
}

linear_grads linear_activation_backward(const matrix *dA, const layer_cache *cache, activation_t activation)
{
    matrix dZ;
    int owns_dZ = 0;
    
    switch (activation)
    {
        case ACTIVATION_RELU:
            dZ = relu_backward(dA, &cache->linear.Z);
            owns_dZ = 1;
            break;
        case ACTIVATION_SOFTMAX:
            // For softmax with cross-entropy, dZ = AL - Y (passed directly as dA)
            dZ = *dA;
            owns_dZ = 0;
            break;
    }

    linear_grads grads = linear_backward(&dZ, &cache->linear);

    if (owns_dZ)
        delete_matrix(&dZ);

    return grads;
}

nn_grads L_model_backward(const matrix *AL, const matrix *Y,
                          const forward_pass *fwd, int L)
{
    nn_grads grads;
    grads.dW = (matrix *)malloc(sizeof(matrix) * L);
    grads.db = (matrix *)malloc(sizeof(matrix) * L);

    // dAL = AL - Y (for softmax + cross-entropy)
    matrix dAL = matrix_sub(AL, Y);

    // Output layer backward (softmax)
    linear_grads current = linear_activation_backward(&dAL,
                                                      &fwd->caches[L - 1],
                                                      ACTIVATION_SOFTMAX);
    grads.dW[L - 1] = current.dW;
    grads.db[L - 1] = current.db;

    matrix dA = current.dA_prev;
    delete_matrix(&dAL);

    // Hidden layers backward (relu)
    for (int l = L - 2; l >= 0; l--)
    {
        current = linear_activation_backward(&dA, &fwd->caches[l], ACTIVATION_RELU);
        grads.dW[l] = current.dW;
        grads.db[l] = current.db;
        delete_matrix(&dA);
        dA = current.dA_prev;
    }
    delete_matrix(&dA);

    return grads;
}

void cleanup_forward_pass(forward_pass *fwd, int L)
{
    if (!fwd || !fwd->caches)
        return;
    
    for (int l = 0; l < L; l++)
    {
        delete_matrix(&fwd->caches[l].linear.Z);
        delete_matrix(&fwd->caches[l].A);
    }
    free(fwd->caches);
    fwd->caches = NULL;
}