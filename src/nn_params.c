#include <stdlib.h>
#include <math.h>

#include "nn_params.h"
#include "config.h"

nn_params initialize_parameters_he(int *layer_dims, int L, int seed_offset)
{
    nn_params params;
    params.L = L;
    params.W = (matrix *)malloc(sizeof(matrix) * L);
    params.b = (matrix *)malloc(sizeof(matrix) * L);

    // Seed random number generator
    srand(RANDOM_SEED + seed_offset);

    for (int l = 0; l < L; l++)
    {
        int rows = layer_dims[l + 1];
        int cols = layer_dims[l];

        // Initialize weights with He initialization: W ~ N(0, sqrt(2/n_prev))
        params.W[l] = new_matrix(rows, cols);
        double std = sqrt(2.0 / cols);
        for (int i = 1; i <= rows; i++)
            for (int j = 1; j <= cols; j++)
            {
                // Box-Muller transform for normal distribution
                double u1 = ((double)rand() / RAND_MAX);
                double u2 = ((double)rand() / RAND_MAX);
                // Avoid log(0)
                if (u1 < 1e-10) u1 = 1e-10;
                double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                mget(params.W[l], i, j) = z * std;
            }

        // Initialize biases to zero
        params.b[l] = new_matrix(rows, 1);
    }

    return params;
}

void update_parameters(nn_params *params, const nn_grads *grads, double learning_rate)
{
    for (int l = 0; l < params->L; l++)
    {
        // W = W - learning_rate * dW
        for (int i = 1; i <= params->W[l].rows; i++)
            for (int j = 1; j <= params->W[l].cols; j++)
                mget(params->W[l], i, j) -= learning_rate * mget(grads->dW[l], i, j);

        // b = b - learning_rate * db
        for (int i = 1; i <= params->b[l].rows; i++)
            mget(params->b[l], i, 1) -= learning_rate * mget(grads->db[l], i, 1);
    }
}

void delete_nn_params(nn_params *params)
{
    if (!params)
        return;
    
    for (int l = 0; l < params->L; l++)
    {
        delete_matrix(&params->W[l]);
        delete_matrix(&params->b[l]);
    }
    
    free(params->W);
    free(params->b);
    params->W = NULL;
    params->b = NULL;
    params->L = 0;
}

void delete_nn_grads(nn_grads *grads, int L)
{
    if (!grads || !grads->dW)
        return;
    
    for (int l = 0; l < L; l++)
    {
        delete_matrix(&grads->dW[l]);
        delete_matrix(&grads->db[l]);
    }
    
    free(grads->dW);
    free(grads->db);
    grads->dW = NULL;
    grads->db = NULL;
}