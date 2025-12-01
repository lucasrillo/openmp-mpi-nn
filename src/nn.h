#ifndef NN_H
#define NN_H

#include "matrix.h"

typedef enum
{
    ACTIVATION_RELU,
    ACTIVATION_SOFTMAX
} activation_t;

typedef struct
{
    int L;
    matrix *W;
    matrix *b;
} nn_params;

typedef struct
{
    matrix A;
    matrix W;
    matrix b;
    matrix Z; // Pre-activation (owned, needs cleanup)
} linear_cache;

typedef struct
{
    linear_cache linear;
    matrix A; // Post-activation (owned, needs cleanup)
} layer_cache;

typedef struct
{
    layer_cache *caches;
    matrix AL; // Final output (alias to last cache.A)
} forward_pass;

typedef struct
{
    matrix dA_prev;
    matrix dW;
    matrix db;
} linear_grads;

typedef struct
{
    matrix *dW;
    matrix *db;
} nn_grads;

// Activation functions
matrix relu(const matrix *Z);
matrix softmax(const matrix *Z);
matrix relu_backward(const matrix *dA, const matrix *Z_cache);

// Forward pass functions
linear_cache linear_forward(const matrix *A, const matrix *W, const matrix *b);
layer_cache linear_activation_forward(const matrix *A_prev, const matrix *W, const matrix *b, activation_t activation);
forward_pass L_model_forward(const matrix *X, const nn_params *params);

// Cost function
double compute_cost(const matrix *AL, const matrix *Y);

// Backward pass functions
linear_grads linear_backward(const matrix *dZ, const linear_cache *cache);
linear_grads linear_activation_backward(const matrix *dA, const layer_cache *cache, activation_t activation);
nn_grads L_model_backward(const matrix *AL, const matrix *Y, const forward_pass *fwd, int L);

// Cleanup forward pass caches
void cleanup_forward_pass(forward_pass *fwd, int L);

#endif // NN_H