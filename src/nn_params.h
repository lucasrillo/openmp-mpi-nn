#ifndef NN_PARAMS_H
#define NN_PARAMS_H

#include "matrix.h"
#include "nn.h"

// Initialize network parameters with He initialization
nn_params initialize_parameters_he(int *layer_dims, int L, int seed_offset);

// Update parameters using gradient descent
void update_parameters(nn_params *params, const nn_grads *grads, double learning_rate);

// Cleanup functions
void delete_nn_params(nn_params *params);
void delete_nn_grads(nn_grads *grads, int L);

#endif // NN_PARAMS_H