#ifndef NN_TRAIN_H
#define NN_TRAIN_H

#include "matrix.h"
#include "nn_params.h"

// Compute accuracy of model predictions
double compute_accuracy(const matrix *X, const matrix *Y, const nn_params *params);

// Train the neural network model
nn_params train_model(const matrix *X_train, const matrix *Y_train,
                      const matrix *X_test, const matrix *Y_test,
                      int *layer_dims, int L,
                      double learning_rate, int num_iterations,
                      int print_every, int num_samples);

#endif // NN_TRAIN_H