#include <stdio.h>
#include <stdlib.h>
#include "nn_train.h"

double compute_accuracy(const matrix *X, const matrix *Y, const nn_params *params)
{
    forward_pass fwd = L_model_forward(X, params);
    
    int m = X->cols;
    int correct = 0;

    // For each example
    for (int j = 1; j <= m; j++)
    {
        // Find predicted class (argmax of AL)
        int pred_class = 0;
        double max_prob = mget(fwd.AL, 1, j);
        for (int i = 2; i <= fwd.AL.rows; i++)
        {
            if (mget(fwd.AL, i, j) > max_prob)
            {
                max_prob = mget(fwd.AL, i, j);
                pred_class = i - 1;
            }
        }

        // Find true class (argmax of Y)
        int true_class = 0;
        for (int i = 1; i <= Y->rows; i++)
        {
            if (mgetp(Y, i, j) > 0.5)
            {
                true_class = i - 1;
                break;
            }
        }

        if (pred_class == true_class)
            correct++;
    }

    // Cleanup
    for (int l = 0; l < params->L; l++)
    {
        delete_matrix(&fwd.caches[l].linear.Z);
        delete_matrix(&fwd.caches[l].A);
    }
    free(fwd.caches);

    return (double)correct / m * 100.0;
}

nn_params train_model(const matrix *X_train, const matrix *Y_train,
                      const matrix *X_test, const matrix *Y_test,
                      int *layer_dims, int L,
                      double learning_rate, int num_iterations,
                      int print_every)
{
    // Initialize parameters
    nn_params params = initialize_parameters_he(layer_dims, L);
    
    printf("\nStarting training...\n");
    printf("Architecture: ");
    for (int l = 0; l <= L; l++)
    {
        printf("%d", layer_dims[l]);
        if (l < L)
            printf(" -> ");
    }
    printf("\n");
    printf("Learning rate: %.4f\n", learning_rate);
    printf("Iterations: %d\n\n", num_iterations);

    // Training loop
    for (int iter = 0; iter < num_iterations; iter++)
    {
        printf("Iteration %d/%d\n", iter + 1, num_iterations);
        // Forward propagation
        forward_pass fwd = L_model_forward(X_train, &params);

        // Compute cost
        double cost = compute_cost(&fwd.AL, Y_train);

        // Backward propagation
        nn_grads grads = L_model_backward(&fwd.AL, Y_train, &fwd, params.L);

        // Update parameters
        update_parameters(&params, &grads, learning_rate);

        // Print progress
        if (print_every > 0 && iter % print_every == 0)
        {
            double train_acc = compute_accuracy(X_train, Y_train, &params);
            double test_acc = compute_accuracy(X_test, Y_test, &params);
            
            printf("Iteration %5d: Cost = %.6f | Train Acc = %.2f%% | Test Acc = %.2f%%\n",
                   iter, cost, train_acc, test_acc);
        }

        // Cleanup iteration
        for (int l = 0; l < params.L; l++)
        {
            delete_matrix(&fwd.caches[l].linear.Z);
            delete_matrix(&fwd.caches[l].A);
        }
        free(fwd.caches);
        delete_nn_grads(&grads);
    }

    // Final evaluation
    printf("\nTraining complete!\n");
    double final_train_acc = compute_accuracy(X_train, Y_train, &params);
    double final_test_acc = compute_accuracy(X_test, Y_test, &params);
    printf("Final Train Accuracy: %.2f%%\n", final_train_acc);
    printf("Final Test Accuracy: %.2f%%\n", final_test_acc);

    return params;
}