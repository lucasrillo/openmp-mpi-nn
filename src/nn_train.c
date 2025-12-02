#include <stdio.h>
#include <stdlib.h>

#include "nn_train.h"
#include "config.h"
#include "timing.h"
#include "mpi_utils.h"

// Compute accuracy across all MPI processes
static double compute_accuracy(const matrix *X, const matrix *Y, const nn_params *params, int num_processes);

nn_params train_model(const matrix *X_train, const matrix *Y_train,
                      const matrix *X_test, const matrix *Y_test,
                      int *layer_dims, int L,
                      double learning_rate, int num_iterations,
                      int print_every, int num_samples, int num_threads,
                      int rank, int num_processes)
{
    // Initialize timing accumulators
    init_timing_accumulators();
    timer_t_custom timer;
    timer_t_custom training_timer;
    TIMER_START(training_timer);

    // Initialize parameters (same on all ranks due to same seed)
    nn_params params = initialize_parameters_he(layer_dims, L);

    if (rank == 0)
    {
        printf("\n========== TRAINING CONFIGURATION ==========\n");
        printf("Architecture: ");
        for (int l = 0; l <= L; l++)
        {
            printf("%d", layer_dims[l]);
            if (l < L)
                printf(" -> ");
        }
        printf("\n");
        printf("Learning rate: %.4f\n", learning_rate);
        printf("Iterations: %d\n", num_iterations);
        printf("Total samples: %d\n", num_samples);
        printf("Samples per process: %d\n", X_train->cols + X_test->cols);
        printf("Local training samples: %d\n", X_train->cols);
        printf("Local test samples: %d\n", X_test->cols);
        printf("MPI processes: %d\n", num_processes);
        printf("OpenMP threads per process: %d\n", num_threads);
        printf("Effective mini-batch size: %d\n", X_train->cols * num_processes);
        printf("=============================================\n\n\n");
    }

    // Training loop
    if (rank == 0)
        printf("========== TRAINING LOOP ==========\n");

    for (int iter = 0; iter < num_iterations; iter++)
    {
        // Forward propagation (local)
        TIMER_START(timer);
        forward_pass fwd = L_model_forward(X_train, &params);
        TIMER_STOP(timer);
        ACCUM_ADD(g_forward_time, timer);

        // Compute cost (local, then allreduce)
        TIMER_START(timer);
        double local_cost = compute_cost(&fwd.AL, Y_train);
        double cost = allreduce_cost(local_cost, num_processes);
        TIMER_STOP(timer);
        ACCUM_ADD(g_cost_time, timer);

        // Backward propagation (local, then allreduce)
        TIMER_START(timer);
        nn_grads grads = L_model_backward(&fwd.AL, Y_train, &fwd, params.L);
        allreduce_gradients(&grads, params.L, num_processes);
        TIMER_STOP(timer);
        ACCUM_ADD(g_backward_time, timer);

        // Update parameters (local, but same on all processes)
        TIMER_START(timer);
        update_parameters(&params, &grads, learning_rate);
        TIMER_STOP(timer);
        ACCUM_ADD(g_update_time, timer);

        // Print progress
        if (print_every > 0 && iter % print_every == 0)
        {
            // Compute accuracy across all processes
            TIMER_START(timer);
            double train_acc = compute_accuracy(X_train, Y_train, &params, num_processes);
            double test_acc = compute_accuracy(X_test, Y_test, &params, num_processes);
            TIMER_STOP(timer);
            ACCUM_ADD(g_accuracy_time, timer);

            if (rank == 0)
                printf("Iter %5d: Cost = %.6f | Train Acc = %6.2f%% | Test Acc = %6.2f%%\n", iter, cost, train_acc, test_acc);
        }

        // Cleanup
        cleanup_forward_pass(&fwd, params.L);
        delete_nn_grads(&grads, params.L);
    }

    TIMER_STOP(training_timer);

    if (rank == 0)
    {
        printf("------------------------------------------------------------\n");
        printf("[TIMER] Total training time: %.2f seconds\n", training_timer.elapsed_ms / 1000.0);
    }

    // Compute final accuracy across all processes
    double final_train_acc = compute_accuracy(X_train, Y_train, &params, num_processes);
    double final_test_acc = compute_accuracy(X_test, Y_test, &params, num_processes);

    if (rank == 0)
    {
        // Print final results
        printf("Final Train Accuracy: %.2f%%\n", final_train_acc);
        printf("Final Test Accuracy:  %.2f%%\n", final_test_acc);
        printf("=======================================\n\n");
        print_timing_summary();

        // Log results to CSV
        log_results_to_csv("training_results.csv", num_samples, num_iterations, learning_rate, final_train_acc, final_test_acc, training_timer.elapsed_ms / 1000.0, num_threads, num_processes);
    }

    return params;
}

static double compute_accuracy(const matrix *X, const matrix *Y, const nn_params *params, int num_processes)
{
    forward_pass fwd = L_model_forward(X, params);

    int m = X->cols;
    int correct_count = 0;

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
        
        // Check if correct
        if (pred_class == true_class)
            correct_count++;
    }

    // Cleanup
    cleanup_forward_pass(&fwd, params->L);

    // Allreduce and return accuracy percentage
    return allreduce_accuracy(correct_count, m);
}
