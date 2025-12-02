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

    // Calculate local batch size per process
    int local_batch_size = BATCH_SIZE / num_processes;
    int num_train_samples = X_train->cols;
    int num_batches = (num_train_samples + local_batch_size - 1) / local_batch_size;

    // Allocate mini-batch matrices
    matrix X_batch = new_matrix(X_train->rows, local_batch_size);
    matrix Y_batch = new_matrix(Y_train->rows, local_batch_size);

    // Initialize parameters (use rank to differentiate seeds and avoid identical initialization)
    nn_params params = initialize_parameters_he(layer_dims, L, rank);

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
        printf("Mini-batch size: %d (global), %d (local per process)\n", BATCH_SIZE, local_batch_size);
        printf("Batches per epoch: %d\n", num_batches);
        printf("=============================================\n\n\n");
    }

    // Training loop
    if (rank == 0)
        printf("========== TRAINING LOOP ==========\n");

    for (int iter = 0; iter < num_iterations; iter++)
    {
        double epoch_cost = 0.0;

        // Process each mini-batch
        for (int batch = 0; batch < num_batches; batch++)
        {
            int start_idx = batch * local_batch_size;
            int current_batch_size = local_batch_size;

            // Handle last batch which might be smaller
            if (start_idx + local_batch_size > num_train_samples)
            {
                current_batch_size = num_train_samples - start_idx;
            }

            // Extract mini-batch (copy columns from training data)
            for (int j = 0; j < current_batch_size; j++)
            {
                int src_col = start_idx + j + 1; // 1-indexed column in source
                int dst_col = j + 1;             // 1-indexed column in batch

                // Copy X column
                for (int i = 1; i <= X_train->rows; i++)
                    mget(X_batch, i, dst_col) = mgetp(X_train, i, src_col);

                // Copy Y column
                for (int i = 1; i <= Y_train->rows; i++)
                    mget(Y_batch, i, dst_col) = mgetp(Y_train, i, src_col);
            }

            // Create matrix views for the current batch size if it's smaller
            matrix X_batch_view = X_batch;
            matrix Y_batch_view = Y_batch;
            if (current_batch_size < local_batch_size)
            {
                X_batch_view.cols = current_batch_size;
                Y_batch_view.cols = current_batch_size;
            }

            // Forward propagation (local)
            TIMER_START(timer);
            forward_pass fwd = L_model_forward(&X_batch_view, &params);
            TIMER_STOP(timer);
            ACCUM_ADD(g_forward_time, timer);

            // Compute cost (local, then allreduce)
            TIMER_START(timer);
            double local_cost = compute_cost(&fwd.AL, &Y_batch_view);
            double cost = allreduce_cost(local_cost, num_processes);
            epoch_cost += cost;
            TIMER_STOP(timer);
            ACCUM_ADD(g_cost_time, timer);

            // Backward propagation (local, then allreduce)
            TIMER_START(timer);
            nn_grads grads = L_model_backward(&fwd.AL, &Y_batch_view, &fwd, params.L);
            allreduce_gradients(&grads, params.L, num_processes);
            TIMER_STOP(timer);
            ACCUM_ADD(g_backward_time, timer);

            // Update parameters (local, but same on all processes)
            TIMER_START(timer);
            update_parameters(&params, &grads, learning_rate);
            TIMER_STOP(timer);
            ACCUM_ADD(g_update_time, timer);

            // Cleanup batch
            cleanup_forward_pass(&fwd, params.L);
            delete_nn_grads(&grads, params.L);
        }

        // Average cost over all batches
        epoch_cost /= num_batches;

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
                printf("Iter %5d: Avg Cost = %.6f | Train Acc = %6.2f%% | Test Acc = %6.2f%%\n", iter, epoch_cost, train_acc, test_acc);
        }
    }

    // Cleanup mini-batch matrices
    delete_matrix(&X_batch);
    delete_matrix(&Y_batch);

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
        log_results_to_csv("training_results.csv", num_samples, num_iterations, learning_rate,
                           final_train_acc, final_test_acc, training_timer.elapsed_ms / 1000.0,
                           num_threads, num_processes);
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