#include <stdio.h>
#include <stdlib.h>

#include "nn_train.h"
#include "config.h"
#include "timing.h"

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
    cleanup_forward_pass(&fwd, params->L);

    return (double)correct / m * 100.0;
}

nn_params train_model(const matrix *X_train, const matrix *Y_train,
                      const matrix *X_test, const matrix *Y_test,
                      int *layer_dims, int L,
                      double learning_rate, int num_iterations,
                      int print_every, int num_samples, int num_threads)
{
    // Initialize timing accumulators
    init_timing_accumulators();
    timer_t_custom timer;
    timer_t_custom training_timer;
    TIMER_START(training_timer);

    // Initialize parameters
    nn_params params = initialize_parameters_he(layer_dims, L);

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
    printf("Training samples: %d\n", X_train->cols);
    printf("Test samples: %d\n", X_test->cols);
    printf("=============================================\n\n\n");

    // Training loop
    printf("========== TRAINING LOOP ==========\n");
    for (int iter = 0; iter < num_iterations; iter++)
    {
        // ========== FORWARD PROPAGATION ==========
        TIMER_START(timer);
        forward_pass fwd = L_model_forward(X_train, &params);
        TIMER_STOP(timer);
        ACCUM_ADD(g_forward_time, timer);

        // ========== COMPUTE COST ==========
        TIMER_START(timer);
        double cost = compute_cost(&fwd.AL, Y_train);
        TIMER_STOP(timer);
        ACCUM_ADD(g_cost_time, timer);

        // ========== BACKWARD PROPAGATION ==========
        TIMER_START(timer);
        nn_grads grads = L_model_backward(&fwd.AL, Y_train, &fwd, params.L);
        TIMER_STOP(timer);
        ACCUM_ADD(g_backward_time, timer);

        // ========== UPDATE PARAMETERS ==========
        TIMER_START(timer);
        update_parameters(&params, &grads, learning_rate);
        TIMER_STOP(timer);
        ACCUM_ADD(g_update_time, timer);

        // ========== PRINT PROGRESS ==========
        if (print_every > 0 && iter % print_every == 0)
        {
            TIMER_START(timer);
            double train_acc = compute_accuracy(X_train, Y_train, &params);
            double test_acc = compute_accuracy(X_test, Y_test, &params);
            TIMER_STOP(timer);
            ACCUM_ADD(g_accuracy_time, timer);

            printf("Iter %5d: Cost = %.6f | Train Acc = %6.2f%% | Test Acc = %6.2f%%\n",
                   iter, cost, train_acc, test_acc);
        }

        // ========== CLEANUP ITERATION ==========
        cleanup_forward_pass(&fwd, params.L);
        delete_nn_grads(&grads, params.L);
    }

    TIMER_STOP(training_timer);
    printf("------------------------------------------------------------\n");
    printf("[TIMER] Total training time: %.2f seconds\n", training_timer.elapsed_ms / 1000.0);

    double final_train_acc = compute_accuracy(X_train, Y_train, &params);
    double final_test_acc = compute_accuracy(X_test, Y_test, &params);
    printf("Final Train Accuracy: %.2f%%\n", final_train_acc);
    printf("Final Test Accuracy:  %.2f%%\n", final_test_acc);
    printf("=======================================\n\n");

    print_timing_summary();

    // Log results to CSV (training_time_sec is just the training loop, not entire program)
    log_results_to_csv("training_results.csv",
                       num_samples,
                       num_iterations,
                       learning_rate,
                       final_train_acc,
                       final_test_acc,
                       training_timer.elapsed_ms / 1000.0,
                       num_threads);

    return params;
}