#include <stdio.h>
#include <stdlib.h>
#include "nn_train.h"
#include "config.h"

// Define timing globals
#define TIMING_IMPL
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
                      int print_every)
{
    // Initialize timing accumulators
#if ENABLE_TIMING
    init_timing_accumulators();
    timer_t_custom timer;
    timer_t_custom total_timer;
    TIMER_START(total_timer);
#endif

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
#if USE_DEBUG_SUBSET
    printf("Mode: DEBUG SUBSET\n");
#else
    printf("Mode: FULL DATASET\n");
#endif
    printf("=============================================\n\n");

    // Training loop
    for (int iter = 0; iter < num_iterations; iter++)
    {
        // ========== FORWARD PROPAGATION ==========
#if ENABLE_TIMING
        TIMER_START(timer);
#endif
        forward_pass fwd = L_model_forward(X_train, &params);
#if ENABLE_TIMING
        TIMER_STOP(timer);
        ACCUM_ADD(g_forward_time, timer);
#if VERBOSE_TIMING
        if (iter % print_every == 0)
            TIMER_PRINT(timer, "Forward");
#endif
#endif

        // ========== COMPUTE COST ==========
#if ENABLE_TIMING
        TIMER_START(timer);
#endif
        double cost = compute_cost(&fwd.AL, Y_train);
#if ENABLE_TIMING
        TIMER_STOP(timer);
        ACCUM_ADD(g_cost_time, timer);
#endif

        // ========== BACKWARD PROPAGATION ==========
#if ENABLE_TIMING
        TIMER_START(timer);
#endif
        nn_grads grads = L_model_backward(&fwd.AL, Y_train, &fwd, params.L);
#if ENABLE_TIMING
        TIMER_STOP(timer);
        ACCUM_ADD(g_backward_time, timer);
#if VERBOSE_TIMING
        if (iter % print_every == 0)
            TIMER_PRINT(timer, "Backward");
#endif
#endif

        // ========== UPDATE PARAMETERS ==========
#if ENABLE_TIMING
        TIMER_START(timer);
#endif
        update_parameters(&params, &grads, learning_rate);
#if ENABLE_TIMING
        TIMER_STOP(timer);
        ACCUM_ADD(g_update_time, timer);
#endif

        // ========== PRINT PROGRESS ==========
        if (print_every > 0 && iter % print_every == 0)
        {
#if ENABLE_TIMING
            TIMER_START(timer);
#endif
            double train_acc = compute_accuracy(X_train, Y_train, &params);
            double test_acc = compute_accuracy(X_test, Y_test, &params);
#if ENABLE_TIMING
            TIMER_STOP(timer);
            ACCUM_ADD(g_accuracy_time, timer);
#endif
            
            printf("Iter %5d: Cost = %.6f | Train Acc = %6.2f%% | Test Acc = %6.2f%%",
                   iter, cost, train_acc, test_acc);
            
#if ENABLE_TIMING && VERBOSE_TIMING
            printf(" | Fwd: %.1fms | Bwd: %.1fms",
                   g_forward_time.total_ms / (iter + 1),
                   g_backward_time.total_ms / (iter + 1));
#endif
            printf("\n");
        }

        // ========== CLEANUP ITERATION ==========
        cleanup_forward_pass(&fwd, params.L);
        delete_nn_grads(&grads, params.L);
    }

    // ========== FINAL EVALUATION ==========
    printf("\n========== TRAINING COMPLETE ==========\n");
    
#if ENABLE_TIMING
    TIMER_STOP(total_timer);
    printf("Total training time: %.2f seconds\n", total_timer.elapsed_ms / 1000.0);
#endif

    double final_train_acc = compute_accuracy(X_train, Y_train, &params);
    double final_test_acc = compute_accuracy(X_test, Y_test, &params);
    printf("Final Train Accuracy: %.2f%%\n", final_train_acc);
    printf("Final Test Accuracy:  %.2f%%\n", final_test_acc);

#if ENABLE_TIMING
    print_timing_summary();
#endif

    return params;
}