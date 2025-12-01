#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "load.h"
#include "transform.h"
#include "matrix.h"
#include "nn.h"
#include "nn_params.h"
#include "nn_train.h"
#include "timing.h"

// Convert labels to one-hot encoding
matrix labels_to_onehot(uint8_t *labels, int n_samples, int n_classes)
{
    matrix Y = new_matrix(n_classes, n_samples);
    for (int j = 0; j < n_samples; j++)
    {
        int label = labels[j];
        mget(Y, label + 1, j + 1) = 1.0;
    }
    return Y;
}

// Wrap raw data into matrix structure (no copy, just wrapping)
matrix wrap_data(double *data_ptr, int rows, int cols)
{
    matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.val = data_ptr;
    return mat;
}

int main(int argc, char *argv[])
{
    timer_t_custom startup_timer;
    TIMER_START(startup_timer);
    
    printf("========== CIFAR-10 Neural Network ==========\n");
#if USE_DEBUG_SUBSET
    printf("Mode: DEBUG (using small i.i.d. subset)\n");
#else
    printf("Mode: FULL DATASET\n");
#endif
    printf("==============================================\n\n");

    // ========== LOAD DATA ==========
    timer_t_custom load_timer;
    TIMER_START(load_timer);
    
    if (init_cifar10_data() != 0)
    {
        fprintf(stderr, "Failed to initialize CIFAR-10 data\n");
        return 1;
    }
    
    TIMER_STOP(load_timer);
    printf("[TIMER] Data loading: %.2f ms\n", load_timer.elapsed_ms);
    printf("Successfully loaded %d total images (%.2f MB)\n", TOTAL_IMAGES, TOTAL_MEMORY_MB);

    // ========== TRANSFORM DATA ==========
    timer_t_custom transform_timer;
    TIMER_START(transform_timer);
    
    if (prepare_cifar10_data() != 0)
    {
        fprintf(stderr, "Failed to prepare CIFAR-10 data\n");
        cleanup_cifar10_data();
        return 1;
    }
    
    TIMER_STOP(transform_timer);
    printf("[TIMER] Data transformation: %.2f ms\n", transform_timer.elapsed_ms);

    // ========== CREATE MATRIX VIEWS ==========
    // Wrap data into matrix structures (no copy, just pointer wrapping)
    matrix X_train = wrap_data(data->x_train, PIXELS_PER_IMAGE, TRAIN_SIZE);
    matrix X_test = wrap_data(data->x_test, PIXELS_PER_IMAGE, TEST_SIZE);
    
    // Convert labels to one-hot encoding (this allocates memory)
    matrix Y_train = labels_to_onehot(data->y_train, TRAIN_SIZE, NUM_CLASSES);
    matrix Y_test = labels_to_onehot(data->y_test, TEST_SIZE, NUM_CLASSES);

    printf("\nData shapes:\n");
    printf("  X_train: %d x %d (features x samples)\n", X_train.rows, X_train.cols);
    printf("  Y_train: %d x %d (classes x samples)\n", Y_train.rows, Y_train.cols);
    printf("  X_test:  %d x %d\n", X_test.rows, X_test.cols);
    printf("  Y_test:  %d x %d\n", Y_test.rows, Y_test.cols);

    TIMER_STOP(startup_timer);
    printf("\n[TIMER] Total startup: %.2f ms\n", startup_timer.elapsed_ms);

    // ========== DEFINE NETWORK ARCHITECTURE ==========
    int layer_dims[] = {PIXELS_PER_IMAGE, 128, 64, NUM_CLASSES};
    int L = 3; // number of layers (excluding input)

    // ========== TRAIN MODEL ==========
    nn_params params = train_model(&X_train, &Y_train, &X_test, &Y_test,
                                   layer_dims, L,
                                   DEFAULT_LEARNING_RATE,
                                   DEFAULT_NUM_ITERATIONS,
                                   DEFAULT_PRINT_EVERY);

    // ========== CLEANUP ==========
    printf("\nCleaning up...\n");
    delete_matrix(&Y_train);
    delete_matrix(&Y_test);
    // Note: X_train and X_test are just wrappers, don't delete their val pointers
    delete_nn_params(&params);
    cleanup_transformed_data();
    cleanup_cifar10_data();
    printf("Done!\n");

    return 0;
}