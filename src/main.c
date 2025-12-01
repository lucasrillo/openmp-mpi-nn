#include <stdio.h>
#include <stdlib.h>

#include "load.h"
#include "transform.h"
#include "matrix.h"
#include "nn.h"
#include "nn_params.h"
#include "nn_train.h"

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

// Wrap raw data into matrix structure (no copy)
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
    // Initialize and load all CIFAR-10 data
    if (init_cifar10_data() != 0)
    {
        fprintf(stderr, "Failed to initialize CIFAR-10 data\n");
        return 1;
    }
    printf("Successfully loaded %d total images (%.2f MB)\n", TOTAL_IMAGES, TOTAL_MEMORY_MB);

    // Transform data: split, transpose, and normalize
    if (prepare_cifar10_data() != 0)
    {
        fprintf(stderr, "Failed to prepare CIFAR-10 data\n");
        cleanup_cifar10_data();
        return 1;
    }

    // Wrap data into matrix structures (no copy)
    matrix X_train = wrap_data(data->x_train, PIXELS_PER_IMAGE, TRAIN_SIZE);
    matrix X_test = wrap_data(data->x_test, PIXELS_PER_IMAGE, TEST_SIZE);
    
    // Convert labels to one-hot encoding
    matrix Y_train = labels_to_onehot(data->y_train, TRAIN_SIZE, NUM_CLASSES);
    matrix Y_test = labels_to_onehot(data->y_test, TEST_SIZE, NUM_CLASSES);

    printf("\nData shapes:\n");
    printf("X_train: %d x %d\n", X_train.rows, X_train.cols);
    printf("Y_train: %d x %d\n", Y_train.rows, Y_train.cols);
    printf("X_test: %d x %d\n", X_test.rows, X_test.cols);
    printf("Y_test: %d x %d\n", Y_test.rows, Y_test.cols);

    // Define network architecture
    int layer_dims[] = {PIXELS_PER_IMAGE, 128, 64, NUM_CLASSES};
    int L = 3; // number of layers (excluding input)

    // Train the model
    nn_params params = train_model(&X_train, &Y_train, &X_test, &Y_test,
                                   layer_dims, L,
                                   0.001,  // learning_rate
                                   1000,   // num_iterations
                                   100);   // print_every

    // Cleanup
    delete_matrix(&Y_train);
    delete_matrix(&Y_test);
    delete_nn_params(&params);
    cleanup_transformed_data();
    cleanup_cifar10_data();

    return 0;
}