#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "transform.h"

// Global variables
CIFAR10Data *data = NULL;

/**
 * Prepare CIFAR-10 data: split, transpose, standardize, and convert to matrices
 * Takes first n samples per category
 * Returns 0 on success, 1 on error
 */
int prepare_cifar10_data(int num_samples)
{
    // Calculate train/test split (90% train, 10% test)
    int train_size = (num_samples * 9) / 10;
    int test_size = num_samples - train_size;

    // Allocate memory for transformed data structure
    data = (CIFAR10Data *)malloc(sizeof(CIFAR10Data));
    if (!data)
    {
        fprintf(stderr, "Error: Memory allocation failed for CIFAR10Data\n");
        return 1;
    }

    data->train_size = train_size;
    data->test_size = test_size;

    // Create matrices using new_matrix (allocates and initializes to zero)
    data->X_train = new_matrix(PIXELS_PER_IMAGE, train_size);
    data->Y_train = new_matrix(NUM_CLASSES, train_size);
    data->X_test = new_matrix(PIXELS_PER_IMAGE, test_size);
    data->Y_test = new_matrix(NUM_CLASSES, test_size);

    // Calculate samples per class
    int train_per_class = train_size / NUM_CLASSES;
    int test_per_class = test_size / NUM_CLASSES;

    // Track how many samples collected per class
    int class_train_count[NUM_CLASSES] = {0};
    int class_test_count[NUM_CLASSES] = {0};
    int train_idx = 0;
    int test_idx = 0;

    // Process all images and distribute to train/test based on class
    for (int i = 0; i < TOTAL_IMAGES && (train_idx < train_size || test_idx < test_size); i++)
    {
        uint8_t label = cifar10_images[i].label;

        // Add to training set if this class needs more training samples
        if (class_train_count[label] < train_per_class && train_idx < train_size)
        {
            // Set one-hot encoding for label (matrices are 1-indexed via mget)
            mget(data->Y_train, label + 1, train_idx + 1) = 1.0;

            // Transpose and normalize pixel data
            for (int pixel = 0; pixel < PIXELS_PER_IMAGE; pixel++)
            {
                mget(data->X_train, pixel + 1, train_idx + 1) = cifar10_images[i].data[pixel] / 255.0;
            }

            class_train_count[label]++;
            train_idx++;
        }
        // Add to test set if this class needs more test samples
        else if (class_test_count[label] < test_per_class && test_idx < test_size)
        {
            // Set one-hot encoding for label
            mget(data->Y_test, label + 1, test_idx + 1) = 1.0;

            // Transpose and normalize pixel data
            for (int pixel = 0; pixel < PIXELS_PER_IMAGE; pixel++)
            {
                mget(data->X_test, pixel + 1, test_idx + 1) = cifar10_images[i].data[pixel] / 255.0;
            }

            class_test_count[label]++;
            test_idx++;
        }
    }

    // Verify we got the expected number of samples
    if (train_idx != train_size || test_idx != test_size)
    {
        fprintf(stderr, "Warning: Expected %d train and %d test samples, got %d and %d\n", 
                train_size, test_size, train_idx, test_idx);
        cleanup_transformed_data();
        return 1;
    }

    return 0;
}

/**
 * Clean up transformed data
 */
void cleanup_transformed_data(void)
{
    if (!data)
        return;
    delete_matrix(&data->X_train);
    delete_matrix(&data->Y_train);
    delete_matrix(&data->X_test);
    delete_matrix(&data->Y_test);
    free(data);
    data = NULL;
}