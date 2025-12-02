#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "transform.h"

// Global variables
CIFAR10Data *data = NULL;

int prepare_cifar10_data(int num_samples, int rank, int num_processes)
{
    int samples_per_process = num_samples / num_processes;

    // Calculate train/test split for this process (90% train, 10% test)
    int local_train_size = (samples_per_process * 9) / 10;
    int local_test_size = samples_per_process - local_train_size;

    // Calculate samples per class for this process
    int samples_per_class_total = num_samples / NUM_CLASSES;
    int samples_per_class_per_process = samples_per_class_total / num_processes;
    int train_per_class = (local_train_size) / NUM_CLASSES;
    int test_per_class = (local_test_size) / NUM_CLASSES;

    // Calculate the starting offset for this rank within each class
    int class_offset = rank * samples_per_class_per_process;

    // Allocate memory for transformed data structure
    data = (CIFAR10Data *)malloc(sizeof(CIFAR10Data));
    if (!data)
    {
        fprintf(stderr, "Error: Memory allocation failed for CIFAR10Data on rank %d\n", rank);
        return 1;
    }

    data->train_size = local_train_size;
    data->test_size = local_test_size;

    // Create matrices for this process's local data
    data->X_train = new_matrix(PIXELS_PER_IMAGE, local_train_size);
    data->Y_train = new_matrix(NUM_CLASSES, local_train_size);
    data->X_test = new_matrix(PIXELS_PER_IMAGE, local_test_size);
    data->Y_test = new_matrix(NUM_CLASSES, local_test_size);

    // Track how many samples collected per class for this process
    int class_train_count[NUM_CLASSES] = {0};
    int class_test_count[NUM_CLASSES] = {0};
    int class_seen_count[NUM_CLASSES] = {0};
    int train_idx = 0;
    int test_idx = 0;

    // Process all images and distribute to train/test based on class and rank offset
    for (int i = 0; i < TOTAL_IMAGES && (train_idx < local_train_size || test_idx < local_test_size); i++)
    {
        uint8_t label = cifar10_images[i].label;
        int seen_in_class = class_seen_count[label];
        class_seen_count[label]++;

        // Check if this sample belongs to this rank's portion
        // Each rank gets samples [class_offset, class_offset + samples_per_class_per_process - 1] for each class
        if (seen_in_class < class_offset || seen_in_class >= class_offset + samples_per_class_per_process)
        {
            continue; // This sample belongs to another rank
        }

        // Add to training set if this class needs more training samples
        if (class_train_count[label] < train_per_class && train_idx < local_train_size)
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
        else if (class_test_count[label] < test_per_class && test_idx < local_test_size)
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
    if (train_idx != local_train_size || test_idx != local_test_size)
    {
        fprintf(stderr, "Rank %d: Expected %d train and %d test samples, got %d and %d\n",
                rank, local_train_size, local_test_size, train_idx, test_idx);
        cleanup_transformed_data();
        return 1;
    }

    return 0;
}

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