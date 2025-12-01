#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "transform.h"

// Global variables
CIFAR10Data *data = NULL;

#if USE_DEBUG_SUBSET
/**
 * Fisher-Yates shuffle to generate random permutation indices
 */
static void generate_random_indices(int *indices, int n, int max_val)
{
    // Initialize with sequential values
    for (int i = 0; i < max_val; i++)
        indices[i] = i;
    
    // Seed random number generator
    srand(42);  // Fixed seed for reproducibility
    
    // Partial Fisher-Yates: only shuffle first n elements
    for (int i = 0; i < n && i < max_val - 1; i++)
    {
        int j = i + rand() % (max_val - i);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}
#endif

/**
 * Prepare CIFAR-10 data: split, transpose, and standardize
 * Returns 0 on success, 1 on error
 */
int prepare_cifar10_data(void)
{
    // Allocate memory for transformed data structure
    data = (CIFAR10Data *)malloc(sizeof(CIFAR10Data));
    if (!data)
    {
        fprintf(stderr, "Error: Memory allocation failed for CIFAR10Data\n");
        return 1;
    }

    // Allocate memory for training and testing data
    data->x_train = (double *)malloc(PIXELS_PER_IMAGE * TRAIN_SIZE * sizeof(double));
    data->y_train = (uint8_t *)malloc(TRAIN_SIZE * sizeof(uint8_t));
    data->x_test = (double *)malloc(PIXELS_PER_IMAGE * TEST_SIZE * sizeof(double));
    data->y_test = (uint8_t *)malloc(TEST_SIZE * sizeof(uint8_t));
    
    if (!data->x_train || !data->y_train || !data->x_test || !data->y_test)
    {
        fprintf(stderr, "Error: Memory allocation failed for transformed data\n");
        cleanup_transformed_data();
        return 1;
    }

#if USE_DEBUG_SUBSET
    printf("Using DEBUG SUBSET: %d train, %d test (i.i.d. sampled)\n", 
           TRAIN_SIZE, TEST_SIZE);
    
    // Generate random indices for i.i.d. sampling
    int *train_indices = (int *)malloc(FULL_TRAIN_SIZE * sizeof(int));
    int *test_indices = (int *)malloc(FULL_TEST_SIZE * sizeof(int));
    
    if (!train_indices || !test_indices)
    {
        fprintf(stderr, "Error: Memory allocation failed for indices\n");
        if (train_indices) free(train_indices);
        if (test_indices) free(test_indices);
        cleanup_transformed_data();
        return 1;
    }
    
    generate_random_indices(train_indices, TRAIN_SIZE, FULL_TRAIN_SIZE);
    generate_random_indices(test_indices, TEST_SIZE, FULL_TEST_SIZE);
    
    // Process training data (randomly sampled subset)
    for (int i = 0; i < TRAIN_SIZE; i++)
    {
        int img_idx = train_indices[i];
        
        // Copy label
        data->y_train[i] = cifar10_images[img_idx].label;

        // Transpose and normalize pixel data
        for (int pixel = 0; pixel < PIXELS_PER_IMAGE; pixel++)
        {
            data->x_train[pixel * TRAIN_SIZE + i] = cifar10_images[img_idx].data[pixel] / 255.0;
        }
    }

    // Process testing data (randomly sampled from last 5000)
    for (int i = 0; i < TEST_SIZE; i++)
    {
        int img_idx = FULL_TRAIN_SIZE + test_indices[i];
        
        // Copy label
        data->y_test[i] = cifar10_images[img_idx].label;

        // Transpose and normalize pixel data
        for (int pixel = 0; pixel < PIXELS_PER_IMAGE; pixel++)
        {
            data->x_test[pixel * TEST_SIZE + i] = cifar10_images[img_idx].data[pixel] / 255.0;
        }
    }
    
    free(train_indices);
    free(test_indices);
    
#else
    // Full dataset processing
    printf("Using FULL DATASET: %d train, %d test\n", TRAIN_SIZE, TEST_SIZE);
    
    // Process training data (first 45000 images)
    for (int i = 0; i < TRAIN_SIZE; i++)
    {
        // Copy label
        data->y_train[i] = cifar10_images[i].label;

        // Transpose and normalize pixel data
        for (int pixel = 0; pixel < PIXELS_PER_IMAGE; pixel++)
        {
            data->x_train[pixel * TRAIN_SIZE + i] = cifar10_images[i].data[pixel] / 255.0;
        }
    }

    // Process testing data (last 5000 images)
    for (int i = 0; i < TEST_SIZE; i++)
    {
        int img_idx = TRAIN_SIZE + i;

        // Copy label
        data->y_test[i] = cifar10_images[img_idx].label;

        // Transpose and normalize pixel data
        for (int pixel = 0; pixel < PIXELS_PER_IMAGE; pixel++)
        {
            data->x_test[pixel * TEST_SIZE + i] = cifar10_images[img_idx].data[pixel] / 255.0;
        }
    }
#endif

    return 0;
}

/**
 * Clean up transformed data
 */
void cleanup_transformed_data(void)
{
    if (!data)
        return;
    if (data->x_train)
    {
        free(data->x_train);
        data->x_train = NULL;
    }
    if (data->y_train)
    {
        free(data->y_train);
        data->y_train = NULL;
    }
    if (data->x_test)
    {
        free(data->x_test);
        data->x_test = NULL;
    }
    if (data->y_test)
    {
        free(data->y_test);
        data->y_test = NULL;
    }
    free(data);
    data = NULL;
}