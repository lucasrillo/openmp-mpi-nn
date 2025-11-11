#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "transform.h"

// Global variables
CIFAR10Data *data = NULL;

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

    // Process training data (first 45000 images)
    for (int i = 0; i < TRAIN_SIZE; i++)
    {
        // Copy label
        data->y_train[i] = cifar10_images[i].label;

        // Transpose and normalize pixel data so that each column is an image, values in [0,1]
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

        // Transpose and normalize pixel data so that each column is an image, values in [0,1]
        for (int pixel = 0; pixel < PIXELS_PER_IMAGE; pixel++)
        {
            data->x_test[pixel * TEST_SIZE + i] = cifar10_images[img_idx].data[pixel] / 255.0;
        }
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