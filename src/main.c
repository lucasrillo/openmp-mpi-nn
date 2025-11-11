#include <stdio.h>
#include <stdlib.h>

#include "load.h"
#include "transform.h"

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

    // Example: Access pixel 0 of training image 0
    printf("\nExample access - first pixel of first training image: %.4f\n",
           data->x_train[0 * TRAIN_SIZE + 0]);
    printf("Label of first training image: %u (%s)\n",
           data->y_train[0], class_names[data->y_train[0]]);

    // TODO: Define sigmoid, rand, forward, backward functions

    // Cleanup
    cleanup_transformed_data();
    cleanup_cifar10_data();

    return 0;
}