#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "load.h"

// Global variables
CIFAR10Image *cifar10_images = NULL;
const char *class_names[NUM_CLASSES] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"};

// Helper functions
static int read_cifar10_file(const char *filename, CIFAR10Image *images);

/**
 * Read CIFAR-10 binary file into pre-allocated array
 * Returns 0 on success, 1 on error
 */
static int read_cifar10_file(const char *filename, CIFAR10Image *images)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return 1;
    }

    // Read all images
    for (int i = 0; i < IMAGES_PER_BATCH; i++)
    {
        // Read label (1 byte)
        if (fread(&images[i].label, 1, 1, file) != 1)
        {
            fprintf(stderr, "Error reading label for image %d in %s\n", i, filename);
            fclose(file);
            return 1;
        }

        // Read pixel data (PIXELS_PER_IMAGE bytes)
        if (fread(images[i].data, 1, PIXELS_PER_IMAGE, file) != PIXELS_PER_IMAGE)
        {
            fprintf(stderr, "Error reading pixel data for image %d in %s\n", i, filename);
            fclose(file);
            return 1;
        }
    }

    fclose(file);
    return 0;
}

/**
 * Initialize CIFAR-10 data by loading all batch files
 * Returns 0 on success, 1 on error
 */
int init_cifar10_data()
{
    // Allocate memory for all images
    cifar10_images = (CIFAR10Image *)malloc(TOTAL_IMAGES * sizeof(CIFAR10Image));
    if (!cifar10_images)
    {
        fprintf(stderr, "Error: Memory allocation failed for images (%.2f MB needed)\n",
                (TOTAL_IMAGES * sizeof(CIFAR10Image)) / (1024.0 * 1024.0));
        return 1;
    }

    // Load all batch files
    for (int batch = 1; batch <= NUM_BATCHES; batch++)
    {
        char batch_path[512];
        snprintf(batch_path, sizeof(batch_path), "%s/data_batch_%d.bin", "cifar-10-batches-bin", batch);

        if (read_cifar10_file(batch_path, &cifar10_images[(batch - 1) * IMAGES_PER_BATCH]) != 0)
        {
            fprintf(stderr, "Error: Failed to read batch %d\n", batch);
            cleanup_cifar10_data();
            return 1;
        }
    }

    return 0;
}

/**
 * Clean up allocated memory
 */
void cleanup_cifar10_data(void)
{
    if (cifar10_images)
    {
        free(cifar10_images);
        cifar10_images = NULL;
    }
}
