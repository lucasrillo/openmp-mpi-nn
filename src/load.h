#ifndef LOAD_H
#define LOAD_H

#include <stdint.h>

#define IMAGE_SIZE 32
#define CHANNELS 3
#define PIXELS_PER_IMAGE (IMAGE_SIZE * IMAGE_SIZE * CHANNELS)
#define RECORD_SIZE (1 + PIXELS_PER_IMAGE) // 1 byte label + 3072 bytes pixels
#define NUM_CLASSES 10
#define IMAGES_PER_BATCH 10000
#define NUM_BATCHES 5
#define TOTAL_IMAGES (IMAGES_PER_BATCH * NUM_BATCHES)
#define TOTAL_MEMORY_MB ((TOTAL_IMAGES * sizeof(CIFAR10Image)) / (1024.0 * 1024.0))

// CIFAR-10 image structure
typedef struct
{
    uint8_t label;
    uint8_t data[PIXELS_PER_IMAGE];
} CIFAR10Image;

// Global arrays (dynamically allocated)
extern CIFAR10Image *cifar10_images;
extern const char *class_names[NUM_CLASSES];

// Functions
int init_cifar10_data(void);
void cleanup_cifar10_data(void);

#endif // LOAD_H