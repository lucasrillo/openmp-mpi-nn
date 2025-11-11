#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <stdint.h>
#include "load.h"

#define TRAIN_SIZE 45000
#define TEST_SIZE 5000

// Transformed data structure
typedef struct {
    double *x_train;  // Training images (PIXELS_PER_IMAGE × TRAIN_SIZE)
    uint8_t *y_train; // Training labels (TRAIN_SIZE)
    double *x_test;   // Testing images (PIXELS_PER_IMAGE × TEST_SIZE)
    uint8_t *y_test;  // Testing labels (TEST_SIZE)
} CIFAR10Data;

// Global transformed data (dynamically allocated)
extern CIFAR10Data *data;

// Functions
int prepare_cifar10_data(void);
void cleanup_transformed_data(void);

#endif // TRANSFORM_H