#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <stdint.h>
#include "load.h"
#include "config.h"
#include "matrix.h"

// Transformed data structure with matrices
typedef struct {
    matrix X_train;   // Training images (PIXELS_PER_IMAGE × train_size)
    matrix Y_train;   // Training labels one-hot (NUM_CLASSES × train_size)
    matrix X_test;    // Testing images (PIXELS_PER_IMAGE × test_size)
    matrix Y_test;    // Testing labels one-hot (NUM_CLASSES × test_size)
    int train_size;   // Actual training size
    int test_size;    // Actual test size
} CIFAR10Data;

// Global transformed data (dynamically allocated)
extern CIFAR10Data *data;

// Functions
int prepare_cifar10_data(int num_samples);
void cleanup_transformed_data(void);

#endif // TRANSFORM_H