#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "matrix.h"
#include "load.h"

// Transformed data structure
typedef struct
{
    matrix X_train;
    matrix Y_train;
    matrix X_test;
    matrix Y_test;
    int train_size;
    int test_size;
} CIFAR10Data;

// Global pointer to transformed data
extern CIFAR10Data *data;

/**
 * Prepare CIFAR-10 data for a specific MPI rank
 * Given num_processes P, rank r (0 to P-1), and total num_samples n,
 * each rank gets a disjoint subset: rank r gets samples [r*n/P, (r+1)*n/P-1] from each class
 * Returns 0 on success, 1 on error
 */
int prepare_cifar10_data(int num_samples, int rank, int num_processes);

// Clean up transformed data
void cleanup_transformed_data(void);

#endif // TRANSFORM_H