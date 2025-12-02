#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <mpi.h>
#include "matrix.h"
#include "nn.h"

void allreduce_matrix(matrix *A, int num_processes); // Averages A across all processes (in-place)
void allreduce_gradients(nn_grads *grads, int L, int num_processes);
double allreduce_cost(double local_cost, int num_processes);
double allreduce_accuracy(int local_correct, int local_total);

#endif // MPI_UTILS_H