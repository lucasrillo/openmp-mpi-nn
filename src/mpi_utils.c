#include <mpi.h>
#include <stdlib.h>

#include "mpi_utils.h"

void allreduce_matrix(matrix *A, int num_processes)
{
    int size = A->rows * A->cols;

    // Allocate buffer for receiving sum
    double *recv_buffer = (double *)malloc(size * sizeof(double));

    // Sum across all processes
    MPI_Allreduce(A->val, recv_buffer, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Average by dividing by number of processes
    double inv_np = 1.0 / num_processes;
    for (int i = 0; i < size; i++)
        A->val[i] = recv_buffer[i] * inv_np;

    free(recv_buffer);
}

void allreduce_gradients(nn_grads *grads, int L, int num_processes)
{
    for (int l = 0; l < L; l++)
    {
        allreduce_matrix(&grads->dW[l], num_processes);
        allreduce_matrix(&grads->db[l], num_processes);
    }
}

double allreduce_cost(double local_cost, int num_processes)
{
    double global_cost;
    MPI_Allreduce(&local_cost, &global_cost, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_cost / num_processes;
}

double allreduce_accuracy(int local_correct, int local_total)
{
    int global_correct, global_total;

    MPI_Allreduce(&local_correct, &global_correct, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_total, &global_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    return (double)global_correct / global_total * 100.0;
}
