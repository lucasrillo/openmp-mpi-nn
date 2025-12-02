#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

#include "config.h"
#include "load.h"
#include "transform.h"
#include "matrix.h"
#include "nn.h"
#include "nn_params.h"
#include "nn_train.h"
#include "timing.h"

static void print_usage(const char *prog_name)
{
    printf("Usage: mpirun -np <num_processes> %s [OPTIONS]\n", prog_name);
    printf("Options:\n");
    printf("  -n, --train-samples <num> Number of training samples (max %d, default %d)\n", MAX_TRAINING_SAMPLES, DEFAULT_TRAINING_SAMPLES);
    printf("                            Must be divisible by BATCH_SIZE (%d) and 90 (for 10 classes, 9:1 split)\n", BATCH_SIZE);
    printf("                            In other words, must be divisible by 2880\n");
    printf("  -i, --iterations <num>    Number of training iterations (default %d)\n", DEFAULT_NUM_ITERATIONS);
    printf("  -p, --print <num>         Print progress every N iterations (default %d)\n", DEFAULT_PRINT_EVERY);
    printf("  -t, --threads <num>       Number of OpenMP threads per process (default %d)\n", DEFAULT_NUM_THREADS);
    printf("  -h, --help                Show this help message\n");
    printf("\nExample:\n");
    printf("  mpirun -np 4 %s -n 2880 -i 10 -p 1 -t 4\n", prog_name);
}

int main(int argc, char *argv[])
{
    // ========== INITIALIZE MPI ==========
    MPI_Init(&argc, &argv);

    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    // ========== START TOTAL PROGRAM TIMER ==========
    TIMER_START(g_total_program_time);

    // Default values
    int num_training_samples = DEFAULT_TRAINING_SAMPLES;
    int num_iterations = DEFAULT_NUM_ITERATIONS;
    int print_every = DEFAULT_PRINT_EVERY;
    int num_threads = DEFAULT_NUM_THREADS;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            if (rank == 0)
                print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
        else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--train-samples") == 0) && i + 1 < argc)
        {
            num_training_samples = atoi(argv[++i]);
            if (num_training_samples <= 0 || num_training_samples > MAX_TRAINING_SAMPLES)
            {
                if (rank == 0)
                    fprintf(stderr, "Error: Number of training samples must be between 1 and %d\n", MAX_TRAINING_SAMPLES);
                MPI_Finalize();
                return 1;
            }
        }
        else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0) && i + 1 < argc)
        {
            num_iterations = atoi(argv[++i]);
            if (num_iterations <= 0)
            {
                if (rank == 0)
                    fprintf(stderr, "Error: Number of iterations must be positive\n");
                MPI_Finalize();
                return 1;
            }
        }
        else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--print") == 0) && i + 1 < argc)
        {
            print_every = atoi(argv[++i]);
            if (print_every < 0)
            {
                if (rank == 0)
                    fprintf(stderr, "Error: Print frequency must be non-negative\n");
                MPI_Finalize();
                return 1;
            }
        }
        else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) && i + 1 < argc)
        {
            num_threads = atoi(argv[++i]);
            if (num_threads <= 0)
            {
                if (rank == 0)
                    fprintf(stderr, "Error: Number of threads must be positive\n");
                MPI_Finalize();
                return 1;
            }
        }
        else
        {
            if (rank == 0)
            {
                fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
                print_usage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // BATCH_SIZE must be divisible by num_processes
    if (BATCH_SIZE % num_processes != 0)
    {
        if (rank == 0)
            fprintf(stderr, "Error: BATCH_SIZE (%d) must be divisible by num_processes (%d)\n", BATCH_SIZE, num_processes);
        MPI_Finalize();
        return 1;
    }


    // Training samples must be divisible by BATCH_SIZE
    if (num_training_samples % BATCH_SIZE != 0)
    {
        if (rank == 0)
            fprintf(stderr, "Error: Training samples (%d) must be divisible by BATCH_SIZE (%d)\n", num_training_samples, BATCH_SIZE);
        MPI_Finalize();
        return 1;
    }

    // Training samples must be divisible by 90 (9:1 ratio plus balanced 10 classes)
    if (num_training_samples % 90 != 0)
    {
        if (rank == 0)
            fprintf(stderr, "Error: Training samples (%d) must be divisible by 90\n", num_training_samples);
        MPI_Finalize();
        return 1;
    }

    // Calculate test samples and total samples
    int num_test_samples = num_training_samples / 9; // 10% of total
    int num_samples = num_training_samples + num_test_samples;

    // Set number of OpenMP threads
    omp_set_num_threads(num_threads);

    timer_t_custom startup_timer;
    TIMER_START(startup_timer);

    int samples_per_process = num_samples / num_processes;
    int train_per_process = num_training_samples / num_processes;
    int test_per_process = num_test_samples / num_processes;

    if (rank == 0)
    {
        printf("\n========== CIFAR-10 Neural Network (MPI + OpenMP) ==========\n");
        printf("Training samples: %d (90%% of total)\n", num_training_samples);
        printf("Test samples: %d (10%% of total)\n", num_test_samples);
        printf("Total samples: %d\n", num_samples);
        printf("MPI processes: %d\n", num_processes);
        printf("Samples per process: %d (train: %d, test: %d)\n", samples_per_process, train_per_process, test_per_process);
        printf("Samples per class (global): train: %d, test: %d\n", num_training_samples / NUM_CLASSES, num_test_samples / NUM_CLASSES);
        printf("Mini-batch size: %d (global), %d (per process)\n", BATCH_SIZE, BATCH_SIZE / num_processes);
        printf("Iterations: %d\n", num_iterations);
        printf("Print every: %d iterations\n", print_every);
        printf("OpenMP threads per process: %d\n", num_threads);
        printf("=============================================================\n\n");
    }

    // Load data (all ranks load all data)
    timer_t_custom load_timer;
    TIMER_START(load_timer);

    if (init_cifar10_data() != 0)
    {
        fprintf(stderr, "Rank %d: Failed to initialize CIFAR-10 data\n", rank);
        MPI_Finalize();
        return 1;
    }

    TIMER_STOP(load_timer);

    // Synchronize after loading
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("\n========= DATA LOADED ==========\n");
        printf("[TIMER] Data loading: %.2f ms\n", load_timer.elapsed_ms);
        printf("Successfully loaded %d total images (%.2f MB)\n", TOTAL_IMAGES, TOTAL_MEMORY_MB);
        printf("================================\n\n");
    }

    // Transform data (each rank gets its subset)
    timer_t_custom transform_timer;
    TIMER_START(transform_timer);

    if (prepare_cifar10_data(num_samples, rank, num_processes) != 0)
    {
        fprintf(stderr, "Rank %d: Failed to prepare CIFAR-10 data\n", rank);
        cleanup_cifar10_data();
        MPI_Finalize();
        return 1;
    }

    TIMER_STOP(transform_timer);

    // Synchronize after transformation
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("\n====== DATA TRANSFORMED ======\n");
        printf("[TIMER] Data transformation: %.2f ms\n", transform_timer.elapsed_ms);
        printf("Each process prepared its data subset\n");
        printf("Local data shapes (per process):\n");
        printf("  X_train: %d x %d (features x samples)\n", data->X_train.rows, data->X_train.cols);
        printf("  Y_train: %d x %d (classes x samples)\n", data->Y_train.rows, data->Y_train.cols);
        printf("  X_test:  %d x %d\n", data->X_test.rows, data->X_test.cols);
        printf("  Y_test:  %d x %d\n", data->Y_test.rows, data->Y_test.cols);
        printf("================================\n\n");
    }

    TIMER_STOP(startup_timer);
    if (rank == 0)
        printf("\n[TIMER] Total startup: %.2f ms\n\n", startup_timer.elapsed_ms);

    // Define neural network architecture
    int layer_dims[] = {PIXELS_PER_IMAGE, 128, 64, NUM_CLASSES};
    int L = 3; // number of layers (excluding input)

    // Train the model
    nn_params params = train_model(&data->X_train, &data->Y_train, &data->X_test, &data->Y_test,
                                   layer_dims, L, DEFAULT_LEARNING_RATE, num_iterations,
                                   print_every, num_samples, num_threads, rank, num_processes);

    // Cleanup
    if (rank == 0)
        printf("\nCleaning up...\n");
    delete_nn_params(&params);
    cleanup_transformed_data();
    cleanup_cifar10_data();

    // Stop total program timer
    TIMER_STOP(g_total_program_time);
    if (rank == 0)
    {
        printf("Done!\n\n");
        printf("========================================\n");
        printf("[TIMER] TOTAL PROGRAM TIME: %.3f seconds\n", g_total_program_time.elapsed_ms / 1000.0);
        printf("========================================\n");
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}