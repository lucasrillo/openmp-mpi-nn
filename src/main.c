#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "load.h"
#include "transform.h"
#include "matrix.h"
#include "nn.h"
#include "nn_params.h"
#include "nn_train.h"
#include "timing.h"

void print_usage(const char *prog_name)
{
    printf("Usage: %s [OPTIONS]\n", prog_name);
    printf("Options:\n");
    printf("  -n, --samples <num>      Number of samples to use (must be multiple of 100, max %d, default %d)\n",
           MAX_NUM_SAMPLES, DEFAULT_NUM_SAMPLES);
    printf("  -i, --iterations <num>   Number of training iterations (default %d)\n", DEFAULT_NUM_ITERATIONS);
    printf("  -p, --print <num>        Print progress every N iterations (default %d)\n", DEFAULT_PRINT_EVERY);
    printf("  -h, --help               Show this help message\n");
    printf("\nExample:\n");
    printf("  %s -n 10000 -i 500 -p 50\n", prog_name);
}

int main(int argc, char *argv[])
{
    // ========== START TOTAL PROGRAM TIMER ==========
    TIMER_START(g_total_program_time);
    
    // Default values
    int num_samples = DEFAULT_NUM_SAMPLES;
    int num_iterations = DEFAULT_NUM_ITERATIONS;
    int print_every = DEFAULT_PRINT_EVERY;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--samples") == 0) && i + 1 < argc)
        {
            num_samples = atoi(argv[++i]);
            if (num_samples <= 0 || num_samples > MAX_NUM_SAMPLES)
            {
                fprintf(stderr, "Error: Number of samples must be between 1 and %d\n", MAX_NUM_SAMPLES);
                return 1;
            }
            if (num_samples % 100 != 0)
            {
                fprintf(stderr, "Error: Number of samples must be a multiple of 100\n");
                return 1;
            }
        }
        else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0) && i + 1 < argc)
        {
            num_iterations = atoi(argv[++i]);
            if (num_iterations <= 0)
            {
                fprintf(stderr, "Error: Number of iterations must be positive\n");
                return 1;
            }
        }
        else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--print") == 0) && i + 1 < argc)
        {
            print_every = atoi(argv[++i]);
            if (print_every < 0)
            {
                fprintf(stderr, "Error: Print frequency must be non-negative\n");
                return 1;
            }
        }
        else
        {
            fprintf(stderr, "Error: Unknown argument '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    timer_t_custom startup_timer;
    TIMER_START(startup_timer);

    int train_size = (num_samples * 9) / 10;
    int test_size = num_samples - train_size;
    printf("\n========== CIFAR-10 Neural Network ==========\n");
    printf("Samples: %d (train: %d, test: %d)\n", num_samples, train_size, test_size);
    printf("Samples per class: %d (train: %d, test: %d)\n", num_samples / NUM_CLASSES, (train_size / NUM_CLASSES), (test_size / NUM_CLASSES));
    printf("Iterations: %d\n", num_iterations);
    printf("Print every: %d iterations\n", print_every);
    printf("==============================================\n\n");

    // ========== LOAD DATA ==========
    timer_t_custom load_timer;
    TIMER_START(load_timer);

    if (init_cifar10_data() != 0)
    {
        fprintf(stderr, "Failed to initialize CIFAR-10 data\n");
        return 1;
    }

    TIMER_STOP(load_timer);
    printf("\n========= DATA LOADED ==========\n");
    printf("[TIMER] Data loading: %.2f ms\n", load_timer.elapsed_ms);
    printf("Successfully loaded %d total images (%.2f MB)\n", TOTAL_IMAGES, TOTAL_MEMORY_MB);
    printf("================================\n\n");

    // ========== TRANSFORM DATA ==========
    timer_t_custom transform_timer;
    TIMER_START(transform_timer);

    if (prepare_cifar10_data(num_samples) != 0)
    {
        fprintf(stderr, "Failed to prepare CIFAR-10 data\n\n");
        cleanup_cifar10_data();
        return 1;
    }

    TIMER_STOP(transform_timer);
    printf("\n====== DATA TRANSFORMED ======\n");
    printf("[TIMER] Data transformation: %.2f ms\n", transform_timer.elapsed_ms);
    printf("Successfully prepared transformed data\n");
    printf("Data shapes:\n");
    printf("  X_train: %d x %d (features x samples)\n", data->X_train.rows, data->X_train.cols);
    printf("  Y_train: %d x %d (classes x samples)\n", data->Y_train.rows, data->Y_train.cols);
    printf("  X_test:  %d x %d\n", data->X_test.rows, data->X_test.cols);
    printf("  Y_test:  %d x %d\n", data->Y_test.rows, data->Y_test.cols);
    printf("================================\n\n");

    TIMER_STOP(startup_timer);
    printf("\n[TIMER] Total startup: %.2f ms\n\n", startup_timer.elapsed_ms);

    // ========== DEFINE NETWORK ARCHITECTURE ==========
    int layer_dims[] = {PIXELS_PER_IMAGE, 128, 64, NUM_CLASSES};
    int L = 3; // number of layers (excluding input)

    // ========== TRAIN MODEL ==========
    nn_params params = train_model(&data->X_train, &data->Y_train, &data->X_test, &data->Y_test, 
                                    layer_dims, L, DEFAULT_LEARNING_RATE, num_iterations, 
                                    print_every, num_samples);

    // ========== CLEANUP ==========
    printf("\nCleaning up...\n");
    delete_nn_params(&params);
    cleanup_transformed_data();
    cleanup_cifar10_data();
    
    // ========== STOP TOTAL PROGRAM TIMER ==========
    TIMER_STOP(g_total_program_time);
    printf("Done!\n\n");
    printf("========================================\n");
    printf("[TIMER] TOTAL PROGRAM TIME: %.3f seconds\n", g_total_program_time.elapsed_ms / 1000.0);
    printf("========================================\n");

    return 0;
}