#ifndef CONFIG_H
#define CONFIG_H

// Maximum and default training dataset size
#define MAX_TRAINING_SAMPLES 43200 // 64 * 675 (divisible by 64 and 90, less than 90% of 50000)
#define DEFAULT_TRAINING_SAMPLES MAX_TRAINING_SAMPLES

// Default hyperparameters
#define DEFAULT_LEARNING_RATE 0.001
#define DEFAULT_NUM_ITERATIONS 1000
#define DEFAULT_PRINT_EVERY 100
#define BATCH_SIZE 64

// Random seed for reproducibility
#define RANDOM_SEED 42

// Default number of OpenMP threads
#define DEFAULT_NUM_THREADS 1

#endif // CONFIG_H