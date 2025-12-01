#ifndef CONFIG_H
#define CONFIG_H

// ============================================
// DEBUG/DEVELOPMENT CONFIGURATION
// ============================================

// Set to 1 to use a smaller subset for faster debugging
#define USE_DEBUG_SUBSET 1

#if USE_DEBUG_SUBSET
    // Small i.i.d. subset for debugging
    #define DEBUG_TRAIN_SIZE 1000   // 1000 training samples
    #define DEBUG_TEST_SIZE  200    // 200 test samples
#endif // #if USE_DEBUG_SUBSET

// ============================================
// TRAINING CONFIGURATION  
// ============================================

// Default hyperparameters (can be overridden in main)
#define DEFAULT_LEARNING_RATE 0.001
#define DEFAULT_NUM_ITERATIONS 10
#define DEFAULT_PRINT_EVERY 1

// ============================================
// TIMING CONFIGURATION
// ============================================

// Set to 1 to enable detailed timing output
#define ENABLE_TIMING 1

// Set to 1 to print per-iteration timing (verbose)
#define VERBOSE_TIMING 1

#endif // CONFIG_H