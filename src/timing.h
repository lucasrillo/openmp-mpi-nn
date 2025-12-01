#ifndef TIMING_H
#define TIMING_H

#include <time.h>
#include <stdio.h>

// High-resolution timer using clock_gettime
typedef struct
{
    struct timespec start;
    struct timespec end;
    double elapsed_ms;
} timer_t_custom;

// Timer macros for easy use
#define TIMER_START(timer) clock_gettime(CLOCK_MONOTONIC, &(timer).start)

#define TIMER_STOP(timer)                                                           \
    do                                                                              \
    {                                                                               \
        clock_gettime(CLOCK_MONOTONIC, &(timer).end);                               \
        (timer).elapsed_ms = ((timer).end.tv_sec - (timer).start.tv_sec) * 1000.0 + \
                             ((timer).end.tv_nsec - (timer).start.tv_nsec) / 1e6;   \
    } while (0)

// Accumulator for tracking totals across iterations
typedef struct
{
    double total_ms;
    int count;
    const char *label;
} timer_accum_t;

#define ACCUM_INIT(accum, name) \
    do                          \
    {                           \
        (accum).total_ms = 0.0; \
        (accum).count = 0;      \
        (accum).label = name;   \
    } while (0)

#define ACCUM_ADD(accum, timer)                 \
    do                                          \
    {                                           \
        (accum).total_ms += (timer).elapsed_ms; \
        (accum).count++;                        \
    } while (0)

#define ACCUM_PRINT(accum)                                             \
    printf("[TOTAL] %-30s: %10.3f ms (avg: %.3f ms, count: %d)\n",     \
           (accum).label, (accum).total_ms,                            \
           (accum).count > 0 ? (accum).total_ms / (accum).count : 0.0, \
           (accum).count)

// Global timing accumulators
extern timer_accum_t g_forward_time;
extern timer_accum_t g_backward_time;
extern timer_accum_t g_update_time;
extern timer_accum_t g_cost_time;
extern timer_accum_t g_accuracy_time;
extern timer_t_custom g_total_program_time;

// Function declarations
void init_timing_accumulators(void);
void print_timing_summary(void);
void log_results_to_csv(const char *filename, int num_samples, int num_iterations, double learning_rate, double final_train_acc, double final_test_acc, double training_time_sec, int num_threads);

#endif // TIMING_H