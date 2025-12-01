#ifndef TIMING_H
#define TIMING_H

#include <time.h>
#include <stdio.h>

// High-resolution timer using clock_gettime
typedef struct {
    struct timespec start;
    struct timespec end;
    double elapsed_ms;
} timer_t_custom;

// Timer macros for easy use
#define TIMER_START(timer) clock_gettime(CLOCK_MONOTONIC, &(timer).start)

#define TIMER_STOP(timer) do { \
    clock_gettime(CLOCK_MONOTONIC, &(timer).end); \
    (timer).elapsed_ms = ((timer).end.tv_sec - (timer).start.tv_sec) * 1000.0 + \
                         ((timer).end.tv_nsec - (timer).start.tv_nsec) / 1e6; \
} while(0)

#define TIMER_PRINT(timer, label) \
    printf("[TIMER] %-30s: %10.3f ms\n", label, (timer).elapsed_ms)

// Accumulator for tracking totals across iterations
typedef struct {
    double total_ms;
    int count;
    const char *label;
} timer_accum_t;

#define ACCUM_INIT(accum, name) do { \
    (accum).total_ms = 0.0; \
    (accum).count = 0; \
    (accum).label = name; \
} while(0)

#define ACCUM_ADD(accum, timer) do { \
    (accum).total_ms += (timer).elapsed_ms; \
    (accum).count++; \
} while(0)

#define ACCUM_PRINT(accum) \
    printf("[TOTAL] %-30s: %10.3f ms (avg: %.3f ms, count: %d)\n", \
           (accum).label, (accum).total_ms, \
           (accum).count > 0 ? (accum).total_ms / (accum).count : 0.0, \
           (accum).count)

// Global timing accumulators (declare in one source file)
#ifdef TIMING_IMPL
timer_accum_t g_forward_time;
timer_accum_t g_backward_time;
timer_accum_t g_update_time;
timer_accum_t g_cost_time;
timer_accum_t g_accuracy_time;
#else
extern timer_accum_t g_forward_time;
extern timer_accum_t g_backward_time;
extern timer_accum_t g_update_time;
extern timer_accum_t g_cost_time;
extern timer_accum_t g_accuracy_time;
#endif

// Initialize all global accumulators
static inline void init_timing_accumulators(void)
{
    ACCUM_INIT(g_forward_time, "Forward Pass");
    ACCUM_INIT(g_backward_time, "Backward Pass");
    ACCUM_INIT(g_update_time, "Parameter Update");
    ACCUM_INIT(g_cost_time, "Cost Computation");
    ACCUM_INIT(g_accuracy_time, "Accuracy Computation");
}

// Print all timing summaries
static inline void print_timing_summary(void)
{
    printf("\n========== TIMING SUMMARY ==========\n");
    ACCUM_PRINT(g_forward_time);
    ACCUM_PRINT(g_backward_time);
    ACCUM_PRINT(g_update_time);
    ACCUM_PRINT(g_cost_time);
    ACCUM_PRINT(g_accuracy_time);
    
    double total = g_forward_time.total_ms + g_backward_time.total_ms + 
                   g_update_time.total_ms + g_cost_time.total_ms;
    printf("------------------------------------\n");
    printf("[TOTAL] %-30s: %10.3f ms\n", "Training Loop", total);
    printf("========================================\n");
}

#endif // TIMING_H