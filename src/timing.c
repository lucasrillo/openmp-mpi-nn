#include <time.h>
#include <stdio.h>

#include "timing.h"

timer_accum_t g_forward_time;
timer_accum_t g_backward_time;
timer_accum_t g_update_time;
timer_accum_t g_cost_time;
timer_accum_t g_accuracy_time;
timer_t_custom g_total_program_time;

// Initialize all global accumulators
void init_timing_accumulators(void)
{
    ACCUM_INIT(g_forward_time, "Forward Pass");
    ACCUM_INIT(g_backward_time, "Backward Pass");
    ACCUM_INIT(g_update_time, "Parameter Update");
    ACCUM_INIT(g_cost_time, "Cost Computation");
    ACCUM_INIT(g_accuracy_time, "Accuracy Computation");
}

// Print all timing summaries
void print_timing_summary(void)
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
    printf("========================================\n\n");
}

// Log results to CSV file
void log_results_to_csv(const char *filename,
                        int num_samples,
                        int num_iterations,
                        double learning_rate,
                        double final_train_acc,
                        double final_test_acc,
                        double total_time_sec)
{
    FILE *file = fopen(filename, "r");
    int file_exists = (file != NULL);
    if (file)
        fclose(file);

    file = fopen(filename, "a");
    if (!file)
    {
        fprintf(stderr, "Warning: Could not open %s for writing\n", filename);
        return;
    }

    // Write header if file is new
    if (!file_exists)
    {
        fprintf(file, "num_samples,num_iterations,learning_rate,train_accuracy,test_accuracy,"
                      "total_time_sec,forward_time_ms,backward_time_ms,update_time_ms,"
                      "cost_time_ms,accuracy_time_ms,avg_forward_ms,avg_backward_ms,"
                      "avg_update_ms\n");
    }

    // Write data row
    fprintf(file, "%d,%d,%.6f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            num_samples,
            num_iterations,
            learning_rate,
            final_train_acc,
            final_test_acc,
            total_time_sec,
            g_forward_time.total_ms,
            g_backward_time.total_ms,
            g_update_time.total_ms,
            g_cost_time.total_ms,
            g_accuracy_time.total_ms,
            g_forward_time.count > 0 ? g_forward_time.total_ms / g_forward_time.count : 0.0,
            g_backward_time.count > 0 ? g_backward_time.total_ms / g_backward_time.count : 0.0,
            g_update_time.count > 0 ? g_update_time.total_ms / g_update_time.count : 0.0);

    fclose(file);
    printf("Results logged to %s\n", filename);
}
