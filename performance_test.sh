#!/bin/bash

# SLURM job script for testing MPI + OpenMP scaling performance
# This script runs multiple configurations to analyze parallel efficiency
# Submit with: sbatch performance_test.sh

#SBATCH --nodes=1                    # Number of nodes to use
#SBATCH --cpus-per-task=32           # Number of processor cores per node
#SBATCH --time=0-2:0:0               # Walltime limit
#SBATCH --job-name="nn_scaling"      # Job name
#SBATCH --output=scaling_results_%j.out   # Output file with job ID
#SBATCH --constraint="nova18"

# Load necessary modules
module load openmpi

# Fixed parameters for all runs
TRAIN_SAMPLES=2880
ITERATIONS=1
PRINT_EVERY=1

run_test() {
    local np=$1  # Number of MPI Processes
    local nt=$2  # Number of OpenMP Threads per Process
    
    export OMP_NUM_THREADS=$nt
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores

    mpirun -np $np \
           --map-by node:PE=$nt \
           --bind-to core \
           ./main.exe -n $TRAIN_SAMPLES -i $ITERATIONS -p $PRINT_EVERY -t $nt
}

run_test 1 1
run_test 1 2
run_test 1 4
run_test 1 8
run_test 1 16
