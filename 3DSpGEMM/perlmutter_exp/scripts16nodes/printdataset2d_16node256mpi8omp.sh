#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --constraint=cpu
#SBATCH -J PD_isom100-1
#SBATCH -o "PD_isom1001_%j.out"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8
cd /pscratch/sd/y/yuxihong/graphclustering/perfrun
echo Node size: $SLURM_NNODES, Ntasks per Node:$SLURM_NTASKS_PER_NODE, Ntasks:$SLURM_NTASKS 
echo CPUs per task:$SLURM_CPUS_PER_TASK, omp threads:$OMP_NUM_THREADS

srun --cpu-bind=cores ./SSMCL/printdataset2d \
 /global/cfs/cdirs/m1982/HipMCL/iso_m100/subgraph1/subgraph1_iso_vs_iso_30_70length_ALL.m100.indexed.mtx
