#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=512
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cpu
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=4
cd /pscratch/sd/y/yuxihong/graphclustering/perfrun
echo Node size: $SLURM_NNODES, Ntasks per Node:$SLURM_NTASKS_PER_NODE, Ntasks:$SLURM_NTASKS 
echo CPUs per task:$SLURM_CPUS_PER_TASK, omp threads:$OMP_NUM_THREADS
export dataset=vir
# 1. convert the mtx to graph 
srun --cpu-bind=cores ./SSMCL/mtx2graph --dataset $dataset --metisgp flops,$SLURM_NTASKS