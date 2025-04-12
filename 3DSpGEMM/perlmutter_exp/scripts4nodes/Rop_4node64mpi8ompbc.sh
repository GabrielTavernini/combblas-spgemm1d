#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --constraint=cpu
#SBATCH -J eukbc64
#SBATCH -o "eukbc64mpi_%j.out"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8
cd /pscratch/sd/y/yuxihong/graphclustering/perfrun
echo Node size: $SLURM_NNODES, Ntasks per Node:$SLURM_NTASKS_PER_NODE, Ntasks:$SLURM_NTASKS 
echo CPUs per task:$SLURM_CPUS_PER_TASK, omp threads:$OMP_NUM_THREADS
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset hv15r --BC 17,4096 --1d
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset hv15r --BC 17,4096 --2d --randperm
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset hv15r --BC 17,4096 --3d 4,16 --randperm

srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset euk --BC 17,4096 --1d --metisgp file,64
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset euk --BC 17,4096 --1d 
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset euk --BC 17,4096 --2d --randperm
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset euk --BC 17,4096 --3d 4,16 --randperm

