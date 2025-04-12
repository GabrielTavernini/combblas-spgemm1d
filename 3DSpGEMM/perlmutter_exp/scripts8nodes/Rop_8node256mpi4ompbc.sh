#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH --constraint=cpu
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=4
cd /pscratch/sd/y/yuxihong/graphclustering/perfrun
echo Node size: $SLURM_NNODES, Ntasks per Node:$SLURM_NTASKS_PER_NODE, Ntasks:$SLURM_NTASKS 
echo CPUs per task:$SLURM_CPUS_PER_TASK, omp threads:$OMP_NUM_THREADS
export dataset=euk,hv15r
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset hv15r --BC 17,4096 --1d
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset hv15r --BC 17,4096 --2d --randperm
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset hv15r --BC 17,4096 --3d 4,16 --randperm

srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset euk --BC 17,4096 --1d --metisgp file,256
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset euk --BC 17,4096 --1d
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset euk --BC 17,4096 --2d --randperm
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset euk --BC 17,4096 --3d 4,16 --randperm


