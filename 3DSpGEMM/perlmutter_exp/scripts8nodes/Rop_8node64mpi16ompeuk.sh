#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --constraint=cpu
#SBATCH -J eukarch
#SBATCH -o "parmetis_eukarch_%j.out"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=16
cd /pscratch/sd/y/yuxihong/graphclustering/perfrun
echo Node size: $SLURM_NNODES, Ntasks per Node:$SLURM_NTASKS_PER_NODE, Ntasks:$SLURM_NTASKS 
echo CPUs per task:$SLURM_CPUS_PER_TASK, omp threads:$OMP_NUM_THREADS

srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset euk --AA --1d --metisgp file,64
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset arch --AA --1d --metisgp file,64
export dataset=euk,arch
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset $dataset --AA --1d --randperm
srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset $dataset --AA --1d 

