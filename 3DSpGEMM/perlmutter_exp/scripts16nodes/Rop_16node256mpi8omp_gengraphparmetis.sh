#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --constraint=cpu
#SBATCH -J PM_mo16
#SBATCH -o "parmetis_mo16_%j.out"
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8
cd /pscratch/sd/y/yuxihong/graphclustering/perfrun
echo Node size: $SLURM_NNODES, Ntasks per Node:$SLURM_NTASKS_PER_NODE, Ntasks:$SLURM_NTASKS 
echo CPUs per task:$SLURM_CPUS_PER_TASK, omp threads:$OMP_NUM_THREADS

# srun --cpu-bind=cores ./SSMCL/mtx2graph --dataset isom100-1,isom100-3,mo16 --metisgp flops,1024
# srun --cpu-bind=cores ./SSMCL/mtx2graph --dataset metaclust50 --metisgp flops,1024

srun --cpu-bind=cores ./parmetis-4.0.3/programs/parmetis $DLOC/mo16_flops.graph 1
srun --cpu-bind=cores ./parmetis-4.0.3/programs/parmetis $DLOC/agatha_flops.graph 1
srun --cpu-bind=cores ./parmetis-4.0.3/programs/parmetis $DLOC/friends_flops.graph 1
# srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset $dataset --AA --Rop --2d --3d 4,16,64,256 --randperm 
# srun --cpu-bind=cores ./3DSpGEMM/Rop --dataset $dataset --AA --Rop --2d --3d 4,16,64,256
