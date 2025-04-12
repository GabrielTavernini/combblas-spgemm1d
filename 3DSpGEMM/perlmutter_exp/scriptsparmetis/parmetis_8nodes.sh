#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --constraint=cpu
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8
cd /pscratch/sd/y/yuxihong/graphclustering/latestdebug
echo Node size: $SLURM_NNODES, Ntasks per Node:$SLURM_NTASKS_PER_NODE, Ntasks:$SLURM_NTASKS 
echo CPUs per task:$SLURM_CPUS_PER_TASK, omp threads:$OMP_NUM_THREADS
export dataset=diereal,cage15,queen,nlpkkt200,stokes,hv15r
srun --cpu-bind=cores ./parmetis-4.0.3/programs/parmetis $DLOC/stokes_flops.graph 1
srun --cpu-bind=cores ./parmetis-4.0.3/programs/parmetis $DLOC/nlpkkt200_flops.graph 1
srun --cpu-bind=cores ./parmetis-4.0.3/programs/parmetis $DLOC/hv15r_flops.graph 1

