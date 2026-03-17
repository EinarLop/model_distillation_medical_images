#!/bin/bash
#SBATCH --account=project_2018357
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:05:00
#SBATCH --gres=gpu:a100_1g.5gb:1,nvme:20

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DATASET_DIR=$LOCAL_SCRATCH

unzip -q /scratch/project_2018357/data/chexpert.zip -d $LOCAL_SCRATCH


module load pytorch
srun python3 ./../src/main.py
