#!/bin/bash
#SBATCH --account=project_2018357
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:a100:1,nvme:20

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DATASET_DIR=$LOCAL_SCRATCH
export BATCH_SIZE=256
export ROOT_PATH = "/projappl/project_2018357/model_distillation_medical_images"

unzip -q /scratch/project_2018357/data/chexpert.zip -d $LOCAL_SCRATCH

module load pytorch
srun python3 /projappl/project_2018357/model_distillation_medical_images/src/main.py

