#!/bin/bash
#SBATCH --account=project_2018357
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              
#SBATCH --time=00:30:00               
#SBATCH --gres=gpu:a100_1g.5gb:1,nvme:20
#SBATCH --mem=16G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DATASET_DIR=$LOCAL_SCRATCH
export BATCH_SIZE=32
export ROOT_PATH="/projappl/project_2018357/model_distillation_medical_images"
export TRAIN_SUBSET_SIZE=2000
export VAL_SUBSET_SIZE=400
export CHECKPOINT_NAME="checkpoint_reduced_gpu_run2403.pth"

unzip -q /scratch/project_2018357/data/chexpert.zip -d $LOCAL_SCRATCH

module load pytorch
srun python3 /projappl/project_2018357/model_distillation_medical_images/src/main.py

