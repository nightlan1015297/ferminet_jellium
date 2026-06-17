#!/bin/bash
#---------------------------------------------------------------------#
# SLURM Options
#---------------------------------------------------------------------#
#SBATCH --account=MST114233
#SBATCH --job-name=jelly                          # Job name with parameters
#SBATCH --partition=8gpus                         # Partition
#SBATCH --time=48:00:00                          # Time limit (HH:MM:SS)
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --cpus-per-task=2                       # CPUs per task
#SBATCH --mem=32G                               # Memory per node
#SBATCH --gres=gpu:4                       # Request 1 h200 GPU
#SBATCH --output=slurm_logs/job%j.out           # Log file with parameters
#SBATCH --error=slurm_logs/job%j.err            # Error file with parameters
#---------------------------------------------------------------------#
# Job Environment Setup
#---------------------------------------------------------------------#
# Create logs directory if it doesn't exist
mkdir -p slurm_logs 

echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running from script: $(realpath $0)"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "nvidia-smi output:"
nvidia-smi
echo ""

echo ""
echo "Activating Conda environment..."
CONDA_INSTALL_PATH="/work/u9996654/miniconda3"
source "${CONDA_INSTALL_PATH}/etc/profile.d/conda.sh"
if [ $? -ne 0 ]; then
    echo "Error sourcing conda profile script. Check CONDA_INSTALL_PATH."
    exit 1
fi

conda activate ferminet_jelly
if [ $? -ne 0 ]; then
    echo "Error activating conda environment. Make sure it exists."
    exit 1
fi

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo ""

#---------------------------------------------------------------------#
# Execute Code
#---------------------------------------------------------------------#
echo "Running script..."
# NOTE: Ensure this path points to your actual python script
# ferminet --config ferminet/configs/jellium_sphere.py --config.system.n_electrons 2 --config.system.r_s 1
ferminet --config ferminet/configs/jellium_sphere.py \
    --config.system.n_electrons 2 \
    --config.system.r_s 1.0 \
    --config.batch_size 8192 \
    --config.optim.kfac.damping 0.0001 \
    --config.mcmc.steps 30 \
    --config.optim.iterations 150000
# Check the exit code of Python script
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Python script exited with error code $EXIT_CODE"
fi
echo ""

#---------------------------------------------------------------------#
# Cleanup
#---------------------------------------------------------------------#
echo "Job finished with exit code $EXIT_CODE"
date
#---------------------------------------------------------------------#
