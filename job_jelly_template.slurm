#!/bin/bash
#---------------------------------------------------------------------#
# SLURM Options
#
# Parametrised version of job_nano.slurm. The number of electrons and the
# density r_s are read from the environment (N_ELECTRONS / R_S), which the
# submit script (submit_sweep.sh) passes in via `sbatch --export`. The
# --job-name / --output / --error below are sensible defaults and are
# overridden per-combination by submit_sweep.sh.
#---------------------------------------------------------------------#
#SBATCH --account=MST114233
#SBATCH --job-name=jelly                          # Job name (overridden by submit_sweep.sh)
#SBATCH --partition=8gpus                         # Partition
#SBATCH --time=48:00:00                          # Time limit (HH:MM:SS)
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --cpus-per-task=2                       # CPUs per task
#SBATCH --mem=32G                               # Memory per node
#SBATCH --gres=gpu:2                       # Request 4 GPUs
#SBATCH --output=slurm_logs/job%j.out           # Log file (overridden by submit_sweep.sh)
#SBATCH --error=slurm_logs/job%j.err            # Error file (overridden by submit_sweep.sh)
#---------------------------------------------------------------------#
# Job Environment Setup
#---------------------------------------------------------------------#
# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Parameters of the sweep. Defaults match the paper's simplest system so the
# template still runs if launched directly with `sbatch job_jelly_template.slurm`.
N_ELECTRONS="${N_ELECTRONS:-2}"
R_S="${R_S:-4}"

echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running from script: $(realpath $0)"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Sweep parameters: n_electrons=${N_ELECTRONS}  r_s=${R_S}"
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
ferminet --config ferminet/configs/jellium_sphere.py \
    --config.system.n_electrons "${N_ELECTRONS}" \
    --config.system.r_s "${R_S}"

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
