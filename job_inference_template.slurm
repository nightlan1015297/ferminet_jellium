#!/bin/bash
#---------------------------------------------------------------------#
# SLURM Options
#
# Inference counterpart of job_jelly_template.slurm. Runs a fixed-parameter
# VMC inference calculation (frozen network, accumulate <H>) on a system that
# has ALREADY been trained -- it restores the trained parameters from
# results/<run-name>/ and writes its own outputs to
# results/<run-name>_inference_b<batch>_s<steps>/.
#
# The system (N_ELECTRONS / R_S) and the run length (STEPS / BATCH_SIZE) are
# read from the environment; pass them via `sbatch --export`, e.g.
#
#   sbatch --export=ALL,N_ELECTRONS=18,R_S=1 job_inference_template.slurm
#
# Defaults below run the requested 2**18 steps x 4096 walkers on N=2, r_s=4.
#---------------------------------------------------------------------#
#SBATCH --account=MST114233
#SBATCH --job-name=jelly_infer                   # Job name (override per run)
#SBATCH --partition=8gpus                         # Partition
#SBATCH --time=48:00:00                          # Time limit (HH:MM:SS)
#SBATCH --nodes=1                               # Single node (walkers staged per host)
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --cpus-per-task=2                       # CPUs per task
#SBATCH --mem=32G                               # Memory per node
#SBATCH --gres=gpu:6                       # Request 6 GPUs
#SBATCH --output=slurm_logs/infer%j.out         # Log file (override per run)
#SBATCH --error=slurm_logs/infer%j.err          # Error file (override per run)
#---------------------------------------------------------------------#
# Job Environment Setup
#---------------------------------------------------------------------#
mkdir -p slurm_logs

# Physical system (must match a previously TRAINED run so its checkpoint exists)
# and the inference run length.
N_ELECTRONS="${N_ELECTRONS:-2}"
R_S="${R_S:-4}"
STEPS="${STEPS:-262144}"        # 2**18 energy-evaluation iterations
BATCH_SIZE="${BATCH_SIZE:-4096}" # walkers (samples) per iteration

echo "Job started on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running from script: $(realpath "$0")"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Inference parameters: n_electrons=${N_ELECTRONS} r_s=${R_S} steps=${STEPS} batch_size=${BATCH_SIZE}"
echo "nvidia-smi output:"
nvidia-smi
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
echo "Running inference..."
# Pass the SAME system flags used for training so the trained checkpoint is
# found; --steps / --batch_size control the inference run (not --config.*).
python ferminet/inference.py --config ferminet/configs/jellium_sphere.py \
    --config.system.n_electrons "${N_ELECTRONS}" \
    --config.system.r_s "${R_S}" \
    --steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}"

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
