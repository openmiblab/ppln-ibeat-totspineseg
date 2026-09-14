#!/bin/bash   
#SBATCH --mem=128G         
#SBATCH --cpus-per-task=8
#SBATCH --time=95:00:00
#SBATCH --mail-user=ergunwhy1@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=ppln
#SBATCH --output=logs/ppln.out
#SBATCH --error=logs/ppln.err

# Set the number of threads environment variables to use cpus-per-task
# This is needed to ensure efficient core usage.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK # OpenMP
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK # Intel's Math Kernel Library (MKL)
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK # OpenBLAS

# Ensures all environment variables from submission 
# environment are passed into job’s environment
export SLURM_EXPORT_ENV=ALL

# Loads the Anaconda module provided by the cluster.
#module load Anaconda3/2024.02-1 # doesn't work on GPU
module load Anaconda3
module load Python/3.10.8-GCCcore-12.2.0 # essential to load latest GCC
module load CUDA/12.4.0 # must match with version in env.yml

# Initialise Conda for this non-interactive shell
eval "$(conda shell.bash hook)"

# Activates Conda environment named ibeat-totspineseg.
# (Older clusters use source activate; newer Conda versions use conda activate venv.)
# Assumes conda environment 'ibeat-totspineseg' has already been created
conda activate ibeat-totspineseg

# Force Slurm to tell PyTorch which card belongs to this task
export CUDA_VISIBLE_DEVICES=0

# Get current username
USERNAME=$(whoami)

# Define path variables
ENV="/mnt/parscratch/users/$USERNAME/.conda/.envs/ibeat-totspineseg"
CODE="/mnt/parscratch/users/$USERNAME/iBEAt/code/ppln-ibeat-totspineseg/src/ibeat_totspineseg"
BUILD="/mnt/parscratch/users/$USERNAME/iBEAt/build/totspineseg"

# Run python scripts for all stages 1-4 on the allocated compute resources managed by Slurm
srun "$ENV/bin/python" "$CODE/stage_01_auto_segment.py" --build="$BUILD"
srun "$ENV/bin/python" "$CODE/stage_03_display.py" --build="$BUILD"
srun "$ENV/bin/python" "$CODE/stage_04_measure.py" --build="$BUILD"
