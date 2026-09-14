#!/bin/bash   
#SBATCH --mem=128G         
#SBATCH --cpus-per-task=12
#SBATCH --time=72:00:00
#SBATCH --mail-user=ergunwhy1@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=ppln
#SBATCH --output=logs/stg3-4.out
#SBATCH --error=logs/stg3-4.err

# Unsets the CPU binding policy.
# Some clusters automatically bind threads to cores; unsetting it can 
# prevent performance issues if code manages threading itself 
# (e.g. OpenMP, NumPy, or PyTorch).
unset SLURM_CPU_BIND

# Ensures all environment variables from the submission 
# environment are passed into the job’s environment
export SLURM_EXPORT_ENV=ALL

# Loads the Anaconda module provided by the cluster.
# (On HPC systems, software is usually installed as “modules” to avoid version conflicts.)
module load Anaconda3/2024.02-1
module load Python/3.10.8-GCCcore-12.2.0 # essential to load latest GCC

# Initialise Conda for this non-interactive shell
eval "$(conda shell.bash hook)"

# Activates Conda environment named ibeat-totspineseg.
# (Older clusters use source activate; newer Conda versions use conda activate venv.)
# Assumes conda environment 'ibeat-totspineseg' has already been created
conda activate ibeat-totspineseg

# Get current username
USERNAME=$(whoami)

# Define path variables
ENV="/mnt/parscratch/users/$USERNAME/.conda/.envs/ibeat-totspineseg"
CODE="/mnt/parscratch/users/$USERNAME/iBEAt/code/ppln-ibeat-totspineseg/src/ibeat_totspineseg"
BUILD="/mnt/parscratch/users/$USERNAME/iBEAt/build/totspineseg"

# Run python scripts for stages 3-4 on the allocated compute resources managed by Slurm
srun "$ENV/bin/python" "$CODE/stage_03_display.py" --build="$BUILD"
srun "$ENV/bin/python" "$CODE/stage_04_measure.py" --build="$BUILD"