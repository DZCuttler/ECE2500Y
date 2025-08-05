#!/bin/bash
#SBATCH --account=def-romang      # Replace with your PI's allocation/account
#SBATCH --job-name=my_project_job
#SBATCH --time=01:00:00                 # Max runtime (HH:MM:SS)
#SBATCH --mem=8G                        # Memory per node
#SBATCH --cpus-per-task=4              # Number of CPU cores
#SBATCH --mail-user=david.cuttler@mail.utoronto.ca       # (Optional) get email updates
#SBATCH --mail-type=ALL           # (Optional) email on end/fail


# Bash settings for debugging and error handling
set -euo pipefail                           # Exit on error, unset variables, failed pipes
trap 'echo "Error occurred at $BASH_SOURCE:$LINENO. Exit code: $?"' ERR

echo "=== Job started at $(date) ==="
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $PWD"

# Load modules
cd ~/scratch/dcuttler/ECE2500Y || { echo "Failed to cd"; exit 1; }
module purge
module load python/3.10 scipy-stack
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip
# source ~/myEnv/bin/activate

# pip install --no-index -r ~/requirements.txt

echo "Activating virtual environment..."
source ~/myEnv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }


# Run your code
echo "Running the Python script..."
python optimization_models/optimization_model_v3_1/grid_search_EE.py