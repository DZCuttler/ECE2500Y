#!/bin/bash
#SBATCH --account=dcuttler      # Replace with your PI's allocation/account
#SBATCH --job-name=my_project_job
#SBATCH --output=logs/%x-%j.out         # Save stdout to logs/jobname-jobid.out
#SBATCH --time=01:00:00                 # Max runtime (HH:MM:SS)
#SBATCH --mem=4G                        # Memory per node
#SBATCH --cpus-per-task=2              # Number of CPU cores
#SBATCH --mail-user=david.cuttler@mail.utoronto.ca       # (Optional) get email updates
# SBATCH --mail-type=ALL           # (Optional) email on end/fail

# Load modules
module load python/3.10  # Or whatever version you need
source ~/myenv/bin/activate  # Activate your conda/venv if needed

# Run your code
python optimization_models/optimization_model_v3/grid_search_EE.py