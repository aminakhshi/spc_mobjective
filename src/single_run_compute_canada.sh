#!/bin/bash

# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=.....       	# Account name
#SBATCH --job-name=MHR_optimization  	# Job name
#SBATCH --output=log_MHR_optimization-%j.txt    	# Standard output and error log
#SBATCH --ntasks=256                	# Number of MPI ranks (tasks)
#SBATCH --cpus-per-task=1           	# Number of cores per MPI rank
#SBATCH --nodes=4                   	# Number of nodes
#SBATCH --time=3-00:00:00           	# Time limit (3 days)
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------

module load python/3.10.12
mkdir -p .venv
export SLURM_TMPDIR=.venv
virtualenv --no-download $SLURM_TMPDIR
source $SLURM_TMPDIR/bin/activate
pip install --no-index --upgrade pip
pip install --no-index h5py numpy scipy matplotlib seaborn pandas joblib
pip install --no-index sklearn
pip install --no-index tqdm
pip install --no-index numba

echo "Current working directory: `pwd`"
python -u parallel_optimization.py > report_run.out


# --- Output and Cleanup ---
echo "Compressing results and output reports..."
tar -czf results_and_reports.tar.gz result report_run.out FI_curve-${SLURM_JOB_ID}.txt

echo "Cleaning up unnecessary files..."
rm -rf .venv result report_run.out FI_curve-${SLURM_JOB_ID}.txt

exit_code=$?
echo "Job finished with exit code $exit_code at: $(date)"

