#!/bin/bash
#SBATCH -A %%project_name%%
#SBATCH -J %%experiment_name%%
#SBATCH -D %%experiment_cwd%%
#SBATCH --mail-type=END
# Please use the complete path details :
#SBATCH -e %%err_file%%
#SBATCH -o %%out_file%%
#
#SBATCH -n %%number_of_jobs%%         # Number of tasks
#SBATCH -c %%number_of_cpu_per_job%%  # Number of cores per task
#SBATCH --mem-per-cpu=%%mem%%         # Main memory in MByte per MPI task
#SBATCH -t %%time_limit%%             # 1:00:00 Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes

# -------------------------------

# Load the required modules
module load gcc openmpi/gcc

# Activate the virtualenv / conda environment
source activate your_env

# cd into the working directory
cd %%experiment_cwd%%

srun hostname > hostfile.$SLURM_JOB_ID

mpiexec -map-by node -hostfile $SLURM_JOB_ID.hostfile --mca mpi_warn_on_fork 0 --display-allocation --display-map python -m mpi4py %%python_script_name%% -c %%yaml_config%% -d -v
