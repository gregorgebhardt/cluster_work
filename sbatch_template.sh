#!/bin/bash
#SBATCH -A {project_name} # project00554
#SBATCH -J {experiment_name}
#SBATCH -D {experiment_cwd} # /home/yy05vipo/git/pyKKF/experiments
#SBATCH --mail-type=END
# Please use the complete path details :
#SBATCH -e {err_file} # /home/yy05vipo/git/pyKKF/experiments/pendulum.err.%j
#SBATCH -o {out_file} # /home/yy05vipo/git/pyKKF/experiments/pendulum.out.%j
#
#SBATCH -n {number_of_jobs}         # Number of tasks
#SBATCH -c {number_of_cpu_per_job}  # Number of cores per task
#SBATCH --mem-per-cpu={mem}  # Main memory in MByte per MPI task
#SBATCH -t {time_limit}     # 1:00:00 Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes

# -------------------------------
# Afterwards you write your own commands, e.g. 
module load gcc openmpi/gcc
source activate kkf
cd {experiment_cwd} #/home/yy05vipo/git/pyKKF/experiments
srun hostname > hostfile.$SLURM_JOB_ID
hostfileconv hostfile.$SLURM_JOB_ID
job_stream --hostfile hostfile.$SLURM_JOB_ID.converted -- python {python_script_name} -c {yaml_config} -d -v
