#!/bin/bash -x
#SBATCH --account=INF25_brainsta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:1
#SBATCH --output=/leonardo_scratch/large/userexternal/bgolosio/test_ngpu_test_all_out.%j
#SBATCH --error=/leonardo_scratch/large/userexternal/bgolosio/test_ngpu_test_all_err.%j
# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun /usr/bin/sh test_all.sh
