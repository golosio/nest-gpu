#!/bin/bash -x
#SBATCH --account=icei-hbp-2020-0007
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
###SBATCH --cpus-per-task=128
#SBATCH --time=00:10:00
#SBATCH --partition=gpus
#SBATCH --output=/p/scratch/icei-hbp-2020-0007/tmp/test_ngpu_out.%j
#SBATCH --error=/p/scratch/icei-hbp-2020-0007/tmp/test_ngpu_err.%j
# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.

###export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun python test_iaf_psc_exp_g.py
