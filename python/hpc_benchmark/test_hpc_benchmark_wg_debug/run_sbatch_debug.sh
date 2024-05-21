#!/bin/bash -x
#SBATCH --account=INF24_brainsta_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
####SBATCH --gres=gpu:4
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --output=/leonardo_scratch/large/userexternal/bgolosio/debug/test_ngpu_hpcb_wg_out.%j
#SBATCH --error=/leonardo_scratch/large/userexternal/bgolosio/debug/test_ngpu_hpcb_wg_err.%j
# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.

###export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun python3 hpc_benchmark_wg_debug.py --seed $1
