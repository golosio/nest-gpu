for i in $(seq 100 199); do
    sbatch run_sbatch_debug.sh 1234$i
done
