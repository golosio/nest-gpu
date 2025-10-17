echo "To run this test you must search the keyword"
echo "test_connect_distributed_fixed_indegree in the file"
echo "src/remote_connect.h, uncomment the next line, rebuild"
echo "and reinstall NEST GPU"
echo "Remember to comment it again, rebuild and reinstall NEST GPU after the test!"
echo
:> tmp1.dat
:> tmp2.dat
N=100
echo "Progress:"
for seed in $(seq 1 $N); do
    echo "$seed / $N"
    (mpirun -np 7 python test_connect_distributed_fixed_indegree.py $seed | grep 'TDFID 2,' | awk '{print $NF}' | while read a; do echo -n "$a "; done; echo "0 0 0") | awk '{print $1, $2, $3}' >> tmp1.dat
    (mpirun -np 7 python test_connect_distributed_fixed_indegree.py $seed | grep 'TDFID 6,' | awk '{print $NF}' | while read a; do echo -n "$a "; done; echo "0 0 0") | awk '{print $1, $2, $3}' >> tmp2.dat
done
python analyze_connect_distributed_fixed_indegree.py
