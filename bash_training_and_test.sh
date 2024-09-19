#!/bin/bash


#training and test CL...
echo " Running Continual Training EXP"
echo $(pwd)


python train_CL.py  --model b2p --buffer_size 500 --debug_mode 0 --replayed_rc 1 & P1=$!
wait $P1
python test_CL.py --model b2p --buffer_size 500 --num_tasks 8 --store_traj True & P2=$!
wait $P2
echo "Ours Finished"

python train_CL.py --model gem --buffer_size 500 --debug_mode 0 --replayed_rc 1 & P3=$!
wait $P3
python test_CL.py --model gem --buffer_size 500 --num_tasks 8 --store_traj True & P4=$!
wait $P4
echo "GEM Finished"


python train_CL.py --model gss --buffer_size 500 --debug_mode 0 --replayed_rc 1 & P5=$
wait $P5
python test_CL.py --model gss --buffer_size 500 --num_tasks 8 --store_traj True & P6=$!
wait $P6
echo "GSS Finished"

python train_CL.py  --model der --buffer_size 500 --debug_mode 0 --replayed_rc 1 & P7=$!
wait $P7
python test_CL.py --model der --buffer_size 500 --num_tasks 8 --store_traj True & P8=$!
wait $P8
echo "DER Finished"

python train_CL.py --model agem --buffer_size 500 --debug_mode 0 --replayed_rc 1 & P9=$!
wait $P9
python test_CL.py --model agem --buffer_size 500 --num_tasks 8 --store_traj True & P10=$!
wait $P10
echo "A-GEM Finished"

python train_CL.py --model vanilla --buffer_size 0 --debug_mode 0 --replayed_rc 0 & P11=$!
wait $P11
python test_CL.py --model vanilla --buffer_size 500 --num_tasks 8 --store_traj True & P12=$!
wait $P12
echo "Vanilla Finished"


echo "-----------All scripts are executed.The index is No.1"
 
