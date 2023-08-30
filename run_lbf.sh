#python main.py --agent ac_mlp    --env lbf --map Foraging-easy-v0   --time_limit 40  --total_epoches 200 --use_multiprocessing --memo ming &
#python main.py --agent ac_att    --env lbf --map Foraging-easy-v0   --time_limit 40  --total_epoches 200 --use_multiprocessing --memo ming &

#python main.py --agent tarmac              --env lbf --map Foraging-easy-v0   --time_limit 40  --seed 123 --total_epoches 100 --use_multiprocessing --memo final_123 &
#python main.py --agent magic               --env lbf --map Foraging-easy-v0   --time_limit 40  --seed 123 --total_epoches 100 --use_multiprocessing --memo final_123 &
#python main.py --agent commnet             --env lbf --map Foraging-easy-v0   --time_limit 40  --seed 123 --total_epoches 100 --use_multiprocessing --memo final_123 &
#python main.py --agent ic3net              --env lbf --map Foraging-easy-v0   --time_limit 40  --seed 123 --total_epoches 100 --use_multiprocessing --memo final_123 &
#python main.py --agent tiecomm             --env lbf --map Foraging-easy-v0   --time_limit 40  --seed 123 --total_epoches 100 --use_multiprocessing --memo final_123 &
#python main.py --agent tiecomm_wo_inter    --env lbf --map Foraging-easy-v0   --time_limit 40  --seed 123 --total_epoches 100 --use_multiprocessing --memo final_123 &
#python main.py --agent tiecomm_wo_intra    --env lbf --map Foraging-easy-v0   --time_limit 40  --seed 123 --total_epoches 100 --use_multiprocessing --memo final_123 &
#python main.py --agent tiecomm_default     --env lbf --map Foraging-easy-v0   --time_limit 40  --seed 123 --total_epoches 100 --use_multiprocessing --memo final_123 &


python main.py --agent ac_mlp   --env lbf --map Foraging-medium-v0  --time_limit 60  --total_epoches 500 --use_multiprocessing --memo pre &
python main.py --agent ac_att   --env lbf --map Foraging-medium-v0  --time_limit 60  --total_epoches 500 --use_multiprocessing --memo pre &
python main.py --agent ac_mlp   --env lbf --map Foraging-hard-v0  --time_limit 80 --total_epoches 500 --use_multiprocessing --memo pre &
python main.py --agent ac_att   --env lbf --map Foraging-hard-v0  --time_limit 80 --total_epoches 500 --use_multiprocessing --memo pre &

#python main.py --agent tarmac --total_epoches 500 --env lbf --map Foraging-easy-v0 --block no --use_multiprocessing --memo aaai &