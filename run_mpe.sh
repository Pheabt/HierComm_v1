#python main.py --agent ac_mlp --env tj --map easy --use_multiprocessing --n_processes 2 --epoch_size 4 --batch_size 200 &
#python main.py --agent ac_att --env tj --map easy --use_multiprocessing --n_processes 2 --epoch_size 4 --batch_size 200 &
#python main.py --agent tiecomm_random --env tj --map easy --use_multiprocessing --n_processes 2 --epoch_size 4 --batch_size 200 &
#python main.py --agent tiecomm_one --env tj --map easy --use_multiprocessing --n_processes 2 --epoch_size 4 --batch_size 200 &


#python main.py --agent ac_mlp --env mpe --map pz-mpe-large-spread-v1 --use_multiprocessing --n_processes 4 --epoch_size 5 --batch_size 200 &
#python main.py --agent ac_att --env mpe --map pz-mpe-large-spread-v1 --use_multiprocessing --n_processes 4 --epoch_size 5 --batch_size 200 &
#python main.py --agent tiecomm_random --env mpe --map pz-mpe-large-spread-v1 --use_multiprocessing --n_processes 4 --epoch_size 5 --batch_size 200 &
#python main.py --agent tiecomm_one --env mpe --map pz-mpe-large-spread-v1 --use_multiprocessing --n_processes 4 --epoch_size 5 --batch_size 200 &

#lbforaging:Foraging-10x10-3p-3f-v2



#v1 50
#python main.py --agent ac_mlp --env mpe --map mpe-large-spread-v1  --total_epoches 5000 --time_limit 50   --use_multiprocessing --memo env_test &
#python main.py --agent ac_att --env mpe --map mpe-large-spread-v1  --total_epoches 5000 --time_limit 50   --use_multiprocessing --memo env_test &
#python main.py --agent magic   --env mpe --map mpe-large-spread-v1   --total_epoches 1000 --time_limit 50   --use_multiprocessing --memo env_test &
#python main.py --agent tiecomm          --env mpe --map mpe-large-spread-v1  --total_epoches 1000 --time_limit 50   --use_multiprocessing --memo ming &
#python main.py --agent tiecomm_default  --env mpe --map mpe-large-spread-v1  --total_epoches 1000 --time_limit 50   --use_multiprocessing --memo ming &
python main.py --agent tiecomm_wo_inter --env mpe --map mpe-large-spread-v1  --total_epoches 1000 --time_limit 50   --use_multiprocessing --memo ming &
python main.py --agent tiecomm_wo_intra --env mpe --map mpe-large-spread-v1  --total_epoches 1000 --time_limit 50   --use_multiprocessing --memo ming &


#python main.py --agent tiecomm   --env mpe --map mpe-large-spread-v1 --time_limit 50   --use_multiprocessing
#
#python main.py --agent tiecomm   --env mpe --map mpe-large-spread-v2 --time_limit 100   --use_multiprocessing
#python main.py --agent tiecomm   --env mpe --map mpe-large-spread-v2 --time_limit 100   --use_multiprocessing