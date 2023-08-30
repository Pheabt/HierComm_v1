#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output.log
#SBATCH --error=error.log

#SBATCH --partition=cpu_partition
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=6

source /home/iotsc_g4/app/miniconda3/bin/activate tie


cd /home/iotsc_g4/aaa/NewHierComm


srun  python main.py --agent ac_att  --att_head 1  --seed 6666 &
srun  python main.py --agent ac_att  --att_head 2  --seed 6666 &
srun  python main.py --agent ac_att  --att_head 3  --seed 6666 &
srun  python main.py --agent ac_att  --att_head 4  --seed 6666 &

wait
