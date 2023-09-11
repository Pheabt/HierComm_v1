#!/bin/bash
#SBATCH --job-name=procgen01
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=cpu_batch 

source /home/iotsc_g4/app/miniconda3/bin/activate tie
cd /home/iotsc_g4/aaa/HierComm

srun python main.py --agent  ac_mlp             --use_multiprocessing &
srun python main.py --agent  ac_att             --use_multiprocessing &
srun python main.py --agent  tiecomm            --use_multiprocessing &
srun python main.py --agent  magic              --use_multiprocessing &
srun python main.py --agent  commnet            --use_multiprocessing &
srun python main.py --agent  ic3net             --use_multiprocessing &
srun python main.py --agent  tarmac             --use_multiprocessing &
srun python main.py --agent  hiercomm_random    --use_multiprocessing &

wait 