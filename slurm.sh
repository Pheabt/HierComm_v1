#!/bin/bash
#SBATCH --job-name=procgen01
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=cpu_batch 
#SBATCH --nodes=2
#SBATCH --cpus-per-task=6
#SBATCH -n 48


source /home/iotsc_g4/app/miniconda3/bin/activate tie
cd /home/iotsc_g4/aaa/HierComm
sh run_mpe.sh

