#!/bin/bash

#SBATCH -J cuda-maxflow			# Job name
#SBATCH -o slurm-out/maxflow.o%j	# Name of stdout output file
#SBATCH -e slurm-out/maxflow.e%j	# Name of stderr error file
#SBATCH -p gpu-a100			# Queue (partition) name
#SBATCH -N 1               		# Total # of nodes 
#SBATCH -t 12:00:00        		# Run time (hh:mm:ss)
#SBATCH --mail-type=all    		# Send email at begin and end of job
#SBATCH --mail-user=andrewnguyen@utexas.edu

# Any other commands must follow all #SBATCH directives...
cd /work/10613/andrewnguyen/ls6/palgos_proj

source venv/bin/activate

pwd
date
module list
pip list
echo "------"

python3 run_cuda.py
