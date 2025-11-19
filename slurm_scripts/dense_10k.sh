#!/bin/bash

#SBATCH -J dense-maxflow-10k			# Job name
#SBATCH -o slurm-out/dense-10k.o%j	# Name of stdout output file
#SBATCH -e slurm-out/dense-10k.e%j	# Name of stderr error file
#SBATCH -p gpu-a100				# Queue (partition) name
#SBATCH -N 1               			# Total # of nodes 
#SBATCH -t 06:00:00        			# Run time (hh:mm:ss)
#SBATCH --mail-type=all    			# Send email at begin and end of job
#SBATCH --mail-user=andrewnguyen@utexas.edu

# Any other commands must follow all #SBATCH directives...
cd /work/10613/andrewnguyen/ls6/palgos_proj

source venv/bin/activate

date
echo "------"

python3 run_cuda_10k.py

