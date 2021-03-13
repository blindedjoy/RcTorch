#!/bin/bash

#
#SBATCH --threads-per-core=8
#SBATCH -n 30		    # Number of cores requested SBATCH --cores-per-socket=15
#SBATCH -N 1 				# Ensure that all cores are on one machine		    # Runtime in minutes
#SBATCH -o output_%j.out 	# Standard out goes to this file
#SBATCH -e error_%j.err 	# Standard err goes to this file

module load Anaconda3/2019.10;
#--cores-per-socket=15	        Number of cores in a socket to dedicate to a job (minimum)
#--threads-per-core=cv_samples	Number of threads in a core to dedicate to a job (minimum)

# n_cores = 30 ==> 15 tasks ; n_experiments = 2 ==> 30

# Labtop uses 8 gb per core, try to go way above that. Let's say, 20 gb per core. 20*8 = 160gb, let's go for 200gb.

echo "Running bayesRC on 20 CPU cores"

# n-tasks per node: n_cores * cv_samples
# -n or num_cores: len(experiment_set) * n-tasks

# 16 tests, 8 cores each. Then we have the cv loop, requesting four cores per run.
# 16 * 8

#install the customized version of Reinier's reservoir package
cd ..; ./reinstall.sh; cd MARIOS; 
#chmod a+x ./reinstall.sh
# 
# ##### asfSBATCH	--cpus-per-task=8

chmod a+x ./build_filesystem.sh
./build_filesystem.sh

python execute.py $1

echo "$1"

#python PyFiles/test.py asdfknl
#asdlfk;SBA -p serial_requeue #SBATCH -p serial_requeue
# asfdsfasjk;nATCH --ntasks-per-node=6 # faSasdfH --ntasks-per-node=15 v#SBACH --mem=200gb 			# Memory in GB (see also --mem-per-cpu)
