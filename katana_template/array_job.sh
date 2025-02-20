#!/bin/bash
#PBS -l select=1:ngpus=0:ncpus=8:mem=8gb
#PBS -l walltime=01:59:00
#PBS -j oe
#PBS -o ./Log/array_job.log
#PBS -J 1-10 # this isthe range of array jobs, here you ask to run job 1 to job 10
item=$(sed -n "${PBS_ARRAY_INDEX} p" "corresponding_data_file_list.txt") # This line and the following one are useful when you want to pass on a specific file to each array job
read -r data_file_j <<<"${item}" #  
module load tensorflow/2.11.0
module load python/3.10.8
python3 file_to_be_executed.py "${data_file_j}" 8 5 123 # passing some arguments to the python file 
exit
