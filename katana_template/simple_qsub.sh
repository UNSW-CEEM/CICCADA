#!/bin/bash
#PBS -l select=1:ngpus=0:ncpus=01:mem=1gb
#PBS -l walltime=00:10:00

#PBS -j oe 
#PBS -o ./Log/Main.log
module load tensorflow/2.11.0
module load python/3.10.8
python3 ./file_to_be_exec.py
exit
