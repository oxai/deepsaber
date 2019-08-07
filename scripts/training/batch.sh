#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH -J bs_likelihood

#small for 1 gpu, big for 4 or 8
#SBATCH --partition=big

# set number of GPUs
#SBATCH --gres=gpu:8

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=guillefix@gmail.com

#Launching the commands within script.sh

/jmain01/apps/docker/tensorflow-batch -v 18.07-py3 -c './script'
