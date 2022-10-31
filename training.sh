#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=luc.lerch@unibe.ch

# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=fail,end
#SBATCH --job-name="cifar acgan"
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=gpu
#SBATCH --qos=job_gpu_preempt
#SBATCH --gres=gpu:teslap100:1
#SBATCH --array=1-4

python cifar_acgan.py 