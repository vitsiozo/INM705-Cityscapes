#!/bin/bash
#SBATCH --job-name work                      # Job name
#SBATCH --partition=preemptgpu                         # Select the correct partition.
#SBATCH --nodes=1                                # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=1                        # Run one task
#SBATCH --cpus-per-task=4                          # Use 4 cores, most of the procesing happens on the GPU
#SBATCH --mem=72GB                                 # Expected ammount CPU RAM needed (Not GPU Memory)
#SBATCH --time=72:00:00                            # Expected ammount of time to run Time limit hrs:min:sec
#SBATCH --gres=gpu:a100:1                               # Use one gpu.
#SBATCH -e results/%x_%j.e                         # Standard output and error log [%j is replaced with the jobid]
#SBATCH -o results/%x_%j.o                         # [%x with the job name], make sure 'results' folder exists.
#SBATCH --error work_preempt.err
#SBATCH --output work_preempt.out
#SBATCH --constraint=cuda12

source /opt/flight/etc/setup.sh
flight env activate gridware
module load libs/nvidia-cuda/11.2.0/bin
module load gnu
export WANDB_API_KEY=9692ff12f6990a08e1a75d22ddd651d0f3de3e95
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export TORCH_HOME=/mnt/data/public/torch
export CUDA_LAUNCH_BLOCKING=1

python work.py "$@" --model Swin2Base
python work.py "$@" --model UNet
