#!/bin/bash
#SBATCH --job-name=speed-r1-zero
#SBATCH --nodes=1       
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=PDS0352


# export WANDB_MODE=offline
export HF_HOME="/fs/scratch/PDS0352/xhan/.cache/huggingface"
export APPTAINER_CACHEDIR="/fs/scratch/PDS0352/xhan/.apptainer"

export SSL_CERT_DIR=/etc/ssl/certs
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

unset GCC_HOME
unset OSC_GCC_DIR
unset CXX
unset OSC_CC
unset CC
unset MPICC
unset MPICXX


cd $HOME/workspace/mini-r1-zero
pwd
nvidia-smi
singularity exec --nv --bind $HOME:$HOME --bind /tmp:/tmp $HOME/sif/sglang.sif bash rl_grpo.sh 