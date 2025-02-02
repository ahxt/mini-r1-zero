#!/bin/sh
set -e -x
nvidia-smi
pip list

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


python3 grpo.py \
    --dataset_name openai/gsm8k \
    --dataset_train_split='train' \
    --dataset_test_split='test' \
    --learning_rate 3e-6 \
    --output_dir /fs/scratch/PDS0352/xhan/mini-r1-zero \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --bf16 \
    --num_generations 18 \
    --logging_steps 1 \
    --report_to wandb \
    --run_name grpo-debug \
    --torch_compile \
    --max_completion_length 384 \
    --num_train_epochs 1 \
    --torch_dtype bfloat16 \
    --attn_implementation sdpa \
    --save_steps 25 \
    --use_vllm \
    --vllm_device cuda:0 \
    --vllm_gpu_memory_utilization 0.5


# meta-llama/Llama-3.2-1B
# meta-llama/Llama-3.2-1B-Instruct