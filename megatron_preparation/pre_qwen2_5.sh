#!/bin/bash

# Download models
cd /mnt
modelscope download --model Qwen/Qwen2.5-7B  --local_dir ./qwen-ckpts/Qwen2.5-7B

# set the Megatron pythonpath
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH=/workspace/FT-cutlass/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250908:/workspace/FT-cutlass/Pai-Megatron-Patch:$PYTHONPATH

# convert HF model to Megatron-Core model
cd /workspace/FT-cutlass/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen2_5/run_8xH20.sh \
7B \
/mnt/qwen-ckpts/Qwen2.5-7B \
/mnt/qwen-ckpts/Qwen2.5-7B-to-mcore  \
false \
true \
bf16 \
whatever \
1 \
1 2>&1 | tee /tmp/qwen2.5_convert.log

# make qwen 2.5 sft datasets
cd /workspace/FT-cutlass/Pai-Megatron-Patch/toolkits/sft_data_preprocessing
bash run_build_idxmap_sft_dataset.sh \
/mnt/qwen-datasets/alpaca_data.json \
Qwen2Tokenizer \
1024 \
/mnt/qwen-datasets/mmap_qwen2_sft_datasets_en \
/mnt/qwen-ckpts/Qwen2.5-7B

