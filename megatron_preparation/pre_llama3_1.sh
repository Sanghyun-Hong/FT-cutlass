#!/bin/bash

# Download models
cd /mnt
mkdir llama3-ckpts
cd llama3-ckpts
modelscope download --model LLM-Research/Meta-Llama-3.1-8B --local_dir ./llama3-ckpts/Meta-Llama-3.1-8B

# set the Megatron pythonpath
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH=/workspace/FT-cutlass/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250908:/workspace/FT-cutlass/Pai-Megatron-Patch:$PYTHONPATH

# convert HF model to Megatron-Core model
cd /workspace/FT-cutlass/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
bash hf2mcore_convertor_llama3_1.sh \
8B \
/mnt/llama3-ckpts/Meta-Llama-3.1-8B    \
/mnt/llama3-ckpts/Meta-Llama-3.1-8B-mcore-tp2-pp2  \
2  \
2  \
false \
true \
false \
bf16 \
1 \
1 2>&1 | tee /tmp/llama3_convert.log

# make qwen 2.5 sft datasets
cd /mnt
mkdir llama3-datasets

cd /workspace/FT-cutlass/Pai-Megatron-Patch/toolkits/sft_data_preprocessing
bash run_build_idxmap_sft_dataset.sh \
/mnt/qwen-datasets/alpaca_data.json \
LLama3Tokenizer \
1024 \
/mnt/llama3-datasets/mmap_llama3_sft_datasets_en \
/mnt/llama3-ckpts/Meta-Llama-3.1-8B  

