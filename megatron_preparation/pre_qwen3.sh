#!/bin/bash


# make model and dataset dir
cd /mnt
mkdir qwen-ckpts qwen-datasets

# download the model checkpoint
modelscope download --model Qwen/Qwen3-8B --local_dir ./qwen-ckpts/Qwen3-8B

# set the Megatron pythonpath
export TORCH_DISABLE_ADDR2LINE=1
export PYTHONPATH=/workspace/FT-cutlass/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250908:/workspace/FT-cutlass/Pai-Megatron-Patch:$PYTHONPATH

# convert HF model to Megatron-Core model
cd /workspace/FT-cutlass/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen3/run_8xH20.sh \
8B \
/mnt/qwen-ckpts/Qwen3-8B \
/mnt/qwen-ckpts/Qwen3-8B-to-mcore  \
false \
true \
bf16 \
whatever \
2 \
2 2>&1 | tee /tmp/qwen3_convert.log
 
# download the dataset
cd /mnt/qwen-datasets
# wget --no-check-certificate https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/qwen_sft.json
wget https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json

# make the datasets
cd /workspace/FT-cutlass/Pai-Megatron-Patch/toolkits/sft_data_preprocessing
bash run_build_idxmap_sft_dataset.sh \
/mnt/qwen-datasets/alpaca_data.json \
Qwen3Tokenizer \
1024 \
/mnt/qwen-datasets/mmap_qwen3_sft_datasets_en \
/mnt/qwen-ckpts/Qwen3-8B


# all three models needs 4 GPUs
