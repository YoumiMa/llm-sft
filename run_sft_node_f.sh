#! /bin/sh
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=20:00:00

HUGGINGFACE_CACHE=/gs/bs/tga-okazaki/ma/cache

export HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_CACHE
export HF_HOME=$HUGGINGFACE_CACHE

#export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME=br01 # for me it is 'br0' interface, you should use yours :)
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate llm-jp-sft

LR=2.5e-5
MINLR=2.5e-6
WD=0.1
NAME=$1; shift
SEED=$1; shift
DATA=("$@")
NAME=llama-3.1-${NAME}_LR_${LR}_MINLR_${MINLR}_WD_${WD}

accelerate launch --config_file configs/my_accelerate_config_zero1.yaml scripts/train_llm_llama3.py --output_dir /gs/bs/tga-okazaki/ma/ckpts/${NAME}_${SEED} \
--run_name $NAME \
--data_files ${DATA[*]} \
--model_name_or_path meta-llama/Llama-3.1-8B \
--tokenizer_name_or_path meta-llama/Llama-3.1-8B-Instruct \
--bf16 \
--num_train_epochs 2 \
--per_device_train_batch 2 \
--gradient_accumulation_steps 64 \
--gradient_checkpointing \
--optim adamw_torch \
--adam_beta2 0.95 \
--learning_rate ${LR} \
--lr_scheduler_type cosine_with_min_lr \
--lr_scheduler_kwargs '{"min_lr":'${MINLR}'}' \
--weight_decay ${WD} \
--warmup_ratio 0.1 \
--logging_steps 10 \
--save_steps 500 \
--seed ${SEED} \
