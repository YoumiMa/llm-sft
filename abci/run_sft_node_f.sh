#! /bin/sh
#PBS -P gag51395  
#PBS -q rt_HF
#PBS -l select=1  -k oe
#PBS -l walltime=07:00:00

module load cuda/12.8/12.8.1

source miniconda3/bin/activate
conda activate llm-jp-sft


LR=2.5e-5
MINLR=2.5e-6
WD=0.1
NAME=test
SEED=42
DATA="/groups/gcb50243/ma/data/lmsys-chat-1m/lmsys-chat-1m-synth-en.jsonl.gz"
NAME=llama-3.1-${NAME}_LR_${LR}_MINLR_${MINLR}_WD_${WD}

cd llm-sft

accelerate launch --config_file abci_configs/my_accelerate_config_zero1.yaml scripts/train_llm_llama3.py --output_dir /groups/gcb50243/ma/ckpts/$NAME \
--run_name $NAME \
--data_files $DATA  \
--model_name_or_path meta-llama/Llama-3.1-8B \
--tokenizer_name_or_path meta-llama/Llama-3.1-8B-Instruct \
--bf16 \
--num_train_epochs 2 \
--per_device_train_batch 4 \
--gradient_accumulation_steps 16 \
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
--seed ${SEED} > out.log
