source ~/llm-sft/.dpo/bin/activate

HUGGINGFACE_CACHE=/gs/bs/tga-okazaki/ma/cache

export HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_CACHE
export HF_HOME=$HUGGINGFACE_CACHE

#export NCCL_IB_DISABLE=1
#export NCCL_SOCKET_IFNAME=br01 # for me it is 'br0' interface, you should use yours :)
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

export WANDB_PROJECT=dpo

LR=5e-7
MINLR=5e-8
WD=0.1
MAIN_PROCESS_IP=$1; shift
NAME=$1; shift
SEED=$1; shift
DATA=("$@")
NAME=Llama-3.1-70B-${NAME}_LR_${LR}_MINLR_${MINLR}_WD_${WD}

# machine_rankを保存してからunset
MACHINE_RANK=${OMPI_COMM_WORLD_RANK}

# OpenMPI環境変数をクリア（DeepSpeedとの衝突を防ぐ）
unset OMPI_COMM_WORLD_LOCAL_RANK
unset OMPI_COMM_WORLD_RANK
unset OMPI_COMM_WORLD_SIZE

echo "Rank: ${MACHINE_RANK}"
echo "task name: ${NAME}"
echo "Main Process: ${MAIN_PROCESS_IP}"
echo "data: ${DATA}"

accelerate launch --config_file configs/my_accelerate_config_zero3.16nodes.yaml \
        --main_process_ip $MAIN_PROCESS_IP \
         --main_process_port 29500 \
         --machine_rank $MACHINE_RANK \
         dpo/dpo_llm_llama3_70b.py --output_dir /gs/bs/tga-ma-act-x/ckpts/${NAME}_${SEED} \
--run_name $NAME \
--data_files ${DATA[*]} \
--model_name_or_path /gs/bs/tga-ma-act-x/ckpts/${NAME}_${SEED}/checkpoint-500 \
--tokenizer_name_or_path /gs/bs/tga-ma-act-x/ckpts/${NAME}_${SEED}/checkpoint-500 \
--resume_from_checkpoint /gs/bs/tga-ma-act-x/ckpts/${NAME}_${SEED}/checkpoint-500 \
--bf16 true \
--num_train_epochs 2 \
--per_device_train_batch 1 \
--gradient_accumulation_steps 8 \
--gradient_checkpointing \
--optim adamw_torch \
--adam_beta2 0.95 \
--learning_rate ${LR} \
--lr_scheduler_type cosine_with_min_lr \
--lr_scheduler_kwargs "{\"min_lr\":${MINLR}}" \
--weight_decay ${WD} \
--warmup_ratio 0.1 \
--logging_steps 10 \
--save_steps 400 \
--seed ${SEED} \
--report_to wandb \
--project dpo
