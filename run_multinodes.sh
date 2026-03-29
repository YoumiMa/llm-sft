#! /bin/sh
#$ -cwd
#$ -l node_f=2
#$ -l h_rt=01:20:00

module load openmpi/5.0.7-gcc

cat $PE_HOSTFILE > ./hostfile
cat $PE_HOSTFILE
echo $NHOSTS
export MASTER=$(cat $PE_HOSTFILE | head -1 | cut -d ' ' -f 1)


TRAIN_SHELL=$1; shift
TASK_NAME=$1; shift
SEED=$1; shift
DATA_DIR="$@"

eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate llm-jp-sft

echo $DATA_DIR
APP="${TRAIN_SHELL} ${MASTER} ${TASK_NAME} ${SEED} ${DATA_DIR}"
MPIOPTS="--npernode 1 -n ${NHOSTS}"

echo $APP
mpirun $MPIOPTS bash $APP
# mpirun $MPIOPTS python ./parallel.py
