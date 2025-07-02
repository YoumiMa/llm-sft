# LLM-jp SFT (Supervised Fine-Tuning)

This repository contains the code for supervised fine-tuning of LLM-jp models.

## Requirements

- Python: 3.10.12
- [torch](https://pytorch.org/)==2.1.2 (should meet with cuda version)
- [transformers](https://huggingface.co/docs/transformers/index)>=4.34.0
- [tokenizers](https://huggingface.co/docs/tokenizers/index)>=0.14.0
- [accelerate](https://huggingface.co/docs/accelerate/index)>=0.23.0
- [peft](https://huggingface.co/docs/peft/index)>=0.5.0

## Installation

Install the necessary packages using `pip`:

### TSUBAME 4.0
```bash
pip install -r requirements_tsubame.txt
```

### ABCI 3.0
```bash
pip install -r requirements_abci.txt
```
### Flash Attention (Optional)

To turn on `use_flash_attention_2` option:
```bash
pip install flash-attn --no-build-isolation
```

## Dataset Preparation

A sample dataset is provided in `data/`. A training example is structured as follows:

```json
{
  "conversation_id": "33f01939a744455c869cb234afca47f1",
  "conversation": [
    {
      "content": "個人情報保護サービスは、個人情報の盗難からどのように私を守ることができるのでしょうか？",
      "role": "user"
    },
    {
      "role": "assistant",
      "content": "個人情報保護サービスは、個人情報の盗難からあなたを守るために、以下のような機能を提供しています。\n\n1.  **個人情報の監視**：あなたの個人情報がインターネット上で共有されているかどうかを監視し、不正な利用を検知します。\n2.  **不正アクセスの防止**：あなたの個人情報に不正アクセスする試みを阻止します。\n3.  **個人情報の暗号化**：あなたの個人情報を暗号化して、第三者がアクセスできないようにします。\n4.  **警告と通知**：不正なアクセスや個人情報の漏洩が検知された場合、即座にあなたに通知します。\n5.  **個人情報の削除**：インターネット上からあなたの個人情報を削除する手助けをします。\n6.  **セキュリティソフトウェアの提供**：マルウェアやウイルスからあなたのデバイスを保護するセキュリティソフトウェアを提供します。\n7.  **個人情報の管理**：あなたの個人情報を安全に管理し、必要な場合には更新や削除を行います。\n8.  **サポートとアドバイス**：個人情報の保護に関するサポートとアドバイスを提供します。\n\nこれらの機能により、個人情報保護サービスはあなたの個人情報を保護し、盗難や漏洩から守ることができます。"
    }
  ]
}
```

During training, loss calculation is focused on tokens in `d["content"]` where `d["role"]=="assistant"`. 

## Training
### TSUBAME 4.0
Bash scripts for submitting batch jobs for SFT on TSUBAME 4.0 are stored in `tsubame/`.
When submitting the batch job for a single node, the example usage is to run below:
```
qsub -g ${GROUP_ID} tsubame/run_sft_node_f.sh ${WANDB_JOB_NAME} ${SEED} ${TRAINING_DATA_PATH_1} ${TRAINING_DATA_PATH_2} ...
```
When submitting the batch job for multiple nodes, the example usage is to run below:
```
qsub -g ${GROUP_ID} tsubame/run_multinodes.sh ${DISTRIBUTED_TRAINING_SCRIPT} ${WANDB_JOB_NAME} ${SEED} ${TRAINING_DATA_PATH_1} ${TRAINING_DATA_PATH_2} ...
```
where an example of `${DISTRIBUTED_TRAINING_SCRIPT}` is [tsubame/run_sft_node_f_llama3.per_node.sh](tsubame/run_sft_node_f_llama3.per_node.sh).
- make sure to `chmod 777 ${DISTRIBUTED_TRAINING_SCRIPT}`, otherwise the script could not be executed by mpirun.

### ABCI 3.0
Bash scripts for submitting batch jobs for SFT on ABCI 3.0 are stored in `abci/`.
The hyperparameters are hard-coded in the bash script. Simply run `qsub abci/run_sft_node_f.sh` after modifying the arguments when necessary.

### Training models other than Llama-3.1-8B
When adding support for a new model, it is necessary to prepare a seperated `.py` training script. We are doing so because different model has different configurations regarding the chat template and stop/padding tokens.

Currently we support Llama-3.1, Llama-3.1-Swallow, Llama-3-Youko, LLM-jp, Gemma-2, Qwen-2.5, and OLMo-2. See the scripts in [scripts/](scripts/).
