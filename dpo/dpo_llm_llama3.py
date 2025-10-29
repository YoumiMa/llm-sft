import torch
import logging
import itertools
import transformers
from typing import Optional
from dataclasses import dataclass
from transformers.trainer_utils import set_seed
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset, concatenate_datasets
from data_collator import DataCollatorForLastTurnOnlyLM

from transformers import (
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    BitsAndBytesConfig,
)


logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_info()


        
def load_dpo_datasets(files, tokenizer):
    datasets = [] # multiple datasets
    
    for data_file in files:
        # Detect file format based on extension
        if data_file.endswith('.parquet'):
            dataset = load_dataset("parquet", data_files=data_file)
        elif data_file.endswith('.json') or data_file.endswith('.jsonl') or data_file.endswith('.jsonl.gz'):
            dataset = load_dataset("json", data_files=data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")
        
        dataset = dataset["train"]
        datasets.append(dataset)
    
    # Concatenate all datasets
    combined_dataset = concatenate_datasets(datasets)
    
    return combined_dataset

@dataclass
class DPOTrainingArguments:
    model_name_or_path: str
    data_files: list[str]
    eval_data_files: list[str] = None
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = True
    additional_special_tokens: list[str] = None
    max_seq_length: int = 4096
    preprocessing_num_workers: int = 8
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    use_peft: bool = False
    peft_target_model: Optional[str] = "llm-jp"
    peft_target_modules: Optional[list[str]] = None
    peft_lora_r: int = 8
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05
    
    def __post_init__(self):
        if self.load_in_8bit and self.loadi_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive")
        if self.peft_target_model and self.peft_target_modules is None:
            if self.peft_target_model == "llm-jp":
                self.peft_target_modules = ["c_attn", "c_proj", "c_fc"]
            elif self.peft_target_model == "llama":
                # https://github.com/serp-ai/LLaMA-8bit-LoRA/blob/main/finetune_peft_8bit.py
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            elif self.peft_target_model == "llama-all":
                # https://note.com/kan_hatakeyama/n/ncd09c52d26c7
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                    "embed_tokens",
                ]
            else:
                logger.warning(
                    f"peft_target_model '{self.peft_target_model}' is not supported, "
                    f"so peft_target_modules is set to None."
                )

    

def load_chat_datasets(files):
    datasets = [] # multiple datasets
    
    for data_file in files:

        dataset = load_dataset("json", data_files=data_file)
        dataset = dataset["train"]
        datasets.append(dataset)
    
    return concatenate_datasets(datasets)


def main():
    parser = HfArgumentParser((DPOConfig, DPOTrainingArguments))
    #print(parser)
    dpo_config, dpo_training_args = parser.parse_args_into_dataclasses()    
    
    set_seed(dpo_config.seed)
    logger.info(f"Set seed: {dpo_config.seed}")
    
    tokenizer_name_or_path: str = (
        dpo_training_args.tokenizer_name_or_path or dpo_training_args.model_name_or_path
    )
    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=dpo_training_args.use_fast,
        additional_special_tokens=dpo_training_args.additional_special_tokens,
        trust_remote_code=True,
    )

    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    print(tokenizer.special_tokens_map, "ids:", tokenizer.all_special_ids)
    logger.info("Loading data")

    # dataset = load_dpo_datasets(dpo_training_args.data_files, tokenizer)
    
    # dataset = load_chat_datasets(dpo_training_args.data_files)
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    
    if dpo_training_args.eval_data_files:
        eval_dataset = load_chat_datasets(dpo_training_args.eval_data_files)
        dpo_config.eval_strategy = "steps"
    else:
        eval_dataset = None
    
    logger.info("Formatting prompts")

    instruction_ids = tokenizer.encode("<|start_header_id|>user<|end_header_id|>\n\n")[1:] # no begin of text
    response_ids = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n")[1:] # no begin of text
    
    collator = DataCollatorForLastTurnOnlyLM(
        last_turn_only=True,
        instruction_template=instruction_ids,
        response_template=response_ids,
        tokenizer=tokenizer,
    )

    # # for debugging purpose
    # batch = collator(tokenized_dataset[:1])
    # input_ids = batch["input_ids"][0]
    # labels = batch["labels"][0]
    # print("入力トークンID:", input_ids)
    # print("正解ラベル:", labels)
    
    
    # segments_to_fit: list[list[int]] = []
    # segments_to_ignore: list[list[int]] = []
    # # ラベルが-100である箇所とそうでない箇所ごとにグルーピング
    # for key, group in itertools.groupby(
    #     range(len(input_ids)), key=lambda i: labels[i] == -100
    # ):
    #     group = list(group)
    #     if key:
    #         segments_to_ignore.append(group)
    #     else:
    #         segments_to_fit.append(group)
    
    # print("---- 損失を計算しない部分 ----")
    # for seg in segments_to_ignore:
    #     print(tokenizer.decode(input_ids[seg]))
    #     print()
    
    # print("---- 損失を計算する部分 ----")
    # for seg in segments_to_fit:
    #     print(tokenizer.decode(input_ids[seg]))
    #     print()
    # ------------debugging------------

    logger.info(f"Loading model from {dpo_training_args.model_name_or_path}")
    
    logger.debug(
        f"AutoModelForCausalLM.from_pretrained({dpo_training_args.model_name_or_path}, trust_remote_code=True)"
    )
    model = AutoModelForCausalLM.from_pretrained(
        dpo_training_args.model_name_or_path,
        use_cache=False,
        trust_remote_code=True,
    )
    
    model.config.eos_token_id = [128001, 128008, 128009]

    logger.info("Setting up trainer")
    trainer = DPOTrainer(
    model,
    train_dataset=dataset,  # トークンID化されたデータセット
    args=dpo_config,  # 訓練の設定
    processing_class=tokenizer,  # パラメータ保存時にトークナイザも一緒に保存するために指定
)

    logger.info("Training")
    trainer.train(resume_from_checkpoint = dpo_config.resume_from_checkpoint)
    
    #model.config.eos_token_id = [128001, 128008, 128009]
    model.generation_config.eos_token_id = [128001, 128008, 128009]
    
    logger.info("Saving model")
    trainer.save_model()
    
    return


if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
