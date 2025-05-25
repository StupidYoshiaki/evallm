import json
import argparse
import datetime
import logging
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from transformers.trainer_utils import set_seed
from datasets import Dataset

from ..myutils.parsing import create_training_examples
from ..myutils.io import write_json

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def train_lora(
    base_model: Path,
    dataset_path: Path,
    user_template: str,
    assistant_template: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    accum_steps: int,
    seed: int,
) -> Path:
    setup_logging()
    # シード固定
    set_seed(seed)

    # トークナイザの設定
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        attn_implementation='eager',
        add_eos_token=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    # collator の設定, resoponse_template の部分のみ損失計算を適用させる
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<start_of_turn>user\n",
        response_template="<start_of_turn>model\n",
        tokenizer=tokenizer,
    )

    # データセットの読み込み
    train_dataset = create_training_examples(
        dataset_path, tokenizer,
        user_template, assistant_template
    )
    train_dataset = Dataset.from_list(train_dataset)

    # import sys
    # print(collator(train_dataset[0]["messages"]))
    # sys.exit()

    # 量子化をモデルに適用
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
        attn_implementation="eager"
    )
    model.resize_token_embeddings(len(tokenizer))

    # LoRA パラメータ設定
    peft_conf = LoraConfig(
        r=128, lora_alpha=128, lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )

    # 学習パラメータ設定
    sft_conf = SFTConfig(
        output_dir=output_dir,
        bf16=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        logging_steps=1000,
        save_steps=1000,
        save_total_limit=1,
    )

    # chat テンプレートの適用関数
    def formatting_func(example):
        return tokenizer.apply_chat_template(example['messages'], tokenize=False)
    
    # 学習
    trainer = SFTTrainer(
        model, 
        train_dataset=train_dataset,
        data_collator=collator,
        args=sft_conf,
        peft_config=peft_conf,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    trainer.train()

    # LoRA マージ
    logging.info("LoRA をベースにマージします")
    if isinstance(model, PeftModel):
        model.merge_and_unload()
    merged_pt = output_dir / "merged_lora.pt"
    torch.save(model.state_dict(), merged_pt)
    logging.info(f"マージ済み重み保存: {merged_pt}")

    return merged_pt

def main():
    p = argparse.ArgumentParser(description="LoRA 学習スクリプト")
    p.add_argument("--base-model", required=True, help="ベースモデルパス")
    p.add_argument("--dataset", required=True, help="JSONL データセットパス")
    p.add_argument("--user-template", required=True, help="ユーザーテンプレート .j2")
    p.add_argument("--assistant-template", required=True, help="アシスタントテンプレート .j2")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # ベースモデルパスの親ディレクトリに出力ディレクトリを作成
    base_model_path = Path(args.base_model)
    output_dir = base_model_path.parent / datetime.datetime.now().strftime("%Y%m%d") / "lora" 
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = train_lora(
        base_model_path,
        args.dataset,
        args.user_template,
        args.assistant_template,
        output_dir,
        args.epochs,
        args.batch_size,
        args.accum_steps,
        args.seed
    )
    print(merged)

    # 引数の内容を全て config.json に保存
    config = {
        "base_model": str(base_model_path),
        "dataset": str(Path(args.dataset).resolve()),
        "user_template": args.user_template,
        "assistant_template": args.assistant_template,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "seed": args.seed,
    }
    config_path = output_dir / "config.json"
    write_json(config_path, config)
    

if __name__ == "__main__":
    main()
