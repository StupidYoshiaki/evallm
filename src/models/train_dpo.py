import os
import json
import argparse
import datetime
import logging
from pathlib import Path
from typing import Optional, Dict

import torch
from peft import LoraConfig, TaskType
from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoModelForCausalLM, 
    AutoConfig,
)
from trl import DPOTrainer, DPOConfig # DPOConfigをインポート
from transformers.trainer_utils import set_seed
from datasets import load_dataset, Dataset

# 外部のユーティリティ関数をインポート（実際のパスに合わせてください）
from ..myutils.io import write_config
from ..myutils.logging import setup_logging
from ..myutils.parsing import render_prompt

# 使用するGPUを指定
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def set_device_map(base_model_path: Path) -> dict:
    """
    モデルのレイヤーを複数のGPUに分散させるためのdevice_mapを構築する。
    """
    config = AutoConfig.from_pretrained(base_model_path)
    num_layers = config.num_hidden_layers
    logging.info(f"モデルのレイヤー数: {num_layers}")
    device_map = {}
    layers_on_gpu0 = num_layers // 2
    device_map["model.embed_tokens"] = 0
    for i in range(layers_on_gpu0):
        device_map[f"model.layers.{i}"] = 0
    for i in range(layers_on_gpu0, num_layers):
        device_map[f"model.layers.{i}"] = 1
    device_map["model.norm"] = 1
    device_map["lm_head"] = 0
    logging.info(f"手動device_mapを構築しました: {device_map}")
    return device_map

def format_dataset_for_dpo(
    example: Dict, 
    tokenizer: AutoTokenizer, 
    user_template: str, 
    assistant_template: str
) -> Dict:
    """
    DPO学習用に、データセットの各サンプルをチャットテンプレート形式に整形する。
    """
    # ユーザープロンプトのコンテンツを作成
    user_content = render_prompt(user_template, context=example['prompt'])
    
    # Hugging Faceのチャット形式に変換
    prompt_messages = [{"role": "user", "content": user_content}]
    
    # トークナイザのチャットテンプレートを適用し、最終的なプロンプト文字列を作成
    prompt_str = tokenizer.apply_chat_template(
        prompt_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # chosenとrejectedにもテンプレートを適用
    chosen_response = render_prompt(assistant_template, question=example['chosen'])
    rejected_response = render_prompt(assistant_template, question=example['rejected'])
    
    return {
        "prompt": prompt_str,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }

def train_dpo(
    base_model: Path,
    resume_from_checkpoint: Optional[Path],
    dataset_path: Path,
    valid_dataset_path: Optional[Path],
    user_template: str,
    assistant_template: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    accum_steps: int,
    beta: float,
    seed: int,
):
    """メインのDPO学習処理を実行する関数。"""
    
    set_seed(seed)

    # 1. トークナイザの準備
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        attn_implementation='eager',
        add_eos_token=True,
    )
    if tokenizer.pad_token is None:
        logging.info("pad_tokenがNoneのため、eos_tokenをpad_tokenとして設定します。")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    # 2. データセットの読み込みと前処理
    logging.info(f"DPOデータセットを読み込み中: {dataset_path}")
    train_dataset_raw = load_dataset("json", data_files=str(dataset_path), split="train")
    
    train_dataset = train_dataset_raw.map(
        format_dataset_for_dpo, 
        fn_kwargs={
            "tokenizer": tokenizer, 
            "user_template": user_template,
            "assistant_template": assistant_template
        }
    )
    
    eval_dataset = None
    if valid_dataset_path:
        logging.info(f"DPO検証データセットを読み込み中: {valid_dataset_path}")
        eval_dataset_raw = load_dataset("json", data_files=str(valid_dataset_path), split="train")
        eval_dataset = eval_dataset_raw.map(
            format_dataset_for_dpo,
            fn_kwargs={
                "tokenizer": tokenizer, 
                "user_template": user_template,
                "assistant_template": assistant_template
            }
        )
    
    # データセットのサンプルを常に表示して確認する
    logging.info("--- 整形後の学習データセットのサンプル（最初の5件） ---")
    num_samples_to_show = min(5, len(train_dataset))
    for i in range(num_samples_to_show):
        example = train_dataset[i]
        print("-" * 50)
        print(f"【サンプル {i+1}】")
        print(f"--- PROMPT ---\n{example['prompt']}")
        print(f"--- CHOSEN ---\n{example['chosen']}")
        print(f"--- REJECTED ---\n{example['rejected']}")
        print("-" * 50 + "\n")

    # 3. モデルの準備
    logging.info(f"ベースモデル（SFT済みモデル）をロード中: {base_model}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map=set_device_map(base_model),
        use_cache=False,
        attn_implementation="eager"
    )
    model.resize_token_embeddings(len(tokenizer))

    # 4. LoRA (PEFT) の設定
    peft_config = LoraConfig(
        r=128, 
        lora_alpha=128, 
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 5. DPOConfigの設定
    eval_strategy = "steps" if eval_dataset is not None else "no"
    logging_and_save_steps = 1000

    dpo_config = DPOConfig(
        # --- DPO固有のパラメータ ---
        beta=beta,
        max_prompt_length=1024,
        max_length=2048,
        
        # --- TrainingArguments相当のパラメータ ---
        output_dir=str(output_dir),
        bf16=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        logging_steps=logging_and_save_steps,
        save_steps=logging_and_save_steps,
        eval_strategy=eval_strategy,
        eval_steps=logging_and_save_steps,
        save_total_limit=1,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
    )

    # 6. DPOTrainerの初期化
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=dpo_config, # DPOConfigオブジェクトを渡す
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 7. 学習の実行
    logging.info("DPO学習を開始します...")
    if resume_from_checkpoint:
        logging.info(f"チェックポイントから学習を再開: {resume_from_checkpoint}")
        dpo_trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
    else:
        logging.info("新規学習を開始します。")
        dpo_trainer.train()
    
    logging.info("学習が完了しました。")
    
    # 8. 最終モデルの保存
    if eval_dataset is not None:
        logging.info("最良モデルを保存します...")
        final_model_path = output_dir / "best_model"
        dpo_trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        logging.info(f"最良モデルを '{final_model_path}' に保存しました。")


def main():
    p = argparse.ArgumentParser(description="DPO 学習スクリプト")
    p.add_argument("--base-model", type=Path, required=True, help="ファインチューニングのベースとなるSFT済みモデルのパス")
    p.add_argument("--resume-from-checkpoint", type=Path, default=None, help="学習を再開するチェックポイントのパス")
    p.add_argument("--train-dataset", type=Path, required=True, help="DPO学習用のJSONLデータセットパス")
    p.add_argument("--valid-dataset", type=Path, default=None, help="DPO検証用のJSONLデータセットパス (オプション)")
    p.add_argument("--user-template", type=str, required=True, help="ユーザープロンプトのJinja2テンプレートファイル名")
    p.add_argument("--assistant-template", type=str, required=True, help="アシスタント応答のJinja2テンプレートファイル名")
    p.add_argument("--epochs", type=int, default=1, help="学習エポック数")
    p.add_argument("--batch-size", type=int, default=1, help="デバイス毎のバッチサイズ")
    p.add_argument("--accum-steps", type=int, default=8, help="勾配累積ステップ数")
    p.add_argument("--beta", type=float, default=0.1, help="DPO損失のβパラメータ")
    p.add_argument("--seed", type=int, default=42, help="乱数シード")
    p.add_argument("--log-filename", type=Path, default=None, help="ログファイル名")
    p.add_argument("--log-type", type=str, default="info", choices=["debug", "info"], help="ログレベル")
    args = p.parse_args()

    setup_logging(filename=args.log_filename, log_type=args.log_type)

    base_model_path = Path(args.base_model)
    output_dir = base_model_path.parent / "dpo" / datetime.datetime.now().strftime("%Y%m%d")
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.info(f"出力ディレクトリを作成しました: {output_dir}")

    train_dpo(
        base_model_path,
        args.resume_from_checkpoint,
        args.train_dataset,
        args.valid_dataset,
        args.user_template,
        args.assistant_template,
        output_dir,
        args.epochs,
        args.batch_size,
        args.accum_steps,
        args.beta,
        args.seed,
    )

    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    logging.info(f"学習設定を保存しました: {config_path}")

if __name__ == "__main__":
    main()
