import os
import json
import argparse
import datetime
import logging
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, TaskType
from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig, 
    AutoModelForCausalLM, 
    AutoConfig,
    TrainingArguments,
)
from trl import DPOTrainer # SFTTrainerの代わりにDPOTrainerをインポート
from transformers.trainer_utils import set_seed
from datasets import load_dataset

# 外部のユーティリティ関数をインポート（実際のパスに合わせてください）
from ..myutils.io import write_json, write_config
from ..myutils.logging import setup_logging

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

def train_dpo(
    base_model: Path,
    resume_from_checkpoint: Optional[Path],
    dataset_path: Path,
    valid_dataset_path: Optional[Path],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    accum_steps: int,
    beta: float, # DPO用のbetaパラメータを追加
    seed: int,
):
    """メインのDPO学習処理を実行する関数。"""
    
    # 再現性のために乱数シードを固定
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
        # DPOではpaddingを左側にするのが一般的
        tokenizer.padding_side = 'left'

    # 2. データセットの読み込み
    logging.info(f"DPOデータセットを読み込み中: {dataset_path}")
    train_dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    
    eval_dataset = None
    if valid_dataset_path:
        logging.info(f"DPO検証データセットを読み込み中: {valid_dataset_path}")
        eval_dataset = load_dataset("json", data_files=str(valid_dataset_path), split="train")

    # 3. モデルの準備 (量子化とロード)
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

    # 5. 学習パラメータ（TrainingArguments）の設定
    eval_strategy = "steps" if eval_dataset is not None else "no"
    logging_and_save_steps = 1000

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        bf16=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=1e-5, # DPOではSFTより学習率を低めに設定するのが一般的
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        logging_steps=logging_and_save_steps,
        save_steps=logging_and_save_steps,
        evaluation_strategy=eval_strategy,
        eval_steps=logging_and_save_steps,
        save_total_limit=1,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
    )

    # 6. DPOTrainerの初期化
    # DPOTrainerは内部で自動的に参照モデルのコピーを作成する
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None, # Noneにすると、trainerが自動で参照モデルを準備してくれる
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=beta, # DPO損失のβパラメータ
        max_prompt_length=1024, # プロンプトの最大長
        max_length=2048, # プロンプト+応答の最大長
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
    final_model_path = output_dir / "best_dpo_model"
    dpo_trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    logging.info(f"最良モデルを '{final_model_path}' に保存しました。")


def main():
    p = argparse.ArgumentParser(description="DPO 学習スクリプト")
    p.add_argument("--base-model", type=Path, required=True, help="ファインチューニングのベースとなるSFT済みモデルのパス")
    p.add_argument("--resume-from-checkpoint", type=Path, default=None, help="学習を再開するチェックポイントのパス")
    p.add_argument("--dataset", type=Path, required=True, help="DPO学習用のJSONLデータセットパス")
    p.add_argument("--valid-dataset", type=Path, default=None, help="DPO検証用のJSONLデータセットパス (オプション)")
    p.add_argument("--epochs", type=int, default=1, help="学習エポック数")
    p.add_argument("--batch-size", type=int, default=1, help="デバイス毎のバッチサイズ")
    p.add_argument("--accum-steps", type=int, default=8, help="勾配累積ステップ数")
    p.add_argument("--beta", type=float, default=0.1, help="DPO損失のβパラメータ")
    p.add_argument("--seed", type=int, default=42, help="乱数シード")
    p.add_argument("--log-filename", type=Path, default=None, help="ログファイル名")
    p.add_argument("--log-type", type=str, default="info", choices=["debug", "info"], help="ログレベル")
    args = p.parse_args()

    # ログ設定
    setup_logging(filename=args.log_filename, log_type=args.log_type)

    # 出力ディレクトリの作成
    base_model_path = Path(args.base_model)
    # SFT済みモデルのディレクトリ構造を想定
    output_dir = base_model_path.parent / f"dpo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.info(f"出力ディレクトリを作成しました: {output_dir}")

    # メインの学習関数を呼び出し
    train_dpo(
        base_model_path,
        args.resume_from_checkpoint,
        args.dataset,
        args.valid_dataset,
        output_dir,
        args.epochs,
        args.batch_size,
        args.accum_steps,
        args.beta,
        args.seed
    )

    # 実行した学習設定をJSONファイルとして保存
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    write_config(output_dir, config)
    logging.info(f"学習設定を保存しました: {output_dir / 'training_config.json'}")

if __name__ == "__main__":
    main()
