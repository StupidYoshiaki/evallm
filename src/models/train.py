#!/usr/bin/env python3
# src/train/cli.py

import argparse
import datetime
import logging
from pathlib import Path

from ..myutils.parsing import create_training_examples
from ..myutils.trainer import (
    prepare_tokenizer, prepare_model_with_lora,
    train_and_export
)
from transformers.trainer_utils import set_seed

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="LoRA付きモデルの学習＋GGUF変換")
    parser.add_argument(
        "--base-model-gguf", type=Path, required=True,
        help="ベースモデルの GGUF ファイルパス"
    )
    parser.add_argument(
        "--lora-template", type=str, required=True,
        help="LoRA適用用プロンプト Jinja2 テンプレート名 (例: train_assistant.j2)"
    )
    parser.add_argument(
        "--user-template", type=str, required=True,
        help="ユーザープロンプト用テンプレート名 (例: train_user.j2)"
    )
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="学習用データ JSONL ファイルパス"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/generator"),
        help="生成モデル出力先ディレクトリ"
    )
    parser.add_argument("--epochs", type=int, default=1, help="学習エポック数")
    parser.add_argument("--batch-size", type=int, default=2, help="バッチサイズ")
    parser.add_argument("--accum-steps", type=int, default=8, help="勾配蓄積ステップ数")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")

    args = parser.parse_args()

    # 乱数シード固定
    set_seed(args.seed)

    # トークナイザ＆事前処理
    tokenizer = prepare_tokenizer(str(args.base_model_gguf))
    examples = create_training_examples(
        str(args.dataset),
        tokenizer,
        args.user_template,
        args.lora_template
    )

    # LoRA 設定
    from peft import LoraConfig, TaskType
    peft_conf = LoraConfig(
        r=128,
        lora_alpha=128,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj"
        ],
    )

    # モデル準備
    model = prepare_model_with_lora(str(args.base_model_gguf), peft_conf)

    # 出力ディレクトリを日付ベースで作成
    # LoRA テンプレート名に日付を含める場合はそちらを使う
    today = datetime.datetime.now().strftime("%Y%m%d")
    model_name = args.base_model_gguf.stem
    out_dir = args.output_dir / model_name / "lora" / today

    # 学習＋マージ＋GGUF 変換
    final_gguf = train_and_export(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        base_gguf_path=args.base_model_gguf
    )

    logging.info(f"最終GGUFモデル: {final_gguf}")

if __name__ == "__main__":
    main()
