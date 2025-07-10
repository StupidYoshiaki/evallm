import argparse
import json
import logging
import datetime
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import set_seed
from trl import GRPOConfig, GRPOTrainer
from sentence_transformers import SentenceTransformer, util
from janome.tokenizer import Tokenizer as JanomeTokenizer

from ..myutils.parsing import render_prompt, parse_json_objects, create_grpo_examples
from ..myutils.io import write_json, write_config
from ..myutils.logging import setup_logging
from .rewards import QARewardSystem

# --- ロギングとシードの設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
torch.manual_seed(42)


# --- メイン実行ブロック ---
def main():
    p = argparse.ArgumentParser(description="SFT済みLoRAモデルをGRPOで追加学習するスクリプト")
    p.add_argument("--base-model-path", type=Path, required=True, help="元のベースモデルのHugging Faceパス")
    p.add_argument("--sft-lora-path", type=Path, default=None, help="SFTで学習済みのアダプタのパス (checkpoint-xxxなど)")
    p.add_argument("--emb-model-path", type=Path, required=True, help="埋め込みモデルのパス")
    p.add_argument("--dataset-path", type=Path, required=True, help="学習に使用するJSONLデータセットのパス")
    p.add_argument("--prompt-template-name", required=True, help="ユーザーテンプレート .j2")
    
    # 学習ハイパーパラメータ
    p.add_argument("--num-generations", type=int, default=8, help="生成するサンプル数")
    p.add_argument("--learning-rate", type=float, default=1e-5, help="学習率")
    p.add_argument("--max-prompt-length", type=int, default=1024, help="プロンプトの最大長")
    p.add_argument("--max-completion-length", type=int, default=256, help="生成する最大トークン数")
    p.add_argument("--context-length", type=int, default=1024, help="コンテキストの最大長 (トークン数)")
    p.add_argument("--epochs", type=int, default=1, help="学習エポック数")
    p.add_argument("--batch-size", type=int, default=1, help="バッチサイズ")
    p.add_argument("--accum-steps", type=int, default=8, help="勾配累積ステップ数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=0.1, help="GRPOのbetaパラメータ (正則化の強さ)")

    # ログ設定
    p.add_argument("--log-filename", type=Path, default=None, help="ログファイル名")
    p.add_argument("--log-type", type=str, default="info", choices=["debug", "info"], help="ログレベル")

    args = p.parse_args()

    # --- ログ設定 ---
    setup_logging(filename=args.log_filename, log_type=args.log_type)

    # --- シードの設定 ---
    set_seed(args.seed)
    
    # --- モデルとトークナイザのロード ---
    logging.info(f"ベースモデルをロード中: {args.base_model_path}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager", # 対応GPUがない場合は "eager" に変更
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.sft_lora_path is not None:
        logging.info(f"SFT学習済みLoRAアダプタをロード中: {args.sft_lora_path}")
        model = PeftModel.from_pretrained(model, str(args.sft_lora_path))
    else:
        logging.info("SFTアダプタが指定されていないため、新しいLoRAアダプタを作成して学習します。")
        # SFT学習時と同様のLoraConfigを定義
        peft_config = LoraConfig(
            r=128,
            lora_alpha=128,
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
            # モデルのモジュール名は要確認 (Qwen3ならこれでOKなことが多い)
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        # ベースモデルに新しいLoRAアダプタを適用
        model = get_peft_model(model, peft_config)
        logging.info("新しいLoRAアダプタをモデルに適用しました。")
    
    # 学習可能なパラメータがあるか確認
    model.print_trainable_parameters()

    # --- データセットと報酬システムの準備 ---
    train_examples = create_grpo_examples(
        jsonl_path=args.dataset_path,
        tokenizer=tokenizer,
        prompt_template=args.prompt_template_name,
        max_length=args.max_prompt_length
    )
    train_dataset = Dataset.from_list(train_examples)
    reward_system = QARewardSystem(device=model.device) # モデルと同じデバイスに報酬計算モデルを配置

    # --- 出力ディレクトリの作成 ---
    base_model_path = Path(args.base_model_path)
    output_dir = base_model_path.parent / "grpo" / datetime.datetime.now().strftime("%Y%m%d") 
    output_dir.mkdir(parents=True, exist_ok=False)

    # --- GRPOTrainerの設定 ---
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=args.learning_rate,
        num_generations=args.num_generations, # 生成するサンプル数
        temperature=1.0,
        max_prompt_length=args.max_prompt_length, # プロンプトの最大長
        max_completion_length=args.max_completion_length, # 生成する最大トークン数
        beta=args.beta, # GRPOの正則化項
        logging_steps=10,
        save_strategy="epoch",
        report_to="none", # wandbなどを使う場合は設定
        remove_unused_columns=False, # context列などを報酬関数で使うためFalseに
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=reward_system, # 報酬計算クラスのインスタンスを渡す
    )
    
    # --- 学習開始 ---
    logging.info("GRPO学習を開始します...")
    trainer.train()
    
    # --- モデル保存 ---
    # logging.info("学習が完了しました。最終モデルを保存します。")
    # trainer.save_model(output_dir)

    # --- config ファイルの書き出し ---
    config = {
        "base_model_path": str(args.base_model_path),
        "sft_lora_path": str(args.sft_lora_path),
        "emb_model_path": str(args.emb_model_path),
        "dataset_path": str(args.dataset_path),
        "prompt_template_name": args.prompt_template_name,
        "learning_rate": args.learning_rate,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "seed": args.seed,
        "beta": args.beta,
    }
    write_config(output_dir, config)


if __name__ == "__main__":
    main()
