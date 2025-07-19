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
    TrainingArguments, # SFTConfigの代わりに直接TrainingArgumentsを使用
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from transformers.trainer_utils import set_seed
from datasets import Dataset

# 外部のユーティリティ関数をインポート（実際のパスに合わせてください）
# これらはユーザーが独自に作成したモジュールと仮定します
from ..myutils.parsing import create_training_examples
from ..myutils.io import write_json, write_config
from ..myutils.logging import setup_logging

# 使用するGPUを指定（例: GPU 0とGPU 1）
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def set_device_map(base_model_path: Path) -> dict:
    """
    モデルのレイヤーを複数のGPUに分散させるためのdevice_mapを構築する。
    メモリ使用量を均等にするため、手動でレイヤー配置を決定する。
    """
    # モデルの設定ファイルを読み込み、レイヤー数を取得
    config = AutoConfig.from_pretrained(base_model_path)
    num_layers = config.num_hidden_layers
    logging.info(f"モデルのレイヤー数: {num_layers}")

    # device_mapを初期化
    device_map = {}
    
    # モデルの半分のレイヤーをGPU 0に、残りをGPU 1に割り当てる
    layers_on_gpu0 = num_layers // 2
    
    # 入力埋め込み層をGPU 0に配置
    device_map["model.embed_tokens"] = 0
    # 前半のトランスフォーマーレイヤーをGPU 0に配置
    for i in range(layers_on_gpu0):
        device_map[f"model.layers.{i}"] = 0
    
    # 後半のトランスフォーマーレイヤーをGPU 1に配置
    for i in range(layers_on_gpu0, num_layers):
        device_map[f"model.layers.{i}"] = 1
    # 最終的な正規化層をGPU 1に配置
    device_map["model.norm"] = 1
    
    # 出力層（言語モデルヘッド）をGPU 0に配置
    # 計算グラフの最初と最後を同じデバイスに置くことで、デバイス間のデータ転送を最適化する
    device_map["lm_head"] = 0
    
    logging.info(f"手動device_mapを構築しました: {device_map}")
    return device_map

def _create_and_tokenize_dataset(
    dataset_path: Optional[Path], 
    model_type: str, 
    tokenizer: AutoTokenizer, 
    user_template: str, 
    assistant_template: str
) -> Optional[Dataset]:
    """
    指定されたパスからデータセットを読み込み、チャットテンプレートを適用し、トークン化して返すヘルパー関数。
    検証データがない場合も考慮し、Optional[Path]とOptional[Dataset]を扱う。
    """
    # データセットのパスが指定されていない場合はNoneを返す
    if not dataset_path:
        return None
    
    logging.info(f"データセットを作成し、チャットテンプレートを適用・トークン化します: {dataset_path}")
    
    # ユーザー定義の関数でJSONLから対話形式のリストを作成
    training_examples_with_messages_key = create_training_examples(
        dataset_path, model_type, tokenizer,
        user_template, assistant_template
    )
    
    # 各対話サンプルをトークン化
    processed_dataset_for_trainer = []
    for example_item in training_examples_with_messages_key:
        tokenized_sample = tokenizer.apply_chat_template(
            example_item['messages'],
            tokenize=True,
            add_generation_prompt=False, # SFTでは通常False
            return_tensors=None, # Pythonのリストとして返す
            return_dict=True,    # 辞書形式で返す
        )
        processed_dataset_for_trainer.append(tokenized_sample)
        
    # PythonのリストからHugging FaceのDatasetオブジェクトを作成
    dataset = Dataset.from_list(processed_dataset_for_trainer)
    logging.info(f"トークン化済みデータセット作成完了。サンプル数: {len(dataset)}")
    return dataset

def train_sft(
    base_model: Path,
    resume_from_checkpoint: Optional[Path],
    model_type: str,
    train_dataset_path: Path,
    valid_dataset_path: Optional[Path], # 検証データセットのパス（オプション）
    user_template: str,
    assistant_template: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    accum_steps: int,
    seed: int,
):
    """メインの学習処理を実行する関数。"""
    
    # 再現性のために乱数シードを固定
    set_seed(seed)

    # 1. トークナイザの準備
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        attn_implementation='eager', # Attentionの実装を指定
        add_eos_token=True,        # シーケンスの終端にEOSトークンを追加
    )
    # pad_tokenが設定されていない場合、eos_tokenで代用する（一般的な対処法）
    if tokenizer.pad_token is None:
        logging.info("pad_tokenがNoneのため、eos_tokenをpad_tokenとして設定します。")
        tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"使用するチャットテンプレート: {tokenizer.chat_template}")

    # 2. データセットの読み込みと前処理
    # ヘルパー関数を使って訓練データと検証データをそれぞれ処理
    train_dataset = _create_and_tokenize_dataset(
        train_dataset_path, model_type, tokenizer, user_template, assistant_template
    )
    eval_dataset = _create_and_tokenize_dataset(
        valid_dataset_path, model_type, tokenizer, user_template, assistant_template
    )
    
    # 3. データコレータの設定
    # 応答部分のみを損失計算の対象とするためのコレータ
    assistant_response_prefix_str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    response_template_ids = tokenizer.encode(assistant_response_prefix_str, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    # 4. モデルの準備 (量子化とロード)
    logging.info(f"ベースモデルをロード中: {base_model}")
    # 4bit量子化の設定
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    # モデルを量子化し、複数のGPUに分散してロード
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map=set_device_map(base_model),
        use_cache=False, # 学習時にはキャッシュを無効化
        attn_implementation="eager"
    )
    # トークナイザの語彙サイズに合わせてモデルの埋め込み層をリサイズ
    model.resize_token_embeddings(len(tokenizer))

    # 5. LoRA (PEFT) の設定
    peft_config = LoraConfig(
        r=128, 
        lora_alpha=128, 
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        # LoRAを適用するモジュール名を指定
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 6. 学習パラメータ（TrainingArguments）の設定
    # 検証データセットの有無に応じて、評価戦略を動的に決定
    eval_strategy = "steps" if eval_dataset is not None else "no"
    logging_and_save_steps = 1000 # ログ出力とチェックポイント保存のステップ数を統一

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        bf16=True, # bfloat16を使用
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        gradient_checkpointing=True, # メモリ節約のため勾配チェックポインティングを有効化
        optim="paged_adamw_8bit", # メモリ効率の良いオプティマイザ
        learning_rate=3e-4,
        lr_scheduler_type="cosine", # 学習率スケジューラ
        max_grad_norm=0.3, # 勾配クリッピング
        warmup_ratio=0.1,
        logging_steps=logging_and_save_steps,
        save_steps=logging_and_save_steps,
        eval_strategy=eval_strategy, # 検証データがあればステップごとに評価
        eval_steps=logging_and_save_steps, # 評価のステップ間隔
        save_total_limit=1, # 保存するチェックポイントは最新の1つのみ
        load_best_model_at_end=eval_dataset is not None, # 検証データがあれば、最後に最良モデルをロード
        metric_for_best_model="loss" if eval_dataset is not None else None, # 最良モデルの判断基準
    )

    # 7. SFTTrainerの初期化
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset is not None else None,
        data_collator=collator,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # SFTTrainerによってpeftモデルが準備されているはずなので、trainer.modelで確認
    peft_model = trainer.model 
    logging.info(f"PEFTモデルクラス: {peft_model.__class__.__name__}")
    logging.info(f"PEFTモデル全体のデバイス (最初のパラメータ): {next(peft_model.parameters()).device}")

    # ベースモデルの主要コンポーネントのデバイスを確認
    base_model_proper = peft_model.base_model.model # 通常、この下に実際のトランスフォーマーブロックがある
    
    logging.info(f"入力埋め込みデバイス: {base_model_proper.get_input_embeddings().weight.device}")

    lm_head_to_check = None
    if hasattr(base_model_proper, 'lm_head') and base_model_proper.lm_head is not None:
        lm_head_to_check = base_model_proper.lm_head
        logging.info(f"LMヘッド (base_model_proper.lm_head) のデバイス: {lm_head_to_check.weight.device}")
    elif hasattr(base_model_proper, 'get_output_embeddings') and base_model_proper.get_output_embeddings() is not None:
        # Gemmaなど、get_output_embeddings()がLMヘッドを返す場合
        lm_head_to_check = base_model_proper.get_output_embeddings()
        logging.info(f"出力埋め込み (LMヘッド) のデバイス: {lm_head_to_check.weight.device}")
    else:
        logging.warning("LMヘッドの特定が困難でした。モデル構造を確認してください。")

    try:
        first_layer_device = next(iter(base_model_proper.model.layers[0].parameters())).device
        last_layer_device = next(iter(base_model_proper.model.layers[-1].parameters())).device
        logging.info(f"最初のトランスフォーマーレイヤーのデバイス: {first_layer_device}")
        logging.info(f"最後のトランスフォーマーレイヤーのデバイス: {last_layer_device}")
    except Exception as e:
        logging.warning(f"レイヤーのデバイス取得中にエラー: {e}")
    
    logging.info(f"--- 確認終了 ---")

    # 8. 学習の実行
    logging.info("LoRA学習を開始します...")
    # チェックポイントが指定されていれば、そこから学習を再開
    if resume_from_checkpoint:
        logging.info(f"チェックポイントから学習を再開: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))
    else:
        logging.info("新規学習を開始します。")
        trainer.train()
    
    logging.info("学習が完了しました。")

    # 9. モデルとトークナイザの保存
    if eval_dataset is not None:
        logging.info("最良モデルを保存します...")
        best_model_path = output_dir / "best_model"
        trainer.save_model(str(best_model_path))
        tokenizer.save_pretrained(str(best_model_path))
        logging.info(f"最良モデルを '{best_model_path}' に保存しました。")

def main():
    # コマンドライン引数のパーサーをセットアップ
    p = argparse.ArgumentParser(description="検証データ付きLoRA学習スクリプト")
    p.add_argument("--base-model", type=Path, required=True, help="ベースモデルのパス")
    p.add_argument("--resume-from-checkpoint", type=Path, default=None, help="学習を再開するチェックポイントのパス")
    p.add_argument("--model-type", type=str, default="qa_generator", choices=["qa_generator", "predictor", "q_generator"], help="モデルタイプ")
    p.add_argument("--train-dataset", type=Path, required=True, help="訓練用JSONLデータセットパス")
    p.add_argument("--valid-dataset", type=Path, default=None, help="検証用JSONLデータセットパス (オプション)")
    p.add_argument("--user-template", required=True, help="ユーザープロンプトのJinja2テンプレートパス")
    p.add_argument("--assistant-template", required=True, help="アシスタント応答のJinja2テンプレートパス")
    p.add_argument("--epochs", type=int, default=1, help="学習エポック数")
    p.add_argument("--batch-size", type=int, default=1, help="デバイス毎のバッチサイズ")
    p.add_argument("--accum-steps", type=int, default=8, help="勾配累積ステップ数")
    p.add_argument("--seed", type=int, default=42, help="乱数シード")
    p.add_argument("--log-filename", type=Path, default=None, help="ログファイル名")
    p.add_argument("--log-type", type=str, default="info", choices=["debug", "info"], help="ログレベル")
    args = p.parse_args()

    # ログ設定
    setup_logging(filename=args.log_filename, log_type=args.log_type)

    # 出力ディレクトリの作成（日付ベース）
    base_model_path = Path(args.base_model)
    output_dir = base_model_path.parent / "sft" / datetime.datetime.now().strftime("%Y%m%d") 
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.info(f"出力ディレクトリを作成しました: {output_dir}")

    # メインの学習関数を呼び出し
    train_sft(
        base_model_path,
        args.resume_from_checkpoint,
        args.model_type,
        args.train_dataset,
        args.valid_dataset, # 検証データパスを渡す
        args.user_template,
        args.assistant_template,
        output_dir,
        args.epochs,
        args.batch_size,
        args.accum_steps,
        args.seed
    )

    # 実行した学習設定をJSONファイルとして保存
    config = {
        "base_model": str(base_model_path),
        "resume_from_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
        "model_type": args.model_type,
        "train_dataset": str(Path(args.train_dataset).resolve()),
        "valid_dataset": str(Path(args.valid_dataset).resolve()) if args.valid_dataset else None,
        "user_template": args.user_template,
        "assistant_template": args.assistant_template,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "seed": args.seed,
    }
    write_config(output_dir, config)
    logging.info(f"学習設定を保存しました: {output_dir / 'config.json'}")

if __name__ == "__main__":
    main()
