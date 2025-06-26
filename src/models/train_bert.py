import argparse
import json
import logging
import datetime
import sys
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import DatasetDict, load_dataset, Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

from ..myutils.logging import setup_logging

# GPU 0とGPU 1を利用可能にする
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def load_data(train_path: Path, valid_path: Optional[Path]) -> DatasetDict:
    """
    指定されたパスから訓練データと検証データを読み込む。

    Args:
        train_path (Path): 訓練データセットのパス (JSONL形式)。
        valid_path (Optional[Path]): 検証データセットのパス (JSONL形式)。

    Returns:
        DatasetDict: 読み込まれたデータセット。
    """
    data_files = {'train': str(train_path)}
    if valid_path:
        data_files['validation'] = str(valid_path)
    
    logging.info(f"データセットを読み込み中: {data_files}")
    return load_dataset('json', data_files=data_files)


def setup_tokenizer_and_model(model_name_or_path: str):
    """
    指定されたモデル名またはパスからトークナイザとモデルを読み込む。

    Args:
        model_name_or_path (str): Hugging Face Hubのモデル名またはローカルのモデルパス。

    Returns:
        tuple: (tokenizer, model)
    """
    logging.info(f"トークナイザとモデルを読み込み中: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
    return tokenizer, model


def preprocess_data(
    dataset: DatasetDict, 
    tokenizer: AutoTokenizer, 
    max_length: int
) -> DatasetDict:
    """
    質問応答タスク用にデータセットを前処理する。

    Args:
        dataset (DatasetDict): 読み込まれたデータセット。
        tokenizer (AutoTokenizer): 使用するトークナイザ。
        max_length (int): 入力の最大トークン長。

    Returns:
        DatasetDict: 前処理されたデータセット。
    """
    logging.info("データセットの前処理を開始します...")

    def _preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        # stride関連の引数を削除し、トランケーションのみを行う
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",  # context側を切り詰める
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        
        start_positions = []
        end_positions = []

        # バッチ内の各サンプルに対してループ (strideがないので1サンプル=1特徴量)
        for i, offsets in enumerate(offset_mapping):
            sequence_ids = inputs.sequence_ids(i)

            # 回答不能なサンプルの場合
            if examples["is_impossible"][i]:
                start_positions.append(0)
                end_positions.append(0)
                continue

            # 回答の文字位置を取得
            answers = examples["answers"][i]
            start_char = answers['answer_start'][0]
            answer_text = answers['text'][0]
            end_char = start_char + len(answer_text)

            # 文脈部分のトークンインデックス範囲を取得
            ctx_start_token_idx = sequence_ids.index(1)
            ctx_end_token_idx = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

            # トランケーションによって回答が文脈から切り捨てられたかチェック
            # offsets[idx][0]はトークンの開始文字位置, offsets[idx][1]は終了文字位置
            if not (offsets[ctx_start_token_idx][0] <= start_char and offsets[ctx_end_token_idx][1] >= end_char):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # 文字位置をトークン位置に変換
                token_start_index = ctx_start_token_idx
                while token_start_index <= ctx_end_token_idx and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)
                
                token_end_index = ctx_end_token_idx
                while token_end_index >= token_start_index - 1 and offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_datasets = dataset.map(
        _preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    logging.info("データセットの前処理が完了しました。")
    return tokenized_datasets


def test_mapping(sample_dataset: Dataset, tokenizer: AutoTokenizer, max_length: int):
    """
    前処理のマッピングが正しいか目視確認するための情報を表示する。
    """
    logging.info("--- マッピングの目視確認を開始します ---")
    
    # テスト用の前処理関数 (remove_columnsなし)
    def _test_preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"], max_length=max_length,
            truncation="only_second", return_offsets_mapping=True, padding="max_length"
        )
        offset_mapping = inputs.pop("offset_mapping")
        start_positions, end_positions = [], []
        for i, offsets in enumerate(offset_mapping):
            sequence_ids = inputs.sequence_ids(i)
            if examples["is_impossible"][i]:
                start_positions.append(0); end_positions.append(0)
                continue
            answers = examples["answers"][i]
            start_char, answer_text = answers['answer_start'][0], answers['text'][0]
            end_char = start_char + len(answer_text)
            ctx_start = sequence_ids.index(1)
            ctx_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
            if not (offsets[ctx_start][0] <= start_char and offsets[ctx_end][1] >= end_char):
                start_positions.append(0); end_positions.append(0)
            else:
                token_start = ctx_start
                while token_start <= ctx_end and offsets[token_start][0] <= start_char:
                    token_start += 1
                start_positions.append(token_start - 1)
                token_end = ctx_end
                while token_end >= token_start - 1 and offsets[token_end][1] >= end_char:
                    token_end -= 1
                end_positions.append(token_end + 1)
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    # 元の情報を保持したまま前処理を実行
    processed_sample = sample_dataset.map(_test_preprocess_function, batched=True)

    for i in range(len(processed_sample)):
        sample = processed_sample[i]
        input_ids = sample["input_ids"]
        start_pos = sample["start_positions"]
        end_pos = sample["end_positions"]
        
        print(f"\n--- サンプル {i} --------------------")
        print(f"質問: {sample['question']}")
        
        original_answer_text = sample["answers"]["text"][0] if not sample["is_impossible"] else "[回答不能]"
        print(f"元の回答テキスト: {original_answer_text}")
        
        print(f" -> 計算されたトークン位置: start={start_pos}, end={end_pos}")

        if start_pos == 0 and end_pos == 0:
            # is_impossible または トランケーションで範囲外になったケース
            decoded_text = "[復元: 回答不能 or 範囲外]"
        else:
            # スライスの終点は含まれないため +1 する
            answer_token_ids = input_ids[start_pos : end_pos + 1]
            decoded_text = tokenizer.decode(answer_token_ids)
        
        print(f" -> 復元されたテキスト: {decoded_text}")

    logging.info("--- マッピングの目視確認が完了しました ---")


def run_training(args: argparse.Namespace):
    """
    メインの学習プロセスを実行する。
    """
    # シード固定による再現性の確保
    set_seed(args.seed)

    # 出力ディレクトリの作成
    output_dir = Path(args.model_path).parent / "finetuned" / datetime.datetime.now().strftime("%Y%m%d") 
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. データの読み込み
    raw_datasets = load_data(args.train_dataset_path, args.valid_dataset_path)

    # 2. トークナイザとモデルの準備
    tokenizer, model = setup_tokenizer_and_model(args.model_path)

    # # 目視確認テスト
    # sample_size = min(10, len(raw_datasets["train"]))
    # test_sample = raw_datasets["train"].select(range(sample_size))
    # test_mapping(test_sample, tokenizer, args.max_length)
    # sys.exit()

    # 3. データの前処理
    tokenized_datasets = preprocess_data(
        raw_datasets,
        tokenizer,
        args.max_length,
    )
    
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets.get("validation") # 検証データが存在する場合に取得

    # 4. 学習引数の設定
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=torch.cuda.is_available(),
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="epoch" if eval_dataset else "no", # 検証データがあればエポック毎に評価
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True if eval_dataset else False, # 評価があれば最良モデルをロード
        metric_for_best_model="loss" if eval_dataset else None, # 評価の指標
    )

    # 5. Trainerの初期化と学習の実行
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    logging.info("学習を開始します...")
    trainer.train()
    logging.info("学習が完了しました。")

    # 6. 最終モデルの保存
    final_model_path = output_dir / "best_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    logging.info(f"最良モデルを '{final_model_path}' に保存しました。")

    # # 7. 学習設定の保存
    # config_path = output_dir / "training_config.json"
    # # argparse.Namespaceを辞書に変換
    # args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    # with open(config_path, "w", encoding="utf-8") as f:
    #     json.dump(args_dict, f, indent=4, ensure_ascii=False)
    # logging.info(f"学習設定を '{config_path}' に保存しました。")


def main():
    parser = argparse.ArgumentParser(description="質問応答モデルのファインチューニングスクリプト (v2)")
    
    # パス関連
    parser.add_argument("--model-path", type=str, required=True, help="HuggingFaceのモデル名またはローカルパス")
    parser.add_argument("--train-dataset-path", type=Path, required=True, help="訓練用データセットのパス (JSONL形式)")
    parser.add_argument("--valid-dataset-path", type=Path, default=None, help="検証用データセットのパス (JSONL形式)")

    # 学習ハイパーパラメータ
    parser.add_argument("--num-train-epochs", type=int, default=1, help="訓練エポック数")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="訓練時のデバイス毎のバッチサイズ")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="【要調整】勾配を累積するステップ数。実質的なバッチサイズ = batch_size * accum_steps")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4, help="評価時のデバイス毎のバッチサイズ")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="学習率") # 技術ブログに最良と記載
    parser.add_argument("--weight-decay", type=float, default=0.01, help="重み減衰")
    parser.add_argument("--logging-steps", type=int, default=50, help="ログ出力のステップ間隔")
    
     # ログ設定
    setup_logging(
        filename=None,  # ログファイルは指定しない
        log_type="info",  # ログレベルをinfoに設定
    )

    # 前処理パラメータ
    parser.add_argument("--max-length", type=int, default=4096, help="入力の最大トークン長")
    
    # その他
    parser.add_argument("--seed", type=int, default=42, help="再現性のための乱数シード")

    args = parser.parse_args()
    
    run_training(args)


if __name__ == "__main__":
    main()
