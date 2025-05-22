#!/usr/bin/env python3
"""
Hugging Face Hubまたは指定URLからデータセットをダウンロードし、
ローカルに保存するスクリプト

使い方例:
  # HF hub からデータセット全体を取得 (default 全スプリット)
  python download_dataset.py --dataset-id your-org/your-dataset --output-dir ./data

  # 特定スプリットのみ取得
  python download_dataset.py --dataset-id your-org/your-dataset --splits train,validation --output-dir ./data

  # URLリストから直接ダウンロード (同一ディレクトリに保存)
  python download_dataset.py --urls urls.txt --output-dir ./data

スクリプト動作:
  - --dataset-id 指定時:
    * Hugging Face datasets ライブラリを使ってロード
    * 指定スプリット(train, validation, test...)を JSON Lines 形式で保存
    * <output-dir>/<dataset_name>/<split>.jsonl に出力
  - --urls 指定時:
    * 各行に書かれた URL からファイルをダウンロード
    * ファイル名は URL の basename
"""
import os
import json
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
import requests
from ...myutils.logging import setup_logging

def download_from_urls(url_file: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for line in url_file.read_text().splitlines():
        url = line.strip()
        if not url:
            continue
        fname = Path(url).name
        dest = output_dir / fname
        logging.info(f"Downloading {url} -> {dest}")
        resp = requests.get(url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    logging.info("All URLs downloaded.")


def download_hf_dataset(dataset_id: str, splits: list[str], output_dir: Path):
    ds = load_dataset(dataset_id)
    base_dir = output_dir / dataset_id.split('/')[-1]
    base_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        if split not in ds:
            logging.warning(f"Split '{split}' not found in dataset. Skipped.")
            continue
        logging.info(f"Saving split '{split}'...")
        out_file = base_dir / f"{split}.jsonl"
        with out_file.open('w', encoding='utf-8') as f:
            for example in ds[split]:
                f.write(f"{json.dumps(example, ensure_ascii=False)}\n")
        logging.info(f"Saved {split} to {out_file}")
    logging.info("Dataset download complete.")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="データセットダウンローダー")
    parser.add_argument("--dataset-id", help="Hugging FaceのデータセットID (例: glue/mrpc)")
    parser.add_argument("--splits", default="train,validation,test",
                        help="カンマ区切りで指定 (デフォ: train,validation,test)")
    parser.add_argument("--urls", help="URLリストファイル (1行1 URL)")
    parser.add_argument("--output-dir", default="./data", help="保存先ディレクトリ")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.dataset_id:
        splits = [s.strip() for s in args.splits.split(',')]
        download_hf_dataset(args.dataset_id, splits, output_dir)
    elif args.urls:
        download_from_urls(Path(args.urls), output_dir)
    else:
        parser.error("--dataset-id または --urls を指定してください。")

if __name__ == '__main__':
    main()
