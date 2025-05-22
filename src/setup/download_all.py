#!/usr/bin/env python3

import csv
import argparse
import logging
from pathlib import Path

# モデル／データセット用モジュールをインポート
from .downloader.model import download_gguf, download_safetensors
from .downloader.data import download_hf_dataset
from ..myutils.logging import setup_logging

def process_models(csv_path: Path):
    """
    CSVフォーマット:
      repo_id,format,filename,output_dir
    format: gguf or safetensors
    filename: ggufの場合のみ必須 (.ggufファイル名)
    """
    with csv_path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo_id    = row['repo_id'].strip()
            fmt        = row['format'].strip()
            filename   = row.get('filename','').strip()
            output_dir = Path(row['output_dir'].strip())
            logging.info(f"モデル: {repo_id} → format={fmt}")

            if fmt == 'gguf':
                if not filename:
                    logging.error(f"  filename が空です: {repo_id}")
                    continue
                download_gguf(
                    repo_id=repo_id,
                    filename=filename,
                    revision='main',
                    output_dir=output_dir
                )
            elif fmt == 'safetensors':
                download_safetensors(
                    repo_id=repo_id,
                    revision='main',
                    output_dir=output_dir
                )
            else:
                logging.error(f"  未知の format: {fmt}")

def process_datasets(csv_path: Path):
    """
    CSVフォーマット:
      dataset_id,splits,output_dir
    splits: train,validation,test...
    """
    with csv_path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds_id      = row['dataset_id'].strip()
            splits     = [s.strip() for s in row['splits'].split(',')]
            output_dir = Path(row['output_dir'].strip())
            logging.info(f"データセット: {ds_id} → splits={splits}")
            download_hf_dataset(
                dataset_id=ds_id,
                splits=splits,
                output_dir=output_dir,
            )

def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="モデル＆データセット一括ダウンロード"
    )
    parser.add_argument(
        "--model-list", type=Path, required=True,
        help="モデル一覧CSV (repo_id,format,filename,output_dir)"
    )
    parser.add_argument(
        "--dataset-list", type=Path, required=True,
        help="データセット一覧CSV (dataset_id,splits,output_dir)"
    )
    args = parser.parse_args()

    logging.info("== モデル一覧処理開始 ==")
    process_models(args.model_list)

    logging.info("== データセット一覧処理開始 ==")
    process_datasets(args.dataset_list)

    logging.info("=== すべてのダウンロードが完了しました ===")

if __name__ == "__main__":
    main()
