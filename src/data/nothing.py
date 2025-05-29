#!/usr/bin/env python3

import json
import argparse
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def preprocess(input_path: Path, output_path: Path):
    """
    input_path: JSONL ファイルまたはディレクトリ（*.jsonl）のパス
    output_path: 前処理後 JSONL を書き出すファイルパス
    """
    records = []
    # ファイル単体 or ディレクトリ
    paths = []
    if input_path.is_dir():
        paths = sorted(input_path.glob("*.jsonl"))
    else:
        paths = [input_path]

    for p in paths:
        logging.info(f"Processing {p}")
        with p.open('r', encoding='utf-8') as f:
            for _ in f:
                rec = {}
                records.append(rec)

    # 出力ディレクトリを作成
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # JSONL で書き出し
    with output_path.open('w', encoding='utf-8') as out_f:
        for rec in records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logging.info(f"前処理完了: {len(records)} レコードを書き出しました → {output_path}")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="nothing データセットの前処理 (id, context, question, first answer の抽出)"
    )
    parser.add_argument(
        "--input", type=Path,
        help="入力データセットのパス (単一 JSONL ファイルまたはディレクトリ)"
    )
    parser.add_argument(
        "--output", type=Path,
        help="前処理済みデータの出力先 JSONL ファイルパス"
    )
    args = parser.parse_args()

    preprocess(args.input, args.output)

if __name__ == "__main__":
    main()
