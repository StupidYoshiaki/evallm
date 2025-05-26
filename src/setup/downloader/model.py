#!/usr/bin/env python3
"""
GGUF形式またはsafetensors形式のモデルをHugging Face Hubまたは直接URLからダウンロードするスクリプト

使い方例:
  # GGUFモデル(.gguf)のみを取得
  python download_model.py \
    --repo-id your-org/your-gguf-model \
    --format gguf \
    --filename base.gguf \
    --output-dir ./models/evaluatee

  # safetensorsモデル全ファイルを 'safetensors/base' に取得
  python download_model.py \
    --repo-id your-org/your-safetensors-model \
    --format safetensors \
    --output-dir ./models/generator

  # URLから直接ファイルを取得
  python download_model.py \
    --url https://example.com/model.safetensors \
    --output ./models/generator/your-model/safetensors/base/model.safetensors

スクリプト動作:
  - --repo-id 指定時、HF Hubから取得
    * gguf: <output-dir>/<model_name>/<filename> に保存
    * safetensors: <output-dir>/<model_name>/safetensors/base/ 以下に全ファイルを保存
    * 各 <model_name> ディレクトリに config.json を生成し、取得元情報を記載
  - --url 指定時、URL直接ダウンロード
"""
import re
import os
import json
import argparse
import logging
import requests
import shutil
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download
from ...myutils.logging import setup_logging
from ...myutils.io import write_config


def download_gguf(repo_id: str, filename: str, revision: str, output_dir: Path):
    """
    Hugging Face Hubから.ggufモデルをダウンロード
    保存先: <output_dir>/<model_name>/<filename>
    """
    repo_name = repo_id.split("/")[-1]
    repo_name = repo_name.replace("-GGUF", "")  # GGUFを削除
    # もし末尾に -id (dは自然数) などがあれば削除
    repo_name = re.sub(r'-i\d*$', '', repo_name)
    model_dir = output_dir / repo_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # 一時ダウンロード
    tmp_path = Path(hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    ))

    # 目的地へコピー
    dest_path = model_dir / "gguf" / "base.gguf"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tmp_path, dest_path)
    logging.info(f"GGUFモデルを保存しました: {dest_path}")

    # config.json 作成
    info = {
        "repo_id": repo_id,
        "filename": filename,
        "revision": revision,
        "source_path": str(tmp_path)
    }
    write_config(model_dir / "gguf", info)
    return dest_path


def download_safetensors(repo_id: str, revision: str, output_dir: Path):
    """
    Hugging Face Hubからモデルスナップショットをダウンロードし、
    スナップショット全ファイルを <output_dir>/<model_name>/safetensors/base/ に保存
    """
    repo_name = repo_id.split("/")[-1]
    base_dir = output_dir / repo_name / "safetensors" / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    # 一時スナップショット取得
    tmp_root = Path(snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(output_dir / "_tmp"),
    ))

    # 全ファイルコピー
    for file_path in tmp_root.rglob('*'):
        if file_path.is_file():
            rel = file_path.relative_to(tmp_root)
            dest = base_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, dest)
    logging.info(f"スナップショット全ファイルを保存しました: {base_dir}")

    # config.json 作成
    info = {
        "repo_id": repo_id,
        "revision": revision,
        "snapshot_root": str(tmp_root)
    }
    write_config(output_dir / repo_name / "safetensors", info)

    # 一時ディレクトリ削除
    shutil.rmtree(output_dir / "_tmp")
    logging.info(f"一時ディレクトリを削除しました: {output_dir / '_tmp'}")
    return base_dir


def download_from_url(url: str, output_path: Path):
    """
    直接URLから単一ファイルをダウンロード（プログレスバー付き）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    try:
        from tqdm import tqdm
        progress = tqdm(total=total, unit='iB', unit_scale=True)
    except ImportError:
        progress = None

    with output_path.open('wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            if progress:
                progress.update(len(chunk))
    if progress:
        progress.close()

    logging.info(f"URLからダウンロードしました: {output_path}")
    return output_path


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="GGUFまたはsafetensors形式のモデルをHF HubまたはURLからダウンロード"
    )
    parser.add_argument("--repo-id", help="Hugging FaceのモデルリポジトリID。URL取得時は不要")
    parser.add_argument("--format", choices=["gguf", "safetensors"], help="モデル形式を指定")
    parser.add_argument("--filename", help=".ggufファイル名（ggufモードで必須）")
    parser.add_argument("--revision", default="main", help="リビジョン（デフォルト: main）")
    parser.add_argument("--output-dir", default="./models", help="ベースとなる出力先ディレクトリ")
    parser.add_argument("--url", help="直接ダウンロードするファイルのURL")
    parser.add_argument("--output", help="URLモードの出力先パス（省略時はPWD）")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.repo_id:
        if args.format == "gguf" and args.filename:
            download_gguf(
                repo_id=args.repo_id,
                filename=args.filename,
                revision=args.revision,
                output_dir=output_dir,
            )
        elif args.format == "safetensors":
            download_safetensors(
                repo_id=args.repo_id,
                revision=args.revision,
                output_dir=output_dir,
            )
        else:
            parser.error("--format と必要なオプションを正しく指定してください。")
    elif args.url:
        output_path = Path(args.output) if args.output else Path.cwd() / Path(args.url).name
        download_from_url(args.url, output_path)
    else:
        parser.error("--repo-id または --url のいずれかを指定してください。")

if __name__ == "__main__":
    main()
