#!/usr/bin/env python3
# src/generator/cli.py

import argparse
import logging
import traceback
from pathlib import Path
from tqdm import tqdm

from ..myutils.api import start_llama_server, stop_llama_server, generate_from_llm
from ..myutils.parsing import build_messages, parse_json_objects
from ..myutils.io import read_jsonl, write_jsonl
from ..myutils.logging import setup_logging

MAX_RETRIES = 10 # 最大リトライ回数

def process_item(item: dict, template: str, model_label: str, max_tokens: int) -> dict:
    messages = build_messages(template, item["context"], item["question"])
    resp = generate_from_llm(
        messages=messages,
        model=model_label,
        max_tokens=max_tokens
    )
    text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")

    objs = parse_json_objects(text)
    if not objs:
        logging.warning(f"id={item['id']} で JSON 抜き出せず")
        return {"id": item["id"], "context": item["context"], "qa": []}
    
    gen = objs[0]
    content = {
        "id": item["id"],
        "context": item["context"],
        "question": item["question"],
        "answer": gen,
    }

    return content

def main():
    setup_logging()
    p = argparse.ArgumentParser(description="QA 生成用 CLI")
    p.add_argument("--base-model", type=Path, required=True,
                   help="Base GGUF モデルファイルパス") # ex: models/generator/gemma-2-9b-it/gguf/base.gguf
    p.add_argument("--template", type=str, required=True,
                   help="プロンプト Jinja2 テンプレート名 (.j2)")
    p.add_argument("--input", type=Path, required=True,
                   help="入力 JSON ファイルパス") # ex: data/JSQuAD/eval/baseline.jsonl, data/JSQuAD/eval/[model_name]/lora/[date]/generated.jsonl
    p.add_argument("--output-dir", type=Path, required=True,
                   help="出力ディレクトリ")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--n_gpu_layers", type=int, default=None,
                   help="GPU レイヤ数")

    args = p.parse_args()

    start_llama_server(str(args.base_model), str(args.lora_model) if args.lora_model else None, args.n_gpu_layers)
    model_label = args.base_model.parts[2]
    
    dataset = read_jsonl(args.input)
    # dataset = dataset[:3]  # デバッグ用に最初の10件だけ処理

    results = []
    for item in tqdm(dataset, desc="QA 生成中"):
        retries = 0
        success = False
        while retries < MAX_RETRIES and not success:
            try:
                content = process_item(
                    item, args.template, model_label, args.max_tokens
                )
                results.append(content)
                success = True
            except Exception as e:
                retries += 1
                logging.error(f"id={item['id']} でエラー ({retries}/{MAX_RETRIES}): {e}")
                logging.debug(traceback.format_exc())
                if retries < MAX_RETRIES:
                    logging.info("再試行します…")
        if not success:
            logging.warning(f"id={item['id']} は最大試行回数に達したためスキップします")

    # 出力パスを決定
    dname = args.base_model.parts[1]
    mname = model_label
    match args.base_model.parts[-1]:
        case "baseline.jsonl":
            out_path = args.output_dir / dname / "baseline" / mname / "prediction.jsonl"
        case "generated.jsonl":
            gmname = args.input.parts[3]
            date_str = args.input.parts[4]
            out_path = args.output_dir / dname / "generated" / gmname / date_str / mname / "prediction.jsonl"
    

    write_jsonl(out_path, results)

    # サーバー停止
    stop_llama_server()

if __name__ == "__main__":
    main()
