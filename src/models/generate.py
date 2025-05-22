import json
import random
import argparse
import datetime
import logging
import traceback
from pathlib import Path
from tqdm import tqdm

from ..myutils.api import start_llama_server, stop_llama_server, generate_from_llm
from ..myutils.parsing import build_messages, parse_json_objects
from ..myutils.io import read_jsonl, write_jsonl, write_json
from ..myutils.logging import setup_logging

MAX_RETRIES = 10 # 最大リトライ回数

def get_few_shot_examples(few_shot_input: Path, shot_num: int) -> list:
    if few_shot_input is None:
        return []
    
    dataset = read_jsonl(few_shot_input)
    dataset = random.sample(dataset, shot_num) if len(dataset) > shot_num else dataset
    
    examples = []
    for item in dataset:
        shot = {
            "question": item["question"],
            "answer": item["answer"]
        }
        examples.append(json.dumps(shot, ensure_ascii=False))
    
    return examples

def process_item(item: dict, template: str, model_label: str, max_tokens: int, few_shots: str = None) -> dict:
    messages = build_messages(template, context=item["context"], examples=few_shots)
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
    content = None
    if "question" in gen and "answer" in gen:
        content = {
            "id": item["id"],
            "context": item["context"],
            "question": gen["question"],
            "answer": gen["answer"],
        }

    return content

def main():
    setup_logging()
    p = argparse.ArgumentParser(description="QA 生成用 CLI")
    p.add_argument("--base-model", type=Path, required=True,
                   help="Base GGUF モデルファイルパス") # ex: models/generator/gemma-2-9b-it/gguf/base.gguf
    p.add_argument("--lora-model", type=Path,
                   help="LoRA 適用後モデルファイルパス（オプション）") # ex: models/generator/gemma-2-9b-it/gguf/2025-05-22/lora.gguf
    
    p.add_argument("--template", type=str, required=True,
                   help="プロンプト Jinja2 テンプレート名 (.j2)")
    
    p.add_argument("--input", type=Path, required=True,
                   help="入力 JSON ファイルパス")
    p.add_argument("--few-shot-input", type=Path, default=None,
                   help="Few-Shot用の入力 JSON ファイルパス")
    p.add_argument("--shot-num", type=int, default=None,
                   help="Few-Shot のサンプル数")
    
    p.add_argument("--output-dir", type=Path, required=True,
                   help="出力ディレクトリ")
    
    p.add_argument("--max-tokens", type=int, default=200,
                   help="最大トークン数")
    p.add_argument("--n_gpu_layers", type=int, default=None,
                   help="GPU レイヤ数")

    args = p.parse_args()

    start_llama_server(str(args.base_model), str(args.lora_model) if args.lora_model else None, args.n_gpu_layers)
    model_label = args.base_model.parts[2]
    
    dataset = read_jsonl(args.input)
    # dataset = dataset[:3]  # デバッグ用に最初の10件だけ処理

    few_shots = None
    if args.few_shot_input is not None:
        few_shots = get_few_shot_examples(args.few_shot_input, args.shot_num)
        few_shots = "\n".join(few_shots)

    results = []
    for item in tqdm(dataset, desc="QA 生成中"):
        retries = 0
        success = False
        while retries < MAX_RETRIES and not success:
            try:
                content = process_item(
                    item, args.template, model_label, args.max_tokens, few_shots=few_shots
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
    mname = model_label
    today = datetime.datetime.now().strftime("%Y%m%d")
    if args.lora_model:
        date_str = args.lora_model.parts[4]
        out_path = args.output_dir / mname / "lora" / date_str / today / "generated.jsonl"
    else:
        out_path = args.output_dir / mname / "base" / today / "generated.jsonl"

    write_jsonl(out_path, results)

    # サーバー停止
    stop_llama_server()

    # 引数の内容を全て config.json に保存
    config = {
        "base_model": str(args.base_model),
        "lora_model": str(args.lora_model) if args.lora_model else None,
        "template": args.template,
        "input": str(args.input),
        "few_shot_input": str(args.few_shot_input) if args.few_shot_input else None,
        "shot_num": args.shot_num if args.shot_num else None,
        "output_dir": str(args.output_dir),
        "max_tokens": args.max_tokens,
        "n_gpu_layers": args.n_gpu_layers
    }
    config_path = out_path.parent / "config.json"
    write_json(config_path, config)

if __name__ == "__main__":
    main()
