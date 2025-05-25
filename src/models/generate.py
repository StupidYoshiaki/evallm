import sys
import json
import random
import argparse
import datetime
import logging
import traceback
from pathlib import Path
from tqdm import tqdm
import asyncio
import httpx

# 以下のimportは、お客様のプロジェクト構成に合わせてください
from ..myutils.api import start_llama_server, stop_llama_server, generate_from_llm
from ..myutils.parsing import build_messages, parse_json_objects # ここでは堅牢なパース関数が使われる
from ..myutils.io import read_jsonl, write_jsonl, write_json
from ..myutils.logging import setup_logging

MAX_RETRIES = 10 # 最大リトライ回数
RETRY_DELAY = 1 # 再試行までの待機時間（秒）

def get_few_shot_examples(few_shot_input: Path, shot_num: int) -> str | None:
    """Few-shotのサンプルを読み込み、単一の文字列に結合する"""
    if few_shot_input is None or shot_num is None or shot_num == 0:
        return None
    
    dataset = read_jsonl(few_shot_input)
    # shot_numがデータセットのサイズより大きい場合、全データを使う
    dataset = random.sample(dataset, min(shot_num, len(dataset)))
    
    examples = []
    for item in dataset:
        shot = {
            "question": item["question"],
            "answer": json.dumps({"question": item["question"], "answer": item["answer"]}, ensure_ascii=False)
        }
        examples.append(json.dumps(shot, ensure_ascii=False))
    
    return "\n".join(examples)

async def generate_and_parse_with_retry(
    item: dict,
    client: httpx.AsyncClient,
    template: str,
    model_label: str,
    max_tokens: int,
    few_shots: str | None,
    temperature: float = 0.7
) -> dict | None:
    """
    1個のアイテムに対してLLMからの生成とJSONパースを行い、失敗した場合はリトライする非同期ワーカー関数。
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            messages = build_messages(template, context=item["context"], examples=few_shots)
            resp_data = await generate_from_llm(
                client=client,
                messages=messages,
                model=model_label,
                max_tokens=max_tokens,
                temperature=temperature
            )
            text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # 堅牢なパース関数でJSONを抽出
            parsed_json_list = parse_json_objects(text)
            
            if parsed_json_list:
                gen = parsed_json_list[0]
                if "question" in gen and "answer" in gen:
                    logging.info(f"id={item['id']} の処理に成功しました。")
                    return {
                        "id": item["id"],
                        "context": item["context"],
                        "question": gen["question"],
                        "answer": gen["answer"],
                    }
                else:
                    raise ValueError("生成されたJSONに必要な'question'または'answer'キーがありません。")
            else:
                # logging.info(f"レスポンステキスト: {text}")
                raise ValueError("生成されたテキストから有効なJSONを抽出できませんでした。")

        except Exception as e:
            retries += 1
            logging.warning(f"id={item['id']} でエラー ({retries}/{MAX_RETRIES}): {e}")
            if retries < MAX_RETRIES:
                logging.info(f"{RETRY_DELAY}秒後に再試行します…")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logging.error(f"id={item['id']} は最大試行回数に達したため強制終了します。")
                sys.exit(0)  # 強制終了
    return None


async def main():
    """非同期処理のメインロジック"""
    setup_logging()
    p = argparse.ArgumentParser(description="QA 生成用 CLI")
    p.add_argument("--base-model", type=Path, required=True, help="Base GGUF モデルファイルパス")
    p.add_argument("--lora-model", type=Path, default=None, help="LoRA 適用後モデルファイルパス（オプション）")
    p.add_argument("--template", type=str, required=True, help="プロンプト Jinja2 テンプレート名 (.j2)")
    p.add_argument("--input", type=Path, required=True, help="入力 JSON ファイルパス")
    p.add_argument("--few-shot-input", type=Path, default=None, help="Few-Shot用の入力 JSON ファイルパス")
    p.add_argument("--shot-num", type=int, default=0, help="Few-Shot のサンプル数")
    p.add_argument("--output-dir", type=Path, required=True, help="出力ディレクトリ")
    p.add_argument("--max-tokens", type=int, default=200, help="最大トークン数")
    p.add_argument("--n-gpu-layers", type=int, default=42, help="GPU レイヤ数")
    p.add_argument("--parallel", type=int, default=8, help="並列処理数")
    p.add_argument("--n-ctx", type=int, default=2048, help="コンテキスト長 (n_ctx)")
    p.add_argument("--temperature", type=float, default=0.7, help="生成時の温度（デフォルトは0.1）")
    args = p.parse_args()

    # サーバー起動
    start_llama_server(
        str(args.base_model), 
        str(args.lora_model) if args.lora_model else None, 
        args.n_gpu_layers,
        args.parallel,
        n_ctx=args.n_ctx
    )
    model_label = args.base_model.parts[2]
    
    dataset = read_jsonl(args.input)
    # dataset = dataset[:10] # デバッグ用
    
    few_shots = get_few_shot_examples(args.few_shot_input, args.shot_num)

    results = []
    async with httpx.AsyncClient(timeout=300.0) as client:
        for i in tqdm(range(0, len(dataset), args.parallel), desc="QA 生成中"):
            batch = dataset[i:i+args.parallel]
            
            tasks = [
                generate_and_parse_with_retry(
                    item, client, args.template, model_label, args.max_tokens, few_shots, args.temperature
                ) 
                for item in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend([res for res in batch_results if res is not None])

    # 出力パス決定とファイル書き込み
    mname = model_label
    today = datetime.datetime.now().strftime("%Y%m%d%H%M")
    out_dir = args.output_dir / mname / today
    # if args.lora_model:
    #     # パス構造が深くなる可能性を考慮し、存在確認と作成を行う
    #     out_dir = args.output_dir / mname / "lora" / args.lora_model.parent.name / today
    # else:
    #     out_dir = args.output_dir / mname / "base" / today
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generated.jsonl"
    write_jsonl(out_path, results)

    # サーバー停止
    stop_llama_server()

    # 設定ファイル保存
    config = {
        "base_model": str(args.base_model),
        "lora_model": str(args.lora_model) if args.lora_model else None,
        "template": args.template,
        "input": str(args.input),
        "few_shot_input": str(args.few_shot_input) if args.few_shot_input else None,
        "shot_num": args.shot_num,
        "output_dir": str(args.output_dir),
        "max_tokens": args.max_tokens,
        "n_gpu_layers": args.n_gpu_layers,
        "parallel": args.parallel,
        "n_ctx": args.n_ctx,
        "temperature": args.temperature,
    }
    config_path = out_path.parent / "config.json"
    write_json(config_path, config)

if __name__ == "__main__":
    asyncio.run(main())
