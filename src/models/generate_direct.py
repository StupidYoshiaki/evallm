import json
import random
import argparse
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import asyncio
import os
import time

# --- 必要なライブラリのインポート ---
import torch
from llama_cpp import Llama 

# --- myutilsからのインポート ---
# api.pyからのインポートは不要になる
from ..myutils.parsing import build_messages, parse_json_objects 
from ..myutils.io import read_jsonl, write_jsonl, write_config
from ..myutils.logging import setup_logging

# --- 定数 ---
MAX_RETRIES = 10
RETRY_DELAY = 2

def get_few_shot_examples(few_shot_input: Path, shot_num: int) -> str | None:
    """Few-shotのサンプルを読み込み、単一の文字列に結合する"""
    if few_shot_input is None or shot_num is None or shot_num == 0:
        return None
    
    dataset = read_jsonl(few_shot_input)
    if not dataset: return None
        
    random.seed(42)  # 再現性のためにシードを固定
    dataset_sample = random.sample(dataset, min(shot_num, len(dataset)))
    
    examples = []
    for item in dataset_sample:
        shot = {
            "question": item.get("question", ""),
            "answer": item.get("answer", "")
        }
        examples.append(json.dumps(shot, ensure_ascii=False))
    
    return "\n".join(examples) if examples else None

async def generate_and_parse_with_retry(
    item: dict,
    llm: Llama, # httpx.AsyncClient の代わりに Llama オブジェクトを受け取る
    template: str,
    max_tokens: int,
    few_shots: str | None,
    temperature: float
) -> dict | None:
    """
    1個のアイテムに対してLLMモデルで生成とパースを行い、失敗した場合はリトライする。
    """
    retries = 0
    context_value = item.get("context")
    # このワーカー関数に渡される前にプロンプトが構築される方が効率的
    messages = build_messages(template, context=context_value, examples=few_shots)
    
    while retries < MAX_RETRIES:
        try:
            # Llamaオブジェクトのメソッドを直接、同期的に呼び出す
            # この呼び出しはブロッキングなので、asyncio.to_threadで別スレッドで実行
            resp_data = await asyncio.to_thread(
                llm.create_chat_completion,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|eot_id|>", "<|im_end|>"] # 一般的な停止トークン
            )

            text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed_json_list = parse_json_objects(text)
            
            if parsed_json_list:
                gen = parsed_json_list[0]
                if "question" in gen and "answer" in gen:
                    logging.info(f"id={item.get('id')} の処理に成功しました。")
                    return {
                        "id": item.get("id"),
                        "context": item.get("context"),
                        "question": gen["question"],
                        "answer": gen["answer"],
                    }
                else:
                    raise ValueError("生成されたJSONに必要なキーがありません。")
            else:
                logging.warning(f"レスポンステキストから有効なJSONを抽出できませんでした。Text: {text}")
                raise ValueError("生成されたテキストから有効なJSONを抽出できませんでした。")

        except Exception as e:
            retries += 1
            logging.warning(f"id={item.get('id')} でエラー ({retries}/{MAX_RETRIES}): {e}")
            if retries < MAX_RETRIES:
                logging.info(f"{RETRY_DELAY}秒後に再試行します…")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logging.error(f"id={item.get('id')} は最大試行回数に達したためスキップします。")
                return None
    return None


async def main():
    """非同期処理のメインロジック"""
    parser = argparse.ArgumentParser(description="QA生成スクリプト（LLM直接呼び出し版）")
    # サーバーを起動しないので、モデルロード用の引数に戻す
    parser.add_argument("--base-model", type=Path, required=True, help="ベースモデルのGGUFファイルパス")
    parser.add_argument("--lora-model", type=Path, default=None, help="適用するLoRAアダプタのGGUFファイルパス（オプション）")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPUにオフロードするレイヤー数 (-1で全て)")
    parser.add_argument("--n-ctx", type=int, default=2048, help="コンテキストサイズ")
    
    # 既存の引数
    parser.add_argument("--template", type=str, required=True, help="プロンプト Jinja2 テンプレート名 (.j2)")
    parser.add_argument("--input", type=Path, required=True, help="入力 JSON ファイルパス")
    parser.add_argument("--few-shot-input", type=Path, default=None, help="Few-Shot用の入力 JSON ファイルパス")
    parser.add_argument("--shot-num", type=int, default=0, help="Few-Shot のサンプル数")
    parser.add_argument("--output-dir", type=Path, required=True, help="出力ディレクトリ")
    parser.add_argument("--max-tokens", type=int, default=200, help="最大トークン数")
    parser.add_argument("--parallel", type=int, default=8, help="並列処理数（ワーカースレッド数）")
    parser.add_argument("--temperature", type=float, default=0.1, help="生成時の温度")
    parser.add_argument("--log-filename", type=Path, default=None, help="ログファイル名")
    parser.add_argument("--log-type", type=str, default="info", choices=["debug", "info"], help="ログレベル")
    
    args = parser.parse_args()
    setup_logging(filename=args.log_filename, log_type=args.log_type.upper())

    # --- モデルを一度だけロードする ---
    logging.info(f"モデルをロードしています... Path: {args.base_model}")
    try:
        llm = Llama(
            model_path=str(args.base_model),
            lora_path=str(args.lora_model) if args.lora_model else None,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx,
            n_threads=os.cpu_count(), # CPUスレッド数を最大化
            n_batch=512, # プロンプト処理のバッチサイズ
            verbose=False # llama.cppの詳細ログは無効化
        )
        logging.info("モデルのロードが完了しました。")
    except Exception as e:
        logging.error(f"モデルのロードに失敗しました: {e}")
        return

    dataset = read_jsonl(args.input)
    if not dataset:
        logging.error("入力データが空か、読み込めませんでした。")
        return
        
    few_shots = get_few_shot_examples(args.few_shot_input, args.shot_num)

    # --- メインループ（並列処理） ---
    results = []
    for i in tqdm(range(0, len(dataset), args.parallel), desc="QA 生成中"):
        batch = dataset[i:i+args.parallel]
        
        tasks = [
            generate_and_parse_with_retry(
                item, llm, args.template, args.max_tokens, few_shots, args.temperature
            ) 
            for item in batch
        ]
        
        batch_results = await asyncio.gather(*tasks)
        results.extend([res for res in batch_results if res is not None])

    # --- stop_llama_server の呼び出しは不要 ---

    # 出力パス決定とファイル書き込み
    model_name = args.base_model.stem
    today = datetime.datetime.now().strftime("%Y%m%d%H%M")
    out_dir = args.output_dir / model_name / today
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generated.jsonl"
    write_jsonl(out_path, results)
    
    # 設定ファイル保存
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_config(out_dir, config)
    logging.info(f"処理が完了しました。出力先: {out_path}")

if __name__ == "__main__":
    main() # asyncio.run は不要、main関数を同期的に呼び出す
