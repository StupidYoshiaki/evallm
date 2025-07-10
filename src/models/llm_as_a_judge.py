import json
import random
import argparse
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import asyncio
import httpx

# 参照するユーティリティ関数をインポート
from ..myutils.api import start_llama_server, stop_llama_server, generate_from_llm, handle_error
from ..myutils.parsing import build_messages, parse_json_objects
from ..myutils.io import read_jsonl, write_jsonl, write_config
from ..myutils.logging import setup_logging
import uuid

MAX_RETRIES = 100 # 最大リトライ回数
RETRY_DELAY = 2 # 再試行までの待機時間（秒）

async def judge_qa_pair_with_retry(
    item: dict,
    client: httpx.AsyncClient,
    template: str,
    model_label: str,
    max_tokens: int,
    temperature: float
) -> dict | None:
    """
    1個のQAペアをLLMに判定させ、失敗した場合はリトライする非同期ワーカー関数。
    """
    retries = 0
    context = item.get("context")
    question = item.get("question")

    # 判定に必要なキーがない場合はスキップ
    if not context or not question:
        logging.warning(f"id={item.get('id')} には 判定に必要な情報がありません。スキップします。")
        return None

    while retries < MAX_RETRIES:
        try:
            # LLM-as-a-Judge用のプロンプトを構築
            messages = build_messages(
                template,
                context=context,
                question=question,
            )
            
            # LLMに判定を依頼
            resp_data = await generate_from_llm(
                client=client,
                messages=messages,
                model=model_label,
                max_tokens=max_tokens,
                temperature=temperature
            )
            text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # LLMの回答をパース
            # decision_text = text.strip()
            # is_appropriate = None

            # LLMの回答（JSON）をパース
            try:
                decision_data = parse_json_objects(text.strip())[0]
                decision_text = decision_data.get("is_appropriate")
                if decision_text is None:
                    raise ValueError("LLMの応答に 'is_appropriate' キーがありません。")
            except json.JSONDecodeError as e:
                raise ValueError(f"LLMの応答をJSONとしてパースできませんでした: {text.strip()}") from e

            if decision_text.startswith("はい"):
                is_appropriate = True
            elif decision_text.startswith("いいえ"):
                is_appropriate = False
            else:
                # 予期しない回答の場合はリトライ
                raise ValueError(f"LLMが予期しない応答をしました: '{decision_text}'")
            
            # if "はい" in decision_text:
            #     is_appropriate = True
            # elif "いいえ" in decision_text:
            #     is_appropriate = False
            # else:
            #     # 予期しない回答の場合はリトライ
            #     raise ValueError(f"LLMが予期しない応答をしました: '{decision_text}'")

            logging.info(f"id={item.get('id')} の判定に成功しました: {'適切' if is_appropriate else '不適切'}")
            
            # 元のデータに判定結果を追加して返す
            new_item = item.copy()
            new_item['is_appropriate'] = is_appropriate
            return new_item

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
    p = argparse.ArgumentParser(description="LLM-as-a-Judge を用いた自己完結QA判定スクリプト")
    p.add_argument("--base-model", type=Path, required=True, help="Base GGUF モデルファイルパス")
    p.add_argument("--lora-model", type=Path, default=None, help="LoRA 適用後モデルファイルパス（オプション）")
    p.add_argument("--template", type=str, required=True, help="プロンプト Jinja2 テンプレート名 (.j2)")
    p.add_argument("--input", type=Path, required=True, help="入力 JSONL ファイルパス")
    p.add_argument("--output-dir", type=Path, required=True, help="出力ディレクトリ")
    p.add_argument("--max-tokens", type=int, default=100, help="最大生成トークン数（「はい」「いいえ」なので小さくて良い）")
    p.add_argument("--n-gpu-layers", type=int, default=42, help="GPU レイヤ数")
    p.add_argument("--parallel", type=int, default=8, help="並列処理数")
    p.add_argument("--n-ctx", type=int, default=2048, help="コンテキスト長 (n_ctx)")
    p.add_argument("--temperature", type=float, default=0.0, help="生成時の温度（判定タスクなので0.0を推奨）")
    p.add_argument("--log-filename", type=Path, default=None, help="ログファイル名")
    p.add_argument("--log-type", type=str, default="info", choices=["debug", "info"], help="ログレベル")
    args = p.parse_args()

    # ログ設定
    setup_logging(filename=args.log_filename, log_type=args.log_type)

    # LLMサーバー起動
    start_llama_server(
        str(args.base_model), 
        str(args.lora_model) if args.lora_model else None, 
        args.n_gpu_layers,
        args.parallel,
        n_ctx=args.n_ctx
    )
    model_label = args.base_model.parts[2]
    
    dataset = read_jsonl(args.input)
    dataset = dataset[:100] # デバッグ用に最初の100件のみ使用
    
    results = []
    async with httpx.AsyncClient(timeout=300.0) as client:
        for i in tqdm(range(0, len(dataset), args.parallel), desc="QAペア判定中"):
            batch = dataset[i:i+args.parallel]
            
            tasks = [
                judge_qa_pair_with_retry(
                    item, client, args.template, model_label, args.max_tokens, args.temperature
                ) 
                for item in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend([res for res in batch_results if res is not None])

    # サーバー停止
    stop_llama_server()

    # 判定結果に基づいてソート（自己完結(True)のものが先頭に来るように）
    results.sort(key=lambda x: x.get('is_appropriate', False), reverse=True)

    # 出力パス決定とファイル書き込み
    mname = model_label
    today = datetime.datetime.now().strftime("%Y%m%d%H%M")
    out_dir = args.output_dir / mname / today
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "judged_qa.jsonl"
    write_jsonl(out_path, results)

    # 設定ファイル保存
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    write_config(out_dir, config)

if __name__ == "__main__":
    asyncio.run(main())
