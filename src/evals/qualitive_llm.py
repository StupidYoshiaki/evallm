import json
import random
import argparse
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import asyncio
import httpx
import signal

from ..myutils.api import start_llama_server, stop_all_llama_servers, generate_from_llm, handle_error
from ..myutils.parsing import build_messages, parse_json_objects 
from ..myutils.io import read_jsonl, write_jsonl, write_config
from ..myutils.logging import setup_logging
import uuid

MAX_RETRIES = 100 # 最大リトライ回数
RETRY_DELAY = 5 # 再試行までの待機時間（秒）

async def generate_and_parse_with_retry(
    item: dict,
    client: httpx.AsyncClient,
    port: int,
    template: str,
    model_label: str,
    max_tokens: int,
    few_shots: str | None,
    temperature: float = 0.7
) -> dict | None:
    """
    1個のアイテムに対してLLMからの生成とJSONパースを行い、失敗した場合はリトライする非同期ワーカー関数。
    """
    context = item.get("context")
    question = item.get("question")
    answer = item.get("answer")

    retries = 0
    while retries < MAX_RETRIES:
        try:
            messages = build_messages(
                template,
                context=context,
                question=question,
                answer=answer,
                examples=few_shots
            )
            resp_data = await generate_from_llm(
                client=client,
                port=port,
                messages=messages,
                model=model_label,
                max_tokens=max_tokens,
                temperature=temperature
            )
            text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # 堅牢なパース関数でJSONを抽出
            parsed_json_list = parse_json_objects(text)
            
            for parsed_json in parsed_json_list:
                if "question_clarity" in parsed_json and "context_grounding" in parsed_json:
                    logging.info(f"id={item.get('id')} の処理に成功しました。")
                    item["question_clarity"] = parsed_json["question_clarity"]
                    item["context_grounding"] = parsed_json["context_grounding"]
                    return item
                else:
                    raise ValueError("生成されたJSONに必要な'question'または'answer'キーがありません。")
            else:
                logging.info(f"レスポンステキスト: {text}")
                raise ValueError("生成されたテキストから有効なJSONを抽出できませんでした。")

        except Exception as e:
            retries += 1
            logging.warning(f"id={item.get('id')} でエラー ({retries}/{MAX_RETRIES}): {e}")
            if retries < MAX_RETRIES:
                logging.info(f"{RETRY_DELAY}秒後に再試行します…")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logging.error(f"id={item.get('id')} は最大試行回数に達したため強制終了します。")
                handle_error() # 強制終了
    return None


async def main_async():
    """非同期処理のメインロジック"""
    p = argparse.ArgumentParser(description="QA 生成用 CLI")
    p.add_argument("--base-model", type=Path, required=True, help="Base GGUF モデルファイルパス")
    p.add_argument("--lora-model", type=Path, default=None, help="LoRA 適用後モデルファイルパス（オプション）")
    p.add_argument("--template", type=str, required=True, help="プロンプト Jinja2 テンプレート名 (.j2)")
    p.add_argument("--input", type=Path, required=True, help="入力 JSON ファイルパス")
    p.add_argument("--output-dir", type=Path, required=True, help="出力ディレクトリ")
    p.add_argument("--max-tokens", type=int, default=32768, help="最大トークン数")
    p.add_argument("--n-gpu-layers", type=int, default=42, help="GPU レイヤ数")
    p.add_argument("--parallel", type=int, default=8, help="並列処理数")
    p.add_argument("--n-ctx", type=int, default=40960, help="コンテキスト長 (n_ctx)")
    p.add_argument("--port", type=int, default=8080, help="llama-serverのポート番号")
    p.add_argument("--temperature", type=float, default=0, help="生成時の温度（デフォルトは0.1）")
    p.add_argument("--log-filename", type=Path, default=None, help="ログファイル名")
    p.add_argument("--log-type", type=str, default="info", choices=["debug", "info"], help="ログレベル")
    args = p.parse_args()

    # ログ設定
    setup_logging(filename=args.log_filename, log_type=args.log_type)

    # サーバー起動
    start_llama_server(
        str(args.base_model), 
        args.port,
        args.n_gpu_layers,
        args.parallel,
        n_ctx=args.n_ctx,
        lora_path=args.lora_model 
    )
    model_label = args.base_model.parts[2]
    
    dataset = read_jsonl(args.input)
    # dataset = dataset[:8] # デバッグ用
    
    results = []
    async with httpx.AsyncClient() as client:
        for i in tqdm(range(0, len(dataset), args.parallel), desc="QA 評価中"):
            batch = dataset[i:i+args.parallel]
            
            tasks = [
                generate_and_parse_with_retry(
                    item, client, args.port, args.template, model_label, args.max_tokens, args.temperature
                ) 
                for item in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend([res for res in batch_results if res is not None])

    # 出力パス決定とファイル書き込み
    today = datetime.datetime.now().strftime("%Y%m%d%H%M")
    out_dir = args.output_dir / "judged" / today
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "qualitive_llm.jsonl"
    write_jsonl(out_path, results)

    # サーバー停止
    stop_all_llama_servers()

    # 実行した学習設定をJSONファイルとして保存
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_config(out_dir, config)
    logging.info(f"学習設定を保存しました: {out_dir / 'config.json'}")

def main():
    # Ctrl-Cなどで中断された場合でもサーバーを確実に停止するためのハンドラ
    def signal_handler(signum, frame):
        logging.info("CTRL-C を検知しました。全てのサーバーを停止して終了します。")
        stop_all_llama_servers()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(main_async())
    except Exception as e:
        logging.error(f"メイン処理で予期せぬエラーが発生しました: {e}", exc_info=True)
    finally:
        # 正常終了時、エラー発生時ともに、必ずサーバーを停止する
        stop_all_llama_servers()

if __name__ == "__main__":
    main()
