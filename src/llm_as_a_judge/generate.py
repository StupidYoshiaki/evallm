import json
import argparse
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import asyncio
import httpx
import signal
import re # 正規表現モジュールをインポート

# myutilsパッケージが親ディレクトリにあると仮定
from ..myutils.api import start_llama_server, stop_all_llama_servers, generate_from_llm, handle_error
from ..myutils.parsing import build_messages, parse_json_objects
from ..myutils.io import read_jsonl, write_jsonl, write_config
from ..myutils.logging import setup_logging

MAX_RETRIES = 10
RETRY_DELAY = 5

async def generate_and_parse_with_retry(
    item: dict,
    client: httpx.AsyncClient,
    port: int,
    template: str,
    model_label: str,
    max_tokens: int,
    temperature: float = 0.1
) -> dict | None:
    """
    1個のアイテムに対してLLMによる評価生成と解析を行い、失敗した場合はリトライする非同期ワーカー関数。
    <|think|>トークンとjudge JSONを別々に抽出する。
    """
    history = item.get("history")
    response_A = item.get("human_ref_A")
    response_B = item.get("human_ref_B")
    item_id = item.get("post_id", "N/A")

    retries = 0
    while retries < MAX_RETRIES:
        try:
            messages = build_messages(
                template,
                history=history,
                response_A=response_A,
                response_B=response_B
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

            # 1. <think> ブロックから思考プロセスを抽出
            thinking_text = ""
            think_match = re.search(r"<think>(.*?)<\/think>", text, re.DOTALL)
            if think_match:
                thinking_text = think_match.group(1).strip()
            else:
                logging.warning(f"id={item_id} のレスポンスに <think> トークンが見つかりませんでした。")

            # 2. テキストから judge JSON を抽出
            parsed_json_list = parse_json_objects(text)
            
            for parsed_json in parsed_json_list:
                # 3. 期待する 'judge' オブジェクトを持っているか検証
                if "is_better" in parsed_json and "criteria" in parsed_json and "reason" in parsed_json:
                    logging.info(f"id={item_id} の処理に成功しました。")
                    
                    # 4. 抽出した思考プロセスとjudgeオブジェクトを結合して結果を格納
                    item["prediction"] = {
                        "prompt": messages[0]["content"], # プロンプトの内容
                        "thinking": thinking_text, # 抽出した思考プロセス
                        "judge": parsed_json
                    }
                    return item

            # 有効なJSONが見つからなかった場合
            logging.warning(f"id={item_id} のレスポンスから有効なjudge JSONを抽出できませんでした。レスポンステキスト: {text}")
            raise ValueError("生成されたテキストから有効なJSONを抽出できませんでした。")

        except Exception as e:
            retries += 1
            logging.warning(f"id={item_id} でエラー ({retries}/{MAX_RETRIES}): {e}")
            if retries < MAX_RETRIES:
                logging.info(f"{RETRY_DELAY}秒後に再試行します…")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logging.error(f"id={item_id} は最大試行回数に達したためスキップします。")
                return None
    return None

async def main_async():
    """非同期処理のメインロジック"""
    p = argparse.ArgumentParser(description="LLM as a Judge 実行用 CLI")
    p.add_argument("--base-model", type=Path, required=True, help="Base GGUF モデルファイルパス")
    p.add_argument("--lora-model", type=Path, default=None, help="LoRA 適用後モデルファイルパス（オプション）")
    p.add_argument("--template", type=str, required=True, help="プロンプト Jinja2 テンプレート名 (.j2)")
    p.add_argument("--input", type=Path, required=True, help="入力 JSONL ファイルパス")
    p.add_argument("--output-dir", type=Path, required=True, help="出力ディレクトリ")
    p.add_argument("--max-tokens", type=int, default=32768, help="最大生成トークン数")
    p.add_argument("--n-gpu-layers", type=int, default=42, help="GPUにオフロードするレイヤ数 (-1はすべて)")
    p.add_argument("--parallel", type=int, default=8, help="並列処理数")
    p.add_argument("--n-ctx", type=int, default=40960, help="コンテキスト長 (n_ctx)")
    p.add_argument("--port", type=int, default=8085, help="llama-serverのポート番号")
    p.add_argument("--temperature", type=float, default=0.1, help="生成時の温度（デフォルトは0.1）")
    p.add_argument("--log-filename", type=Path, default=None, help="ログファイル名")
    p.add_argument("--log-type", type=str, default="info", choices=["debug", "info"], help="ログレベル")
    args = p.parse_args()

    setup_logging(filename=args.log_filename, log_type=args.log_type)

    start_llama_server(
        str(args.base_model), 
        args.port,
        args.n_gpu_layers,
        args.parallel,
        n_ctx=args.n_ctx,
        lora_path=args.lora_model 
    )
    model_label = args.base_model.name
    
    dataset = read_jsonl(args.input)
    dataset = dataset[:10]  # デバッグ用に最初の10件だけ処理
    
    results = []
    async with httpx.AsyncClient(timeout=300.0) as client:
        for i in tqdm(range(0, len(dataset), args.parallel), desc="LLMによる評価を実行中"):
            batch = dataset[i:i+args.parallel]
            
            tasks = [
                generate_and_parse_with_retry(
                    item, client, args.port, args.template, model_label, args.max_tokens, args.temperature
                ) 
                for item in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend([res for res in batch_results if res is not None])

    mname = args.base_model.parts[2]
    today = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / mname / today
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"judged_output.jsonl"
    write_jsonl(out_path, results)
    logging.info(f"評価結果を保存しました: {out_path}")

    stop_all_llama_servers()

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_config(out_dir, config)
    logging.info(f"実行設定を保存しました: {out_dir / 'config.json'}")

def main():
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
        stop_all_llama_servers()

if __name__ == "__main__":
    main()
