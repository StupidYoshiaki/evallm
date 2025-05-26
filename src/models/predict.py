import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import asyncio
import httpx

from ..myutils.api import start_llama_server, stop_llama_server, generate_from_llm, handle_error
from ..myutils.parsing import build_messages
from ..myutils.io import read_jsonl, write_jsonl, write_json
from ..myutils.logging import setup_logging

MAX_RETRIES = 10 # 最大リトライ回数
RETRY_DELAY = 1 # 再試行までの待機時間（秒）

async def generate_and_parse_with_retry(
    item: dict, 
    client: httpx.AsyncClient, 
    template: str, 
    model_label: str, 
    max_tokens: int,
    temperature: float = 0.7
) -> dict | None:
    """
    1個のアイテムに対してLLMからの生成とJSONパースを行い、失敗した場合はリトライする。
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            # LLMにリクエストを送信
            messages = build_messages(template, question=item["question"], context=item["context"])
            resp_data = await generate_from_llm(
                client=client,
                messages=messages,
                model=model_label,
                max_tokens=max_tokens,
                temperature=temperature
            )
            text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "id": item["id"],
                "context": item["context"],
                "question": item["question"],
                "answer": text, # パースした結果のanswerを使う
            }
            
        except Exception as e:
            retries += 1
            logging.warning(f"id={item['id']} でエラー ({retries}/{MAX_RETRIES}): {e}")
            if retries < MAX_RETRIES:
                logging.info(f"{RETRY_DELAY}秒後に再試行します…")
                await asyncio.sleep(RETRY_DELAY) # サーバー負荷軽減のため少し待つ
            else:
                logging.error(f"id={item['id']} は最大試行回数に達したため強制終了します。")
                handle_error() # 強制終了
            
    return None

async def main():
    setup_logging()
    p = argparse.ArgumentParser(description="QA 生成用 CLI")
    p.add_argument("--base-model", type=Path, required=True, help="Base GGUF モデルファイルパス")
    p.add_argument("--template", type=str, required=True, help="プロンプト Jinja2 テンプレート名 (.j2)")
    p.add_argument("--input", type=Path, required=True, help="入力 JSON ファイルパス")
    p.add_argument("--output-dir", type=Path, required=True, help="出力ディレクトリ")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--n-gpu-layers", type=int, default=None, help="GPU レイヤ数")
    p.add_argument("--parallel", type=int, default=8, help="並列処理数")
    p.add_argument("--n-ctx", type=int, default=2048, help="コンテキスト長 (n_ctx)")
    p.add_argument("--temperature", type=float, default=0.7, help="生成の温度")

    args = p.parse_args()

    # parallel引数をサーバー起動時に渡す
    start_llama_server(str(args.base_model), n_gpu_layers=args.n_gpu_layers, parallel=args.parallel, n_ctx=args.n_ctx)
    model_label = args.base_model.parts[-3]
    
    dataset = read_jsonl(args.input)
    # dataset = dataset[:10]  # デバッグ用

    results = []
    # httpx.AsyncClientを一度だけ生成し、使い回す
    async with httpx.AsyncClient(timeout=300.0) as client:
        for i in tqdm(range(0, len(dataset), args.parallel), desc="QA 生成中"):
            batch = dataset[i:i+args.parallel]
            
            # バッチ内の各アイテムに対して、リトライ機能付きのワーカータスクを作成
            tasks = [
                generate_and_parse_with_retry(
                    item, client, args.template, model_label, args.max_tokens, args.temperature
                ) 
                for item in batch
            ]
            
            # バッチ内のタスクを並列実行
            batch_results = await asyncio.gather(*tasks)
            
            # Noneでない成功した結果だけをresultsに追加
            results.extend([res for res in batch_results if res is not None])

    dname = args.input.parts[1]
    mname = model_label
    match args.input.parts[-1]:
        case "baseline.jsonl":
            out_path = args.output_dir / dname / "baseline" / mname / "prediction.jsonl"
        case "generated.jsonl":
            gmname = args.input.parts[3]
            date_str = args.input.parts[4]
            out_path = args.output_dir / dname / "generated" / gmname / date_str / mname / "prediction.jsonl"
        case _:
            logging.error("不明な入力ファイル形式です")
            stop_llama_server()
            return
        
    write_jsonl(out_path, results)
    stop_llama_server()

    config_path = out_path.parent / "config.json"
    config = {
        "base_model": str(args.base_model),
        "template": args.template,
        "input": str(args.input),
        "max_tokens": args.max_tokens,
        "n_gpu_layers": args.n_gpu_layers,
        "parallel": args.parallel,
        "n_ctx": args.n_ctx,
    }
    write_json(config_path, config)


if __name__ == "__main__":
    asyncio.run(main())
