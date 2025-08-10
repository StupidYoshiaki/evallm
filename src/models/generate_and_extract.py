import json
import random
import argparse
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import asyncio
import httpx
import uuid

# --- 追加するインポート ---
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
# -------------------------

from ..myutils.api import start_llama_server, stop_all_llama_servers, generate_from_llm, handle_error
from ..myutils.parsing import build_messages, parse_json_objects 
from ..myutils.io import read_jsonl, write_jsonl, write_config
from ..myutils.logging import setup_logging
from ..myutils.extractor import extract_single_answer, setup_bert_model

MAX_RETRIES = 100 # 最大リトライ回数
RETRY_DELAY = 2 # 再試行までの待機時間（秒）

def get_few_shot_examples(few_shot_input: Path, shot_num: int) -> str | None:
    """Few-shotのサンプルを読み込み、単一の文字列に結合する"""
    if few_shot_input is None or shot_num is None or shot_num == 0:
        return None
    
    dataset = read_jsonl(few_shot_input)
    random.seed(42)
    dataset = random.sample(dataset, min(shot_num, len(dataset)))
    
    examples = []
    for item in dataset:
        shot = {"question": item["question"]} # 質問のみをFew-shotに含める
        examples.append(json.dumps(shot, ensure_ascii=False))
    
    return "\n".join(examples)

async def generate_and_extract_with_retry(
    item: dict,
    client: httpx.AsyncClient,
    port: int,
    template: str,
    model_label: str,
    max_tokens: int,
    few_shots: str | None,
    bert_model,
    bert_tokenizer,
    device,
    temperature: float = 0.7
) -> dict | None:
    """
    LLMで質問を生成し、BERTで回答を抽出する。失敗した場合はリトライする。
    """
    retries = 0
    context_value = item.get("context")
    while retries < MAX_RETRIES:
        try:
            # 1. LLMで質問を生成
            messages = build_messages(template, context=context_value, examples=few_shots)
            resp_data = await generate_from_llm(
                client=client,
                port=port,
                messages=messages,
                model=model_label,
                max_tokens=max_tokens,
                temperature=temperature
            )
            question = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # デバッグ用に生成質問をログ出力
            logging.debug(f"生成された質問: {question}")

            # 2. BERTで回答を抽出 (CPU/GPUバウンドな処理なので別スレッドで実行)
            extracted_answer, score = await asyncio.to_thread(
                extract_single_answer,
                question,
                context_value,
                bert_model,
                bert_tokenizer,
                device
            )

            # 3. 回答が抽出できたかチェック
            if not extracted_answer:
                logging.info(f"抽出された回答が空です。質問: {question}")
                raise ValueError("BERTが回答を抽出できませんでした。質問を再生成します。")

            # 成功
            logging.info(f"id={item.get('id')} の処理に成功しました。")
            return {
                "id": item.get("id") or str(uuid.uuid4()),
                "context": context_value,
                "question": question,
                "answer": extracted_answer,
            }

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


async def main():
    """非同期処理のメインロジック"""
    p = argparse.ArgumentParser(description="QA 生成・抽出パイプライン CLI")
    # LLM関連
    p.add_argument("--base-model", type=Path, required=True, help="Base GGUF モデルファイルパス")
    p.add_argument("--lora-model", type=Path, default=None, help="LoRA 適用後モデルファイルパス（オプション）")
    p.add_argument("--template", type=str, required=True, help="プロンプト Jinja2 テンプレート名 (.j2)")
    p.add_argument("--n-gpu-layers", type=int, default=42, help="GPU レイヤ数")
    p.add_argument("--n-ctx", type=int, default=2048, help="コンテキスト長 (n_ctx)")
    p.add_argument("--temperature", type=float, default=0.7, help="生成時の温度")
    p.add_argument("--port", type=int, default=8080, help="llama-serverのポート番号")
    
    # BERT関連
    p.add_argument("--bert-model", type=Path, required=True, help="回答抽出用のBERTモデルのパス")
    p.add_argument("--bert-gpu-id", type=int, default=0, help="BERTモデルが使用するGPU ID")

    # データ・実行関連
    p.add_argument("--input", type=Path, required=True, help="入力 JSON ファイルパス")
    p.add_argument("--output-dir", type=Path, required=True, help="出力ディレクトリ")
    p.add_argument("--few-shot-input", type=Path, default=None, help="Few-Shot用の入力 JSON ファイルパス")
    p.add_argument("--shot-num", type=int, default=0, help="Few-Shot のサンプル数")
    p.add_argument("--max-tokens", type=int, default=200, help="最大トークン数")
    p.add_argument("--parallel", type=int, default=8, help="並列処理数")
    
    # ログ関連
    p.add_argument("--log-filename", type=Path, default=None, help="ログファイル名")
    p.add_argument("--log-type", type=str, default="info", choices=["debug", "info"], help="ログレベル")
    args = p.parse_args()

    setup_logging(filename=args.log_filename, log_type=args.log_type)

    # 1. BERTモデルのセットアップ
    bert_model, bert_tokenizer, device = setup_bert_model(args.bert_model, args.bert_gpu_id)

    # 2. LLMサーバー起動
    start_llama_server(
        str(args.base_model), 
        args.port,
        args.n_gpu_layers,
        args.parallel,
        n_ctx=args.n_ctx,
        lora_path=str(args.lora_model) 
    )
    model_label = args.base_model.parts[2]
    
    dataset = read_jsonl(args.input)
    few_shots = get_few_shot_examples(args.few_shot_input, args.shot_num)

    # 3. メイン処理
    results = []
    async with httpx.AsyncClient(timeout=300.0) as client:
        for i in tqdm(range(0, len(dataset), args.parallel), desc="QA 生成・抽出中"):
            batch = dataset[i:i+args.parallel]
            
            tasks = [
                generate_and_extract_with_retry(
                    item, client, args.port, args.template, model_label, args.max_tokens, few_shots, 
                    bert_model, bert_tokenizer, device, args.temperature
                ) 
                for item in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend([res for res in batch_results if res is not None])

    # 4. 出力
    mname = model_label
    today = datetime.datetime.now().strftime("%Y%m%d%H%M")
    out_dir = args.output_dir / mname / today
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generated.jsonl"
    write_jsonl(out_path, results)

    # 5. LLMサーバー停止
    stop_all_llama_servers()

    # 6. 設定ファイル保存
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    write_config(out_dir, config)

if __name__ == "__main__":
    asyncio.run(main())
