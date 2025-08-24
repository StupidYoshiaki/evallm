import os
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
import signal
from typing import Optional

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# 外部ユーティリティ関数のインポート（api.py, parsing.py, io.py, logging.pyが適切に配置されていると仮定）
from ..myutils.api import start_llama_server, stop_all_llama_servers, generate_from_llm, handle_error
from ..myutils.parsing import build_messages, parse_json_objects
from ..myutils.io import read_jsonl, write_jsonl, write_config
from ..myutils.logging import setup_logging
from ..myutils.extractor import extract_single_answer, setup_bert_model

# --- グローバル設定 ---
MAX_RETRIES = 100 # 1つのアイテム処理あたりの最大リトライ回数
RETRY_DELAY = 5  # 再試行までの待機時間（秒）

def get_few_shot_examples(few_shot_input: Path, shot_num: int) -> str | None:
    """Few-shotのサンプルを読み込み、単一の文字列に結合する"""
    if few_shot_input is None or shot_num is None or shot_num == 0:
        return None
    
    dataset = read_jsonl(few_shot_input)
    # shot_numがデータセットのサイズより大きい場合、全データを使う
    random.seed(42)  # 再現性のためにシードを固定
    dataset = random.sample(dataset, min(shot_num, len(dataset)))
    
    examples = []
    for item in dataset:
        shot = {
            "question": item["question"],
            "answer": item["answer"]
        }
        examples.append(json.dumps(shot, ensure_ascii=False))
    
    return "\n".join(examples)

async def generate_qa_pair(
    client: httpx.AsyncClient,
    port: int,
    template: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    context: str,
    hint: Optional[str] = None,
    few_shots: Optional[str] = None,
    bert_model: Optional[str] = None,
    bert_tokenizer: Optional[AutoTokenizer] = None,
    bert_device: Optional[torch.device] = None
) -> dict:
    """
    指定されたポートのLLMサーバーから単一のQAペアを生成し、パースするヘルパー関数。
    """
    # プロンプトに渡す変数を構築
    prompt_kwargs = {"context": context}
    if hint:
        prompt_kwargs["hint"] = hint
    if few_shots:
        prompt_kwargs["examples"] = few_shots

    messages = build_messages(template, **prompt_kwargs)
    
    resp_data = await generate_from_llm(
        client=client,
        port=port,
        messages=messages,
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature
    )
    text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")

    # BERTモデルが指定されている場合、回答を抽出
    if bert_model and model_name == "generator-model":
        question = text
        answer, _ = await asyncio.to_thread(
            extract_single_answer,
            question,
            context,
            bert_model,
            bert_tokenizer,
            bert_device
        )
        return {"question": question, "answer": answer}

    parsed_json_list = parse_json_objects(text)
    
    if parsed_json_list and "answer" in parsed_json_list[0] and "question" in parsed_json_list[0]:
        return parsed_json_list[0]
    else:
        logging.warning(f"ポート {port} からのレスポンスのパースに失敗しました。レスポンス: {text}")
        raise ValueError("生成されたテキストから有効なQA JSONを抽出できませんでした。")


async def directional_stimulus_pipeline_with_retry(
    item: dict,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    few_shots: Optional[str] = None,
    bert_model=None,
    bert_tokenizer=None,
    bert_device=None
) -> dict | None:
    """
    1. 小規模モデルでヒントQAを生成し、2. 大規模モデルで高品質なQAを生成するパイプライン。
    """
    retries = 0
    context = item.get("context")

    while retries < MAX_RETRIES:
        try:
            # --- Step 1: 小規模モデルでヒントを生成 ---
            hint_qa = await generate_qa_pair(
                client,
                args.generator_port,
                args.generator_template,
                "generator-model",
                args.max_tokens_gen,
                args.temperature_gen,
                context=context,
                hint=None,
                few_shots=few_shots,
                bert_model=bert_model,
                bert_tokenizer=bert_tokenizer,
                bert_device=bert_device
            )
            
            # ヒントを文字列として整形
            hint_str = str(hint_qa)

            # --- Step 2: 大規模モデルで高品質なQAを生成 ---
            refined_qa = await generate_qa_pair(
                client,
                args.refiner_port,
                args.refiner_template,
                "refiner-model",
                args.max_tokens_refine,
                args.temperature_refine,
                context=context,
                hint=hint_str,
                few_shots=None,
            )

            logging.info(f"id={item.get('id')} の処理に成功しました。")
            logging.debug(f"生成されたQAペア: {refined_qa}")
            return {
                "id": item.get("id") or str(uuid.uuid4()),
                "context": context,
                "hint_question": hint_qa["question"],
                "hint_answer": hint_qa["answer"],
                "question": refined_qa["question"],
                "answer": refined_qa["answer"],
            }

        except Exception as e:
            retries += 1
            logging.warning(f"id={item.get('id')} のパイプライン処理でエラー ({retries}/{MAX_RETRIES}): {e}")
            if retries < MAX_RETRIES:
                logging.info(f"{RETRY_DELAY}秒後に再試行します…")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logging.error(f"id={item.get('id')} は最大試行回数に達したため強制終了します。")
                handle_error() # 強制終了
    return None


async def main_async():
    """非同期処理のメインロジック"""
    p = argparse.ArgumentParser(description="方向性刺激プロンプティングによるQA生成パイプライン")
    # --- Generator Model Arguments ---
    p.add_argument("--generator-model", type=Path, required=True, help="ヒント生成用（小規模）モデルのGGUFパス")
    p.add_argument("--generator-lora", type=Path, default=None, help="ヒント生成用LoRAパス（オプション）")
    p.add_argument("--generator-template", type=str, required=True, help="ヒント生成用プロンプトのJinja2テンプレート名")
    p.add_argument("--generator-port", type=int, default=8080, help="ヒント生成用サーバーのポート番号")
    p.add_argument("--temperature-gen", type=float, default=0.7, help="ヒント生成時の温度")
    p.add_argument("--max-tokens-gen", type=int, default=200, help="ヒント生成時の最大トークン数")
    p.add_argument("--few-shot-input", type=Path, default=None, help="Few-Shot用の入力 JSON ファイルパス")
    p.add_argument("--shot-num", type=int, default=0, help="Few-Shot のサンプル数")

    # --- Refiner Model Arguments ---
    p.add_argument("--refiner-model", type=Path, required=True, help="高品質化用（大規模）モデルのGGUFパス")
    p.add_argument("--refiner-lora", type=Path, default=None, help="高品質化用LoRAパス（オプション）")
    p.add_argument("--refiner-template", type=str, required=True, help="高品質化用プロンプトのJinja2テンプレート名")
    p.add_argument("--refiner-port", type=int, default=8081, help="高品質化用サーバーのポート番号")
    p.add_argument("--temperature-refine", type=float, default=0.7, help="高品質化時の温度")
    p.add_argument("--max-tokens-refine", type=int, default=250, help="高品質化時の最大トークン数")

    # --- BERT Model Arguments ---
    p.add_argument("--bert-model", type=Path, default=None, help="回答抽出用のBERTモデルのパス")
    p.add_argument("--bert-gpu-id", type=int, default=0, help="BERTモデルが使用するGPU ID")

    # --- Common Arguments ---
    p.add_argument("--input", type=Path, required=True, help="入力JSONLファイルパス（contextキーを含む）")
    p.add_argument("--output-dir", type=Path, required=True, help="出力ディレクトリ")
    p.add_argument("--n-gpu-layers", type=int, default=42, help="両モデル共通のGPUレイヤ数")
    p.add_argument("--parallel", type=int, default=8, help="並列処理数")
    p.add_argument("--n-ctx", type=int, default=2048, help="両モデル共通のコンテキスト長")
    p.add_argument("--log-filename", type=Path, default=None, help="ログファイル名")
    p.add_argument("--log-type", type=str, default="info", choices=["debug", "info"], help="ログレベル")
    args = p.parse_args()

    setup_logging(filename=args.log_filename, log_type=args.log_type)

    # 1. 2つのLLMサーバーを起動
    start_llama_server(
        str(args.generator_model), args.generator_port, args.n_gpu_layers, 
        args.parallel, args.n_ctx, str(args.generator_lora) if args.generator_lora else None
    )
    start_llama_server(
        str(args.refiner_model), args.refiner_port, args.n_gpu_layers,
        args.parallel, args.n_ctx, str(args.refiner_lora) if args.refiner_lora else None
    )

    # 2. BERTモデルのセットアップ（オプション）
    bert_model, bert_tokenizer, bert_device = setup_bert_model(
        model_path=args.bert_model,
        gpu_id=args.bert_gpu_id
    ) if args.bert_model else (None, None, None)
    
    dataset = read_jsonl(args.input)
    few_shots = get_few_shot_examples(args.few_shot_input, args.shot_num)
    
    # 2. メインのパイプライン処理
    results = []
    async with httpx.AsyncClient() as client:
        for i in tqdm(range(0, len(dataset), args.parallel), desc="方向性刺激QA生成中"):
            batch = dataset[i:i+args.parallel]
            tasks = [
                directional_stimulus_pipeline_with_retry(
                    item, 
                    client, 
                    args,
                    few_shots=few_shots,
                    bert_model=bert_model,
                    bert_tokenizer=bert_tokenizer,
                    bert_device=bert_device
                ) for item in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend([res for res in batch_results if res is not None])

    # 3. 出力
    today = datetime.datetime.now().strftime("%Y%m%d%H%M")
    out_dir = args.output_dir / "mixture" / today
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generated.jsonl"
    write_jsonl(out_path, results)

    # 4. 設定ファイル保存
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_config(out_dir, config)

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
