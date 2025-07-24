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

# --- 追加するインポート ---
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import Levenshtein # レーベンシュタイン距離の計算用

# --- 外部ユーティリティ関数のインポート（実際のパスに合わせてください） ---
from ..myutils.api import start_llama_server, stop_llama_server, generate_from_llm, handle_error
from ..myutils.parsing import build_messages
from ..myutils.io import read_jsonl, write_jsonl, write_config
from ..myutils.logging import setup_logging

# --- グローバル設定 ---
MAX_RETRIES = 100 # 1つの候補生成タスクあたりの最大リトライ回数
RETRY_DELAY = 2  # 再試行までの待機時間（秒）

# --- ここからがメインのロジックです ---

def setup_bert_model(model_path: Path, gpu_id: int):
    """回答抽出用のBERTモデルとトークナイザ、デバイスをセットアップする。"""
    logger = logging.getLogger(__name__)
    if torch.cuda.is_available() and gpu_id >= 0:
        try:
            device = torch.device(f"cuda:{gpu_id}")
            logger.info(f"BERTモデル用にCUDAデバイス {gpu_id} ({torch.cuda.get_device_name(gpu_id)}) を使用します。")
        except (AssertionError, IndexError):
            logger.warning(f"指定されたGPU ID {gpu_id} はBERTモデルで無効です。CPUを使用します。")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logger.info("BERTモデル用にCPUを使用します。")
    
    try:
        logger.info(f"BERTモデルをロードしています: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForQuestionAnswering.from_pretrained(str(model_path))
        model.to(device)
        model.eval()
        logger.info("BERTモデルとトークナイザの準備が完了しました。")
        return model, tokenizer, device
    except Exception as e:
        logger.error(f"BERTモデルまたはトークナイザの初期化中にエラーが発生しました: {e}")
        raise

def extract_single_answer(question: str, context: str, model, tokenizer, device) -> str:
    """単一のQAペアに対してBERTモデルで回答を抽出する。"""
    with torch.no_grad():
        inputs = tokenizer(
            question, context, max_length=512, truncation="only_second",
            padding="max_length", return_tensors="pt", return_offsets_mapping=True
        )
        offset_mapping = inputs.pop("offset_mapping").squeeze(0)
        inputs = inputs.to(device)

        outputs = model(**inputs)
        start_logits = outputs.start_logits.squeeze(0)
        end_logits = outputs.end_logits.squeeze(0)

        sequence_ids = inputs.sequence_ids(0)
        if sequence_ids is None or 1 not in sequence_ids:
            return ""

        # 文脈部分のトークンインデックス範囲を取得
        ctx_start_idx = sequence_ids.index(1)
        ctx_end_idx = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # 最適な回答スパンを見つける
        best_score = -float('inf')
        best_start_idx, best_end_idx = 0, 0
        
        start_indices = torch.topk(start_logits, k=10, dim=-1).indices
        end_indices = torch.topk(end_logits, k=10, dim=-1).indices

        for start_idx in start_indices:
            for end_idx in end_indices:
                # 有効なスパンかチェック
                if (ctx_start_idx <= start_idx <= ctx_end_idx and
                    ctx_start_idx <= end_idx <= ctx_end_idx and
                    start_idx <= end_idx):
                    
                    score = (start_logits[start_idx] + end_logits[end_idx]).item()
                    if score > best_score:
                        best_score = score
                        best_start_idx = start_idx.item()
                        best_end_idx = end_idx.item()
        
        # CLSトークンのスコアと比較して、回答不能かを判断
        cls_score = (start_logits[0] + end_logits[0]).item()
        if cls_score > best_score:
            return "" # 回答不能と判断

        # トークン位置から元のテキストの文字位置に変換して回答を抽出
        char_start = offset_mapping[best_start_idx][0].item()
        char_end = offset_mapping[best_end_idx][1].item()
        return context[char_start:char_end]

def normalized_levenshtein(s1: str, s2: str) -> float:
    """正規化されたレーベンシュタイン距離を計算する。値が0に近いほど類似度が高い。"""
    if not s1 and not s2:
        return 0.0
    if len(s1) == 0 or len(s2) == 0:
        return 1.0 # 片方が空の場合は最大の距離
    # Levenshtein.distanceは編集距離を返す
    distance = Levenshtein.distance(s1, s2)
    # 長い方の文字列長で割って正規化
    return distance / max(len(s1), len(s2))

async def generate_and_evaluate_candidate(
    item: dict,
    client: httpx.AsyncClient,
    template: str,
    model_label: str,
    max_tokens: int,
    bert_model,
    bert_tokenizer,
    device,
    temperature: float,
    max_retries: int
) -> dict | None:
    """
    単一の候補（質問と回答）を生成・評価するコルーチン。
    """
    retries = 0
    context = item.get("context")
    while retries < max_retries:
        try:
            # 1. LLMで質問を生成
            messages = build_messages(template, context=context)
            resp_data = await generate_from_llm(
                client=client, messages=messages, model=model_label,
                max_tokens=max_tokens, temperature=temperature
            )
            generated_question = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not generated_question:
                raise ValueError("LLMが空の質問を生成しました。")

            # 2. BERTで回答を抽出
            extracted_answer = await asyncio.to_thread(
                extract_single_answer, generated_question, context, bert_model, bert_tokenizer, device
            )

            if not extracted_answer:
                raise ValueError("BERTが回答を抽出できませんでした。")

            # 成功したら、生成された質問と抽出された回答を返す
            return {"question": generated_question, "answer": extracted_answer}

        except Exception as e:
            retries += 1
            logging.warning(f"候補生成中にエラー ({retries}/{MAX_RETRIES}): {e}")
            if retries < max_retries:
                await asyncio.sleep(RETRY_DELAY)
            else:
                logging.error(f"候補生成が最大試行回数に達しました。")
                return None # 失敗した場合はNoneを返す
    return None

async def create_dpo_sample_for_item(
    item: dict,
    client: httpx.AsyncClient,
    template: str,
    model_label: str,
    max_tokens: int,
    bert_model,
    bert_tokenizer,
    device,
    temperature: float,
    num_candidates: int
) -> dict | None:
    """
    単一のJSQuADアイテムから、1つのDPO学習サンプル（prompt, chosen, rejected）を作成する。
    【変更】有効な候補が10個集まるまでリトライするロジックに変更。
    """
    context = item.get("context")
    original_question = item.get("question")
    original_answer = item["answers"]["text"][0] if item.get("answers", {}).get("text") else ""

    if not all([context, original_question, original_answer]):
        logging.warning(f"ID {item.get('id')} は必須キーが不足しているためスキップします。")
        return None

    # 1. 有効な候補が10個集まるまでリトライ
    valid_candidates = []
    total_attempts = 0
    # 無限ループを避けるための全体的な試行回数の上限
    max_total_attempts = num_candidates * MAX_RETRIES

    logging.info(f"ID {item.get('id')} のために{num_candidates}個の有効な候補を収集します...")
    
    while len(valid_candidates) < num_candidates and total_attempts < max_total_attempts:
        # 不足している候補の数を計算
        needed = num_candidates - len(valid_candidates)
        
        # 不足分だけタスクを作成して並列実行
        tasks_to_run = [
            generate_and_evaluate_candidate(
                item, client, template, model_label, max_tokens,
                bert_model, bert_tokenizer, device, temperature, num_candidates
            ) for _ in range(needed)
        ]
        
        newly_generated_candidates = await asyncio.gather(*tasks_to_run)
        
        # 成功した結果のみをリストに追加
        for cand in newly_generated_candidates:
            if cand is not None:
                valid_candidates.append(cand)
        
        total_attempts += needed
        logging.info(f"ID {item.get('id')}: {len(valid_candidates)}/{num_candidates} 個の候補を収集済み。総試行回数: {total_attempts}")

    # 上限に達しても候補が一つも得られなかった場合
    if not valid_candidates:
        logging.error(f"ID {item.get('id')} は最大試行回数({max_total_attempts})に達しましたが、候補を1つも生成できませんでした。")
        return None
    
    # 候補がnum_candidates個に満たなかった場合（警告を出すが処理は続行）
    if len(valid_candidates) < num_candidates:
        logging.warning(f"ID {item.get('id')} は{num_candidates}個の候補を収集できませんでした（収集数: {len(valid_candidates)}）。収集済みの候補で処理を続行します。")

    # 2. 最適な負例を選定
    best_candidate = None
    min_distance = float('inf')

    for candidate in valid_candidates:
        distance = normalized_levenshtein(original_answer, candidate["answer"])
        if distance < min_distance:
            min_distance = distance
            best_candidate = candidate
    
    logging.info(f"ID {item.get('id')} の最適な負例を選定しました。距離: {min_distance:.4f}")

    # 3. DPO形式でデータを整形
    return {
        "id": item.get("id"),
        "prompt": context,
        "chosen": original_question,
        "chosen_answer": original_answer,
        "rejected": best_candidate["question"],
        "rejected_answer": best_candidate["answer"],
    }


async def main():
    """非同期処理のメインロジック"""
    p = argparse.ArgumentParser(description="DPO用合成データ作成パイプライン")
    p.add_argument("--base-model", type=Path, required=True, help="Base GGUF モデルファイルパス")
    p.add_argument("--lora-model", type=Path, default=None, help="LoRA 適用後モデルファイルパス（オプション）")
    p.add_argument("--template", type=str, required=True, help="プロンプト Jinja2 テンプレート名 (.j2)")
    p.add_argument("--n-gpu-layers", type=int, default=42, help="GPU レイヤ数")
    p.add_argument("--n-ctx", type=int, default=2048, help="コンテキスト長 (n_ctx)")
    p.add_argument("--temperature", type=float, default=1.2, help="生成時の温度")
    p.add_argument("--bert-model", type=Path, required=True, help="回答抽出用のBERTモデルのパス")
    p.add_argument("--bert-gpu-id", type=int, default=0, help="BERTモデルが使用するGPU ID")
    p.add_argument("--input", type=Path, required=True, help="入力 JSONL ファイルパス (JSQuAD形式)")
    p.add_argument("--output-dir", type=Path, required=True, help="出力ディレクトリ")
    p.add_argument("--parallel", type=int, default=8, help="並列処理数")
    p.add_argument("--log-filename", type=Path, default=None, help="ログファイル名")
    p.add_argument("--log-type", type=str, default="info", choices=["debug", "info"], help="ログレベル")
    p.add_argument("--num-candidates", type=int, default=8, help="1アイテムあたりの有効な候補数")
    args = p.parse_args()

    setup_logging(filename=args.log_filename, log_type=args.log_type)

    # 1. BERTモデルのセットアップ
    bert_model, bert_tokenizer, device = setup_bert_model(args.bert_model, args.bert_gpu_id)

    # 2. LLMサーバー起動
    start_llama_server(
        str(args.base_model), 
        str(args.lora_model) if args.lora_model else None, 
        args.n_gpu_layers,
        args.parallel,
        n_ctx=args.n_ctx
    )
    model_label = args.base_model.parts[-1] if args.base_model.is_dir() else args.base_model.stem
    
    dataset = read_jsonl(args.input)
    dataset = dataset[:20] # デバッグ用に最初の5件のみを使用

    # 3. メイン処理
    dpo_results = []
    async with httpx.AsyncClient(timeout=300.0) as client:
        for i in tqdm(range(0, len(dataset), args.parallel), desc="DPOデータ作成中"):
            batch_items = dataset[i:i+args.parallel]
            
            tasks = [
                create_dpo_sample_for_item(
                    item, client, args.template, model_label, 200, # max_tokensは固定値
                    bert_model, bert_tokenizer, device, args.temperature, args.num_candidates
                ) 
                for item in batch_items
            ]
            
            batch_results = await asyncio.gather(*tasks)
            dpo_results.extend([res for res in batch_results if res is not None])

    # 4. 出力
    today = datetime.datetime.now().strftime("%Y%m%d%H%M")
    out_dir = args.output_dir / "dpo" / today
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generated.jsonl"
    write_jsonl(out_path, dpo_results)

    # 5. LLMサーバー停止
    stop_llama_server()

    # 6. 設定ファイル保存
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    write_config(out_dir, config)

if __name__ == "__main__":
    asyncio.run(main())
