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

from ..myutils.api import start_llama_server, stop_llama_server, generate_from_llm, handle_error
from ..myutils.parsing import build_messages, parse_json_objects 
from ..myutils.io import read_jsonl, write_jsonl, write_config
from ..myutils.logging import setup_logging

MAX_RETRIES = 100 # 最大リトライ回数
RETRY_DELAY = 2 # 再試行までの待機時間（秒）

# --- ここからが追加/修正されたセクション ---

def setup_bert_model(model_path: Path, gpu_id: int):
    """BERTモデルとトークナイザ、デバイスをセットアップする"""
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

def extract_single_answer(question: str, context: str, model, tokenizer, device):
    """
    単一のQAペアに対してBERTモデルで回答を抽出する。
    以前作成した自前の推論ロジックをベースにしています。
    """
    with torch.no_grad():
        inputs = tokenizer(
            question,
            context,
            max_length=512,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True
        )
        offset_mapping = inputs.pop("offset_mapping").squeeze(0)
        inputs = inputs.to(device)

        outputs = model(**inputs)
        start_logits = outputs.start_logits.squeeze(0)
        end_logits = outputs.end_logits.squeeze(0)

        cls_score = (start_logits[0] + end_logits[0]).item()
        sequence_ids = inputs.sequence_ids(0)
        
        # sequence_idsがNoneの場合や1を含まない場合はエラー回避
        if sequence_ids is None or 1 not in sequence_ids:
            return "", 0.0

        ctx_start_idx = sequence_ids.index(1)
        ctx_end_idx = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        start_indexes = torch.topk(start_logits, k=10).indices
        end_indexes = torch.topk(end_logits, k=10).indices
        
        best_span = {"start": 0, "end": 0, "score": -float('inf')}
        max_answer_len = 50

        for start_idx in start_indexes:
            for end_idx in end_indexes:
                if (ctx_start_idx <= start_idx <= ctx_end_idx and
                    ctx_start_idx <= end_idx <= ctx_end_idx and
                    start_idx <= end_idx and
                    end_idx - start_idx + 1 <= max_answer_len):
                    
                    score = (start_logits[start_idx] + end_logits[end_idx]).item()
                    if score > best_span["score"]:
                        best_span["start"] = start_idx.item()
                        best_span["end"] = end_idx.item()
                        best_span["score"] = score

        if cls_score > best_span["score"]:
            return "", 0.0
        else:
            char_start = offset_mapping[best_span["start"]][0].item()
            char_end = offset_mapping[best_span["end"]][1].item()
            final_answer = context[char_start:char_end]
            
            span_logits = start_logits.unsqueeze(1) + end_logits.unsqueeze(0)
            probs = torch.softmax(span_logits, dim=-1)
            final_score = probs[best_span["start"], best_span["end"]].item()
            
            return final_answer, final_score

# --- ここまでが追加/修正されたセクション ---

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
                messages=messages,
                model=model_label,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            # parsed_json_list = parse_json_objects(text)
            
            # if not parsed_json_list or "question" not in parsed_json_list[0]:
            #     raise ValueError("LLMが有効な質問JSONを生成しませんでした。")
            
            # question = parsed_json_list[0]["question"]

            question = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")

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
        str(args.lora_model) if args.lora_model else None, 
        args.n_gpu_layers,
        args.parallel,
        n_ctx=args.n_ctx
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
                    item, client, args.template, model_label, args.max_tokens, few_shots, 
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
    stop_llama_server()

    # 6. 設定ファイル保存
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    write_config(out_dir, config)

if __name__ == "__main__":
    asyncio.run(main())
