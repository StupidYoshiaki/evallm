import argparse
import logging
import json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def read_jsonl(file_path: Path) -> list:
    """JSONLファイルを読み込み、辞書のリストとして返す。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        logger.error(f"入力ファイルが見つかりません: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSONのパース中にエラーが発生しました: {file_path} - {e}")
        return []

def write_jsonl(file_path: Path, data: list):
    """辞書のリストをJSONLファイルに書き込む。"""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"結果をファイルに保存しました: {file_path}")
    except IOError as e:
        logger.error(f"ファイルへの書き込み中にエラーが発生しました: {e}")

def run_batch_inference(args: argparse.Namespace):
    """
    JSONLファイルを読み込み、全てのQAペアに対してバッチ処理で回答抽出を行い、結果を保存する。
    (pipeline不使用版)
    """
    # デバイスの決定
    if torch.cuda.is_available() and args.gpu_id >= 0:
        try:
            device = torch.device(f"cuda:{args.gpu_id}")
            logger.info(f"CUDAデバイス {args.gpu_id} ({torch.cuda.get_device_name(args.gpu_id)}) を使用します。")
        except (AssertionError, IndexError):
            logger.warning(f"指定されたGPU ID {args.gpu_id} は無効です。CPUを使用します。")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        logger.info("CPUを使用します。")

    # 1. モデルとトークナイザを直接ロード
    try:
        logger.info(f"モデルをロードしています: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
        model = AutoModelForQuestionAnswering.from_pretrained(str(args.model_path))
        model.to(device)
        model.eval()
        logger.info("モデルとトークナイザの準備が完了しました。")
    except Exception as e:
        logger.error(f"モデルまたはトークナイザの初期化中にエラーが発生しました: {e}")
        return

    # 2. 入力データの読み込み
    input_data = read_jsonl(args.input_file)
    if not input_data:
        return

    # 3. バッチ処理による推論の実行
    logger.info(f"{len(input_data)}件のデータに対して、バッチサイズ{args.batch_size}で推論を実行します...")
    
    all_results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(input_data), args.batch_size), desc="Extracting Answers"):
            batch = input_data[i:i + args.batch_size]
            
            questions = [item["question"] for item in batch]
            contexts = [item["context"] for item in batch]
            
            inputs = tokenizer(
                questions,
                contexts,
                max_length=512,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt",
                return_offsets_mapping=True
            )
            
            offset_mapping = inputs.pop("offset_mapping")
            inputs = inputs.to(device)

            outputs = model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            for j in range(len(batch)):
                start_logit = start_logits[j]
                end_logit = end_logits[j]
                offsets = offset_mapping[j]
                context = contexts[j]
                
                cls_score = (start_logit[0] + end_logit[0]).item()

                sequence_ids = inputs.sequence_ids(j)
                ctx_start_idx = sequence_ids.index(1)
                ctx_end_idx = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
                
                start_indexes = torch.topk(start_logit, k=10).indices
                end_indexes = torch.topk(end_logit, k=10).indices
                
                best_score = -float('inf')
                best_span = {"start": 0, "end": 0, "score": -float('inf')}
                max_answer_len = 50 # 回答の最大トークン長

                for start_idx in start_indexes:
                    for end_idx in end_indexes:
                        if (ctx_start_idx <= start_idx <= ctx_end_idx and
                            ctx_start_idx <= end_idx <= ctx_end_idx and
                            start_idx <= end_idx and
                            end_idx - start_idx + 1 <= max_answer_len):
                            
                            score = (start_logit[start_idx] + end_logit[end_idx]).item()
                            if score > best_span["score"]:
                                best_span["start"] = start_idx
                                best_span["end"] = end_idx
                                best_span["score"] = score
                
                if cls_score > best_span["score"]:
                    final_answer = ""
                    final_score = torch.softmax(start_logit + end_logit, dim=-1)[0].item()
                else:
                    char_start = offsets[best_span["start"]][0]
                    char_end = offsets[best_span["end"]][1]
                    final_answer = context[char_start:char_end]
                    
                    # ▼▼▼【修正箇所】▼▼▼
                    # スコアを事後確率風の値に変換
                    # 1. 正しいブロードキャストで全スパンのロジット和を計算
                    #    (start_logit[i] + end_logit[j] が (i, j)要素になるように)
                    span_logits = start_logit.unsqueeze(1) + end_logit.unsqueeze(0)
                    
                    # 2. softmaxを適用 (これは厳密な確率分布ではないがスコアの正規化にはなる)
                    #    各開始位置に対して、全ての終了位置の確率を計算
                    probs = torch.softmax(span_logits, dim=-1)
                    
                    # 3. 正しい2次元インデックスでアクセス
                    final_score = probs[best_span["start"], best_span["end"]].item()

                original_item = batch[j]
                all_results.append({
                    "id": original_item.get("id"),
                    "context": original_item.get("context"),
                    "question": original_item.get("question"),
                    "ground_truth_answer": original_item.get("answer"),
                    "extracted_answer": final_answer,
                    "score": final_score
                })

    write_jsonl(args.output_file, all_results)

def main():
    parser = argparse.ArgumentParser(description="ファインチューニング済みQAモデルで、JSONLファイル全体に対して推論を行うスクリプト (pipeline不使用)")
    
    parser.add_argument("--model_path", type=Path, required=True,
                        help="ファインチューニング済みモデルが格納されているディレクトリのパス。")
    parser.add_argument("--input_file", type=Path, required=True,
                        help="推論対象の入力JSONLファイルパス。'context'と'question'キーを含む必要があります。")
    parser.add_argument("--output_file", type=Path, required=True,
                        help="推論結果を保存する出力JSONLファイルパス。")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="推論時のバッチサイズ。GPUのVRAMに応じて調整してください。")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="推論に使用するGPUのID。CPUを使用する場合は-1を指定。 (default: 0)")

    args = parser.parse_args()
    run_batch_inference(args)

if __name__ == "__main__":
    main()
