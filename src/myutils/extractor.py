import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

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
