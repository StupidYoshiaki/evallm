# src/models/evaluate.py
import json
import argparse
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from ..myutils.logging import setup_logging


def read_jsonl_to_dict(file_path: Path) -> Dict[str, Dict]:
    """
    JSONLファイルを読み込み、各行のJSONオブジェクトを"id"をキーとする辞書に格納して返す。
    """
    data_dict = {}
    if not file_path.is_file():
        logging.error(f"ファイルが見つかりません: {file_path}")
        return data_dict # 空の辞書を返す

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line.strip())
                if "id" in record:
                    data_dict[record["id"]] = record
                else:
                    logging.warning(f"{file_path} の {i+1}行目: 'id'フィールドのないレコードが見つかりました: {record}")
            except json.JSONDecodeError:
                logging.warning(f"{file_path} の {i+1}行目: JSONとしてパースできない行がありました: {line.strip()}")
    return data_dict

def normalize_answer(answer: str) -> str:
    """
    回答文字列を正規化する。
    - 前後の空白（改行含む）を除去
    - 末尾の句読点（。、.）を除去
    """
    if not isinstance(answer, str):
        return ""
    normalized = answer.strip()
    normalized = re.sub(r"[。\.]+$", "", normalized) # 末尾の句読点を除去
    return normalized

def calculate_single_model_accuracy(
    ground_truth_data: Dict[str, Dict],
    prediction_file: Path
) -> Tuple[float, int, int]:
    """
    単一モデルの予測ファイルに対する正解率を計算する。
    Returns: (accuracy, correct_count, total_in_gt)
    """
    prediction_data = read_jsonl_to_dict(prediction_file)

    if not prediction_data:
        logging.warning(f"予測ファイルにデータがないか、読み込めませんでした: {prediction_file}")
        return 0.0, 0, len(ground_truth_data)

    correct_predictions = 0
    predictions_found_in_gt = 0

    for pred_id, pred_record in prediction_data.items():
        if pred_id in ground_truth_data:
            predictions_found_in_gt +=1
            gt_record = ground_truth_data[pred_id]
            
            gt_answer = normalize_answer(gt_record.get("answer", ""))
            pred_answer = normalize_answer(pred_record.get("answer", ""))
            
            if gt_answer == pred_answer:
                correct_predictions += 1
    
    total_in_gt = len(ground_truth_data)
    if total_in_gt == 0:
        return 0.0, 0, 0
        
    accuracy = (correct_predictions / total_in_gt) * 100
    logging.info(f"モデル: {prediction_file.parent.name} - 正解率: {accuracy:.2f}% ({correct_predictions}/{total_in_gt})")
    return accuracy, correct_predictions, total_in_gt

def evaluate_all_predictions(ground_truth_file: Path, prediction_base_dir: Path) -> None:
    """
    指定されたベースディレクトリ内の全てのprediction.jsonlを評価し、ランキングを出力する。
    """
    ground_truth_data = read_jsonl_to_dict(ground_truth_file)
    if not ground_truth_data:
        logging.error(f"正解ファイルにデータがないか、読み込めませんでした: {ground_truth_file}")
        return

    model_accuracies = []

    # prediction_base_dir (例: output/JSQuAD/baseline) の直下にある各モデルディレクトリを探索
    for model_dir in prediction_base_dir.iterdir():
        if model_dir.is_dir():
            prediction_file = model_dir / "prediction.jsonl"
            if prediction_file.is_file():
                logging.info(f"評価中: {prediction_file}")
                accuracy, correct, total = calculate_single_model_accuracy(ground_truth_data, prediction_file)
                model_accuracies.append({
                    "model_name": model_dir.name,
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "prediction_file": str(prediction_file)
                })
            else:
                logging.warning(f"予測ファイルが見つかりません: {prediction_file}")
    
    if not model_accuracies:
        logging.warning(f"{prediction_base_dir} 配下に評価対象の予測ファイルが見つかりませんでした。")
        return

    # 正解率で降順ソート
    sorted_models = sorted(model_accuracies, key=lambda x: x["accuracy"], reverse=True)

    # --- ランキング出力 ---
    logging.info(f"\n=== モデル別 正解率ランキング ({prediction_base_dir}内の結果) ===") # どのディレクトリの結果か分かるようにログを修正
    
    # Markdownテーブルヘッダー
    md_table_header = "| Rank | Model Name | Accuracy (%) | Score (Correct/Total) |"
    md_table_separator = "| --- | --- | --- | --- |"
    
    markdown_output_lines = [md_table_header, md_table_separator]
    
    for i, result in enumerate(sorted_models):
        rank = i + 1
        model_name_display = result['model_name'][:30] # モデル名が長い場合切り詰める
        accuracy_str = f"{result['accuracy']:.2f}"
        score_str = f"{result['correct']}/{result['total']}"
        
        markdown_line = f"| {rank} | {model_name_display} | {accuracy_str} | {score_str} |"
        markdown_output_lines.append(markdown_line)

    # Markdownファイルへの書き出し
    markdown_file_path = prediction_base_dir / "ranking.md"
    try:
        with open(markdown_file_path, 'w', encoding='utf-8') as f:
            for line in markdown_output_lines:
                f.write(line + "\n")
        logging.info(f"ランキング結果をMarkdownファイルに保存しました: {markdown_file_path}")
    except IOError as e:
        logging.error(f"Markdownファイルへの書き出しに失敗しました: {markdown_file_path} - {e}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指定されたディレクトリ内の全てのモデルの予測結果を評価し、ランキングを出力します。")
    parser.add_argument("--ground-truth-file", type=Path, required=True, help="正解データのJSONLファイルパス")
    parser.add_argument("--prediction-base-dir", type=Path, required=True, help="予測結果ファイル群(prediction.jsonl)が格納されているベースディレクトリのパス。この直下にモデル名のディレクトリがあると想定。")
    
    args = parser.parse_args()

    setup_logging()
    
    evaluate_all_predictions(args.ground_truth_file, args.prediction_base_dir)
