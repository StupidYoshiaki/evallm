# src/models/evaluate.py
import json
import argparse
import re
import logging
import asyncio
import datetime
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any

import httpx
from tqdm import tqdm

# お客様のプロジェクト構成に合わせてmyutilsの各モジュールをインポート
from ..myutils.logging import setup_logging
from ..myutils.io import read_jsonl, write_jsonl, write_config
from ..myutils.parsing import build_messages, parse_json_objects
from ..myutils.api import start_llama_server, stop_llama_server, generate_from_llm, handle_error

# --- 定数 ---
MAX_RETRIES = 50
RETRY_DELAY = 2

# === 共通ヘルパー関数 ===

def read_jsonl_to_dict(file_path: Path) -> Dict[str, Dict]:
    # (この関数は前回のコードから変更なし)
    data_dict = {}
    if not file_path.is_file():
        logging.error(f"ファイルが見つかりません: {file_path}")
        return data_dict
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
    # (この関数は前回のコードから変更なし)
    if not isinstance(answer, str):
        return ""
    normalized = answer.strip()
    normalized = re.sub(r"[。\.]+$", "", normalized)
    return normalized

# === LLM-as-a-Judgeモード用の関数 ===

async def judge_single_item_with_retry(
    item_id: str,
    context: str,
    question: str,
    ground_truth_answer: str,
    model_answer: str,
    client: httpx.AsyncClient,
    judge_template: str,
    judge_model_label: str
) -> Dict[str, Any] | None:
    """1つの回答ペアを評価者LLMで評価し、パースとリトライを行う非同期ワーカー"""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            messages = build_messages(
                judge_template,
                context=context,
                question=question,
                answer=ground_truth_answer,
                prediction=model_answer
            )
            resp_data = await generate_from_llm(
                client=client,
                messages=messages,
                model=judge_model_label,
                max_tokens=256, # 評価理由も生成するので少し多めに
                temperature=0.7,
            )
            text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed_json_list = parse_json_objects(text)

            if parsed_json_list:
                gen = parsed_json_list[0]
                if "score" in gen and "reason" in gen:
                    logging.info(f"ID={item_id} の評価に成功。スコア: {gen['score']}")
                    return {"id": item_id, "score": gen["score"], "reason": gen["reason"]}
                else:
                    raise ValueError("評価者LLMの生成JSONに必要な'score'または'reason'キーがありません。")
            else:
                logging.warning(f"ID={item_id} の評価レスポンスから有効なJSONを抽出できませんでした。Text: {text}")
                raise ValueError("評価者LLMからの応答をパースできませんでした。")

        except Exception as e:
            retries += 1
            logging.warning(f"ID={item_id} の評価でエラー ({retries}/{MAX_RETRIES}): {e}")
            if retries < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
            else:
                logging.error(f"ID={item_id} の評価が最大試行回数に達しました。")
                return None
    return None

async def evaluate_single_model_with_llm_judge(
    ground_truth_data: Dict[str, Dict],
    prediction_file: Path,
    client: httpx.AsyncClient,
    judge_template: str,
    judge_model_label: str,
    parallel: int
) -> Tuple[float, int]:
    """単一モデルの予測結果をLLM-as-a-Judgeで評価し、平均スコアを計算する"""
    prediction_data = read_jsonl_to_dict(prediction_file)
    if not prediction_data:
        logging.warning(f"予測ファイルにデータがないか、読み込めませんでした: {prediction_file}")
        return 0.0, 0

    items_to_judge = []
    for pred_id, pred_record in prediction_data.items():
        if pred_id in ground_truth_data:
            items_to_judge.append({
                "id": pred_id,
                "context": ground_truth_data[pred_id].get("context", ""),
                "question": ground_truth_data[pred_id].get("question", ""),
                "ground_truth_answer": ground_truth_data[pred_id].get("answer", ""),
                "model_answer": pred_record.get("answer", "")
            })

    if not items_to_judge:
        logging.warning(f"{prediction_file.name} には正解データと共通のIDを持つ項目がありません。")
        return 0.0, 0

    all_results = []
    for i in tqdm(range(0, len(items_to_judge), parallel), desc=f"Evaluating {prediction_file.parent.name}"):
        batch = items_to_judge[i:i+parallel]
        tasks = [
            judge_single_item_with_retry(
                item["id"], item["context"], item["question"], item["ground_truth_answer"], item["model_answer"],
                client, judge_template, judge_model_label
            ) for item in batch
        ]
        batch_results = await asyncio.gather(*tasks)
        all_results.extend([res for res in batch_results if res is not None and isinstance(res.get("score"), (int, float))])

    total_score = sum(item['score'] for item in all_results)
    judged_count = len(all_results)
    average_score = total_score / judged_count if judged_count > 0 else 0.0
    
    logging.info(f"モデル: {prediction_file.parent.name} - 平均スコア: {average_score:.2f} ({judged_count}項目を評価)")
    return average_score, judged_count

# === 完全一致評価モード用の関数 ===

def calculate_single_model_accuracy(
    ground_truth_data: Dict[str, Dict],
    prediction_file: Path
) -> Tuple[float, int, int]:
    # (この関数は前回のコードから変更なし)
    prediction_data = read_jsonl_to_dict(prediction_file)
    if not prediction_data:
        logging.warning(f"予測ファイルにデータがないか、読み込めませんでした: {prediction_file}")
        return 0.0, 0, len(ground_truth_data)
    correct_predictions = 0
    for pred_id, pred_record in prediction_data.items():
        if pred_id in ground_truth_data:
            gt_record = ground_truth_data[pred_id]
            if normalize_answer(gt_record.get("answer", "")) == normalize_answer(pred_record.get("answer", "")):
                correct_predictions += 1
    total_in_gt = len(ground_truth_data)
    if total_in_gt == 0:
        return 0.0, 0, 0
    accuracy = (correct_predictions / total_in_gt) * 100
    logging.info(f"モデル: {prediction_file.parent.name} - 正解率: {accuracy:.2f}% ({correct_predictions}/{total_in_gt})")
    return accuracy, correct_predictions, total_in_gt

# === メイン実行関数 ===

async def evaluate_all_predictions(args: argparse.Namespace):
    """全ての予測ファイルを評価し、ランキングを出力するメインロジック"""
    ground_truth_data = read_jsonl_to_dict(args.ground_truth_file)
    if not ground_truth_data:
        logging.error(f"正解ファイルにデータがないか、読み込めませんでした: {args.ground_truth_file}")
        return

    # LLM-as-a-Judgeモードの場合
    if args.llm_judge_model:
        logging.info(f"LLM-as-a-Judgeモードで評価を開始します。評価者モデル: {args.llm_judge_model}")
        start_llama_server(
            base_model=str(args.llm_judge_model),
            parallel=args.judge_parallel,
            n_gpu_layers=args.judge_n_gpu_layers,
            n_ctx=args.judge_n_ctx
        )
        
        model_scores = []
        judge_model_label = args.llm_judge_model.stem # サーバーに渡すモデル名
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                for model_dir in sorted(args.prediction_base_dir.iterdir()):
                    if model_dir.is_dir():
                        prediction_file = model_dir / "prediction.jsonl"
                        if prediction_file.is_file():
                            avg_score, judged_count = await evaluate_single_model_with_llm_judge(
                                ground_truth_data, prediction_file, client,
                                args.judge_template, judge_model_label, args.judge_parallel
                            )
                            model_scores.append({
                                "model_name": model_dir.name,
                                "score": avg_score,
                                "judged_count": judged_count
                            })
                        else:
                            logging.warning(f"予測ファイルが見つかりません: {prediction_file}")
        finally:
            # 評価が終わったら必ずサーバーを停止
            stop_llama_server()
            
        if not model_scores:
            logging.warning(f"{args.prediction_base_dir} 配下に評価対象の予測ファイルが見つかりませんでした。")
            return
            
        # 平均スコアで降順ソート
        sorted_models = sorted(model_scores, key=lambda x: x["score"], reverse=True)
        # Markdownテーブルの準備
        md_header = "| Rank | Model Name | Avg Score (1-5) | Judged Count |"
        md_separator = "| --- | --- | --- | --- |"
        output_lines = [md_header, md_separator]
        for i, result in enumerate(sorted_models):
            line = f"| {i+1} | {result['model_name']} | {result['score']:.2f} | {result['judged_count']} |"
            output_lines.append(line)

    # 完全一致（Exact Match）モードの場合
    else:
        logging.info("完全一致（Exact Match）モードで評価を開始します。")
        model_accuracies = []
        for model_dir in sorted(args.prediction_base_dir.iterdir()):
            if model_dir.is_dir():
                prediction_file = model_dir / "prediction.jsonl"
                if prediction_file.is_file():
                    accuracy, correct, total = calculate_single_model_accuracy(ground_truth_data, prediction_file)
                    model_accuracies.append({
                        "model_name": model_dir.name,
                        "accuracy": accuracy,
                        "score": f"{correct}/{total}"
                    })
        
        if not model_accuracies:
            logging.warning(f"{args.prediction_base_dir} 配下に評価対象の予測ファイルが見つかりませんでした。")
            return

        sorted_models = sorted(model_accuracies, key=lambda x: x["accuracy"], reverse=True)
        # Markdownテーブルの準備
        md_header = "| Rank | Model Name | Accuracy (%) | Score (Correct/Total) |"
        md_separator = "| --- | --- | --- | --- |"
        output_lines = [md_header, md_separator]
        for i, result in enumerate(sorted_models):
            line = f"| {i+1} | {result['model_name']} | {result['accuracy']:.2f} | {result['score']} |"
            output_lines.append(line)

    # Markdownファイルへの書き出し
    markdown_file_path = args.prediction_base_dir / f"ranking_{datetime.datetime.now().strftime('%Y%m%d')}.md"
    try:
        with open(markdown_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines) + "\n")
        logging.info(f"ランキング結果をMarkdownファイルに保存しました: {markdown_file_path}")
    except IOError as e:
        logging.error(f"Markdownファイルへの書き出しに失敗しました: {markdown_file_path} - {e}")

# === スクリプトのエントリーポイント ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="モデルの予測結果を評価し、ランキングを出力します。")
    # 共通の引数
    parser.add_argument("--ground-truth-file", type=Path, required=True, help="正解データのJSONLファイルパス")
    parser.add_argument("--prediction-base-dir", type=Path, required=True, help="予測結果ファイル群(prediction.jsonl)が格納されているベースディレクトリのパス。")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="ログレベル")
    
    # LLM-as-a-Judgeモード用の引数
    judge_group = parser.add_argument_group('LLM-as-a-Judge Options')
    judge_group.add_argument("--llm-judge-model", type=Path, default=None, help="評価者として使用するGGUFモデルのパス。これを指定するとLLM-as-a-Judgeモードになります。")
    judge_group.add_argument("--judge-template", type=str, default="llm_as_a_judge.j2", help="評価者LLM用のプロンプトテンプレート名。")
    judge_group.add_argument("--judge-parallel", type=int, default=8, help="評価者LLMへの並列リクエスト数。")
    judge_group.add_argument("--judge-n-gpu-layers", type=int, default=42, help="評価者LLMのGPUオフロードレイヤー数。")
    judge_group.add_argument("--judge-n-ctx", type=int, default=2048, help="評価者LLMのコンテキストサイズ。")

    args = parser.parse_args()
    
    # setup_logging は main_async 内で呼び出すのが確実
    try:
        setup_logging(log_type=args.log_level.upper())
        asyncio.run(evaluate_all_predictions(args))
    except (RuntimeError, TimeoutError, FileNotFoundError) as e:
        logging.error(f"スクリプトの実行中に致命的なエラーが発生しました: {e}")
        # stop_llama_server() は evaluate_all_predictions の finally ブロックで呼ばれる
    except Exception:
        logging.error("予期せぬエラーでプログラムが停止しました。")
        logging.error(traceback.format_exc())
        stop_llama_server() # 念のため

    config = {
        "ground_truth_file": str(args.ground_truth_file.resolve()),
        "prediction_base_dir": str(args.prediction_base_dir.resolve()),
        "llm_judge_model": str(args.llm_judge_model.resolve()) if args.llm_judge_model else None,
        "judge_template": args.judge_template,
        "judge_parallel": args.judge_parallel,
        "judge_n_gpu_layers": args.judge_n_gpu_layers,
        "judge_n_ctx": args.judge_n_ctx,
    }
    path = args.prediction_base_dir / f"config_{datetime.datetime.now().strftime('%Y%m%d')}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
