import argparse
import logging
import time  # リトライ待機のためにtimeモジュールをインポート
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..myutils.parsing import build_messages, parse_json_objects
from ..myutils.io import read_jsonl, write_jsonl

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMAsJudge:
    """
    vLLMとChat Templateを使用してローカルでQAデータを評価する最終版クラス。
    リトライ機能を追加。
    """
    def __init__(
        self,
        model_path: str,
        template_name: str,
        tensor_parallel_size: int = 1,
        max_retries: int = 100,       # 追加: 最大リトライ回数
        retry_delay: int = 5,       # 追加: リトライ間の待機時間（秒）
    ):
        self.template_name = template_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(f"モデル '{model_path}' をロードしています...")
        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
        
        logger.info("トークナイザをロードしています...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.sampling_params = SamplingParams(
            temperature=0, max_tokens=8192
        )
        logger.info("モデルとトークナイザの準備が完了しました。")

    def evaluate_batch(self, batch: list[dict]) -> list[dict]:
        """
        QAデータのバッチを評価します。
        vLLMでの推論や結果のパースに失敗した場合、指定回数リトライします。
        """
        prompts = []
        valid_items_indices = []
        
        for i, item in enumerate(batch):
            context = item.get("context")
            question = item.get("question")
            answer = item.get("answer")

            if all([context, question, answer]):
                prompt = build_messages(
                    template_name=self.template_name,
                    context=context,
                    question=question,
                    answer=answer
                )
                try:
                    # prompt = self.tokenizer.apply_chat_template(
                    #     messages,
                    #     tokenize=False,
                    #     add_generation_prompt=True
                    # )
                    prompts.append(prompt)
                    valid_items_indices.append(i)
                except Exception as e:
                    logger.error(f"ID {item.get('id')}: チャットテンプレートの適用に失敗: {e}")
                    item["judge_error"] = f"Failed to apply chat template: {e}"
            else:
                item["judge_error"] = "context, question, or answer is missing."
        
        if not prompts:
            return batch

        retries = 0
        while retries < self.max_retries:
            try:
                # --- vLLMでの推論と結果のパース（リトライ対象） ---
                outputs = self.llm.generate(prompts, self.sampling_params)

                num_successful_parses = 0
                for i, output in enumerate(outputs):
                    item_index = valid_items_indices[i]
                    item = batch[item_index]
                    result_text = output.outputs[0].text.strip()
                    
                    parsed_list = parse_json_objects(result_text)

                    if parsed_list:
                        judge_result = parsed_list[0]
                        item["judge_score"] = judge_result.get("score")
                        item["judge_reason"] = judge_result.get("reason")
                        # 成功した場合、以前のエラー情報をクリア
                        item.pop("judge_error", None)
                        item.pop("judge_raw_output", None)
                        num_successful_parses += 1
                    else:
                        logger.warning(f"ID {item.get('id')}: Judge LLMの出力から有効なJSONを抽出できませんでした。 Raw: '{result_text}'")
                        item["judge_raw_output"] = result_text
                        item["judge_error"] = "Failed to extract valid JSON from judge output."

                # バッチ内のすべてのアイテムでパースが失敗した場合、リトライのために例外を発生させる
                if num_successful_parses == 0 and prompts:
                    raise ValueError("バッチ内のすべてのアイテムでJSONのパースに失敗しました。")

                # 少なくとも1つ成功したら、処理成功とみなしループを抜ける
                logger.info(f"バッチ処理成功。{num_successful_parses}/{len(prompts)} 件のパースに成功。")
                return batch

            except Exception as e:
                retries += 1
                logger.warning(f"バッチ処理でエラーが発生しました (試行 {retries}/{self.max_retries}): {e}")
                if retries < self.max_retries:
                    logger.info(f"{self.retry_delay}秒待機して再試行します...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"最大リトライ回数({self.max_retries}回)に達しました。このバッチの処理を失敗とします。")
                    # 最終的なエラーメッセージをバッチ内の全該当アイテムに設定
                    for i in valid_items_indices:
                        if "judge_score" not in batch[i]: # まだ成功していないアイテムのみ
                            batch[i]["judge_error"] = f"Processing failed after {self.max_retries} retries: {e}"
                    return batch
        
        return batch


def main():
    """メインの実行関数。"""
    parser = argparse.ArgumentParser(
        description="ローカルLLMモデルを使用してQAデータを定性評価します（最終版 V2）。"
    )
    parser.add_argument(
        "--template-name", type=str, required=True,
        help="評価に使用するJinja2テンプレートのファイル名 (例: 'judge_prompt.j2')。",
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="評価に使用するHugging Faceモデル名。",
    )
    parser.add_argument(
        "--input-file", type=Path, required=True,
        help="評価対象の入力JSONLファイルパス。",
    )
    parser.add_argument(
        "--output-file", type=Path, required=True,
        help="評価結果を保存する出力JSONLファイルパス。",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="一度に評価するバッチサイズ。"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="GPUの数に応じたテンソル並列サイズ。",
    )
    # --- 追加した引数 ---
    parser.add_argument(
        "--max-retries", type=int, default=100,
        help="失敗時の最大リトライ回数。"
    )
    parser.add_argument(
        "--retry-delay", type=int, default=5,
        help="リトライ間の待機時間（秒）。"
    )
    args = parser.parse_args()

    data_to_evaluate = read_jsonl(args.input_file)
    if not data_to_evaluate:
        return

    try:
        evaluator = LLMAsJudge(
            model_path=args.model_path,
            template_name=args.template_name,
            tensor_parallel_size=args.tensor_parallel_size,
            max_retries=args.max_retries,          # 引数を渡す
            retry_delay=args.retry_delay,        # 引数を渡す
        )
    except Exception as e:
        logger.critical(f"評価器の初期化に失敗しました: {e}", exc_info=True)
        return
        
    evaluated_results = []
    for i in tqdm(range(0, len(data_to_evaluate), args.batch_size), desc="QAデータを評価中"):
        batch = data_to_evaluate[i : i + args.batch_size]
        processed_batch = evaluator.evaluate_batch(batch)
        evaluated_results.extend(processed_batch)

    evaluated_results.sort(key=lambda x: str(x.get("id", "")))
    write_jsonl(args.output_file, evaluated_results)


if __name__ == "__main__":
    main()
