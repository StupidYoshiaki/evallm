import argparse
import asyncio
import json
import logging
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_jsonl(file_path: Path) -> list[dict]:
    """JSONLファイルを読み込み、辞書のリストとして返します。"""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        logger.error(f"入力ファイルが見つかりません: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"ファイルのJSONデコード中にエラーが発生しました: {e}")
        return []


def write_jsonl(file_path: Path, data: list[dict]):
    """辞書のリストをJSONLファイルに書き込みます。"""
    try:
        with file_path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"{len(data)} 件の評価結果を {file_path} に保存しました。")
    except IOError as e:
        logger.error(f"ファイルへの書き込み中にエラーが発生しました: {e}")


class LLMAsJudge:
    """
    vLLMサーバー上のLLMを使用してQAデータを評価するクラス。
    """

    def __init__(
        self,
        model_name: str,
        prompt_template: str,
        vllm_host: str = "localhost",
        vllm_port: int = 8000,
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.api_url = f"http://{vllm_host}:{vllm_port}/generate"
        self.headers = {"Content-Type": "application/json"}

    async def evaluate_item(self, client: httpx.AsyncClient, item: dict) -> dict:
        """単一のQAデータ項目を評価します。"""
        context = item.get("context")
        question = item.get("question")
        answer = item.get("answer")

        if not all([context, question, answer]):
            item["judge_error"] = "context, question, or answer is missing."
            return item

        prompt = self.prompt_template.format(
            context=context, question=question, answer=answer
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.1,
            "n": 1,
            "stop": ["\n"],
        }

        try:
            response = await client.post(
                self.api_url, headers=self.headers, json=payload, timeout=60.0
            )
            response.raise_for_status()

            # vLLMのレスポンスからテキストを抽出
            result_text = response.json()["text"][0][len(prompt):].strip()

            # LLMの出力（JSON形式を期待）をパース
            try:
                judge_result = json.loads(result_text)
                item["judge_score"] = judge_result.get("score")
                item["judge_reason"] = judge_result.get("reason")
            except json.JSONDecodeError:
                logger.warning(f"ID {item.get('id')}: Judge LLMの出力がJSON形式ではありません。")
                item["judge_raw_output"] = result_text
                item["judge_error"] = "Failed to parse judge output as JSON."

        except httpx.RequestError as e:
            logger.error(f"ID {item.get('id')}: vLLMサーバーへのリクエストに失敗しました: {e}")
            item["judge_error"] = f"Request to vLLM failed: {e}"
        except Exception as e:
            logger.error(f"ID {item.get('id')}: 評価中に予期せぬエラーが発生しました: {e}")
            item["judge_error"] = f"An unexpected error occurred: {e}"

        return item


async def main():
    """メインの実行関数。"""
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judgeを使用してQAデータを定性評価します。"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="vLLMで起動しているモデル名 (例: 'elyza/ELYZA-japanese-Llama-2-7b-instruct')",
    )
    parser.add_argument(
        "--prompt-template-file",
        type=Path,
        required=True,
        help="評価に使用するプロンプトテンプレートのファイルパス。",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="評価対象の入力JSONLファイルパス。",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="評価結果を保存する出力JSONLファイルパス。",
    )
    parser.add_argument(
        "--vllm-host", type=str, default="localhost", help="vLLMサーバーのホスト名。"
    )
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLMサーバーのポート番号。")
    parser.add_argument(
        "--parallel", type=int, default=8, help="並列リクエスト数。"
    )
    args = parser.parse_args()

    # プロンプトテンプレートを読み込み
    try:
        with args.prompt_template_file.open("r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        logger.error(f"プロンプトテンプレートファイルが見つかりません: {args.prompt_template_file}")
        return

    # 入力データを読み込み
    data_to_evaluate = read_jsonl(args.input_file)
    if not data_to_evaluate:
        return

    # 評価器を初期化
    evaluator = LLMAsJudge(
        model_name=args.model_name,
        prompt_template=prompt_template,
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
    )

    evaluated_results = []
    # 非同期HTTPクライアントを使用して並列処理
    async with httpx.AsyncClient() as client:
        tasks = [evaluator.evaluate_item(client, item) for item in data_to_evaluate]
        
        # tqdmで進捗を表示
        for future in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="QAデータを評価中",
        ):
            result = await future
            evaluated_results.append(result)

    # 結果をIDでソートして書き込み（任意）
    evaluated_results.sort(key=lambda x: x.get("id", ""))
    write_jsonl(args.output_file, evaluated_results)


if __name__ == "__main__":
    asyncio.run(main())
