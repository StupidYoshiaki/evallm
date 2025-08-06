import os
import logging
import json
import re
from typing import Dict, List
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
from transformers import PreTrainedTokenizer
from ..myutils.io import read_jsonl

# JSONオブジェクトを抽出するための正規表現パターン
JSON_OBJ_PATTERN = re.compile(r"\{(?:[^{}]|\n|\r)*\}")

# テンプレートファイルが格納されているディレクトリを設定
TEMPLATE_DIR = Path(__file__).parent.parent / "prompt"
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["j2"])
)

def render_prompt(template_name: str, **kwargs) -> str:
    """
    Jinja2テンプレートをレンダリングしてプロンプト文字列を生成する。
    """
    try:
        # テンプレートファイルを読み込む
        tmpl = jinja_env.get_template(template_name)
    except Exception as e:
        logging.error(f"テンプレート読み込み失敗 {template_name}: {e}")
        raise
    # 値が None の場合はレンダリングに使わないようにkwargsから削除
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return tmpl.render(**kwargs)

def build_messages(template_name: str, **kwargs) -> List[Dict[str, str]]:
    """
    プロンプトをレンダリングし、Hugging Faceのチャット形式のメッセージリストを作成する。
    """
    content = render_prompt(template_name, **kwargs)
    logging.debug(f"生成プロンプト: {content}")
    return [{"role": "user", "content": content}]

def parse_json_objects(text: str) -> List[Dict]:
    """
    テキスト内からJSON形式のオブジェクトを全て抽出し、辞書のリストとして返す。
    """
    objs = []
    for m in JSON_OBJ_PATTERN.finditer(text):
        try:
            objs.append(json.loads(m.group()))
        except json.JSONDecodeError:
            logging.warning(f"不正な JSON 部分: {m.group()}")
    return objs

def cut_token_length(tokenizer: PreTrainedTokenizer, text: str, max_length: int) -> str:
    """
    テキストをトークナイザに基づいて指定された最大トークン長に切り詰める。
    """
    # テキストをトークンIDのリストに変換
    tokens = tokenizer(text).input_ids
    # トークン数が最大長を超えている場合
    if len(tokens) > max_length:
        # 最大長までで切り詰め、再度テキストに戻す
        return tokenizer.decode(tokens[:max_length])
    # 超えていない場合は元のテキストを返す
    return text

# ▼▼▼【ここからが修正された関数です】▼▼▼
def create_training_examples(
    jsonl_path: str,
    model_type: str,
    tokenizer: PreTrainedTokenizer,
    user_template: str,
    assistant_template: str,
    context_length: int = 1024
) -> List[Dict]:
    """
    JSONLデータからチャット形式の学習サンプルリストを作成する。（リファクタリング版）

    Args:
        jsonl_path (str): 入力となるJSONLファイルのパス。
        model_type (str): プロンプトの構成を決定するモデルタイプ。
        tokenizer (PreTrainedTokenizer): テキストの長さを計算するためのトークナイザ。
        user_template (str): ユーザープロンプトのテンプレートファイル名。
        assistant_template (str): アシスタント応答のテンプレートファイル名。
        context_length (int): 文脈の最大トークン長。

    Returns:
        List[Dict]: Hugging Faceのチャットテンプレート形式に整形された辞書のリスト。
    """
    # 1. 元データを読み込む
    raw_data = read_jsonl(Path(jsonl_path))
    
    # 整形後の学習例を格納するリスト
    formatted_examples = []
    
    # 2. 各レコードをループ処理
    for record in raw_data:
        # 2-1. コンテキストを指定されたトークン長に切り詰める
        context = cut_token_length(tokenizer, record['context'], context_length)
        
        # 2-2. model_typeに応じて、プロンプトに渡す引数を準備する。
        #      この方法により、条件分岐が1箇所にまとまり、可読性が向上する。
        user_kwargs = {}
        assistant_kwargs = {}

        if model_type == "predictor":
            # predictor: ユーザーは文脈と質問を提示し、アシスタントは答えを返す
            user_kwargs = {"context": context, "question": record['question']}
            assistant_kwargs = {"answer": record['answers']["text"][0]}

        elif model_type == "qa_generator":
            # generator: ユーザーは文脈のみを提示し、アシスタントは質問と答えを生成
            user_kwargs = {"context": context}
            try:
                assistant_kwargs = {"question": record['question'], "answer": record['answers']["text"][0]}
            except KeyError:
                assistant_kwargs = {"question": record['question'], "answer": record['answer']}

        elif model_type == "q_generator":
            # q_generator: ユーザーは文脈のみを提示し、アシスタントは質問のみを生成
            user_kwargs = {"context": context}
            assistant_kwargs = {"question": record['question']}
            
        else:
            # 未知のモデルタイプの場合は警告を出し、このレコードはスキップする
            logging.warning(f"未知のモデルタイプです: {model_type}。レコードをスキップします。")
            continue

        # 2-3. 準備した引数を使ってプロンプトをレンダリング
        user_prompt = render_prompt(user_template, **user_kwargs)
        assistant_prompt = render_prompt(assistant_template, **assistant_kwargs)

        # 2-4. Hugging Faceのチャットテンプレート形式に整形してリストに追加
        formatted_examples.append({
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_prompt}
            ]
        })

    # テストとして最初の3つの例を表示
    if len(formatted_examples) > 3:
        logging.info(f"最初の5つの例: {formatted_examples[:3]}")
        
    return formatted_examples
# ▲▲▲【ここまでが修正された関数です】▲▲▲

def create_grpo_examples(
    jsonl_path: Path,
    tokenizer: PreTrainedTokenizer,
    prompt_template: str,
    max_length: int = 1024
) -> List[Dict]:
    """
    GRPOの学習用に、プロンプトとコンテキストのペアを作成する。
    """
    raw = read_jsonl(Path(jsonl_path))
    examples = []
    for rec in raw:
        ctx = cut_token_length(tokenizer, rec['context'], max_length)
        prompt = render_prompt(
            prompt_template, 
            context=ctx, 
        )
        examples.append({
            "prompt": prompt,
            "context": rec['context'],
        })
    return examples
