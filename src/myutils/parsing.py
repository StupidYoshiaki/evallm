import os
import logging
import json
import re
from typing import Dict, List
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
from transformers import PreTrainedTokenizer
from ..myutils.io import read_jsonl

# JSON 抽出用正規表現
JSON_OBJ_PATTERN = re.compile(r"\{(?:[^{}]|\n|\r)*\}")

# テンプレートディレクトリを環境変数で指定
TEMPLATE_DIR = Path(__file__).parent.parent / "prompt"
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["j2"])
)

def render_prompt(template_name: str, **kwargs) -> str:
    try:
        tmpl = jinja_env.get_template(template_name)
    except Exception as e:
        logging.error(f"テンプレート読み込み失敗 {template_name}: {e}")
        raise
    # 値が None の場合は kwargs から削除
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return tmpl.render(**kwargs)

def build_messages(template_name: str, **kwargs) -> List[Dict[str, str]]:
    content = render_prompt(template_name, **kwargs)
    logging.debug(f"生成プロンプト: {content}")
    return [{"role": "user", "content": content}]

def parse_json_objects(text: str) -> List[Dict]:
    objs = []
    for m in JSON_OBJ_PATTERN.finditer(text):
        try:
            objs.append(json.loads(m.group()))
        except json.JSONDecodeError:
            logging.warning(f"不正な JSON 部分: {m.group()}")
    return objs

def cut_token_length(tokenizer: PreTrainedTokenizer, text: str, max_length: int):
    tokens = tokenizer(text).input_ids
    if len(tokens) > max_length:
        return tokenizer.decode(tokens[:max_length])
    return text

def create_training_examples(
    jsonl_path: str,
    model_type: str,
    tokenizer: PreTrainedTokenizer,
    user_template: str,
    assistant_template: str,
    context_length: int = 1024
) -> List[Dict]:
    raw = read_jsonl(Path(jsonl_path))
    examples = []
    for rec in raw:
        ctx = cut_token_length(tokenizer, rec['context'], context_length)
        user = render_prompt(
            user_template, 
            context=ctx, 
            question=rec['question'] if model_type == "predictor" else None
        )
        assistant = render_prompt(
            assistant_template,
            question=rec['question'] if model_type in ["generator", "q_generator"] else None,
            answer=rec['answer'] if model_type != "q_generator" else None,
        )
        examples.append({
            "messages": [
                {"role":"user","content":user},
                {"role":"assistant","content":assistant}
            ]
        })
        # import sys
        # print(examples)
        # sys.exit()
    return examples

def create_grpo_examples(
    jsonl_path: Path,
    tokenizer: PreTrainedTokenizer,
    prompt_template: str,
    max_length: int = 1024
) -> List[Dict]:
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

