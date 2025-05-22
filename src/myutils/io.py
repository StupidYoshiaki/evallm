import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Generator

def read_jsonl(path: Path) -> List[Dict]:
    """
    JSONL ファイルを読み込み、各行を JSON としてパースしてリストで返す。
    """
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError as e:
                logging.warning(f"行 {i} の JSONL パースに失敗: {e}")  # 行ごとに異常を通知
    logging.info(f"JSONL を読み込みました: {path} ({len(records)} レコード)")  # :contentReference[oaicite:0]{index=0}
    return records

def write_jsonl(path: Path, data: Union[List[Dict], Generator[Dict, None, None]]) -> None:
    """
    JSONL 形式で書き出す。1 要素ごとに json.dumps() + '\\n'。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    logging.info(f"JSONL を書き出しました: {path} ({count} レコード)")  # :contentReference[oaicite:1]{index=1}

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info(f"入力データを読み込みました: {path}")
    return data

def write_json(path: Path, data: Union[Dict, List]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info(f"出力データを書き出しました: {path}")
