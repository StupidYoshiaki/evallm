import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import json
from pathlib import Path
from typing import List, Dict, Union

PROMPT = """\
### 指示:
文脈に基づいて以下の質問に答えてください。

### 文脈:
{context}

### 質問:
{question}

### 注意:
- 質問に対して、具体的かつ明確な回答をしてください。
- 回答は端的に記述してください。説明は一切不要です。
- 回答は文脈から直接導き出せる内容にしてください。
- 日本語で回答してください。

### 応答:
"""

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
                print(f"行 {i} の JSONL パースに失敗: {e}")  # 行ごとに異常を通知
    print(f"JSONL を読み込みました: {path} ({len(records)} レコード)")  # :contentReference[oaicite:0]{index=0}
    return records

def calculate_log_probability(model, tokenizer, text_sequence, context_sequence=None):
    """
    与えられたコンテキストのもとで、テキストシーケンスの対数確率を計算する。
    context_sequenceがNoneの場合、テキストシーケンス自身の文脈で計算（Direct Answer Score用）。
    context_sequenceがある場合、そのコンテキストに続けてテキストシーケンスが生成される確率を計算（Conditioned Answer Score用）。
    """
    if context_sequence:
        # 指示Q + 回答A の形式
        full_text = context_sequence + tokenizer.eos_token + text_sequence # EOSトークンを挟むかはモデルの学習方法による
        input_ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)
        context_ids = tokenizer.encode(context_sequence + tokenizer.eos_token, return_tensors="pt").to(model.device)
        context_length = context_ids.shape[1]
        target_ids = input_ids[:, context_length:] # 回答部分のみをターゲットとする
        labels = input_ids.clone()
        labels[:, :context_length] = -100 # 指示部分の損失は無視
    else:
        # 回答A のみの形式
        input_ids = tokenizer.encode(text_sequence, return_tensors="pt").to(model.device)
        context_length = 0 # コンテキストなし
        target_ids = input_ids
        labels = input_ids.clone() # 全てのトークンがターゲット

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        # lossは平均クロスエントロピーなので、トークン数で割られた値になっている
        # IFDの定義におけるs_theta(A)やs_theta(A|Q)は平均クロスエントロピー損失そのもの
        # outputs.lossはバッチ全体の平均損失。ここでは単一シーケンスなので、これがそのままスコアになる。
        # ただし、Hugging FaceのCausalLMのlossは、shift_logitsとshift_labelsで計算されるため、
        # sequenceの最初のトークンに対する予測は含まない。
        # 論文の定義と厳密に合わせるには、トークンごとの確率から手動で計算する必要がある場合がある。
        # ここでは簡易的にモデルの出力するlossを利用する。
        # 重要なのは、A|QとAで同じ方法でスコアを計算し比較すること。
        log_prob_score = outputs.loss.item()

    # 論文のs_theta(A|Q)やs_theta(A)は負の対数尤度の平均なので、
    # Hugging FaceのCrossEntropyLossはそのまま使える。
    return log_prob_score

def calculate_ifd_score(model, tokenizer, instruction, answer):
    """
    IFDスコアを計算する
    """
    # 1. Conditioned Answer Score s_theta(A|Q) の計算
    # 指示Qが与えられた条件のもとで、モデルが回答Aを生成する際の平均クロスエントロピー損失
    s_A_given_Q = calculate_log_probability(model, tokenizer, answer, context_sequence=instruction)
    print(f"s_theta(A|Q) (Conditioned Answer Score): {s_A_given_Q}")

    # 2. Direct Answer Score s_theta(A) の計算
    # 指示Qなしで、モデルが回答Aを生成する際の平均クロスエントロピー損失
    s_A = calculate_log_probability(model, tokenizer, answer, context_sequence=None)
    print(f"s_theta(A) (Direct Answer Score): {s_A}")

    # 3. IFDスコアの計算
    if s_A == 0: # ゼロ除算を避ける
        # s_Aが0になるケースは、回答が極めて短い、またはモデルが完全にそのシーケンスを記憶している場合など。
        # 実際の運用では、このような場合にどう対処するか別途検討が必要。
        # (例: ごく小さな値をs_Aとする、IFDを非常に大きな値とする、またはエラーとするなど)
        # 論文では特にこの場合の扱いに言及がないため、ここではエラーまたはNaNを返す想定。
        print("Warning: Direct Answer Score s_theta(A) is zero. IFD score cannot be calculated reliably.")
        return float('nan')

    ifd_score = s_A_given_Q / s_A
    print(f"IFD_theta(Q,A): {ifd_score}")

    # 論文に基づき、IFDスコアが1より大きい場合は指示と応答の不整合の可能性がある
    if ifd_score > 1.0:
        print(f"Note: IFD score ({ifd_score:.4f}) > 1.0, which may indicate misalignment between instruction and response.")
    # 実際のデータ選択では、この後IFDスコアに基づいてフィルタリングやソーティングを行う

    return ifd_score

# ---- メイン処理の例 ----
if __name__ == "__main__":
    # モデル名とトークナイザー名を指定 (Hugging Face Hubから)
    model_name = "models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/lora/20250606/checkpoint-2000" # 例: Llama-2-7B (アクセスには注意)
    # model_name = "stabilityai/japanese-stablelm-instruct-gamma-7b" # 例: 日本語モデル

    # トークナイザーとモデルのロード
    # (注意: モデルのダウンロードには時間がかかり、十分なディスク容量とメモリが必要です)
    # (注意: Llama2などの一部モデルはアクセス許可が必要です)
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading model: {model_name}...")
    # GPUが利用可能ならGPUへ、なければCPUへ
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval() # 評価モードにする

    print(f"Model loaded on: {model.device}")

    # dataset_path = Path("data/JSQuAD/eval/baseline.jsonl")
    dataset_path = Path("data/JSQuAD/eval/Llama-3.1-Swallow-8B-Instruct-v0.3/202505270159/generated.jsonl")
    dataset = read_jsonl(dataset_path)
    dataset = dataset[:100]  # 最初の100件をサンプルとして使用

    result = []  # 結果を格納するリスト

    # データセットの各ペアに対してIFDスコアを計算
    for i, item in enumerate(dataset):
        context = item.get("context", "")
        question = item.get("question", "")
        instruction = PROMPT.format(context=context, question=question)
        answer = item.get("answer", "")
        if not instruction or not answer:
            print(f"Skipping item {i} due to missing instruction or answer.")
            continue
        
        print(f"\n--- Processing Item {i+1} ---")
        print(f"Instruction: {instruction}")
        print(f"Answer: {answer}")

        ifd_score = calculate_ifd_score(model, tokenizer, instruction, answer)
        print(f"Calculated IFD Score for Item {i+1}: {ifd_score if ifd_score is not None else 'N/A'}")
        result.append({
            "item_index": i,
            "question": question,
            "answer": answer,
            "context": context,
            "ifd_score": ifd_score
        })

    # 結果をJSONL形式で保存
    output_path = Path("data/exp/ifd_scores.jsonl")

    # IFDスコアの降順でソート
    result.sort(key=lambda x: x["ifd_score"], reverse=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in result:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
