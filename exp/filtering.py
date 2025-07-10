import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
import math

# ロギング設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_log_likelihood(prompt: str, answer: str, model, tokenizer, device) -> float:
    """
    指定された回答テキストの対数尤度を、プロンプトを条件として計算する。
    プロンプト部分の尤度は計算から除外する。
    """
    # プロンプトのトークン長を取得
    # add_special_tokens=False で、プロンプト単体でのトークン数を正確に測る
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens.input_ids)
    
    # モデルへの入力として、プロンプトと回答を結合したテキストをトークナイズ
    full_text = prompt + answer
    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs.input_ids
    
    # ラベルを作成し、プロンプト部分を-100（ignore_index）でマスクする
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        
        # outputs.lossは、マスクされなかったトークン（＝回答部分）の平均負対数尤度
        loss = outputs.loss.item()
        
        # 回答部分のトークン数を計算
        num_answer_tokens = (labels != -100).sum().item()
        
        # 回答が空文字列などでトークンがない場合は0を返す
        if num_answer_tokens == 0:
            return 0.0
            
        # 平均損失から合計損失を計算し、-1を掛けて合計対数尤度を求める
        # 合計NLL = 平均NLL * トークン数
        total_log_likelihood = -loss
        
    return total_log_likelihood

def calculate_dependency_scores(
    qa_data: list,
    model,
    tokenizer,
    device,
    prompt_template_with_context: str,
    prompt_template_without_context: str
):
    """
    各QAペアの文脈依存度スコア（対数尤度差）を計算する。
    """
    results_with_scores = []

    for item in tqdm(qa_data, desc="Calculating dependency scores"):
        question = item.get("question")
        answer = item.get("answer")
        context = item.get("context")

        if not all([question, answer, context]):
            continue

        # 1. 文脈ありプロンプトを作成
        prompt_with_context = prompt_template_with_context.format(
            context=context, question=question
        )

        # 2. 文脈なしプロンプトを作成
        prompt_without_context = prompt_template_without_context.format(
            question=question
        )

        # 3. それぞれの条件下での「回答」の対数尤度を計算
        log_prob_with_context = calculate_log_likelihood(prompt_with_context, answer, model, tokenizer, device)
        log_prob_without_context = calculate_log_likelihood(prompt_without_context, answer, model, tokenizer, device)
        
        # 4. 文脈依存度スコア（対数尤度差）を計算
        # 値が大きいほど文脈への依存度が高い
        dependency_score = log_prob_without_context / log_prob_with_context
        
        # 5. 元のデータにスコアを追加してリストに格納
        new_item = item.copy()
        new_item['context_dependency_score'] = dependency_score
        results_with_scores.append(new_item)
        
        logging.debug(f"ID: {item.get('id')}, Score: {dependency_score:.4f}, Q: {question[:30]}...")
            
    return results_with_scores

def main():
    parser = argparse.ArgumentParser(description="Calculate context dependency score for QA pairs and save sorted results.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the Hugging Face model.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input JSONL file containing QA pairs.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the sorted QA pairs with scores.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use.")
    args = parser.parse_args()

    # デバイス設定
    device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # モデルとトークナイザのロード
    logging.info(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()
    logging.info("Model loaded successfully.")

    # 入力データのロード
    with open(args.input_file, 'r', encoding='utf-8') as f:
        qa_data = [json.loads(line) for line in f]
    
    # シャッフル
    # import random
    # random.seed(42)  # 再現性のためのシード設定
    # random.shuffle(qa_data)
    qa_data = qa_data[:100]  # 最初の100件をサンプルとして使用

    # プロンプトテンプレートの定義
    prompt_with_context = """### 指示:
文脈に基づいて以下の質問に答えてください。

### 文脈:
{context}

### 質問:
{question}

### 応答:
"""
    prompt_without_context = """### 指示:
以下の質問に答えてください。

### 質問:
{question}

### 応答:
"""

    # スコア計算の実行
    results = calculate_dependency_scores(
        qa_data, model, tokenizer, device,
        prompt_with_context, prompt_without_context
    )

    # スコアに基づいて降順にソート
    results.sort(key=lambda x: x['context_dependency_score'], reverse=True)

    # 結果の保存
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logging.info(f"Processing complete.")
    logging.info(f"Processed {len(results)} pairs.")
    logging.info(f"Results with scores saved to {args.output_file}")

if __name__ == "__main__":
    main()
