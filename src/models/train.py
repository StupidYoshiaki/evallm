import os
import json
import argparse
import datetime
import logging
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from transformers.trainer_utils import set_seed
from datasets import Dataset

from ..myutils.parsing import create_training_examples
from ..myutils.io import write_json, write_config
from ..myutils.logging import setup_logging

# GPU 0とGPU 1を利用可能にする
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

TEMPLATE = {
    "gemma": "<start_of_turn>model\n",
    "swallow": "<|start_header_id|>assistant<|end_header_id|>\n\n",
}

def set_device_map(base_model_path: Path) -> dict:
    config = AutoConfig.from_pretrained(base_model_path)
    num_layers = config.num_hidden_layers # Llama-Swallowの場合、ログから42と分かっている
    
    logging.info(f"モデルのレイヤー数: {num_layers}")

    # device_mapを手動で構築
    device_map = {}
    # 入力埋め込みと最初の半分のレイヤーをGPU 0に
    device_map["model.embed_tokens"] = 0
    layers_on_gpu0 = num_layers // 2 
    for i in range(layers_on_gpu0):
        device_map[f"model.layers.{i}"] = 0
    
    # 残りのレイヤーと最終ノーマライゼーション層をGPU 1に
    for i in range(layers_on_gpu0, num_layers):
        device_map[f"model.layers.{i}"] = 1
    device_map["model.norm"] = 1
    
    # ★★★ LMヘッドをGPU 0に強制的に配置 ★★★
    device_map["lm_head"] = 0
    
    logging.info(f"手動device_mapを構築しました: {device_map}")
    return device_map

def train_lora(
    base_model: Path,
    model_type: str,
    dataset_path: Path,
    user_template: str,
    assistant_template: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    accum_steps: int,
    seed: int,
) -> Path:
    # シード固定
    set_seed(seed)

    # トークナイザの設定
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        attn_implementation='eager',
        add_eos_token=True,
    )
    if tokenizer.pad_token is None:
        logging.info("pad_tokenがNoneのため、eos_tokenをpad_tokenとして設定します。")
        tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"Chat template: {tokenizer.chat_template}")

    logging.info("データセットを作成し、チャットテンプレートを適用・トークン化します...")
    training_examples_with_messages_key = create_training_examples(
        dataset_path, model_type, tokenizer,
        user_template, assistant_template
    )
    processed_dataset_for_trainer = []
    for example_item in training_examples_with_messages_key:
        tokenized_sample = tokenizer.apply_chat_template(
            example_item['messages'],
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None, # リストを返すように
            return_dict=True, # {'input_ids': ..., 'attention_mask': ...} の形式
            # max_length=1024, # 必要に応じてここでmax_lengthを指定
            # truncation=True,   # max_lengthを超える場合切り捨て
        )
        processed_dataset_for_trainer.append(tokenized_sample)
    train_dataset = Dataset.from_list(processed_dataset_for_trainer)
    logging.info(f"トークン化済みデータセット作成完了。サンプル数: {len(train_dataset)}")
    
    assistant_response_prefix_str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    response_template_ids = tokenizer.encode(assistant_response_prefix_str, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids, # トークンIDのリスト
        tokenizer=tokenizer,
        # instruction_template もIDで指定するとより堅牢だが、response_templateがIDなら通常不要
    )

    # import sys
    # print(train_dataset[:2])
    # print(collator(train_dataset[:2]["input_ids"])["labels"])
    # sys.exit()

    # 量子化をモデルに適用
    logging.info(f"ベースモデルをロード中: {base_model}")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map=set_device_map(base_model),
        use_cache=False,
        attn_implementation="eager"
    )
    model.resize_token_embeddings(len(tokenizer))

    # LoRA パラメータ設定
    peft_conf = LoraConfig(
        r=128, lora_alpha=128, lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )

    # 学習パラメータ設定
    sft_conf = SFTConfig(
        output_dir=output_dir,
        bf16=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        logging_steps=1000,
        save_steps=1000,
        save_total_limit=1,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # 学習
    trainer = SFTTrainer(
        model, 
        train_dataset=train_dataset,
        data_collator=collator,
        args=sft_conf,
        peft_config=peft_conf,
        processing_class=tokenizer,
    )

    # ... (SFTTrainerの初期化後、trainer.train() の直前) ...
    logging.info(f"--- Llama-Swallow デバイス配置確認 ---")
    # SFTTrainerによってpeftモデルが準備されているはずなので、trainer.modelで確認
    peft_model = trainer.model 
    logging.info(f"PEFTモデルクラス: {peft_model.__class__.__name__}")
    logging.info(f"PEFTモデル全体のデバイス (最初のパラメータ): {next(peft_model.parameters()).device}")

    # ベースモデルの主要コンポーネントのデバイスを確認
    base_model_proper = peft_model.base_model.model # 通常、この下に実際のトランスフォーマーブロックがある
    
    logging.info(f"入力埋め込みデバイス: {base_model_proper.get_input_embeddings().weight.device}")

    lm_head_to_check = None
    if hasattr(base_model_proper, 'lm_head') and base_model_proper.lm_head is not None:
        lm_head_to_check = base_model_proper.lm_head
        logging.info(f"LMヘッド (base_model_proper.lm_head) のデバイス: {lm_head_to_check.weight.device}")
    elif hasattr(base_model_proper, 'get_output_embeddings') and base_model_proper.get_output_embeddings() is not None:
        # Gemmaなど、get_output_embeddings()がLMヘッドを返す場合
        lm_head_to_check = base_model_proper.get_output_embeddings()
        logging.info(f"出力埋め込み (LMヘッド) のデバイス: {lm_head_to_check.weight.device}")
    else:
        logging.warning("LMヘッドの特定が困難でした。モデル構造を確認してください。")

    try:
        first_layer_device = next(iter(base_model_proper.model.layers[0].parameters())).device
        last_layer_device = next(iter(base_model_proper.model.layers[-1].parameters())).device
        logging.info(f"最初のトランスフォーマーレイヤーのデバイス: {first_layer_device}")
        logging.info(f"最後のトランスフォーマーレイヤーのデバイス: {last_layer_device}")
    except Exception as e:
        logging.warning(f"レイヤーのデバイス取得中にエラー: {e}")
    
    logging.info(f"--- 確認終了 ---")

    logging.info("LoRA学習を開始します...")

    trainer.train()


def main():
    p = argparse.ArgumentParser(description="LoRA 学習スクリプト")
    p.add_argument("--base-model", required=True, help="ベースモデルパス")
    p.add_argument("--model-type", type=str, default="generator", choices=["generator", "predictor"], help="モデルタイプ")
    p.add_argument("--dataset", required=True, help="JSONL データセットパス")
    p.add_argument("--user-template", required=True, help="ユーザーテンプレート .j2")
    p.add_argument("--assistant-template", required=True, help="アシスタントテンプレート .j2")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-filename", type=Path, default=None, help="ログファイル名")
    p.add_argument("--log-type", type=str, default="info", choices=["debug", "info"], help="ログレベル")
    args = p.parse_args()

    # ログ設定
    setup_logging(filename=args.log_filename, log_type=args.log_type)

    # ベースモデルパスの親ディレクトリに出力ディレクトリを作成
    base_model_path = Path(args.base_model)
    output_dir = base_model_path.parent / "lora" / datetime.datetime.now().strftime("%Y%m%d") 
    output_dir.mkdir(parents=True, exist_ok=False)

    train_lora(
        base_model_path,
        args.model_type,
        args.dataset,
        args.user_template,
        args.assistant_template,
        output_dir,
        args.epochs,
        args.batch_size,
        args.accum_steps,
        args.seed
    )

    # 引数の内容を全て config.json に保存
    config = {
        "base_model": str(base_model_path),
        "model_type": args.model_type,
        "dataset": str(Path(args.dataset).resolve()),
        "user_template": args.user_template,
        "assistant_template": args.assistant_template,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "seed": args.seed,
    }
    write_config(output_dir, config)
    

if __name__ == "__main__":
    main()
