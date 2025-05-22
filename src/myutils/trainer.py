#!/usr/bin/env python3
import os
import logging
import subprocess
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

# llama.cpp のパス（convert script があるディレクトリ）
LLAMA_CPP_ROOT = Path(os.getenv("LLAMA_CPP_ROOT", "/opt/llama"))

def prepare_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return tok

def prepare_model_with_lora(model_name: str, peft_config: LoraConfig):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
        device_map="auto",
        use_cache=False
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def train_and_export(
    model,
    tokenizer,
    examples,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    accum_steps: int,
    base_gguf_path: Path
):
    # --- SFT Training ---
    tokenized = tokenizer.apply_chat_template(
        examples,
        tokenize=True,
        add_generation_prompt=False
    )
    response_ids = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False
    )[1:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_ids, tokenizer=tokenizer
    )

    peft_conf = SFTConfig(
        output_dir=str(output_dir),
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
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    trainer = SFTTrainer(
        model, train_dataset=tokenized, data_collator=collator,
        args=peft_conf, tokenizer=tokenizer
    )
    trainer.train()

    # --- LoRA マージ ---
    logging.info("LoRA をベースモデルにマージします")
    if isinstance(model, PeftModel):
        model.merge_and_unload()
    merged_pt = output_dir / "merged_lora.pt"
    torch.save(model.state_dict(), merged_pt)
    logging.info(f"マージ済モデルを保存: {merged_pt}")

    # --- GGUF 変換 ---
    gguf_out = output_dir / "final_model.gguf"
    convert_script = LLAMA_CPP_ROOT / "convert_lora_to_gguf.py"
    cmd = [
        "python3", str(convert_script),
        "--base-model-path", str(base_gguf_path),
        "--lora-pt-path", str(merged_pt),
        "--outfile", str(gguf_out),
    ]
    logging.info(f"GGUF 生成コマンド: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logging.info(f"GGUF モデルを生成: {gguf_out}")

    return gguf_out
