import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset, load_dataset
import math
import re
import logging
from typing import List, Tuple, Dict, Any
import os

# --- 0. ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. 設定クラス ---
class TrainingConfig:
    # Paths
    base_generator_model_name_or_path: str = "models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base"
    generator_lora_adapter_path: str = "models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/20250530/lora/checkpoint-7858" # 存在しない場合はNoneまたは空文字列
    predictor_model_path: str = "models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base"
    dataset_path: str = "./dummy_contexts.jsonl"
    output_dir: str = "./ppo_generator_output_v2"

    # Models
    use_qlora_for_generator_base: bool = True # True for QLoRA

    # LoRA (if generator_lora_adapter_path is not provided or fails to load)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_task_type: str = "CAUSAL_LM"

    # PPO
    ppo_learning_rate: float = 1.41e-5
    ppo_batch_size: int = 2
    ppo_mini_batch_size: int = 1
    ppo_epochs: int = 4 # Renamed from num_ppo_epochs based on PPOConfig docs
    ppo_log_with: str = None
    gradient_accumulation_steps: int = 1
    # PPOConfigに存在する他の重要なパラメータもここに追加可能
    # 例: kl_coef, cliprange, vf_coef, etc.
    kl_coef: float = 0.05 # ドキュメントにあるPPOConfigのデフォルト値の一つ

    # Generation
    max_context_length: int = 512
    max_new_tokens_generation: int = 128

    # Training
    num_training_epochs: int = 3
    
    # Hardware
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bnb_config: BitsAndBytesConfig = None

    def __init__(self):
        if self.use_qlora_for_generator_base and self.device.type == "cuda":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            self.use_qlora_for_generator_base = False
            self.bnb_config = None
        
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        if self.dataset_path == "./dummy_contexts.jsonl" and not os.path.exists(self.dataset_path):
            with open(self.dataset_path, "w", encoding="utf-8") as f:
                f.write('{"text": "日本の首都は東京です。東京には多くの観光名所があります。"}\n')
                f.write('{"text": "大規模言語モデルは、自然言語処理の分野で注目されています。"}\n')
                f.write('{"text": "この文脈はテスト用です。"}\n')
                f.write('{"text": "今日は晴天で、絶好の行楽日和です。ピクニックに行きませんか。"}\n')

# --- 2. モデルロード関数 ---
def load_generator_model(config: TrainingConfig) -> Tuple[AutoModelForCausalLMWithValueHead, AutoTokenizer]:
    logger.info(f"Loading generator base model from: {config.base_generator_model_name_or_path}")
    
    model_kwargs = {
        "quantization_config": config.bnb_config,
        "torch_dtype": torch.bfloat16 if (config.use_qlora_for_generator_base or (config.device.type == "cuda" and torch.cuda.is_bf16_supported())) else torch.float16,
        "trust_remote_code": True
    }
    if config.use_qlora_for_generator_base and config.device.type == "cuda":
        model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["device_map"] = "auto"

    base_model = AutoModelForCausalLM.from_pretrained(config.base_generator_model_name_or_path, **model_kwargs)

    if config.use_qlora_for_generator_base and config.device.type == "cuda":
        logger.info("Preparing base model for k-bit training (QLoRA).")
        base_model = prepare_model_for_kbit_training(base_model)

    tokenizer = AutoTokenizer.from_pretrained(config.base_generator_model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    peft_model_path = config.generator_lora_adapter_path
    if peft_model_path and os.path.exists(peft_model_path):
        logger.info(f"Loading LoRA adapter for generator from: {peft_model_path}")
        try:
            peft_model = PeftModel.from_pretrained(base_model, peft_model_path, is_trainable=True)
            logger.info("Successfully loaded and attached LoRA adapter.")
        except Exception as e:
            logger.error(f"Error loading LoRA adapter from {peft_model_path}: {e}", exc_info=True)
            logger.info("Initializing new LoRA layers as a fallback.")
            lora_obj_config = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, bias="none", task_type=config.lora_task_type)
            peft_model = get_peft_model(base_model, lora_obj_config)
    else:
        logger.info("No LoRA adapter path provided or path does not exist. Initializing new LoRA layers.")
        lora_obj_config = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, bias="none", task_type=config.lora_task_type)
        peft_model = get_peft_model(base_model, lora_obj_config)
    
    logger.info("Creating PPO model (Actor/Critic) with ValueHead...")
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model)
    
    logger.info("Generator PPO model loaded. Trainable parameters:")
    # ppo_model.print_trainable_parameters()
    return ppo_model, tokenizer

def load_predictor_model(config: TrainingConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info(f"Loading predictor model from: {config.predictor_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.predictor_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.predictor_model_path,
        torch_dtype=torch.bfloat16 if (config.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    logger.info("Predictor model loaded.")
    return model, tokenizer

# --- 3. IFDスコア算出関連関数 ---
# (calculate_log_probability, calculate_normalized_ifd_reward, calculate_batch_rewards は前回のものを流用)
def calculate_log_probability(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    text_sequence: str, 
    context_sequence: str = None, 
    device: torch.device = torch.device("cpu")
) -> float:
    model.to(device)
    max_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length is not None else 512

    if context_sequence:
        full_text = context_sequence + tokenizer.eos_token + text_sequence
        input_ids = tokenizer.encode(full_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        
        context_plus_instruction_text = context_sequence
        context_plus_instruction_ids = tokenizer.encode(context_plus_instruction_text + tokenizer.eos_token, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        context_plus_instruction_length = context_plus_instruction_ids.shape[1]
        
        labels = input_ids.clone()
        if input_ids.shape[1] > context_plus_instruction_length:
            labels[:, :context_plus_instruction_length] = -100
        else:
            logger.debug(f"LogProb: Answer part empty/short. Full: {input_ids.shape[1]}, Ctx+Ins: {context_plus_instruction_length}. Ctx: '{context_sequence[:30]}...', Ans: '{text_sequence[:30]}...'")
            return 1e9 
    else:
        input_ids = tokenizer.encode(text_sequence, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        labels = input_ids.clone()

    if input_ids.shape[1] == 0 or (labels == -100).all().item():
        logger.debug(f"LogProb: No valid tokens for loss. Input shape: {input_ids.shape}, Text: '{text_sequence[:30]}...'")
        return 1e9

    with torch.no_grad():
        try:
            outputs = model(input_ids, labels=labels)
            log_prob_score = outputs.loss.item()
        except Exception as e:
            logger.error(f"Error during model inference in calculate_log_probability: {e}", exc_info=True)
            logger.error(f"Input IDs shape: {input_ids.shape}, Labels shape: {labels.shape}")
            logger.error(f"Input IDs: {input_ids}")
            logger.error(f"Labels: {labels}")
            return 1e9

    if math.isnan(log_prob_score) or math.isinf(log_prob_score):
        logger.debug(f"LogProb: Score is NaN/Inf. Text: '{text_sequence[:30]}...'. Defaulting to large loss.")
        return 1e9
    return log_prob_score

def calculate_normalized_ifd_reward(
    predictor_model: AutoModelForCausalLM, 
    predictor_tokenizer: AutoTokenizer, 
    context_plus_question: str, 
    answer: str, 
    device: torch.device
) -> torch.Tensor:
    if not context_plus_question or not answer:
        logger.debug(f"IFDReward: Empty Ctx+Q or Ans. Ctx+Q: '{context_plus_question[:30]}...', A: '{answer[:30]}...'. Reward 0.")
        return torch.tensor(0.0, device=device)

    s_A_given_Q = calculate_log_probability(predictor_model, predictor_tokenizer, answer, context_sequence=context_plus_question, device=device)
    s_A = calculate_log_probability(predictor_model, predictor_tokenizer, answer, context_sequence=None, device=device)

    logger.debug(f"s_A_given_Q: {s_A_given_Q:.4f}, s_A: {s_A:.4f} for Ctx+Q: '{context_plus_question[:50]}...', A: '{answer[:50]}...'")

    if abs(s_A) < 1e-9:
        ifd_score = float('inf') if abs(s_A_given_Q) > 1e-9 else 1.0
    else:
        ifd_score = s_A_given_Q / s_A
    
    if math.isinf(ifd_score) or ifd_score > 1e9:
        normalized_score = 0.0
    elif ifd_score < 0:
        logger.warning(f"Negative IFD score ({ifd_score:.4f}). Defaulting reward to 1.0.")
        normalized_score = 1.0 
    else:
        normalized_score = 1.0 / (1.0 + ifd_score)
    
    logger.debug(f"IFD: {ifd_score:.4f}, Normalized Reward: {normalized_score:.4f}")
    return torch.tensor(normalized_score, device=device)

def calculate_batch_rewards(
    predictor_model: AutoModelForCausalLM,
    predictor_tokenizer: AutoTokenizer,
    parsed_predictor_inputs: List[Tuple[str, str]],
    device: torch.device
) -> List[torch.Tensor]:
    rewards = []
    for context_plus_question, answer in parsed_predictor_inputs:
        reward = calculate_normalized_ifd_reward(predictor_model, predictor_tokenizer, context_plus_question, answer, device)
        rewards.append(reward)
    return rewards

# --- 4. データ処理とテキストパース ---
def load_contexts_from_file(dataset_path: str, num_samples: int = None) -> List[str]:
    contexts = []
    try:
        # load_dataset を使う場合
        # dataset = load_dataset("json", data_files=dataset_path, split="train")
        # for item in dataset:
        #     if "text" in item and item["text"].strip():
        #         contexts.append(item["text"].strip())
        # if num_samples:
        #     contexts = contexts[:num_samples]
        with open(dataset_path, "r", encoding="utf-8") as f:
             for line in f:
                 try:
                     import json
                     record = json.loads(line)
                     if "text" in record and record["text"].strip():
                         contexts.append(record["text"].strip())
                 except json.JSONDecodeError:
                     logger.warning(f"Skipping malformed JSON line in {dataset_path}: {line.strip()}")
                     continue
        if num_samples:
            contexts = contexts[:num_samples]
        logger.info(f"Loaded {len(contexts)} contexts from {dataset_path}.")
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {dataset_path}. Please check the path.")
    except Exception as e:
        logger.error(f"Error loading dataset from {dataset_path}: {e}", exc_info=True)
    return contexts

def prepare_generator_prompts(contexts: List[str], config: TrainingConfig) -> List[str]:
    return [
        f"以下の文脈から主要な情報に関する質問とその質問に対する最も端的な回答を一つ作成してください。\n文脈の後に、まず「質問:」と記述し質問を続け、次に改行して「回答:」と記述し回答を続けてください。\n\n文脈:\n{ctx[:config.max_context_length]}\n\n質問:"
        for ctx in contexts
    ]

def parse_generated_text_to_qa(
    original_prompt_for_gen: str,
    generated_text_from_gen: str
) -> Tuple[str, str]:
    # TODO: このパースロジックは、Generatorの出力形式に合わせて精교に実装する必要がある最重要箇所です。
    #       現状は単純な分割ですが、より頑健な方法を検討してください。
    #       (例: Generatorに<QUESTION_END>, <ANSWER_START>のような特殊トークンを出力させる)
    question_part = ""
    answer_part = ""
    
    # "回答:" を基準に分割
    match_ans = re.search(r"回答:(.*)", generated_text_from_gen, re.DOTALL | re.IGNORECASE)
    if match_ans:
        answer_part = match_ans.group(1).strip()
        question_candidate = generated_text_from_gen[:match_ans.start()].strip()
        if question_candidate.lower().startswith("質問:"): # プロンプトの末尾が "質問:" のため
            question_part = question_candidate[len("質問:"):].strip()
        else:
            question_part = question_candidate # プロンプトの "質問:" の後に直接質問が生成された場合
    else:
        # "回答:" が見つからない場合、単純に全体を質問とみなし、回答は空
        logger.warning(f"Could not find '回答:' separator in: '{generated_text_from_gen[:100]}...'. Treating as question only.")
        question_part = generated_text_from_gen.strip()
        if question_part.lower().startswith("質問:"):
            question_part = question_part[len("質問:"):].strip()
        answer_part = ""

    if not question_part: logger.warning(f"Parsed question is empty. Gen text: '{generated_text_from_gen[:100]}...'")
    if not answer_part: logger.warning(f"Parsed answer is empty. Gen text: '{generated_text_from_gen[:100]}...'")
    return question_part, answer_part

def create_predictor_inputs_from_parsed(
    original_prompts_for_gen: List[str],
    parsed_qas_from_gen: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    predictor_inputs = []
    for i, (gen_q, gen_a) in enumerate(parsed_qas_from_gen):
        original_prompt = original_prompts_for_gen[i]
        context_match = re.search(r"文脈:\n(.*?)\n\n質問:", original_prompt, re.DOTALL)
        if not context_match:
            logger.warning(f"Could not parse context from original_prompt: {original_prompt[:100]}...")
            predictor_inputs.append(("", gen_a))
            continue
        actual_context = context_match.group(1).strip()
        context_plus_question_for_predictor = actual_context + "\n質問: " + gen_q
        predictor_inputs.append((context_plus_question_for_predictor, gen_a))
    return predictor_inputs

# --- 5. メイントレーニング関数 ---
def train():
    cfg = TrainingConfig()
    logger.info(f"Using device: {cfg.device}")

    # モデルとトークナイザーのロード
    try:
        generator_ppo_model, generator_tokenizer = load_generator_model(cfg)
        predictor_model, predictor_tokenizer = load_predictor_model(cfg)
    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        return

    # PPOConfig の設定 (TRLドキュメントに記載の引数を参照)
    ppo_config = PPOConfig(
        # exp_name="ppo_qa_generator", # 実験名 (任意)
        learning_rate=cfg.ppo_learning_rate,
        batch_size=cfg.ppo_batch_size, # rollout buffer size (PPO step毎に収集するサンプル数)
        mini_batch_size=cfg.ppo_mini_batch_size, # PPOのミニバッチサイズ
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_ppo_epochs=cfg.ppo_epochs, # TRLドキュメントではppo_epochs
        # log_with=cfg.ppo_log_with,
        kl_coef=cfg.kl_coef,
        # 他のPPOConfigパラメータ (例: cliprange, vf_coef, gamma, lam) はデフォルト値を使用
        # seed=42, # 再現性のため
        # remove_unused_columns=False # Hugging Face Datasetを使う場合に重要
    )
    logger.info(f"PPOConfig initialized: {ppo_config}")

    # データセットの準備 (PPOTrainerに渡すため)
    # PPOTrainerの `dataset` 引数は、通常、クエリ（プロンプト）のリストを含むHugging Face Datasetを期待します。
    # ここでは、学習ループ内で動的にプロンプトを生成するため、初期化時には空のDatasetか、
    # 代表的なプロンプトを含む少数のサンプルデータセットを渡すのが一般的です。
    # TRLの多くの例では、`tokenizer` と `dataset` の両方を `PPOTrainer` に渡しています。
    
    # ダミーの初期データセット (プロンプト文字列のカラム名 'query' を持つ)
    # 実際の学習では、このデータセット自体は直接使われず、ループ内で生成したプロンプトを使う
    dummy_prompts = prepare_generator_prompts(load_contexts_from_file(cfg.dataset_path, num_samples=cfg.ppo_batch_size or 4), cfg)
    if not dummy_prompts:
        logger.error("Failed to create dummy prompts for PPOTrainer initialization. Check dataset loading and prompt preparation.")
        return
        
    initial_dataset_dict = {"query": dummy_prompts}
    # TRLの一部の例では、トークン化されたinput_idsも要求することがある
    # initial_dataset_dict["input_ids"] = [generator_tokenizer.encode(q, truncation=True, max_length=cfg.max_context_length) for q in dummy_prompts]
    initial_dataset = Dataset.from_dict(initial_dataset_dict)


    # PPOTrainer の初期化
    try:
        ppo_trainer = PPOTrainer(
            args=ppo_config,
            model=generator_ppo_model,
            ref_model=None, # TRLが内部で作成
            processing_class=generator_tokenizer,
            dataset=initial_dataset, # プロンプトのリストを含むHugging Face Dataset
            # data_collator=DataCollatorForLanguageModeling(tokenizer=generator_tokenizer, mlm=False) # 必要に応じて
        )
        logger.info("PPOTrainer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize PPOTrainer: {e}", exc_info=True)
        return

    # 学習ループ
    generation_kwargs = {
        "top_k": 0.0, "top_p": 1.0, "do_sample": True,
        "pad_token_id": generator_tokenizer.pad_token_id,
        "max_new_tokens": cfg.max_new_tokens_generation,
        "eos_token_id": generator_tokenizer.eos_token_id,
    }

    logger.info("Starting PPO training...")
    all_contexts = load_contexts_from_file(cfg.dataset_path) # 全てのコンテキストをロード
    if not all_contexts:
        logger.error("No contexts loaded for training. Exiting.")
        return

    for epoch in range(cfg.num_training_epochs):
        logger.info(f"Epoch {epoch+1}/{cfg.num_training_epochs}")
        
        # データセットをシャッフル（任意だが推奨）
        # import random
        # random.shuffle(all_contexts)

        for i in range(0, len(all_contexts), cfg.ppo_batch_size):
            batch_contexts = all_contexts[i : i + cfg.ppo_batch_size]
            if not batch_contexts: continue
            
            current_batch_size = len(batch_contexts)
            logger.info(f"  Processing batch {i // cfg.ppo_batch_size + 1}, size: {current_batch_size}")

            prompt_texts_for_gen = prepare_generator_prompts(batch_contexts, cfg)
            
            # PPOTrainer.generate はトークナイズされたプロンプトのリストを期待 (各要素は1D Tensor)
            query_tensors_for_gen = [generator_tokenizer.encode(prompt, return_tensors="pt").to(cfg.device).squeeze(0) 
                                     for prompt in prompt_texts_for_gen]

            # Generatorによる生成
            response_tensors_list: List[torch.Tensor] = []
            try:
                # ppo_trainer.generate はリスト内の各テンソルに対して生成を行う
                # TRLのバージョンや使い方により、バッチ化されたテンソルを期待する場合もある
                for query_tensor in query_tensors_for_gen:
                    response_tensor = ppo_trainer.generate(query_tensor, **generation_kwargs)
                    response_tensors_list.append(response_tensor.squeeze(0))
            except Exception as e:
                logger.error(f"Error during ppo_trainer.generate: {e}", exc_info=True)
                continue

            generated_texts_from_gen = [generator_tokenizer.decode(r_tensor, skip_special_tokens=True) if r_tensor.numel() > 0 else ""
                                        for r_tensor in response_tensors_list]

            # パース処理
            parsed_qas = [parse_generated_text_to_qa(prompt_texts_for_gen[j], generated_texts_from_gen[j]) 
                          for j in range(len(generated_texts_from_gen))]
            
            predictor_inputs_for_reward = create_predictor_inputs_from_parsed(prompt_texts_for_gen, parsed_qas)
            
            # 有効なサンプルのみフィルタリング
            valid_indices = [
                idx for idx, ((_ctx_q, ans), resp_t) in enumerate(zip(predictor_inputs_for_reward, response_tensors_list)) 
                if ans and resp_t.numel() > 0 # 回答がパースでき、かつresponse_tensorが空でない
            ]
            
            if not valid_indices:
                logger.warning("    No valid samples after parsing and generation checks in this batch. Skipping PPO step.")
                continue

            valid_query_tensors = [query_tensors_for_gen[idx] for idx in valid_indices]
            valid_response_tensors = [response_tensors_list[idx] for idx in valid_indices]
            valid_predictor_inputs_for_reward = [predictor_inputs_for_reward[idx] for idx in valid_indices]
            
            # 報酬計算
            rewards_list = calculate_batch_rewards(predictor_model, predictor_tokenizer, valid_predictor_inputs_for_reward, cfg.device)
            
            if not rewards_list or len(rewards_list) != len(valid_query_tensors):
                 logger.warning(f"    Reward calculation/filtering mismatch. Expected {len(valid_query_tensors)}, got {len(rewards_list)}. Skipping PPO step.")
                 continue
            rewards_tensor = torch.stack(rewards_list).to(cfg.device)
            
            # PPOステップ
            try:
                logger.debug(f"    PPO Step - Query Tensors ({len(valid_query_tensors)}): {[t.shape for t in valid_query_tensors] if valid_query_tensors else 'Empty'}")
                logger.debug(f"    PPO Step - Response Tensors ({len(valid_response_tensors)}): {[t.shape for t in valid_response_tensors] if valid_response_tensors else 'Empty'}")
                logger.debug(f"    PPO Step - Rewards Tensor: {rewards_tensor.shape if rewards_tensor.numel() > 0 else 'Empty'}")

                stats = ppo_trainer.step(valid_query_tensors, valid_response_tensors, rewards_tensor)
                
                log_output = {"epoch": epoch + 1, "batch_step": i // cfg.ppo_batch_size + 1}
                if rewards_tensor.numel() > 0: log_output["mean_reward"] = rewards_tensor.mean().item()
                if stats:
                    for stat_key in ["ppo/loss/policy", "ppo/loss/value", "ppo/policy/entropy", "ppo/policy/approxkl", "ppo/mean_non_score_reward", "objective/kl", "objective/entropy", "objective/reward"]:
                        if stat_key in stats: log_output[stat_key] = stats[stat_key]
                logger.info(f"    Log: {log_output}")

                if cfg.ppo_log_with == "wandb":
                    # W&Bにログを送る場合の整形
                    log_queries = [prompt_texts_for_gen[idx] for idx in valid_indices]
                    log_responses = [generated_texts_from_gen[idx] for idx in valid_indices]
                    ppo_trainer.log_stats(stats, {"query": log_queries}, rewards_tensor, {"response": log_responses})

            except Exception as e:
                logger.error(f"    Error during PPO step: {e}", exc_info=True)
                continue 
        
        # エポックごとのモデル保存
        if cfg.output_dir:
            current_epoch_output_dir = os.path.join(cfg.output_dir, f"epoch_{epoch+1}")
            if not os.path.exists(current_epoch_output_dir): os.makedirs(current_epoch_output_dir, exist_ok=True)
            try:
                logger.info(f"Saving model for epoch {epoch+1} to {current_epoch_output_dir}")
                ppo_trainer.save_model(current_epoch_output_dir) # TRLの推奨する方法で保存
                generator_tokenizer.save_pretrained(current_epoch_output_dir)
            except Exception as e:
                logger.error(f"Error saving model at epoch {epoch+1}: {e}", exc_info=True)

    # 学習後の最終モデルの保存
    if cfg.output_dir:
        final_output_dir = os.path.join(cfg.output_dir, "final")
        if not os.path.exists(final_output_dir): os.makedirs(final_output_dir, exist_ok=True)
        logger.info(f"Saving final model to {final_output_dir}")
        try:
            ppo_trainer.save_model(final_output_dir)
            generator_tokenizer.save_pretrained(final_output_dir)
        except Exception as e:
            logger.error(f"Error saving final model: {e}", exc_info=True)

    logger.info("PPO training finished.")

if __name__ == "__main__":
    train()
