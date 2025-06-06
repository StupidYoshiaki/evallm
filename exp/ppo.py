import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
# GRPOTrainer と GRPOConfig をインポート (TRLのバージョンによってGRPOConfigがない場合、直接引数で指定するか、TrainingArgumentsを流用)
# 最新のTRLでは GRPOConfig が明示的にない場合、GRPOTrainerの引数で直接設定するか、
# TrainingArgumentsをベースにした設定を使うことがあります。
# ここでは、PPOConfigに似たパラメータを持つことを想定し、後でGRPOTrainerの引数に合わせます。
from trl import GRPOTrainer # GRPOConfig がなければ TrainingArguments や直接引数
from trl import GRPOConfig # もし存在すれば。なければPPConfigやTrainingArgumentsで代替or直接引数
# from transformers import TrainingArguments # GRPOConfigがない場合の代替として

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
    base_generator_model_name_or_path: str = "../models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base"
    generator_lora_adapter_path: str = "../models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/20250530/lora/checkpoint-7858"
    predictor_model_path: str = "../models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base"
    dataset_path: str = "./dummy_contexts.jsonl"
    output_dir: str = "./grpo_generator_output"

    # Models
    use_qlora_for_generator_base: bool = True

    # LoRA
    lora_r: int = 128
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_task_type: str = "CAUSAL_LM"

    # grpo/GRPO Config (PPOConfigから流用し、GRPOに合わせて調整)
    grpo_learning_rate: float = 1.41e-5
    grpo_batch_size: int = 8 # rollout buffer size (GRPOでは1ステップで処理するプロンプト数)
    # grpo_mini_batch_size: int = 1 # grpoではPPOのようなミニバッチ内部ループは通常ない
    grpo_epochs: int = 1 # GRPOのtrain()がイテレーションをどう扱うかによる (total_episodesのようなものがあるかも)
    grpo_log_with: str = None
    gradient_accumulation_steps: int = 1
    # GRPO特有のパラメータ (ドキュメント参照)
    beta: float = 0.1 # grpoのbeta (KLペナルティ係数に近い役割)
    max_length: int = 1024 # GRPOTrainerのgenerate時の最大長 (プロンプト+生成)
    max_prompt_length: int = 512 # プロンプト自体の最大長
    max_new_tokens: int = 128 # 生成する新規トークン数 (GRPOTrainerが内部で使う場合)

    # Training
    num_training_steps: int = 100 # GRPOではエピソード数やステップ数で指定することが多い

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
                f.write('{"prompt": "日本の首都は東京です。東京には多くの観光名所があります。"}\n')
                f.write('{"prompt": "大規模言語モデルは、自然言語処理の分野で注目されています。"}\n')

# グローバル変数としてpredictorモデルとトークナイザーを保持 (報酬関数内でアクセスするため)
# より良い設計は、報酬関数をクラスにし、コンストラクタでこれらを受け取る形
PREDICTOR_MODEL = None
PREDICTOR_TOKENIZER = None
PREDICTOR_DEVICE = None

# --- 2. モデルロード関数 ---
def load_models_and_tokenizers(config: TrainingConfig):
    global PREDICTOR_MODEL, PREDICTOR_TOKENIZER, PREDICTOR_DEVICE
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

    generator_tokenizer = AutoTokenizer.from_pretrained(config.base_generator_model_name_or_path, trust_remote_code=True)
    if generator_tokenizer.pad_token is None:
        generator_tokenizer.pad_token = generator_tokenizer.eos_token
        generator_tokenizer.pad_token_id = generator_tokenizer.eos_token_id

    peft_model_path = config.generator_lora_adapter_path
    if peft_model_path and os.path.exists(peft_model_path):
        logger.info(f"Loading LoRA adapter for generator from: {peft_model_path}")
        try:
            # GRPOTrainerはWithValueHeadなしのモデルを期待することが多い
            generator_peft_model = PeftModel.from_pretrained(base_model, peft_model_path, is_trainable=True)
            logger.info("Successfully loaded and attached LoRA adapter.")
        except Exception as e:
            logger.error(f"Error loading LoRA adapter: {e}", exc_info=True)
            logger.info("Initializing new LoRA layers as a fallback.")
            lora_obj_config = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, bias="none", task_type=config.lora_task_type)
            generator_peft_model = get_peft_model(base_model, lora_obj_config)
    else:
        logger.info("No LoRA adapter path. Initializing new LoRA layers for the generator.")
        lora_obj_config = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, bias="none", task_type=config.lora_task_type)
        generator_peft_model = get_peft_model(base_model, lora_obj_config)
    
    logger.info("Generator PEFT model loaded. Trainable parameters:")
    generator_peft_model.print_trainable_parameters()

    # Predictorモデルのロード
    logger.info(f"Loading predictor model from: {config.predictor_model_path}")
    PREDICTOR_TOKENIZER = AutoTokenizer.from_pretrained(config.predictor_model_path, trust_remote_code=True)
    if PREDICTOR_TOKENIZER.pad_token is None:
        PREDICTOR_TOKENIZER.pad_token = PREDICTOR_TOKENIZER.eos_token
        PREDICTOR_TOKENIZER.pad_token_id = PREDICTOR_TOKENIZER.eos_token_id

    PREDICTOR_MODEL = AutoModelForCausalLM.from_pretrained(
        config.predictor_model_path,
        torch_dtype=torch.bfloat16 if (config.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16,
        device_map="auto", # PredictorもGPUに乗せるか、CPUにするか注意
        trust_remote_code=True
    )
    PREDICTOR_MODEL.eval()
    PREDICTOR_DEVICE = PREDICTOR_MODEL.device # Predictorがロードされたデバイスを保持
    logger.info(f"Predictor model loaded on device: {PREDICTOR_DEVICE}")

    return generator_peft_model, generator_tokenizer


# --- 3. IFDスコア算出関連関数 (変更なし) ---
# calculate_log_probability, calculate_normalized_ifd_reward は前回のものを流用

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
) -> float: # torch.Tensorではなくfloatを返すように変更
    if not context_plus_question or not answer:
        logger.debug(f"IFDReward: Empty Ctx+Q or Ans. Reward 0.")
        return 0.0

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
        normalized_score = 1.0 
    else:
        normalized_score = 1.0 / (1.0 + ifd_score)
    
    logger.debug(f"IFD: {ifd_score:.4f}, Normalized Reward: {normalized_score:.4f}")
    return normalized_score


# --- 4. データ処理とテキストパース ---
# (load_contexts_from_file, prepare_generator_prompts, parse_generated_text_to_qa, create_predictor_inputs_from_parsed は前回のものを流用)
def load_contexts_from_file(dataset_path: str, num_samples: int = None) -> List[str]:
    contexts = []
    try:
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
        f"以下の文脈から主要な情報に関する質問とその質問に対する最も端的な回答を一つ作成してください。\n文脈の後に、まず「質問:」と記述し質問を続け、次に改行して「回答:」と記述し回答を続けてください。\n\n文脈:\n{ctx[:config.max_prompt_length]}\n\n質問:"
        for ctx in contexts
    ]

def parse_generated_text_to_qa(
    original_prompt_for_gen: str,
    generated_text_from_gen: str
) -> Tuple[str, str]:
    # TODO: このパースロジックは最重要。実際の出力に合わせて改善必須。
    question_part = ""
    answer_part = ""
    match_ans = re.search(r"回答:(.*)", generated_text_from_gen, re.DOTALL | re.IGNORECASE)
    if match_ans:
        answer_part = match_ans.group(1).strip()
        question_candidate = generated_text_from_gen[:match_ans.start()].strip()
        if question_candidate.lower().startswith("質問:"):
            question_part = question_candidate[len("質問:"):].strip()
        else:
            question_part = question_candidate
    else:
        logger.warning(f"Could not find '回答:' separator in: '{generated_text_from_gen[:100]}...'")
        question_part = generated_text_from_gen.strip() # 全体を質問とみなす
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

# --- 5. GRPOTrainer用の報酬関数 ---
def ifd_reward_function(generated_samples: List[str], prompts: List[str], **kwargs) -> List[float]:
    """
    GRPOTrainer から呼び出される報酬関数。
    generated_samples: Generatorがプロンプトの後に生成したテキスト (「質問: ... 回答: ...」形式を期待)
    prompts: Generatorに入力された元のプロンプト (「文脈: ... 質問:」形式を期待)
    """
    global PREDICTOR_MODEL, PREDICTOR_TOKENIZER, PREDICTOR_DEVICE
    if PREDICTOR_MODEL is None or PREDICTOR_TOKENIZER is None or PREDICTOR_DEVICE is None:
        raise ValueError("Predictor model/tokenizer/device not loaded. Call load_models_and_tokenizers first.")

    rewards = []
    
    # generated_samples (Generatorの出力) と prompts (Generatorへの入力) から、
    # IFD計算に必要な (context_plus_question, answer) を再構築
    parsed_qas_from_gen = [parse_generated_text_to_qa(prompts[i], generated_samples[i]) for i in range(len(generated_samples))]
    predictor_inputs_for_reward = create_predictor_inputs_from_parsed(prompts, parsed_qas_from_gen)

    for ctx_q, ans in predictor_inputs_for_reward:
        if not ans: # パース失敗などで回答がなければ報酬0
            rewards.append(0.0)
            continue
        reward_val = calculate_normalized_ifd_reward(PREDICTOR_MODEL, PREDICTOR_TOKENIZER, ctx_q, ans, PREDICTOR_DEVICE)
        rewards.append(reward_val.item()) # float値をリストに追加
    
    logger.info(f"Calculated rewards for batch: {rewards[:5]}...") # 最初の5件をログ表示
    return rewards


# --- 6. メイントレーニング関数 ---
def train():
    global PREDICTOR_MODEL, PREDICTOR_TOKENIZER, PREDICTOR_DEVICE # グローバル変数を更新するため

    cfg = TrainingConfig()
    logger.info(f"Using device: {cfg.device}")

    try:
        generator_peft_model, generator_tokenizer = load_models_and_tokenizers(cfg)
        # PREDICTOR_MODEL, PREDICTOR_TOKENIZER, PREDICTOR_DEVICE は load_models_and_tokenizers 内で設定される
    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        return

    # データセットの準備 (プロンプトのリスト)
    all_contexts = load_contexts_from_file(cfg.dataset_path)
    if not all_contexts:
        logger.error("No contexts loaded for training. Exiting.")
        return
    
    # GRPOTrainerはプロンプト文字列のリストを持つDatasetを期待することが多い
    # 'query' カラムにプロンプト文字列を入れる
    train_prompts = prepare_generator_prompts(all_contexts, cfg)
    train_dataset_dict = {}
    train_dataset_dict["prompt"] = train_prompts
    # GRPOTrainerが内部でトークン化する場合、input_idsは不要かもしれない
    # train_dataset_dict["input_ids"] = [generator_tokenizer.encode(p, truncation=True, max_length=cfg.max_prompt_length) for p in train_prompts]
    train_dataset = Dataset.from_dict(train_dataset_dict)

    logger.info(f"Prepared train_dataset with {len(train_dataset)} prompts.")

    # GRPOConfig の設定
    # TRLのドキュメントで GRPOTrainer に渡せる GRPOConfig の引数を確認
    grpo_config = GRPOConfig(
        # model_name_or_path=cfg.base_generator_model_name_or_path, # 必須の場合がある
        beta=cfg.beta,
        learning_rate=cfg.grpo_learning_rate,
        per_device_train_batch_size=cfg.grpo_batch_size, # GRPOTrainerではこれがロールアウトごとのプロンプト数になることが多い
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_completion_length=cfg.max_length,             # 生成される最大長 (プロンプト + 生成)
        max_prompt_length=cfg.max_prompt_length, # プロンプトの最大長
        # max_new_tokens=cfg.max_new_tokens,     # GRPOTrainerが内部生成時に使う
        logging_steps=1, # どのくらいの頻度でログを出すか
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.grpo_epochs, # GRPOTrainerのtrain()がこれをどう解釈するか (総ステップ数やエピソード数で指定することも多い)
        # save_steps=50, # 保存頻度
        remove_unused_columns=False, # Datasetのカラムを削除しない
        # label_pad_token_id = generator_tokenizer.pad_token_id, # 損失計算時に無視するID
        # GRPOTrainerはPPOと異なり、ppo_epochsやmini_batch_sizeは直接使わないことが多い
    )
    logger.info(f"GRPOConfig: {grpo_config}")


    # GRPOTrainer の初期化
    try:
        grpo_trainer = GRPOTrainer(
            model=generator_peft_model, # PEFT適用済みのモデル (WithValueHeadなし)
            args=grpo_config,          # GRPOConfigオブジェクト
            processing_class=generator_tokenizer,
            train_dataset=train_dataset, # プロンプトのデータセット
            reward_funcs=ifd_reward_function, # カスタム報酬関数
            # peft_config=lora_obj_config # モデルロード時に適用済みなので不要な場合が多い
        )
        logger.info("GRPOTrainer initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize GRPOTrainer: {e}", exc_info=True)
        return

    # 学習の実行
    logger.info("Starting GRPO training...")
    try:
        grpo_trainer.train(resume_from_checkpoint=None) # resume_from_checkpoint は適宜設定
    except Exception as e:
        logger.error(f"Error during GRPOTrainer.train(): {e}", exc_info=True)
        return

    # 学習後の最終モデルの保存
    if cfg.output_dir:
        final_output_dir = os.path.join(cfg.output_dir, "final_grpo_model")
        if not os.path.exists(final_output_dir): os.makedirs(final_output_dir, exist_ok=True)
        logger.info(f"Saving final model to {final_output_dir}")
        try:
            grpo_trainer.save_model(final_output_dir) # PEFTアダプタが保存される
            generator_tokenizer.save_pretrained(final_output_dir)
        except Exception as e:
            logger.error(f"Error saving final model: {e}", exc_info=True)

    logger.info("GRPO training finished.")

if __name__ == "__main__":
    train()
