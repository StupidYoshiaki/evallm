import argparse
import json
import logging
import datetime
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import set_seed
from trl import GRPOConfig, GRPOTrainer
from sentence_transformers import SentenceTransformer, util
from janome.tokenizer import Tokenizer as JanomeTokenizer

from ..myutils.parsing import render_prompt, parse_json_objects, create_grpo_examples
from ..myutils.logging import setup_logging

# --- ロギングとシードの設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
torch.manual_seed(42)

# --- 報酬システムの定義 ---
class QARewardSystem:
    """
    QA生成タスクのための複数の報酬を計算するクラス（リファクタリング版）。
    各報酬計算ロジックを独立したメソッドに分割。
    """
    def __init__(self, device: str = "cuda"):
        logging.info("報酬システムを初期化中...")
        self.__name__ = "qa_reward_system"
        
        # 形態素解析器の準備
        self.morph_tokenizer = JanomeTokenizer()
        logging.info("Janome Tokenizerをロードしました。")

        # 埋め込みモデルの準備
        model_name = "intfloat/multilingual-e5-large"
        self.embedding_model = SentenceTransformer(model_name, device=device)
        logging.info(f"埋め込みモデル '{model_name}' をデバイス '{device}' にロードしました。")

        # 各報酬の重みを設定（調整可能）
        self.w_extract = 0.30
        self.w_noun = 0.30
        self.w_relevance = 0.40

    # --- 個別の報酬を計算するヘルパーメソッド ---

    def _get_parsed_json(self, completion_text: str) -> dict | None:
        """JSONパースを試み、成功すれば辞書を、失敗すればNoneを返す。"""
        parsed_list = parse_json_objects(completion_text)
    
        # リストが空でなく、最初の要素が存在する場合
        if parsed_list:
            first_json_obj = parsed_list[0]
            if "question" in first_json_obj and "answer" in first_json_obj:
                return first_json_obj # 最初の有効なJSONオブジェクトを返す
        
        # 有効なJSONが見つからなかった場合
        logging.warning(f"有効なQAペアを含むJSONが見つかりませんでした。テキスト: {completion_text[:100]}...")
        return None

    def _get_extractability_reward(self, generated_answer: str, context: str) -> float:
        """答えが文脈から抽出可能か評価する。"""
        return 1.0 if generated_answer in context else 0.0

    def _get_noun_phrase_reward(self, generated_answer: str) -> float:
        """答えが名詞句で終わっているか評価する。"""
        morphemes = list(self.morph_tokenizer.tokenize(generated_answer.strip()))
        if morphemes and '名詞' in morphemes[-1].part_of_speech:
            return 1.0
        return 0.0

    def _get_relevance_reward(self, generated_question: str, context: str) -> float:
        """生成された質問と文脈の関連性を評価する。"""
        query_embedding = self.embedding_model.encode(f"query: {generated_question}", convert_to_tensor=True)
        passage_embedding = self.embedding_model.encode(f"passage: {context}", convert_to_tensor=True)
        return util.cos_sim(query_embedding, passage_embedding).item()
    
    # --- メインの報酬計算関数（オーケストレーター） ---

    def __call__(self, prompts: list[str], completions: list[str], reference_completions: list[str] | None = None, **kwargs) -> list[float]:
        """
        GRPOTrainerから呼び出され、各報酬を計算して統合する。
        """
        final_rewards = []
        contexts = kwargs["context"] # データセットからcontext列を取得

        for i in range(len(completions)):
            completion = completions[i]
            context = contexts[i]

            # ステップ1: JSONパースを試みる (これができないと報酬は0)
            parsed_json = self._get_parsed_json(completion)
            if parsed_json is None:
                final_rewards.append(0.0) # 強制的にスコア0
                continue # 次のサンプルの評価へ

            gen_question = parsed_json["question"]
            gen_answer = parsed_json["answer"]

            # ステップ2: 各報酬を計算
            extract_reward = self._get_extractability_reward(gen_answer, context)
            noun_phrase_reward = self._get_noun_phrase_reward(gen_answer)
            relevance_reward = self._get_relevance_reward(gen_question, context)
            
            # ステップ3: 重み付けして最終的な報酬を計算
            weighted_reward = (
                self.w_extract * extract_reward +
                self.w_noun * noun_phrase_reward +
                self.w_relevance * relevance_reward
            )
            final_rewards.append(weighted_reward)
            
            logging.debug(
                f"Sample {i}: R_extract={extract_reward:.2f}, R_noun={noun_phrase_reward:.2f}, R_relevance={relevance_reward:.2f} -> Final={weighted_reward:.2f}"
            )
            
        return final_rewards


# --- メイン実行ブロック ---
def main():
    p = argparse.ArgumentParser(description="SFT済みLoRAモデルをGRPOで追加学習するスクリプト")
    p.add_argument("--base-model-path", type=Path, required=True, help="元のベースモデルのHugging Faceパス")
    p.add_argument("--sft-lora-path", type=Path, required=True, help="SFTで学習済みのアダプタのパス (checkpoint-xxxなど)")
    p.add_argument("--emb-model-path", type=Path, required=True, help="埋め込みモデルのパス")
    p.add_argument("--dataset-path", type=Path, required=True, help="学習に使用するJSONLデータセットのパス")
    p.add_argument("--prompt-template-name", required=True, help="ユーザーテンプレート .j2")
    
    # 学習ハイパーパラメータ
    p.add_argument("--learning-rate", type=float, default=1e-5, help="学習率")
    p.add_argument("--max-prompt-length", type=int, default=1024, help="プロンプトの最大長")
    p.add_argument("--max-completion-length", type=int, default=256, help="生成する最大トークン数")
    p.add_argument("--epochs", type=int, default=1, help="学習エポック数")
    p.add_argument("--batch-size", type=int, default=1, help="バッチサイズ")
    p.add_argument("--accum-steps", type=int, default=8, help="勾配累積ステップ数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=0.1, help="GRPOのbetaパラメータ (正則化の強さ)")

    # ログ設定
    p.add_argument("--log-filename", type=Path, default=None, help="ログファイル名")
    p.add_argument("--log-type", type=str, default="info", choices=["debug", "info"], help="ログレベル")

    args = p.parse_args()

    # --- ログ設定 ---
    setup_logging(filename=args.log_filename, log_type=args.log_type)

    # --- シードの設定 ---
    set_seed(args.seed)
    
    # --- モデルとトークナイザのロード ---
    logging.info(f"ベースモデルをロード中: {args.base_model_path}")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager", # 対応GPUがない場合は "eager" に変更
    )
    
    logging.info(f"SFT学習済みLoRAアダプタをロード中: {args.sft_lora_path}")
    model = PeftModel.from_pretrained(model, str(args.sft_lora_path))
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # --- データセットと報酬システムの準備 ---
    train_examples = create_grpo_examples(
        jsonl_path=args.dataset_path,
        tokenizer=tokenizer,
        prompt_template=args.prompt_template_name,
        max_length=args.max_prompt_length
    )
    train_dataset = Dataset.from_list(train_examples)
    reward_system = QARewardSystem(device=model.device) # モデルと同じデバイスに報酬計算モデルを配置

    # --- 出力ディレクトリの作成 ---
    base_model_path = Path(args.base_model_path)
    output_dir = base_model_path.parent / "grpo" / datetime.datetime.now().strftime("%Y%m%d") 
    output_dir.mkdir(parents=True, exist_ok=False)

    # --- GRPOTrainerの設定 ---
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=args.learning_rate,
        num_generations=8,
        temperature=1.0,
        max_prompt_length=args.max_prompt_length, # プロンプトの最大長
        max_completion_length=args.max_completion_length, # 生成する最大トークン数
        beta=args.beta, # GRPOの正則化項
        logging_steps=10,
        save_strategy="epoch",
        report_to="none", # wandbなどを使う場合は設定
        remove_unused_columns=False, # context列などを報酬関数で使うためFalseに
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=reward_system, # 報酬計算クラスのインスタンスを渡す
    )
    
    # --- 学習開始 ---
    logging.info("GRPO学習を開始します...")
    trainer.train()
    
    # --- モデル保存 ---
    logging.info("学習が完了しました。最終モデルを保存します。")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    main()
