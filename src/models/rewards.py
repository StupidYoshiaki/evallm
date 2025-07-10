import logging

import torch
from sentence_transformers import SentenceTransformer, util
from janome.tokenizer import Tokenizer as JanomeTokenizer

from ..myutils.parsing import parse_json_objects
from .generate_and_extract import setup_bert_model, extract_single_answer

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
            if extract_reward == 0.0:
                final_rewards.append(0.0)
                logging.debug(f"Sample {i}: 抽出可能でないため、報酬は0")
                continue
            
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


class QuesionImprovementRewardSystem:
    """
    QA生成タスクのための複数の報酬を計算するクラス（リファクタリング版）。
    各報酬計算ロジックを独立したメソッドに分割。
    """
    def __init__(self, device: str = "cuda"):
        logging.info("報酬システムを初期化中...")
        self.__name__ = "qa_reward_system"
        
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
            if extract_reward == 0.0:
                final_rewards.append(0.0)
                logging.debug(f"Sample {i}: 抽出可能でないため、報酬は0")
                continue
            
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
