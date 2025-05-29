import argparse
import re
from scipy.stats import spearmanr
from pathlib import Path
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def parse_markdown_ranking(file_path: Path) -> dict:
    """
    Markdown形式のランキングファイルをパースし、モデル名をキー、ランクを値とする辞書を返す。
    """
    model_ranks = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # ヘッダー行とセパレータ行をスキップ (通常最初の2行)
            # 内容のある行だけを処理対象とする
            data_lines = [line for line in lines if line.strip() and line.startswith('|')]
            if len(data_lines) < 2: # ヘッダーと区切り線は最低でも必要
                logging.warning(f"ファイル '{file_path}' に十分なテーブル行がありません。")
                return model_ranks

            for line in data_lines[2:]: # 最初の2行はヘッダーと区切り線と仮定
                parts = [part.strip() for part in line.split('|')]
                # ['', 'Rank', 'Model Name', 'Accuracy (%)', 'Score (Correct/Total)', ''] のような形式を期待
                if len(parts) > 3: # 少なくともRankとModel Nameが取れること
                    try:
                        rank = int(parts[1])
                        model_name = parts[2]
                        if model_name: # モデル名が空でないこと
                            model_ranks[model_name] = rank
                    except ValueError:
                        logging.warning(f"ファイル '{file_path}' の行でランクを数値に変換できませんでした: {line.strip()}")
                    except IndexError:
                        logging.warning(f"ファイル '{file_path}' の行の形式が不正です: {line.strip()}")
            
            if not model_ranks:
                logging.warning(f"ファイル '{file_path}' からランキングデータを抽出できませんでした。")

    except FileNotFoundError:
        logging.error(f"ファイルが見つかりません: {file_path}")
    except Exception as e:
        logging.error(f"ファイル '{file_path}' の処理中にエラーが発生しました: {e}")
    return model_ranks

def calculate_rank_correlation(file1_path: Path, file2_path: Path) -> None:
    """
    2つのMarkdownランキングファイルから順位相関係数を計算して表示する。
    """
    logging.info(f"ランキングファイル1をパース中: {file1_path}")
    ranks1_dict = parse_markdown_ranking(file1_path)
    logging.info(f"ランキングファイル2をパース中: {file2_path}")
    ranks2_dict = parse_markdown_ranking(file2_path)

    if not ranks1_dict or not ranks2_dict:
        logging.error("両方のファイルからランキングデータを正しく読み込めなかったため、相関係数を計算できません。")
        return

    common_models = sorted(list(set(ranks1_dict.keys()) & set(ranks2_dict.keys())))

    if not common_models:
        logging.warning("2つのランキングファイルに共通のモデルが見つかりませんでした。")
        return
    
    if len(common_models) < 2:
        logging.warning("順位相関係数を計算するには、少なくとも2つの共通モデルが必要です。")
        print("\n共通モデルが1つ以下のため、順位相関係数は計算できません。")
        print(f"共通モデル: {common_models if common_models else 'なし'}")
        return

    aligned_ranks1 = [ranks1_dict[model] for model in common_models]
    aligned_ranks2 = [ranks2_dict[model] for model in common_models]

    logging.info(f"共通モデル数: {len(common_models)}")
    logging.debug(f"共通モデルリスト: {common_models}")
    logging.debug(f"ファイル1の対応ランク: {aligned_ranks1}")
    logging.debug(f"ファイル2の対応ランク: {aligned_ranks2}")

    correlation, p_value = spearmanr(aligned_ranks1, aligned_ranks2)

    print("\n===== 順位相関係数 結果 =====")
    print(f"ファイル1: {file1_path.name}")
    print(f"ファイル2: {file2_path.name}")
    print(f"共通モデル数: {len(common_models)}")
    print(f"スピアマンの順位相関係数 (ρ): {correlation:.4f}")
    print(f"P値: {p_value:.4f}")
    
    if p_value < 0.05:
        print("P値が0.05未満であり、相関は統計的に有意であると考えられます。")
    else:
        print("P値が0.05以上であり、相関は統計的に有意ではない可能性があります。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2つのMarkdown形式のランキングファイルの間のスピアマン順位相関係数を計算します。")
    parser.add_argument("--file1", type=Path, help="1つ目のMarkdownランキングファイルへのパス")
    parser.add_argument("--file2", type=Path, help="2つ目のMarkdownランキングファイルへのパス")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="ログレベル (default: INFO)")
    
    args = parser.parse_args()

    # ログレベルを設定
    logging.getLogger().setLevel(args.loglevel.upper())
    
    calculate_rank_correlation(args.file1, args.file2)
