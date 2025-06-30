import argparse
import json
import logging
import random
import re # 正規表現を扱うためにインポート
import sys # ロギングで使うためインポート
from pathlib import Path

# from ..myutils.logging import setup_logging
# myutilsが利用できない環境を想定し、シンプルなロギング設定関数を定義
def setup_simple_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

def split_and_clean_jsonl(
    input_file: Path,
    output_path1: str,
    output_path2: str,
    ratio1: int,
    seed: int | None = None
):
    """
    JSONLファイルをクリーニングし、指定された比率で2つのファイルにランダムに分割する。
    - contextフィールド内の ' [SEP] ' という文字列は常に '\n\n' に置換される。
    - 上記の置換に伴い、answer_startの位置も自動的に補正される。
    """
    if not (1 <= ratio1 <= 99):
        logging.error("比率は1から99の間で指定してください。")
        return

    # --- 1. 入力ファイルの読み込みとクリーニング ---
    try:
        logging.info(f"入力ファイルを読み込んでいます: {input_file}")
        processed_lines = []
        
        # 置換前後の文字列と、その文字数の差分を定義
        sep_pattern = ' [SEP] '
        replace_with = '\n\n'
        len_diff = len(sep_pattern) - len(replace_with) # 5 - 2 = 3

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                # --- ★★★ [SEP]の置換とanswer_startの補正ロジック ★★★ ---
                if 'context' in data and isinstance(data['context'], str):
                    context = data['context']
                    answer_start = data['answers']['answer_start'][0]

                    # context内にある全ての ' [SEP] ' の開始位置をリストアップ
                    sep_indices = [m.start() for m in re.finditer(re.escape(sep_pattern), context)]
                    
                    # 回答開始位置よりも「前」にある [SEP] の数をカウント
                    seps_before_answer = sum(1 for sep_idx in sep_indices if sep_idx < answer_start)
                    
                    # カウント数と文字数の差分を使って、answer_startを補正
                    adjusted_start = answer_start - (seps_before_answer * len_diff)
                    
                    # 実際にcontextの置換を行う
                    cleaned_context = context.replace(sep_pattern, replace_with)
                    
                    # 補正した値でデータを更新
                    data['context'] = cleaned_context
                    data['answers']['answer_start'] = [adjusted_start]

                # 処理後のデータをJSON文字列に戻してリストに追加
                processed_lines.append(json.dumps(data, ensure_ascii=False) + '\n')

        if not processed_lines:
            logging.warning("入力ファイルが空です。処理を終了します。")
            return
            
        logging.info(f"合計 {len(processed_lines)} 行を読み込み、クリーニングと位置補正を完了しました。")
    except FileNotFoundError:
        logging.error(f"入力ファイルが見つかりません: {input_file}")
        return
    except Exception as e:
        logging.error(f"ファイルの読み込みまたは処理中にエラーが発生しました: {e}", exc_info=True)
        return
    
    # --- 2. データのシャッフル ---
    if seed is not None:
        random.seed(seed)
        logging.info(f"乱数シード {seed} を使ってデータをシャッフルします。")
    else:
        logging.info("データをランダムにシャッフルします。")
    
    random.shuffle(processed_lines)

    # --- 3. 分割点を計算 ---
    total_lines = len(processed_lines)
    split_index = int(total_lines * (ratio1 / 100.0))
    
    if split_index == 0 and total_lines > 0:
        split_index = 1
    if split_index == total_lines and total_lines > 0:
        split_index = total_lines - 1

    lines1 = processed_lines[:split_index]
    lines2 = processed_lines[split_index:]

    logging.info(f"分割比率 {ratio1}:{100-ratio1} に基づき、{len(lines1)}行 と {len(lines2)}行 に分割します。")

    # --- 4. 出力ファイルへの書き込み ---
    try:
        logging.info(f"1つ目のファイルに書き込んでいます: {output_path1}")
        with open(output_path1, 'w', encoding='utf-8') as f:
            f.writelines(lines1)
        
        logging.info(f"2つ目のファイルに書き込んでいます: {output_path2}")
        with open(output_path2, 'w', encoding='utf-8') as f:
            f.writelines(lines2)
            
        logging.info("ファイルの分割が正常に完了しました。")

    except IOError as e:
        logging.error(f"ファイルへの書き込み中にエラーが発生しました: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JSONLファイルをクリーニングし、指定された比率でランダムに2つのファイルに分割します。\n"
                    "このスクリプトは、contextフィールド内の ' [SEP] ' を'\\n\\n'に置換し、\n"
                    "それに応じてanswer_startの位置を自動的に補正します。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input-file",
        type=Path,
        required=True,
        help="分割元の入力JSONLファイルのパス。"
    )
    parser.add_argument(
        "-r", "--ratio",
        type=int,
        required=True,
        help="1つ目のファイルに割り当てる割合（パーセント）。\n例: 80 を指定すると、約80%%と20%%に分割されます。"
    )
    parser.add_argument(
        "-o1", "--output-path1",
        type=str,
        required=True,
        help="1つ目の出力パス名。"
    )
    parser.add_argument(
        "-o2", "--output-path2",
        type=str,
        required=True,
        help="2つ目の出力パス名。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="シャッフルに使用する乱数シード。\n(デフォルト: 42)"
    )

    args = parser.parse_args()

    # ログの設定
    # setup_logging(filename=None, log_type="info")
    setup_simple_logging()

    split_and_clean_jsonl(
        input_file=args.input_file,
        output_path1=args.output_path1,
        output_path2=args.output_path2,
        ratio1=args.ratio,
        seed=args.seed
    )
