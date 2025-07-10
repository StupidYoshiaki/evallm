import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from collections import defaultdict

# myutilsが利用できない環境を想定し、シンプルなロギング設定関数を定義
def setup_simple_logging():
    """シンプルなロギング設定を標準出力に行う"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

def split_by_unique_context(
    input_file: Path,
    output_path1: str,
    output_path2: str,
    ratio1: int,
    seed: int | None = None
):
    """
    JSONLファイルを、ユニークなコンテキストを基準に指定された比率で2つのファイルに分割する。
    - contextフィールド内の ' [SEP] ' という文字列は常に '\n\n' に置換される。
    - 上記の置換に伴い、answer_startの位置も自動的に補正される。
    """
    if not (1 <= ratio1 <= 99):
        logging.error("比率は1から99の間で指定してください。")
        return

    # --- 1. 入力ファイルを読み込み、コンテキストごとにグループ化 ---
    try:
        logging.info(f"入力ファイルを読み込んでいます: {input_file}")
        # defaultdictを使うと、キーが存在しない場合に自動で空のリストを作成してくれる
        context_groups = defaultdict(list)
        
        # 置換前後の文字列と、その文字数の差分を定義
        sep_pattern = ' [SEP] '
        replace_with = '\n\n'
        len_diff = len(sep_pattern) - len(replace_with)

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                # --- [SEP]の置換とanswer_startの補正ロジック ---
                if 'context' in data and isinstance(data['context'], str):
                    context = data['context']
                    # answerフィールドが存在しないデータ形式に対応
                    if 'answers' in data and data['answers']['answer_start']:
                        answer_start = data['answers']['answer_start'][0]
                        # context内にある全ての ' [SEP] ' の開始位置をリストアップ
                        sep_indices = [m.start() for m in re.finditer(re.escape(sep_pattern), context)]
                        # 回答開始位置よりも「前」にある [SEP] の数をカウント
                        seps_before_answer = sum(1 for sep_idx in sep_indices if sep_idx < answer_start)
                        # カウント数と文字数の差分を使って、answer_startを補正
                        adjusted_start = answer_start - (seps_before_answer * len_diff)
                        data['answers']['answer_start'] = [adjusted_start]

                    # 実際にcontextの置換を行う
                    cleaned_context = context.replace(sep_pattern, replace_with)
                    data['context'] = cleaned_context
                    
                    # クリーニング後のコンテキストをキーとしてQAデータをグループ化
                    context_groups[cleaned_context].append(data)
        
        if not context_groups:
            logging.warning("有効なデータが見つかりませんでした。処理を終了します。")
            return
            
        logging.info(f"合計 {len(context_groups)} 件のユニークなコンテキストが見つかりました。")

    except FileNotFoundError:
        logging.error(f"入力ファイルが見つかりません: {input_file}")
        return
    except Exception as e:
        logging.error(f"ファイルの読み込みまたは処理中にエラーが発生しました: {e}", exc_info=True)
        return
    
    # --- 2. ユニークなコンテキストのリストをシャッフル ---
    unique_contexts = list(context_groups.keys())
    
    if seed is not None:
        random.seed(seed)
        logging.info(f"乱数シード {seed} を使ってユニークなコンテキストをシャッフルします。")
    else:
        logging.info("ユニークなコンテキストをランダムにシャッフルします。")
    
    random.shuffle(unique_contexts)

    # --- 3. コンテキストを基準に分割点を計算 ---
    total_contexts = len(unique_contexts)
    split_index = int(total_contexts * (ratio1 / 100.0))
    
    # エッジケースの処理
    if split_index == 0 and total_contexts > 0:
        split_index = 1
    if split_index == total_contexts and total_contexts > 0:
        split_index = total_contexts - 1

    contexts1 = unique_contexts[:split_index]
    contexts2 = unique_contexts[split_index:]

    logging.info(f"ユニークコンテキストを {len(contexts1)}件 と {len(contexts2)}件 に分割します。")

    # --- 4. 分割されたコンテキストに基づいてデータセットを再構築 ---
    lines1 = []
    for context in contexts1:
        lines1.extend(context_groups[context])
        
    lines2 = []
    for context in contexts2:
        lines2.extend(context_groups[context])
        
    logging.info(f"最終的な行数: {len(lines1)}行 と {len(lines2)}行")

    # --- 5. 出力ファイルへの書き込み ---
    try:
        logging.info(f"1つ目のファイルに書き込んでいます: {output_path1}")
        with open(output_path1, 'w', encoding='utf-8') as f:
            for item in lines1:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logging.info(f"2つ目のファイルに書き込んでいます: {output_path2}")
        with open(output_path2, 'w', encoding='utf-8') as f:
            for item in lines2:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
        logging.info("ファイルの分割が正常に完了しました。")

    except IOError as e:
        logging.error(f"ファイルへの書き込み中にエラーが発生しました: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JSONLファイルを、ユニークなコンテキストを基準に指定された比率でランダムに2つのファイルに分割します。\n"
                    "contextフィールド内の ' [SEP] ' を'\\n\\n'に置換し、\n"
                    "それに応じてanswer_startの位置を自動的に補正する機能も含まれます。",
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
        help="1つ目のファイルに割り当てるユニークコンテキストの割合（パーセント）。\n例: 80 を指定すると、コンテキスト数が約80%%と20%%になるように分割されます。"
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
    setup_simple_logging()

    split_by_unique_context(
        input_file=args.input_file,
        output_path1=args.output_path1,
        output_path2=args.output_path2,
        ratio1=args.ratio,
        seed=args.seed
    )
