import argparse
import random
import logging
import datetime
from pathlib import Path

from ..myutils.logging import setup_logging


def split_jsonl_file(
    input_file: Path,
    output_path1: str,
    output_path2: str,
    ratio1: int,
    seed: int | None = None
):
    """
    JSONLファイルを指定された比率で2つのファイルにランダムに分割する。

    Args:
        input_file (Path): 分割元の入力JSONLファイルパス。
        output_file1_name (str): 1つ目の出力ファイル名。
        output_file2_name (str): 2つ目の出力ファイル名。
        ratio1 (int): 1つ目のファイルに割り当てる行の割合 (1-99の整数)。
        seed (int | None, optional): 乱数シード。指定すると分割の再現性が保たれる。
    """
    if not (1 <= ratio1 <= 99):
        logging.error("比率は1から99の間で指定してください。")
        return

    # --- 1. 入力ファイルの読み込み ---
    try:
        logging.info(f"入力ファイルを読み込んでいます: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            logging.warning("入力ファイルが空です。処理を終了します。")
            return
            
        logging.info(f"合計 {len(lines)} 行を読み込みました。")
    except FileNotFoundError:
        logging.error(f"入力ファイルが見つかりません: {input_file}")
        return
    except Exception as e:
        logging.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
        return

    # --- 2. データのシャッフル ---
    if seed is not None:
        random.seed(seed)
        logging.info(f"乱数シード {seed} を使ってデータをシャッフルします。")
    else:
        logging.info("データをランダムにシャッフルします。")
    
    random.shuffle(lines)

    # --- 3. 分割点を計算 ---
    total_lines = len(lines)
    split_index = int(total_lines * (ratio1 / 100.0))
    
    # 0行や全行が片方に寄らないように調整
    if split_index == 0:
        split_index = 1
    if split_index == total_lines:
        split_index = total_lines - 1

    lines1 = lines[:split_index]
    lines2 = lines[split_index:]

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
        description="JSONLファイルを指定された比率でランダムに2つのファイルに分割します。",
        formatter_class=argparse.RawTextHelpFormatter # ヘルプメッセージの改行を維持
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
        default=42, # デフォルトのシード値を設定
        help="シャッフルに使用する乱数シード。指定しない場合は実行ごとに結果が変わります。\n(デフォルト: 42)"
    )

    args = parser.parse_args()

    # ログの設定
    setup_logging(filename=None, log_type="info")

    split_jsonl_file(
        input_file=args.input_file,
        output_path1=args.output_path1,
        output_path2=args.output_path2,
        ratio1=args.ratio,
        seed=args.seed
    )
