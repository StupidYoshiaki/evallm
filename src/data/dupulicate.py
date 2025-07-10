import argparse
import json
import random
import logging
from pathlib import Path

# ロギングの基本設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def deduplicate_and_sample_jsonl(
    input_file: Path,
    output_file: Path,
    num_records: int | None = None,
    seed: int | None = None
):
    """
    JSONLファイルを読み込み、'context'フィールドで重複を除去した後、
    結果をシャッフルし、指定された数のレコードを新しいJSONLファイルに保存する。

    Args:
        input_file (Path): 分割元の入力JSONLファイルパス。
        output_file (Path): 結果を保存する出力JSONLファイルパス。
        num_records (int | None, optional): 保存するレコード数。Noneの場合は全て保存。
        seed (int | None, optional): 乱数シード。指定すると分割の再現性が保たれる。
    """
    # --- 1. 入力ファイルの読み込みと重複除去 ---
    try:
        logging.info(f"入力ファイルを読み込んでいます: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            logging.warning("入力ファイルが空です。処理を終了します。")
            return

        seen_contexts = set()
        unique_records = []
        for line in lines:
            try:
                record = json.loads(line.strip())
                context = record.get("context")
                if context is not None and context not in seen_contexts:
                    seen_contexts.add(context)
                    unique_records.append(record)
            except (json.JSONDecodeError, KeyError):
                # パースできない行やcontextがない行は無視
                continue
        
        logging.info(f"重複除去後のユニークなレコード数: {len(unique_records)} 件")

    except FileNotFoundError:
        logging.error(f"入力ファイルが見つかりません: {input_file}")
        return
    except Exception as e:
        logging.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
        return

    # --- 2. シャッフルとサンプリング ---
    output_records = unique_records # デフォルトでは全てのユニークなレコードを対象とする

    if num_records is not None and num_records > 0:
        logging.info(f"ユニークなレコードをシャッフルし、最大 {num_records} 件を抽出します。")
        
        # 乱数シードを設定
        if seed is not None:
            random.seed(seed)
            logging.info(f"乱数シード {seed} を使用します。")
        
        # リストをシャッフル
        random.shuffle(output_records)
        
        # 指定された数だけスライスして抜き出す
        output_records = output_records[:num_records]
    
    logging.info(f"最終的に {len(output_records)} 件をファイルに書き込みます。")

    # --- 3. 出力ファイルへの書き込み ---
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        logging.info(f"結果を出力ファイルに書き込んでいます: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for record in output_records:
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
        logging.info("書き込みが完了しました。")
    except IOError as e:
        logging.error(f"ファイルへの書き込み中にエラーが発生しました: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JSONLファイルから'context'の重複を除去し、ランダムに指定数をサンプリングして保存します。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input-file",
        type=Path,
        required=True,
        help="処理対象の入力JSONLファイルのパス。"
    )
    parser.add_argument(
        "-o", "--output-file",
        type=Path,
        required=True,
        help="結果を保存する出力JSONLファイルパス。"
    )
    parser.add_argument(
        "-n", "--num-records",
        type=int,
        default=None, # ★ デフォルトはNone（全件保存）
        help="保存する最大レコード数。指定しない場合は全てのユニークなレコードを保存します。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="シャッフルに使用する乱数シード。再現性が必要な場合に指定します。(デフォルト: 42)"
    )

    args = parser.parse_args()

    deduplicate_and_sample_jsonl(
        input_file=args.input_file,
        output_file=args.output_file,
        num_records=args.num_records,
        seed=args.seed
    )
