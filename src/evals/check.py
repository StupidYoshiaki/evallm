import json
import argparse
from pathlib import Path

def calculate_answer_containment_rate(file_path: Path):
    """
    JSONLファイルを読み込み、'answer'が'context'に含まれる割合（ACR）を計算します。

    Args:
        file_path (Path): 入力するJSONLファイルのパス。

    Returns:
        dict: 計算結果を含む辞書。
    """
    total_items = 0
    contained_items = 0

    try:
        with file_path.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    context = data.get("context", "")
                    answer = data.get("answer", "")

                    # contextとanswerの両方が存在する場合のみを対象とする
                    try:
                        if context and answer:
                            total_items += 1
                            if answer in context:
                                contained_items += 1
                    except:
                        total_items += 1
                        continue
                
                except json.JSONDecodeError:
                    print(f"警告: 不正なJSON形式の行をスキップしました: {line.strip()}")

    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {file_path}")
        return None
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

    # ゼロ除算を避ける
    if total_items == 0:
        rate = 0.0
    else:
        rate = (contained_items / total_items) * 100
        
    return {
        "total_items": total_items,
        "contained_items": contained_items,
        "containment_rate": rate,
    }

def main():
    """
    コマンドライン引数を処理し、ACRの計算を実行して結果を表示します。
    """
    parser = argparse.ArgumentParser(
        description="JSONLファイル内の'answer'が'context'に含まれる割合 (Answer Containment Rate) を計算します。"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="評価対象の入力JSONLファイルパス。"
    )
    
    args = parser.parse_args()

    results = calculate_answer_containment_rate(args.input_file)
    
    if results:
        print("\n--- 評価結果 ---")
        print("指標名: Answer Containment Rate (ACR)")
        print(f"  処理対象アイテム数: {results['total_items']}件")
        print(f"  'answer'が'context'に含まれていた数: {results['contained_items']}件")
        print(f"  含有率 (ACR): {results['containment_rate']:.2f}%")
        print("----------------\n")

if __name__ == "__main__":
    main()
