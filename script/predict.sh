#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") -i INPUT_JSONL -t TEMPLATE -p LLM_PORT [--predict-args...]

Options:
  -i, --input         入力 JSONL ファイルパス
  -t, --template      プロンプトテンプレート名
  -p, --port         LLM サーバーのポート番号 (default: 8081)
  -o, --output-root   出力先ルートディレクトリ (default: output)
  -h, --help          このヘルプを表示

以降の引数はそのまま predict.py に渡されます。
EOF
  exit 1
}

# デフォルト設定
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVALUATEE_DIR="models/evaluatee"
OUTPUT_ROOT="output"

# 引数パース
INPUT_FILE=""
TEMPLATE=""
OUTPUT_ROOT_OVERRIDE=""
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input)
      INPUT_FILE="$2"; shift 2;;
    -t|--template)
      TEMPLATE="$2"; shift 2;;
    -p|--port)
      LLM_PORT="$2"; shift 2;;
    -o|--output-root)
      OUTPUT_ROOT_OVERRIDE="$2"; shift 2;;
    -h|--help)
      usage;;
    --)
      shift; break;;
    -*)
      echo "Unknown option: $1" >&2; usage;;
    *)
      POSITIONAL+=("$1"); shift;;
  esac
done

if [[ -z "$INPUT_FILE" || -z "$TEMPLATE" ]]; then
  echo "Error: --input と --template は必須です" >&2
  usage
fi

if [[ -n "$OUTPUT_ROOT_OVERRIDE" ]]; then
  OUTPUT_ROOT="$OUTPUT_ROOT_OVERRIDE"
fi

# ポート番号の設定
export LLM_PORT="${LLM_PORT:-8081}"

echo "入力ファイル: $INPUT_FILE"
echo "テンプレート: $TEMPLATE"
echo "出力ルート: $OUTPUT_ROOT"
echo "ポート番号: $LLM_PORT"
echo

# 各モデルごとに predict.py を呼び出し
find "$EVALUATEE_DIR" -mindepth 3 -maxdepth 3 -type f -name "base.gguf" | while read -r BASE_GGUF; do
  MODEL_NAME="$(basename "$(dirname "$(dirname "$BASE_GGUF")")")"
  echo "▶ モデル: $MODEL_NAME"

  # 出力先を決定
  # predict.py 側で入力ファイル名に応じてサブディレクトリを作るので、ここではモデル単位のディレクトリだけ用意
  OUT_DIR="$OUTPUT_ROOT"
  mkdir -p "$OUT_DIR"

  echo "  → 実行: python -m src.models.predict \\"
  echo "       --base-model \"$BASE_GGUF\" \\"
  echo "       --template \"$TEMPLATE\" \\"
  echo "       --input \"$INPUT_FILE\" \\"
  echo "       --output-dir \"$OUT_DIR\" \\"
  echo "       --n-gpu-layers 42 \\"
  printf "       %s\n" "${POSITIONAL[@]}"

  python -m src.models.predict \
    --base-model "$BASE_GGUF" \
    --template "$TEMPLATE" \
    --input "$INPUT_FILE" \
    --output-dir "$OUT_DIR" \
    --n-gpu-layers 42 \
    "${POSITIONAL[@]}"

  echo "✔ 完了: $MODEL_NAME → $OUT_DIR"
  echo
done

echo "=== 全モデルの predict.py 実行が完了しました ==="
