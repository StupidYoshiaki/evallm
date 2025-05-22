#!/usr/bin/env bash
set -euo pipefail

# プロジェクトルートからの相対パスを設定
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREDICT_SCRIPT="${PROJECT_ROOT}/src/models/predict.py"
EVALUATEE_DIR="${PROJECT_ROOT}/models/evaluatee"
DATA_ROOT="${PROJECT_ROOT}/data/JSQuAD/eval"
OUTPUT_ROOT="${PROJECT_ROOT}/output"
TEMPLATE="qa_generator.j2"
MAX_TOKENS=200

echo "=== evaluatee 以下の全モデルに対して predict.py を実行します ==="

# models/evaluatee/<MODEL>/gguf/base.gguf を探してループ
find "$EVALUATEE_DIR" -mindepth 2 -maxdepth 2 -type f -name "base.gguf" | while read -r BASE_GGUF; do
  MODEL_NAME="$(basename "$(dirname "$(dirname "$BASE_GGUF")")")"
  echo
  echo "▶ モデル: $MODEL_NAME"

  INPUT_JSONL="$DATA_ROOT/$MODEL_NAME/baseline.jsonl"
  if [[ ! -f "$INPUT_JSONL" ]]; then
    echo "  ⚠ 入力ファイルがありません: $INPUT_JSONL"
    continue
  fi

  OUT_DIR="$OUTPUT_ROOT/$MODEL_NAME"
  mkdir -p "$OUT_DIR"

  echo "  → 実行: python3 $PREDICT_SCRIPT \\"
  echo "       --base-model \"$BASE_GGUF\" \\"
  echo "       --template \"$TEMPLATE\" \\"
  echo "       --input \"$INPUT_JSONL\" \\"
  echo "       --output-dir \"$OUT_DIR\" \\"
  echo "       --max-tokens $MAX_TOKENS"
  python3 "$PREDICT_SCRIPT" \
    --base-model "$BASE_GGUF" \
    --template "$TEMPLATE" \
    --input "$INPUT_JSONL" \
    --output-dir "$OUT_DIR" \
    --max-tokens $MAX_TOKENS

  echo "✔ 完了: $MODEL_NAME → $OUT_DIR"
done

echo
echo "=== 全モデルの predict.py 実行が完了しました ==="
