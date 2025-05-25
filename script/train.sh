#!/usr/bin/env bash
set -euo pipefail

# プロジェクトルートからの相対パス取得
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# モジュール名
TRAIN_MODULE="src.train"
# 公式変換スクリプトへのパス（リポジトリ直下に clone してある想定）
CONVERT_SCRIPT="${PROJECT_ROOT}/llama.cpp/convert_hf_to_gguf.py"

usage() {
  cat <<EOF
Usage: $(basename "$0") \
  --base-safetensors BASE_SAFETENSORS \
  --base-gguf BASE_GGUF \
  --dataset DATA_JSONL \
  --user-tpl USER_J2 \
  --assist-tpl ASSIST_J2 \
  --output-path OUTPUT_PATH \
  [--epochs N] [--batch-size N] [--accum-steps N]

LoRA学習→マージ→公式convert_hf_to_gguf.py呼び出しでGGUF生成。
EOF
  exit 1
}

# デフォルト値
EPOCHS=1; BATCH=2; ACCUM=8

# 引数パース
while [[ $# -gt 0 ]]; do
  case $1 in
    --base-safetensors) BASE_SAFETENSORS="$2"; shift 2;;
    --base-gguf) BASE_GGUF="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --user-tpl) U_TPL="$2"; shift 2;;
    --assist-tpl) A_TPL="$2"; shift 2;;
    --output-path) OUTPUT_PATH="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch-size) BATCH="$2"; shift 2;;
    --accum-steps) ACCUM="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown option: $1" >&2; usage;;
  esac
done

# 必須チェック
if [[ -z "${BASE_GGUF:-}" || -z "${DATASET:-}" || -z "${U_TPL:-}" || -z "${A_TPL:-}" ]]; then
  echo "Error: 必須パラメータが足りません" >&2
  usage
fi

echo "▶ LoRA 学習を開始"
MERGED_PT=$(python -m "$TRAIN_MODULE" \
  --base-model "$BASE_SAFETENSORS" \
  --dataset "$DATASET" \
  --user-template "$U_TPL" \
  --assistant-template "$A_TPL" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --accum-steps "$ACCUM")
echo "→ マージ済み重み: $MERGED_PT"
echo

echo "▶ GGUF 変換（公式スクリプト呼び出し）"
python3 "$CONVERT_SCRIPT" \
  --base-model-path "$BASE_GGUF" \
  --lora-pt-path "$MERGED_PT" \
  --outfile "$OUTPUT_PATH" \
echo "→ 生成された GGUF モデル: $OUTPUT_PATH"
echo

echo "✅ 完了"
