#!/usr/bin/env bash
set -euo pipefail

# --- グローバル変数とデフォルト設定 ---
# スクリプトのルートディレクトリを基準にパスを設定
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly EVALUATEE_DIR="${PROJECT_ROOT}/models/evaluatee"
readonly OUTPUT_ROOT="${PROJECT_ROOT}/output"

# --- デフォルトのパラメータ ---
# これらはコマンドライン引数で上書き可能
INPUT_FILE=""
TEMPLATE=""
LLM_PORT="8081"
N_GPU_LAYERS=42
PARALLEL=8
N_CTX=2048
DRY_RUN=false
POSITIONAL_ARGS=()

# --- ヘルプ関数 ---
usage() {
  cat <<EOF
すべての評価対象モデルに対して predict.py を実行します。

Usage: $(basename "$0") -i INPUT_JSONL -t TEMPLATE [OPTIONS] [-- PREDICT_ARGS...]

必須オプション:
  -i, --input <path>      入力JSONLファイルのパス
  -t, --template <name>   プロンプトテンプレート名

オプション:
  -p, --port <port>       LLMサーバーのポート番号 (default: ${LLM_PORT})
  --n-gpu-layers <num>    GPUにオフロードするレイヤー数 (default: ${N_GPU_LAYERS})
  --parallel <num>        並列処理数 (default: ${PARALLEL})
  --n-ctx <num>           コンテキストサイズ (default: ${N_CTX})
  --output-root <path>    出力のルートディレクトリ (default: ${OUTPUT_ROOT})
  --dry-run               コマンドを実行せずに表示のみ行う
  -h, --help              このヘルプを表示

PREDICT_ARGS:
  '--'以降の引数は、そのまま predict.py に渡されます。
EOF
  exit 1
}

# --- メインの実行関数 ---
run_prediction() {
  local base_gguf="$1"
  local model_name
  model_name="$(basename "$(dirname "$(dirname "$base_gguf")")")"

  echo "--------------------------------------------------"
  echo "▶ モデルを処理中: ${model_name}"
  echo "--------------------------------------------------"

  # predict.pyに渡すコマンドを配列として構築
  local cmd=(
    python -m src.models.predict
    --base-model "$base_gguf"
    --template "$TEMPLATE"
    --input "$INPUT_FILE"
    --output-dir "$OUTPUT_ROOT" # predict.py側で詳細パスを生成
    --n-gpu-layers "$N_GPU_LAYERS"
    --parallel "$PARALLEL"
    --n-ctx "$N_CTX"
  )
  # 追加の引数があれば結合
  if [ ${#POSITIONAL_ARGS[@]} -gt 0 ]; then
    cmd+=("${POSITIONAL_ARGS[@]}")
  fi

  echo "  実行コマンド:"
  # コマンドを安全に表示（スペースなどを含む引数も正しく表示される）
  printf "    %q\n" "${cmd[@]}"
  echo

  if [ "$DRY_RUN" = false ]; then
    # コマンドを実行
    "${cmd[@]}"
    echo "✔ 完了: ${model_name}"
  else
    echo "ℹ ドライランモードのため、コマンドは実行されませんでした。"
  fi
  echo
}

# --- メインロジック ---
main() {
  # 引数パース
  while [[ $# -gt 0 ]]; do
    case $1 in
      -i|--input) INPUT_FILE="$2"; shift 2;;
      -t|--template) TEMPLATE="$2"; shift 2;;
      -p|--port) LLM_PORT="$2"; shift 2;;
      --n-gpu-layers) N_GPU_LAYERS="$2"; shift 2;;
      --parallel) PARALLEL="$2"; shift 2;;
      --n-ctx) N_CTX="$2"; shift 2;;
      --output-root) OUTPUT_ROOT="$2"; shift 2;;
      --dry-run) DRY_RUN=true; shift 1;;
      -h|--help) usage;;
      --) shift; POSITIONAL_ARGS=("$@"); break;;
      -*) echo "不明なオプション: $1" >&2; usage;;
      *) POSITIONAL_ARGS+=("$1"); shift;;
    esac
  done

  # 必須引数のチェック
  if [[ -z "$INPUT_FILE" || -z "$TEMPLATE" ]]; then
    echo "エラー: --input と --template は必須です。" >&2
    usage
  fi

  # ポート番号を環境変数に設定
  export LLM_PORT

  echo "=== 全モデルの予測を開始します ==="
  echo "入力ファイル: $INPUT_FILE"
  echo "テンプレート: $TEMPLATE"
  echo "出力ルート: $OUTPUT_ROOT"
  echo "設定: n_gpu_layers=${N_GPU_LAYERS}, parallel=${PARALLEL}, n_ctx=${N_CTX}"
  echo

  # モデルファイルを探してループ処理
  local model_files
  mapfile -t model_files < <(find "$EVALUATEE_DIR" -mindepth 3 -maxdepth 3 -type f -name "base.gguf")

  if [ ${#model_files[@]} -eq 0 ]; then
    echo "エラー: ${EVALUATEE_DIR} 内に base.gguf ファイルが見つかりませんでした。" >&2
    exit 1
  fi

  for model_file in "${model_files[@]}"; do
    run_prediction "$model_file"
  done

  echo "✅ 全モデルの predict.py 実行が完了しました。"
}

# --- スクリプト実行 ---
main "$@"
