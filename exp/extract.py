import argparse
import logging
from pathlib import Path
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def run_inference(args: argparse.Namespace):
    """
    ファインチューニング済みのQAモデルをロードし、与えられた質問と文脈から回答を抽出する。
    """
    # デバイスの決定 (GPUが利用可能であればGPUを使用)
    device_id = -1 # CPU
    if torch.cuda.is_available():
        if args.gpu_id < torch.cuda.device_count():
            device_id = args.gpu_id
            logger.info(f"CUDAデバイス {device_id} ({torch.cuda.get_device_name(device_id)}) を使用します。")
        else:
            logger.warning(f"指定されたGPU ID {args.gpu_id} は無効です。CPUを使用します。")
    else:
        logger.info("CUDAが利用できないため、CPUを使用します。")

    # 1. モデル・トークナイザをロード
    try:
        # 1. まず、モデルとトークナイザをそれぞれ明示的にロードする
        logger.info(f"トークナイザをロード中: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        logger.info(f"モデルをロード中: {args.model_path}")
        model = AutoModelForQuestionAnswering.from_pretrained(args.model_path, trust_remote_code=True)
        
        # 2. pipelineには、ロード済みのオブジェクトを渡す
        logger.info("QAパイプラインを構築しています...")
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=device_id,
        )
        logger.info("モデルとパイプラインの準備が完了しました。")

    except Exception as e:
        logger.error(f"モデルまたはパイプラインの初期化中にエラーが発生しました: {e}")
        return

    # 2. 推論の実行
    logger.info("推論を実行します...")
    # pipelineに質問と文脈を渡すだけ
    result = qa_pipeline(
        question=args.question,
        context=args.context,
    )
    logger.info("推論が完了しました。")

    # 3. 結果の表示
    print("\n--- 推論結果 ---")
    print(f"質問: {args.question}")
    print(f"回答: {result['answer']}")
    print(f"スコア: {result['score']:.4f}")
    print(f"文脈中の位置: Start={result['start']}, End={result['end']}")
    print("----------------")
    print(f"抽出元文脈: ...{args.context[max(0, result['start']-10):result['end']+10]}...")


def main():
    parser = argparse.ArgumentParser(description="ファインチューニング済みQAモデルで推論を行うスクリプト")
    
    parser.add_argument("--model_path", type=Path, required=True,
                        help="ファインチューニング済みモデルが格納されているディレクトリのパス。")
    parser.add_argument("--question", type=str, required=True,
                        help="モデルに尋ねる質問。")
    parser.add_argument("--context", type=str, required=True,
                        help="回答の抽出元となる文脈テキスト。")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="推論に使用するGPUのID。CPUを使用する場合は-1を指定。 (default: 0)")

    args = parser.parse_args()

    # 文脈と質問が長すぎる場合に警告
    if len(args.context) + len(args.question) > 10000: # 目安
        logger.warning("文脈と質問の合計が非常に長いです。処理に時間がかかるか、メモリが不足する可能性があります。")
        
    run_inference(args)

if __name__ == "__main__":
    main()
