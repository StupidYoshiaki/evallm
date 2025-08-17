## preprocess
```
python -m src.data.preprocess --input data/squad/train.jsonl --output data/squad/train/preprocessed.jsonl
```
```
python -m src.data.split -i data/JSQuAD/train.jsonl -r 80 -o1 data/JSQuAD/pipeline/train/step1/raw.jsonl -o2 data/JSQuAD/pipeline/train/step2/raw.jsonl
```

## download
```
python -m src.setup.downloader.model --repo-id mradermacher/Gemma-2-Llama-Swallow-9b-it-v0.1-i1-GGUF --format gguf --filename Gemma-2-Llama-Swallow-9b-it-v0.1.i1-Q4_K_M
.gguf --output-dir models/generator 
```
```
python -m src.setup.downloader.model --repo-id Qwen/Qwen3-8B --format safetensors --output-dir models/generator
```
```
python -m src.setup.downloader.data --dataset-id rajpurkar/squad --splits train,validation --output-dir data
```
```
python -m src.setup.download_all --model-list src/setup/list/model_list.csv --dataset-list src/setup/list/dataset_list.csv
```
```
python -m src.data.split -i data/JSQuAD/train.jsonl -r 80 -o1 data/JSQuAD/train/bert_train.jsonl -o2 data/JSQuAD/train/bert_valid.jsonl
```

## train
```
python -m src.data.preprocess --input data/JSQuAD/train.jsonl --output data/JSQuAD/train/preprocessed.jsonl
```
```
python -m src.models.train_sft --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --user-template train_generator_user.j2 --assistant-template train_generator_assistant.j2 --dataset data/JSQuAD/train/sft_202506141209.jsonl
```
```
python -m src.models.train --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --resume-from-checkpoint models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/lora/20250602/checkpoint-10950 --model-type generator --user-template train_generator_user.j2 --assistant-template train_generator_assistant.j2 --dataset data/JSQuAD/train/preprocessed.jsonl --epochs 2
```
```
python -m src.models.train_grpo --base-model-path models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --sft-lora-path models/generator/Llama-3.1-Swallow-
8B-Instruct-v0.3/safetensors/sft/20250530/checkpoint-7858 --dataset-path data/ja-wiki/grpo.jsonl --prompt-template-name train_generator_user.j2 --emb-model-path models/embedding/multilingual-e5-large/safetensors/base
```
```
python -m src.models.train_bert --model-path models/extractor/modernbert-ja-310m/safetensors/base --train-dataset-path data/JSQuAD/train/bert_train.jsonl --valid-dataset-path data/JSQuAD/train/bert_valid.jsonl
```
```
python -m src.models.train_sft --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --user-template question_generator_user.j2 --assistant-template question_generator_assistant.j2 --dataset data/JSQuAD/pipeline/train/step1/raw.jsonl
```
```
python -m src.models.train_sft --base-model models/generator/llm-jp-3.1-1.8b-instruct4/safetensors/base --user-template question_generator_user.j2 --assistant-template question_generator_assistant.j2 --model-type q_generator --train-dataset data/JSQuAD/train/preprocessed.jsonl
```
```
python -m src.models.train_sft --base-model models/generator/llm-jp-3.1-1.8b-instruct4/safetensors/base --user-template train_generator_user.j2 --assistant-template train_generator_assistant.j2 --model-type qa_generator --train-dataset data/JSQuAD/train/preprocessed.jsonl
```

## generate
```
python -m src.models.generate --base-model models/generator/gemma-2-9b-it/gguf/base.gguf --template qa_generator_few_shot.j2 --input data/JSQuAD/eval/baseline.jsonl --few-shot-input data/JSQuAD/eval/few_shot.jsonl --shot-num 10 --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```
```
python -m src.models.generate --base-model models/generator/Qwen3-8B/gguf/base.gguf --template src/prompt/qa_generator_nothing.j2 --input data/nothing/eval/baseline.jsonl data/nothing/eval --n-gpu-layers 42 --parallel 8 --n-ctx 40960 --max-tokens 32768

```
```
python -m src.models.generate --base-model models/generator/llm-jp-3.1-13b-instruct4/gguf/base.gguf --template qa_generator_few_shot.j2 --input data/JSQuAD/eval/baseline.jsonl --few-shot-input data/JSQuAD/eval/few_shot.jsonl --shot-num 10 --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```
```
python -m src.models.generate --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --lora-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/lora-20250602.gguf --template qa_generator_ja.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```
```
python -m src.models.generate --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --lora-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/grpo-20250612.gguf --template qa_generator.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```
```
python -m src.models.generate --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --lora-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/lora-20250610.gguf --template qa_generator.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```
```
python -m src.models.generate_and_extract --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --lora-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/sft-20250627.gguf --template question_generator_user.j2 --bert-model models/extractor/roberta-base-japanese/safetensors/finetuned/20250708/best_model --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```

### llm-jp
```
python -m src.models.generate --base-model models/generator/llm-jp-3.1-1.8b-instruct4/gguf/base.gguf --template qa_generator.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```
```
python -m src.models.generate --base-model models/generator/llm-jp-3.1-1.8b-instruct4/gguf/base.gguf --lora-model models/generator/llm-jp-3.1-1.8b-instruct4/gguf/sft-20250805.gguf --template qa_generator.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048 --log-type debug --max-tokens 1000
```
```
python -m src.models.generate --base-model models/generator/llm-jp-3.1-1.8b-instruct4/gguf/base.gguf --template qa_generator_few_shot.j2 --input data/JSQuAD/eval/baseline.jsonl --few-shot-input data/JSQuAD/eval/few_shot.jsonl --shot-num 10 --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```

## convert
```
python ../opt/llama/convert_lora_to_gguf.py models/generator/gemma-2-9b-it/safetensors/20250526/lora/checkpoint-7858 --outfile models/generator/gemma-2-9b-it/gguf/lora-20250526.gguf --base models/generator/gemma-2-9b-it/safetensors/base
```
```
python ../opt/llama/convert_lora_to_gguf.py models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/20250602/lora/checkpoint-10950 --outfile models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/lora-20250602.gguf --base models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base
```
```
python ../opt/llama/convert_lora_to_gguf.py models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/lora/20250610/checkpoint-15000 --outfile models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/lora-20250610.gguf --base models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base
```
```
python ../opt/llama/convert_hf_to_gguf.py models/generator/llm-jp-3.1-13b-instruct4/safetensors/base --outfile models/generator/llm-jp-3.1-13b-instruct4/gguf/base.gguf --outtype f16
```
```
python ../opt/llama/convert_lora_to_gguf.py models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/grpo/20250612/checkpoint-1134 --outfile models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/grpo-20250612.gguf --base models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base
```
```
llama-quantize models/generator/llm-jp-3.1-13b-instruct4/gguf/base.gguf models/generator/llm-jp-3.1-13b-instruct4/gguf/base-Q4_K_M.gguf Q4_K_M
```
```
 curl https://raw.githubusercontent.com/llm-jp/llm-jp-tokenizer/main/models/ver3.1/llm-jp-tokenizer-100k.ver3.1.model > tokenizer.model
```
```
python ../opt/llama/convert_lora_to_gguf.py models/generator/llm-jp-3.1-1.8b-instruct4/safetensors/sft/20250804/checkpoint-7858 --outfile models/generator/llm-jp-3.1-1.8b-instruct4/gguf/sft-20250804.gguf --base models/generator/llm-jp-3.1-1.8b-instruct4/safetensors/base
```


### 注意点
- convertのために、llama-cppのコードの一部を改変しているので注意
    - 参考: https://zenn.dev/matsuolab/articles/2857bf0feeeb5d
    - これ参考にしないでください
- https://note.com/npaka/n/n3e99d2a45a4b
    - こっちを参考にしてください
- https://qiita.com/7shi/items/14d24a25aa26dcf97d2d
    - これも参考にしてください

## predict
```
python -m src.models.predict --base-model models/generator/gemma-2-9b-it/gguf/base.gguf --template evaluatee.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir output --n-gpu-layers 42
```

## script
```
./script/predict.sh -i data/JSQuAD/eval/baseline.jsonl -t evaluatee.j2
```
```
./script/predict.sh -i data/JSQuAD/eval/Llama-3.1-Swallow-8B-Instruct-v0.3/202506141141/generated.jsonl -t evaluatee.j2
```

## evaluate
```
python -m src.evals.corr --file1 output/JSQuAD/baseline/ranking.md --file2 output/JSQuAD/generated/gemma-2-9b-it/202505251436/ranking.md
```
```
python -m src.evals.rank --ground-truth-file data/JSQuAD/eval/Llama-3.1-Swallow-8B-Instruct-v0.3/202505270601/generated.jsonl --prediction-base-dir output/JSQuAD/generated/L
lama-3.1-Swallow-8B-Instruct-v0.3/202505270601
```
```
python -m src.evals.rank --ground-truth-file data/JSQuAD/eval/Llama-3.1-Swallow-8B-Instruct-v0.3/202506020258/generated.jsonl --prediction-base-dir output/JSQuAD/generated/Llama-3.1-Swallow-8B-Instruct-v0.3/202506020258 --llm-judge-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf
```
```
LLM_PORT=8081 python -m src.evals.rank --ground-truth-file data/JSQuAD/eval/Llama-3.1-Swallow-8B-Instruct-v0.3/202506020258/generated.jsonl --prediction-base-dir output/JSQuAD/generated/Llama-3.1-Swallow-8B-Instruct-v0.3/202506020258 --llm-judge-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf
```

## うまくllama-serverが立てられない時
```
ps aux | grep llama-server
```

## 容量
```
df -h
```

## ggufモデルのテスト
あらかじめtemplateファイルを定義しないと、llama-cppに登録されていないモデルは使えない
```
llama-cli -m models/generator/llm-jp-3.1-1.8b-instruct4/gguf/base.gguf -i --jinja --chat-template-file chat_tmp.tmpl
```
















# pipelineのコマンドの流れ
## 質問生成モデル開発
```
python -m src.data.split -i data/JSQuAD/train.jsonl -r 50 -o1 data/JSQuAD/pipeline/train/step1/raw.jsonl -o2 data/JSQuAD/pipeline/train/step2/raw.jsonl

python -m src.data.split -i data/JSQuAD/pipeline/train/step1/raw.jsonl -r 95 -o1 data/JSQuAD/pipeline/train/step1/sft_train.jsonl -o2 data/JSQuAD/pipeline/train/step1/sft_valid.jsonl

python -m src.data.split -i data/JSQuAD/pipeline/train/step1/raw.jsonl -r 80 -o1 data/JSQuAD/pipeline/train/step1/bert_train.jsonl -o2 data/JSQuAD/pipeline/train/step1/bert_valid.jsonl

python -m src.data.split -i data/JSQuAD/pipeline/train/step2/raw.jsonl -r 80 -o1 data/JSQuAD/pipeline/train/step2/bert_train.jsonl -o2 data/JSQuAD/pipeline/train/step2/bert_valid.jsonl

python -m src.models.train_sft --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --user-template question_generator_user.j2 --assistant-template question_generator_assistant.j2 --train-dataset data/JSQuAD/pipeline/train/step1/raw.jsonl --model-type q_generator  
(
python -m src.models.train_sft --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --user-template question_generator_user.j2 --assistant-template question_generator_assistant.j2 --train-dataset data/JSQuAD/pipeline/train/step1/sft_train.jsonl --valid-dataset data/JSQuAD/pipeline/train/step1/sft_valid.jsonl --model-type q_generator
)

python ../opt/llama/convert_lora_to_gguf.py models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/sft/20250713/checkpoint-3929 --outfile models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/sft-20250713.gguf --base models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base

python -m src.models.train_bert --model-path models/extractor/modernbert-ja-310m/safetensors/base --train-dataset-path data/JSQuAD/pipeline/train/step1/bert_train.jsonl --valid-dataset-path data/JSQuAD/pipeline/train/step1/bert_valid.jsonl --num-train-epochs 3

python -m src.models.generate_dpo_dataset --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --lora-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/sft-20250713.gguf --template question_generator_user.j2 --bert-model models/extractor/modernbert-ja-310m/safetensors/finetuned/20250712/best_model --input data/JSQuAD/pipeline/train/step2/raw.jsonl --output-dir data/JSQuAD/pipeline/train/step2 --num-candidates 5 --temperature 1.0

python -m src.models.train_dpo --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --resume-from-checkpoint models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/sft/20250713/checkpoint-3929 --user-template question_generator_user.j2 --assistant-template question_generator_assistant.j2 --train-dataset data/JSQuAD/pipeline/train/step2/dpo/202507170608/generated.jsonl --epochs 2

python ../opt/llama/convert_lora_to_gguf.py models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/dpo/20250717/checkpoint-7858 --outfile models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/dpo-20250717.gguf --base models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base

python -m src.models.train_bert --model-path models/extractor/modernbert-ja-310m/safetensors/finetuned/20250712/best_model --train-dataset-path data/JSQuAD/pipeline/train/step2/bert_train.jsonl --valid-dataset-path data/JSQuAD/pipeline/train/step2/bert_valid.jsonl --num-train-epochs 3

python -m src.models.generate_and_extract --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --lora-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/dpo-20250717.gguf --template question_generator_user.j2 --bert-model models/extractor/modernbert-ja-310m/safetensors/finetuned/20250712/best_model --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```

## QA生成モデル開発
```
python -m src.data.split -i data/JSQuAD/train.jsonl -r 50 -o1 data/JSQuAD/pipeline/train/step1/raw.jsonl -o2 data/JSQuAD/pipeline/train/step2/raw.jsonl

python -m src.data.split -i data/JSQuAD/pipeline/train/step1/raw.jsonl -r 95 -o1 data/JSQuAD/pipeline/train/step1/sft_train.jsonl -o2 data/JSQuAD/pipeline/train/step1/sft_valid.jsonl

python -m src.data.split -i data/JSQuAD/pipeline/train/step1/raw.jsonl -r 80 -o1 data/JSQuAD/pipeline/train/step1/bert_train.jsonl -o2 data/JSQuAD/pipeline/train/step1/bert_valid.jsonl

python -m src.data.split -i data/JSQuAD/pipeline/train/step2/raw.jsonl -r 80 -o1 data/JSQuAD/pipeline/train/step2/bert_train.jsonl -o2 data/JSQuAD/pipeline/train/step2/bert_valid.jsonl

python -m src.models.train_sft --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --user-template train_generator_user.j2 --assistant-template train_generator_assistant.j2 --train-dataset data/JSQuAD/pipeline/train/step1/raw.jsonl --model-type qa_generator  
(
python -m src.models.train_sft --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --user-template train_generator_user.j2 --assistant-template train_generator_assistant.j2 --train-dataset data/JSQuAD/pipeline/train/step1/sft_train.jsonl --valid-dataset data/JSQuAD/pipeline/train/step1/sft_valid.jsonl --model-type qa_generator
)

python ../opt/llama/convert_lora_to_gguf.py models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/sft/20250713/checkpoint-3929 --outfile models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/sft-20250713.gguf --base models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base

python -m src.models.train_bert --model-path models/extractor/modernbert-ja-310m/safetensors/base --train-dataset-path data/JSQuAD/pipeline/train/step1/bert_train.jsonl --valid-dataset-path data/JSQuAD/pipeline/train/step1/bert_valid.jsonl --num-train-epochs 3

python -m src.models.generate_dpo_dataset --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --lora-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/sft-20250713.gguf --template question_generator_user.j2 --bert-model models/extractor/modernbert-ja-310m/safetensors/finetuned/20250712/best_model --input data/JSQuAD/pipeline/train/step2/raw.jsonl --output-dir data/JSQuAD/pipeline/train/step2 --num-candidates 5 --temperature 1.0

python -m src.models.train_dpo --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --resume-from-checkpoint models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/sft/20250713/checkpoint-3929 --user-template question_generator_user.j2 --assistant-template question_generator_assistant.j2 --train-dataset data/JSQuAD/pipeline/train/step2/dpo/202507170608/generated.jsonl --epochs 2

python ../opt/llama/convert_lora_to_gguf.py models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/dpo/20250717/checkpoint-7858 --outfile models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/dpo-20250717.gguf --base models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base

python -m src.models.train_bert --model-path models/extractor/modernbert-ja-310m/safetensors/finetuned/20250712/best_model --train-dataset-path data/JSQuAD/pipeline/train/step2/bert_train.jsonl --valid-dataset-path data/JSQuAD/pipeline/train/step2/bert_valid.jsonl --num-train-epochs 3

python -m src.models.generate_and_extract --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --lora-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/dpo-20250717.gguf --template question_generator_user.j2 --bert-model models/extractor/modernbert-ja-310m/safetensors/finetuned/20250712/best_model --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```

## モデル評価
```
python -m src.models.generate_and_extract --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --lora-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/sft-20250713.gguf --template question_generator_user.j2 --bert-model models/extractor/modernbert-ja-310m/safetensors/finetuned/20250712/best_model --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/pipeline/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048

./script/predict.sh -i data/JSQuAD/pipeline/eval/Llama-3.1-Swallow-8B-Instruct-v0.3/202507170808/generated.jsonl -t evaluatee.j2

python -m src.evals.rank --ground-truth-file data/JSQuAD/pipeline/eval/Llama-3.1-Swallow-8B-Instruct-v0.3/202507170808/generated.jsonl --prediction-base-dir output/JSQuAD/generated/Llama-3.1-Swallow-8B-Instruct-v0.3/202507170808

python -m src.evals.corr --file1 output/JSQuAD/baseline/ranking.md --file2 output/JSQuAD/generated/Llama-3.1-Swallow-8B-Instruct-v0.3/202507170808/ranking_20250718.md
```


- 回答を完璧に抽出したいというよりは、ある程度大雑把に回答の見当をつけてほしい
    - レーベンシュタインで距離を測ってマッチングさせるので
    - だからmodern-bertを使う

# QA生成モデルのコマンドの流れ
```
python -m src.data.split -i data/JSQuAD/train.jsonl -r 95 -o1 data/JSQuAD/train/sft_train.jsonl -o2 data/JSQuAD/train/sft_valid.jsonl

python -m src.models.train_sft --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --user-template train_generator_user.j2 --assistant-template train_generator_assistant.j2 --train-dataset data/JSQuAD/train/sft_train.jsonl --valid-dataset data/JSQuAD/train/sft_valid.jsonl --model-type qa_generator  
```

### 方向性刺激生成
```
python -m src.models.generate_dsp --generator-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --generator-lora models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/sft-20250530.gguf --generator-template qa_generator.j2 --refiner-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --refiner-template qa_generator_dsp.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 4096
```
```
python -m src.models.generate_dsp --generator-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --generator-lora models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/sft-20250530.gguf --generator-template qa_generator.j2 --refiner-model models/generator/llm-jp-3.1-13b-instruct4/gguf/base.gguf --refiner-template qa_generator_dsp.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 4096
```
```
--log-filename log/generate_dsp.log --log-type debug
```
```
./script/predict.sh -i data/JSQuAD/eval/mixture/202508061309/generated.jsonl -t evaluatee.j2
```
```
python -m src.evals.rank --ground-truth-file data/JSQuAD/eval/mixture/202508061309/generated.jsonl --prediction-base-dir output/JSQuAD/generated/mixture/202508061309
```
```
python -m src.evals.corr --file1 output/JSQuAD/baseline/ranking.md --file2 output/JSQuAD/generated/mixture/202508061309/ranking_20250806.md
```

```
python -m src.models.train_sft --base-model models/generator/llm-jp-3.1-13b-instruct4/safetensors/base --user-template train_generator_user.j2 --assistant-template train_generator_assistant.j2 --model-type qa_generator --train-dataset data/JSQuAD/train/preprocessed.jsonl
```

```
python -m src.models.generate_dsp --generator-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --generator-lora models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/sft-20250602.gguf --generator-template qa_generator.j2 --refiner-model models/generator/llm-jp-3.1-13b-instruct4/gguf/base.gguf --refiner-template qa_generator_dsp_en2ja.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 4096
```

```
python -m src.models.generate --base-model models/generator/llm-jp-3.1-13b-instruct4/gguf/base.gguf --template qa_generator_few_shot.j2 --input data/JSQuAD/eval/baseline.jsonl --few-shot-input data/JSQuAD/eval/few_shot.jsonl --shot-num 10 --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 8192
```

```
python -m src.models.generate_dsp --generator-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --generator-lora models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/sft-20250627.gguf --generator-template qa_generator.j2 --refiner-model models/generator/llm-jp-3.1-13b-instruct4/gguf/base.gguf --refiner-template qa_generator_dsp_en2ja.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 10240 --bert-model models/extractor/modernbert-ja-310m/safetensors/finetuned/20250712/best_model --generator-port 8002 --refiner-port 8003
```
```
python ../opt/llama/convert_lora_to_gguf.py models/generator/llm-jp-3.1-13b-instruct4/safetensors/sft/20250806/checkpoint-7858 --outfile models/generator/llm-jp-3.1-13b-instruct4/gguf/sft-20250806.gguf --base models/generator/llm-jp-3.1-13b-instruct4/safetensors/base
```
```
python -m src.models.generate --base-model models/generator/llm-jp-3.1-13b-instruct4/gguf/base.gguf --lora-model models/generator/llm-jp-3.1-13b-instruct4/gguf/sft-20250806.gguf --template qa_generator.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 8192
```


# LLM-as-a-Judgeによる定性評価
```
python -m src.evals.qualitive_llm --base-model models/evaluator/Qwen3-30B-A3B-Instruct-2507/gguf/base.gguf --template qualitive.j2 --input data/JSQuAD/eval/mixture/202508061309/generated.jsonl --output-dir data/JSQuAD/eval/mixture/202508061309
```

# Qwen3-30B-A3B-Instruct-2507の実力
```
python -m src.models.generate --base-model models/evaluator/Qwen3-30B-A3B-Instruct-2507/gguf/base.gguf --template qa_generator.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --max-tokens 32768 --n-ctx 40960
```

# thinkingモデルによるllm-as-a-judge
```
python -m src.llm_as_a_judge.generate --base-model models/evaluator/Qwen3-30B-A3B-Thinking-2507/gguf/base.gguf --template llm_as_a_judge.j2 --input data/SHP/validation.jsonl --output-dir data/SHP/judged 
```
