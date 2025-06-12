## preprocess
```
python -m src.data.preprocess --input data/squad/train.jsonl --output data/squad/train/preprocessed.jsonl
```

## download
```
python -m src.setup.downloader.model --repo-id mradermacher/Gemma-2-Llama-Swallow-9b-it-v0.1-i1-GGUF --format gguf --filename Gemma-2-Llama-Swallow-9b-it-v0.1.i1-Q4_K_M
.gguf --output-dir models/generator 
```
```
python -m src.setup.downloader.data --dataset-id rajpurkar/squad --splits train,validation --output-dir data
```
```
python -m src.setup.download_all --model-list src/setup/list/model_list.csv --dataset-list src/setup/list/dataset_list.csv
```

## train
```
python -m src.data.preprocess --input data/JSQuAD/train.jsonl --output data/JSQuAD/train/preprocessed.jsonl
```
```
python -m src.models.train --base-model models/generator/gemma-2-9b-it/safetensors/base --user-template train_user.j2 --assistant-template train_assistant.j2 --dataset data/JSQuAD/train/preprocessed.jsonl
```
```
python -m src.models.train --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --resume-from-checkpoint models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/lora/20250602/checkpoint-10950 --model-type generator --user-template train_generator_user.j2 --assistant-template train_generator_assistant.j2 --dataset data/JSQuAD/train/preprocessed.jsonl --epochs 2
```
```
python -m src.models.train_grpo --base-model-path models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --sft-lora-path models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/lora/20250530/checkpoint-7858 --dataset-path data/ja-wiki/grpo.jsonl --prompt-template-name train_generator_user.j2 --emb-model-path models/embedding/multilingual-e5-large/safetensors/base
```

## generate
```
python -m src.models.generate --base-model models/generator/gemma-2-9b-it/gguf/base.gguf --template qa_generator_few_shot.j2 --input data/JSQuAD/eval/baseline.jsonl --few-shot-input data/JSQuAD/eval/few_shot.jsonl --shot-num 10 --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```
```
python -m src.models.generate --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base --lora-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/20250602/lora/checkpoint-10950 --template qa_generator.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```
```
python -m src.models.generate --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --lora-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/lora-20250602.gguf --template qa_generator_ja.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```
```
python -m src.models.generate --base-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf --lora-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/lora-20250610.gguf --template qa_generator.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```

# convert
```
python ../opt/llama/convert_lora_to_gguf.py models/generator/gemma-2-9b-it/safetensors/20250526/lora/checkpoint-7858 --outfile models/generator/gemma-2-9b-it/gguf/lora-20250526.gguf --base models/generator/gemma-2-9b-it/safetensors/base
```
```
python ../opt/llama/convert_lora_to_gguf.py models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/20250602/lora/checkpoint-10950 --outfile models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/lora-20250602.gguf --base models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base
```
```
python ../opt/llama/convert_lora_to_gguf.py models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/lora/20250610/checkpoint-15000 --outfile models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/lora-20250610.gguf --base models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/safetensors/base
```

## predict
```
python -m src.models.predict --base-model models/generator/gemma-2-9b-it/gguf/base.gguf --template evaluatee.j2 --input data/JSQuAD/eval/baseline.jsonl --output-dir output --n-gpu-layers 42
```

## script
```
./script/predict.sh -i data/JSQuAD/eval/baseline.jsonl -t evaluatee.j2
```

## evaluate
```
python -m src.data.corr --file1 output/JSQuAD/baseline/ranking.md --file2 output/JSQuAD/generated/gemma-2-9b-it/202505251436/ranking.md
```
```
python -m src.models.evaluate --ground-truth-file data/JSQuAD/eval/Llama-3.1-Swallow-8B-Instruct-v0.3/202505270601/generated.jsonl --prediction-base-dir output/JSQuAD/generated/L
lama-3.1-Swallow-8B-Instruct-v0.3/202505270601
```
```
python -m src.data.evaluate --ground-truth-file data/JSQuAD/eval/Llama-3.1-Swallow-8B-Instruct-v0.3/202506020258/generated.jsonl --prediction-base-dir output/JSQuAD/generated/Llama-3.1-Swallow-8B-Instruct-v0.3/202506020258 --llm-judge-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf
```
```
LLM_PORT=8081 python -m src.data.evaluate --ground-truth-file data/JSQuAD/eval/Llama-3.1-Swallow-8B-Instruct-v0.3/202506020258/generated.jsonl --prediction-base-dir output/JSQuAD/generated/Llama-3.1-Swallow-8B-Instruct-v0.3/202506020258 --llm-judge-model models/generator/Llama-3.1-Swallow-8B-Instruct-v0.3/gguf/base.gguf
```

## うまくllama-serverが立てられない時
```
ps aux | grep llama-server
```
