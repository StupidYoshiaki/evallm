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

## generate
```
python -m src.models.generate --base-model models/generator/gemma-2-9b-it/gguf/base.gguf --template qa_generator_few_shot.j2 --input data/JSQuAD/eval/baseline.jsonl --few-shot-input data/JSQuAD/eval/few_shot.jsonl --shot-num 10 --output-dir data/JSQuAD/eval --n-gpu-layers 42 --parallel 8 --n-ctx 2048
```

# convert
```
python ../opt/llama/convert_lora_to_gguf.py models/generator/gemma-2-9b-it/safetensors/20250526/lora/checkpoint-7858 --outfile models/generator/gemma-2-9b-it/gguf/lora-20250526.gguf --base models/generator/gemma-2-9b-it/safetensors/base
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

## うまくllama-serverが立てられない時
```
ps aux | grep llama-server
```
