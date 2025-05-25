

## download
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

## script
```
./script/predict.sh -i data/JSQuAD/eval/baseline.jsonl -t evaluatee.j2 -p 8081
```
