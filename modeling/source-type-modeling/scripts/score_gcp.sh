python src/predict.py \
  --dataset_name data/test_data_to_score.jsonl \
  --model_name_or_path alex2awesome/quote-detection__roberta-base__background-excluded \
  --outfile quote-detection-scored.jsonl \
  --tokenizer_name roberta-base \



# to run locally after training
python src/predict.py \
  --dataset_name data/test_data_to_score.jsonl \
  --model_name_or_path /dev/shm/roberta-base__sentence-model \
  --outfile quote-detection-scored.jsonl