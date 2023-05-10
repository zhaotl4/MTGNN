# python feature_extraction.py --bert_model bert-base-uncased --data ./data_tweet/exp_data --output ./bert_features_tweet/bert_features_train --batch_size 100

python feature_extraction.py --bert_model bert-base-uncased --data ./data_tweet/exp_data/train.label.jsonl --output ./bert_features_tweet/bert_features_train --batch_size 100
python feature_extraction.py --bert_model bert-base-uncased --data ./data_tweet/exp_data/val.label.jsonl --output ./bert_features_tweet/bert_features_val --batch_size 100
python feature_extraction.py --bert_model bert-base-uncased --data ./data_tweet/exp_data/test.label.jsonl --output ./bert_features_tweet/bert_features_test --batch_size 100
