from transformers import BertModel, BertTokenizer

bert = BertModel.from_pretrained("./bert_chinese")
tokenizer = BertTokenizer.from_pretrained("./bert_chinese")

test_sentence = "我在测试bert"
tokens = tokenizer.encode_plus(text=test_sentence)
print(tokens)

tokenizer = BertTokenizer.from_pretrained("./bert_chinese")
test_sentence = "我在测试bert"
tokens = tokenizer.encode_plus(text=test_sentence)
print(tokens)
print(tokenizer.convert_ids_to_tokens(tokens['input_ids']))
