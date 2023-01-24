import numpy as np

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')

tokenizer.add_special_tokens({"sep_token": "<|STK_SP|>"})

input_file = 'train.txt'
output_file = 'train.npy'

print(f'Tokenizing {input_file} (VERY slow. please wait)')

data_raw = open(input_file, encoding="utf-8").read()
print(f'Raw length = {len(data_raw)}')

data_code = tokenizer.encode(data_raw)
print(f'Tokenized length = {len(data_code)}')

out = np.array(data_code, dtype='uint16')
np.save(output_file, out, allow_pickle=False)

tokenizer._tokenizer.save('20B_tokenizer_openchatgpt.json')