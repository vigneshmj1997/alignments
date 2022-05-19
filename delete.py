from transformers import BertTokenizer, BertModel, BertConfig
import torch
from torch import nn

threshold = 0.001
device = "cpu"
bert = "bert-base-multilingual-cased"
config = BertConfig.from_pretrained(bert, output_hidden_states=True)
bert_tokenizer = BertTokenizer.from_pretrained(bert)
bert_model = BertModel.from_pretrained(bert, config=config).to(device)
source_text = "Hello, my dog is cute"
translated_text = "Hello, my dog is cute"
source_tokens = bert_tokenizer(source_text, return_tensors="pt")
print(source_tokens)
source_tokens_len = len(bert_tokenizer.tokenize(source_text))
target_tokens_len = len(bert_tokenizer.tokenize(translated_text))
target_tokens = bert_tokenizer(translated_text, return_tensors="pt")
bpe_source_map = []
for i in source_text.split():
    bpe_source_map += len(bert_tokenizer.tokenize(i)) * [i]
bpe_target_map = []
for i in translated_text.split():
    bpe_target_map += len(bert_tokenizer.tokenize(i)) * [i]
source_embedding = bert_model(**source_tokens).hidden_states[8]
target_embedding = bert_model(**target_tokens).hidden_states[8]
target_embedding = target_embedding.transpose(-1, -2)
source_target_mapping = nn.Softmax(dim=-1)(
    torch.matmul(source_embedding, target_embedding)
)
print(source_target_mapping.shape)
target_source_mapping = nn.Softmax(dim=-2)(
    torch.matmul(source_embedding, target_embedding)
)
print(target_source_mapping.shape)

align_matrix = (source_target_mapping > threshold) * (target_source_mapping > threshold)
align_prob = (2 * source_target_mapping * target_source_mapping) / (
    source_target_mapping + target_source_mapping + 1e-9
)
non_zeros = torch.nonzero(align_matrix)
print(non_zeros)
for i, j, k in non_zeros:
    if j + 1 < source_tokens_len - 1 and k + 1 < target_tokens_len - 1:
        print(bpe_source_map[j + 1], bpe_target_map[k + 1])
