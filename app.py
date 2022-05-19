from select import devpoll
import streamlit as st
import os
import io
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
import time
import json
from typing import List
import torch
import random
import logging
from transformers import BertTokenizer, BertModel, BertConfig


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    logging.warning("GPU not found, using CPU, translation will be very slow.")

st.cache(suppress_st_warning=True, allow_output_mutation=True)
st.set_page_config(page_title="M2M100 Translator")

lang_id = {
    "Afrikaans": "af",
    "Amharic": "am",
    "Arabic": "ar",
    "Asturian": "ast",
    "Azerbaijani": "az",
    "Bashkir": "ba",
    "Belarusian": "be",
    "Bulgarian": "bg",
    "Bengali": "bn",
    "Breton": "br",
    "Bosnian": "bs",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Czech": "cs",
    "Welsh": "cy",
    "Danish": "da",
    "German": "de",
    "Greeek": "el",
    "English": "en",
    "Spanish": "es",
    "Estonian": "et",
    "Persian": "fa",
    "Fulah": "ff",
    "Finnish": "fi",
    "French": "fr",
    "Western Frisian": "fy",
    "Irish": "ga",
    "Gaelic": "gd",
    "Galician": "gl",
    "Gujarati": "gu",
    "Hausa": "ha",
    "Hebrew": "he",
    "Hindi": "hi",
    "Croatian": "hr",
    "Haitian": "ht",
    "Hungarian": "hu",
    "Armenian": "hy",
    "Indonesian": "id",
    "Igbo": "ig",
    "Iloko": "ilo",
    "Icelandic": "is",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Georgian": "ka",
    "Kazakh": "kk",
    "Central Khmer": "km",
    "Kannada": "kn",
    "Korean": "ko",
    "Luxembourgish": "lb",
    "Ganda": "lg",
    "Lingala": "ln",
    "Lao": "lo",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Malagasy": "mg",
    "Macedonian": "mk",
    "Malayalam": "ml",
    "Mongolian": "mn",
    "Marathi": "mr",
    "Malay": "ms",
    "Burmese": "my",
    "Nepali": "ne",
    "Dutch": "nl",
    "Norwegian": "no",
    "Northern Sotho": "ns",
    "Occitan": "oc",
    "Oriya": "or",
    "Panjabi": "pa",
    "Polish": "pl",
    "Pushto": "ps",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Sindhi": "sd",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Albanian": "sq",
    "Serbian": "sr",
    "Swati": "ss",
    "Sundanese": "su",
    "Swedish": "sv",
    "Swahili": "sw",
    "Tamil": "ta",
    "Thai": "th",
    "Tagalog": "tl",
    "Tswana": "tn",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Wolof": "wo",
    "Xhosa": "xh",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "Chinese": "zh",
    "Zulu": "zu",
}


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_model(
    pretrained_model: str = "facebook/m2m100_1.2B",
    cache_dir: str = "models/",
    bert: str = "bert-base-multilingual-cased",
):
    tokenizer = M2M100Tokenizer.from_pretrained(pretrained_model, cache_dir=cache_dir)
    model = M2M100ForConditionalGeneration.from_pretrained(
        pretrained_model, cache_dir=cache_dir
    ).to(device)
    config = BertConfig.from_pretrained(bert, output_hidden_states=True)
    bert_tokenizer: str = BertTokenizer.from_pretrained(bert, config=config)
    bert_model: str = BertModel.from_pretrained(bert).to(device)
    model.eval()
    bert_model.eval()
    return tokenizer, model, bert_tokenizer, bert_model


def find_algnments(
    source_text, translated_text, bert_tokenizer, bert_model, threshold=0.001
):
    source_tokens = bert_tokenizer(source_text, return_tensors="pt")
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
    target_source_mapping = nn.Softmax(dim=-1)(
        torch.matmul(target_embedding, source_embedding)
    )
    align_matrix = (source_target_mapping > threshold) * (
        target_source_mapping > threshold
    )
    non_zeros = torch.nonzero(align_matrix)
    align_words = []
    for i, j, k in non_zeros:
        if j + 1 < source_tokens_len - 1 and k + 1 < target_tokens_len - 1:
            align_words.append([bpe_source_map[j + 1], bpe_target_map[k + 1]])
    return align_words


st.title("M2M100 Translator")
st.write(
    "M2M100 is a multilingual encoder-decoder (seq-to-seq) model trained for Many-to-Many multilingual translation. It was introduced in this paper https://arxiv.org/abs/2010.11125 and first released in https://github.com/pytorch/fairseq/tree/master/examples/m2m_100 repository. The model that can directly translate between the 9,900 directions of 100 languages.\n"
)
st.write(
    "The BERT model was proposed in BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It’s a bidirectional transformer pretrained using a combination of masked language modeling objective and next sentence prediction on a large corpus comprising the Toronto Book Corpus and Wikipedia."
)
st.write(
    " This demo uses the facebook/m2m100_1.2B model. For local inference see https://github.com/ikergarcia1996/Easy-Translate"
)
st.write("This demo uses bert-base-multilingual-cased ")

user_input: str = st.text_area(
    "Input text",
    height=200,
    max_chars=5120,
)

source_lang = st.selectbox(label="Source language", options=list(lang_id.keys()))
target_lang = st.selectbox(label="Target language", options=list(lang_id.keys()))

if st.button("Run"):
    time_start = time.time()
    tokenizer, model, bert_tokenizer, bert_model = load_model()

    src_lang = lang_id[source_lang]
    trg_lang = lang_id[target_lang]
    tokenizer.src_lang = src_lang
    with torch.no_grad():
        encoded_input = tokenizer(user_input, return_tensors="pt").to(device)
        generated_tokens = model.generate(
            **encoded_input, forced_bos_token_id=tokenizer.get_lang_id(trg_lang)
        )
        translated_text = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

    time_end = time.time()
    alignments = find_algnments(user_input, translated_text, bert_tokenizer, bert_model)
    for i, j in alignments:
        st.success(f"{i}->{j}")

    st.write(f"Computation time: {round((time_end-time_start),3)} sec")
