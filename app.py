# библиотеки

import streamlit as st
!pip install transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import re
import string
import torch



# функции, справочники
def preprocess_text(text):
    symbols_articles = set()
    for i in range(len(text)):
        if not text[i].isalnum() and not text[i].isspace():
            symbols_articles.add(text[i])
    symbols_articles = ''.join(symbols_articles)
    text = text.lower()
    pattern_1 = 'https://t.co/[\w]*'
    text = re.sub(pattern_1, '', text)
    pattern_2 = '&[\w]*;'
    text = re.sub(pattern_2, '', text)
    pattern_3 = '&#[\d]*;'
    text = re.sub(pattern_3, '', text)
    for p in string.punctuation + string.whitespace + symbols_articles:
        text = text.replace(p, ' ')
        text = text.strip()
        text = ' '.join([w for w in text.split(' ') if w != ''])
    return text

idx_to_label = {0: "Analyst Update",
                1: "Fed | Central Banks",
                2: "Company | Product News",
                3: "Treasuries | Corporate Debt",
                4: "Dividend",
                5: "Earnings",
                6: "Energy | Oil",
                7: "Financials",
                8: "Currencies",
                9: "General News | Opinion",
                10: "Gold | Metals | Materials",
                11: "IPO",
                12: "Legal | Regulation",
                13: "M&A | Investments",
                14: "Macro",
                15: "Markets",
                16: "Politics",
                17: "Personnel Change",
                18: "Stock Commentary",
                19: "Stock Movement"}



# инициализация pre-train модели и токенизатора

model_trans = AutoModelForSequenceClassification.from_pretrained('my_model', num_labels=20)

tokenizer = AutoTokenizer.from_pretrained("my_tokenizer")



# загрузка данных

user_text = st.text_input('Текст для классификации')

text_pp = preprocess_text(user_text)
text_token = tokenizer([text_pp], truncation = True, max_length=100, padding='max_length')

for k in text_token.keys():
    text_token[k] = torch.tensor(text_token[k])
    


# предсказание

with torch.no_grad():
    outputs = model_trans(**text_token)
    logits = outputs.logits
    label_predict = np.argmax(logits.detach().cpu().numpy(), axis=1)[0]
    st.write(f"Topic: {idx_to_label[label_predict]}")
