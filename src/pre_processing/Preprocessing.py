import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

import re

import string

import emoji

import contractions

import os
import sys

# Import file setup nltk
from nltk_setup import setup_nltk

#  Import dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_path)
from load_dataset import datatset

df = datatset.load_dataset(name_file='train.csv')


# Remove missing values
df = df.dropna(subset=['text'])
# Remove duplicate
df =df.drop_duplicates(subset=['text'])

# print(df.info())

# Clean text
remove_character = string.punctuation

def clean_text(text: string):

    text = text.lower() # chuyen ve chu thuong

    text = text.replace('`',"'")
    text = contractions.fix(text) # chuan hao cac tu viet tat

    # xoa cac ki tu dac biet
    for character in remove_character: 
        text = text.replace(character, '')

    # chuyen cac emoji thanh chu
    text = emoji.demojize(text,delimiters=(' ',' '))

    # xoa link
    text = re.sub(r'http\S+|www\S+|https\S+', '',text,flags= re.MULTILINE)

    # xoa cac ki tu lap
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # text = re.sub(r'(.)\1{2,}', r'\1', text)

    # xoa cac chu so 
    text = re.sub(r'\d+', 'num',text)

    # xoa cac khoang trang trong cau
    text = re.sub(r'\s+',' ',text).strip()

    # xoa cac khoang trang o dau cau va cuoi cau
    text = text.strip()

    return text


def get_wordnet_pos(tag):
    
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # mặc định

# Remove stopword & lemmation

setup_nltk() # install package NLTK

lemmatizer = WordNetLemmatizer()

def lemmation(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(w,get_wordnet_pos(tag)) for w,tag in tagged]

    return ' '.join(tokens)

# Full function
def Preprocessing_text(text):
    text = clean_text(text)
    text = lemmation(text)

    return text

# Apply full function
df['text'] = df['text'].apply(Preprocessing_text)
df['label'] = df['sentiment']
df = df[['text','label']]

# Save file 
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir,'../../dataset/train_processed.csv')
df.to_csv(output_path, index=False, encoding='utf-8')