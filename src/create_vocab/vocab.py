import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Create vocab
nltk.data.path.append('../../nltk_data')

all_tokens = [word for text in df.text for word in word_tokenize(text)]
vocab = {word : idx + 2 for idx, (word, _) in enumerate(Counter(all_tokens).most_common())}

vocab['<PAD>'] = 0 # khi padding
vocab['<UNK>'] = 1 # tu khong co trong vocab