import nltk
import nltk.data
import os

def setup_nltk():
    nltk_data_dir = os.path.join(os.path.dirname(__file__), '../../nltk_data')
    os.makedirs(nltk_data_dir,exist_ok=True)

    nltk.data.path.append(nltk_data_dir)
    
    packages = ['punkt', 'wordnet','omw-1.4','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']

    for pkg in packages:
        try:
            nltk.data.find(pkg) #kiem tra xem da ton tai goi chua
        except:
            nltk.download(pkg,download_dir=nltk_data_dir,quiet=True)
   
