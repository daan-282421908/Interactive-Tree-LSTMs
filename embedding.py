import os
from gensim.models import Word2Vec
from pandas.io.parsers import read_csv
import pandas as pd

def read_files(folder):
    all_paper = []
    for root, _, fnames in os.walk(folder):
        for fname in fnames:
            path = os.path.join(root, fname)
            df = pd.read_csv(path)
            # df = read_csv(path)
            content = []
            for i in range(0,len(df)):
                word = df['word'][i]
                content.append(word)
                all_paper.append(content)

    return all_paper

path_ = os.path.abspath('.')
paper_root = path_+'/table/'
sentences = read_files(paper_root)
paper_root = path_+'/table_test/'
sentences_ = read_files(paper_root)
sentences.extend(sentences_)

