import numpy as np
import topmost
import preprocess
import topmost.utils.file_handling as fh
import pandas
from tqdm import tqdm
from topmost.preprocessing import Preprocessing
import os
import scipy
import json
from tqdm import tqdm
import sentence_transformers

"""
    This file make bert embeddings for the data corressponding to their bow
"""

if __name__ == "__main__":
    bert_model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')
   
    match_argmin = np.load(os.path.join(
        'datasets', 'AGNews', 'train_matching_argmin.npz'))['argmin']
    train_dataset = fh.read_jsonlist(os.path.join(
        'datasets', 'AGNews', 'train.jsonlist'))
    train_text_list = [train_dataset[i]['text'] for i in match_argmin]
    print(train_text_list[:10])

    train_bert_emb = bert_model.encode(
        train_text_list, batch_size=256, show_progress_bar=True)
    
    np.savez(os.path.join('datasets', 'AGNews', 'train_bert.npz'), arr_0=train_bert_emb)


    match_argmin = np.load(os.path.join(
        'datasets', 'AGNews', 'test_matching_argmin.npz'))['argmin']
    test_dataset = fh.read_jsonlist(os.path.join(
        'datasets', 'AGNews', 'test.jsonlist'))
    test_text_list = [test_dataset[i]['text'] for i in match_argmin]
    print(test_text_list[:10])

    test_bert_emb = bert_model.encode(
        test_text_list, batch_size=256, show_progress_bar=True)
    
    np.savez(os.path.join('datasets', 'AGNews', 'test_bert.npz'), arr_0=test_bert_emb)
