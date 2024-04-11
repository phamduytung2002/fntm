import numpy as np
import topmost
import preprocess
import topmost.utils.file_handling as fh
import pandas
from tqdm import tqdm
from topmost.preprocessing import Preprocessing
import os
import scipy
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Pool

vocab_intersection_set = set()

def process_line(line):
    return ' '.join(word for word in line.split() if word in vocab_intersection_set)

def process_file(file_path):
    x = None
    with open(file_path, 'r') as file:
        lines = file.readlines()
    with Pool(processes=8) as pool:
        return pool.map(process_line, lines)

def pairwise_manhattan_distance_chunked(name, X, Y, chunk_size_o=100, chunk_size_i=10000):
    """
    Calculate pairwise Manhattan distance between two sets of points using chunking.

    Parameters:
        X (numpy.ndarray): First set of points, shape (n_samples_X, n_features).
        Y (numpy.ndarray): Second set of points, shape (n_samples_Y, n_features).
        chunk_size (int): Chunk size for processing.

    Returns:
        numpy.ndarray: Pairwise Manhattan distances, shape (n_samples_X, n_samples_Y).
    """
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    distances = np.zeros((n_samples_X, n_samples_Y), dtype=np.int8)

    for i in tqdm(range(0, n_samples_X, chunk_size_o)):
        X_chunk = X[i:i + chunk_size_o]
        for j in tqdm(range(0, n_samples_Y, chunk_size_i)):
            Y_chunk = Y[j:j + chunk_size_i]
            distances[i:i + chunk_size_o, j:j + chunk_size_i] = np.sum(
                np.abs(X_chunk[:, np.newaxis] - Y_chunk), axis=2)
        np.savez(os.path.join(name, 'distances_i.npz'), distances)

    return distances


my_AG = os.path.join('datasets', 'YahooAnswer')
their_AG = os.path.join('data', 'YahooAnswer')

"""
    This file calculate pairwise manhattan distance between their bow and my bow
"""

if __name__ == "__main__":
    # my_vocab_path = os.path.join(my_AG, 'vocab.txt')
    # my_vocab_list = []
    # for line in open(my_vocab_path, 'r'):
    #     my_vocab_list.append(line.strip())

    # their_vocab_path = os.path.join(their_AG, 'vocab.txt')
    # their_vocab_list = []
    # for line in open(their_vocab_path, 'r'):
    #     their_vocab_list.append(line.strip())

    # vocab_intersection_list = sorted(
    #     list(set(my_vocab_list) & set(their_vocab_list)))
    # vocab_intersection_set = set(vocab_intersection_list)
    # print('Vocab Intersection:', len(vocab_intersection_list))

    # # translate to vocab intersection
    # vectorizer = CountVectorizer(
    #     vocabulary=vocab_intersection_list, tokenizer=lambda x: x.split())

    # # test texts
    # print('bowing my test')
    # my_test_text_path = os.path.join(my_AG, 'test_texts.txt')
    # my_test_text = process_file(my_test_text_path)
    # my_test_bow = vectorizer.fit_transform(my_test_text).toarray()
    # scipy.sparse.save_npz(os.path.join(
    #     'datasets', 'YahooAnswer', 'my_test_bow.npz'), scipy.sparse.csr_matrix(my_test_bow))
    # print('my_test_bow:', my_test_bow.shape)

    # print('bowing their test')
    # their_test_text_path = os.path.join(their_AG, 'test_texts.txt')
    # their_test_text = process_file(their_test_text_path)
    # their_test_bow = vectorizer.fit_transform(their_test_text).toarray()
    # scipy.sparse.save_npz(os.path.join(
    #     'datasets', 'YahooAnswer', 'their_test_bow.npz'), scipy.sparse.csr_matrix(their_test_bow))
    # print('their_test_bow:', their_test_bow.shape)


    # # train texts
    # print('bowing my train')
    # my_train_text_path = os.path.join(my_AG, 'train_texts.txt')
    # my_train_text = process_file(my_train_text_path)
    # my_train_bow = vectorizer.fit_transform(my_train_text).toarray()
    # scipy.sparse.save_npz(os.path.join(
    #     'datasets', 'YahooAnswer', 'my_train_bow.npz'), scipy.sparse.csr_matrix(my_train_bow))
    # print('my_train_bow:', my_train_bow.shape)

    # print('bowing their train')
    # their_train_text_path = os.path.join(their_AG, 'train_texts.txt')
    # their_train_text = process_file(their_train_text_path)
    # their_train_bow = vectorizer.fit_transform(their_train_text).toarray()
    # scipy.sparse.save_npz(os.path.join(
    #     'datasets', 'YahooAnswer', 'their_train_bow.npz'), scipy.sparse.csr_matrix(their_train_bow))
    # print('their_train_bow:', their_train_bow.shape)


    # compute manhattan distance
    print('load my train bow')
    my_train_bow_path = os.path.join(my_AG, 'my_train_bow.npz')
    my_train_bow = scipy.sparse.load_npz(
        my_train_bow_path).toarray().astype(np.int8)

    print('my_train_bow:', my_train_bow.shape)

    print('load their train bow')
    their_train_bow_path = os.path.join(my_AG, 'their_train_bow.npz')
    their_train_bow = scipy.sparse.load_npz(
        their_train_bow_path).toarray().astype(np.int8)

    print('their_train_bow:', their_train_bow.shape)


    print('calculating manhattan distance')
    manhattan_distance = pairwise_manhattan_distance_chunked(
        'train', their_train_bow, my_train_bow)

    np.savez(os.path.join(my_AG, 'train_matching.npz'),
             manhattan_distance=manhattan_distance)

    print('load my test bow')
    my_test_bow_path = os.path.join(my_AG, 'my_test_bow.npz')
    my_test_bow = scipy.sparse.load_npz(
        my_test_bow_path).toarray().astype(np.int8)
    print('my_test_bow:', my_test_bow.shape)

    print('load their test bow')
    their_test_bow_path = os.path.join(my_AG, 'their_test_bow.npz')
    their_test_bow = scipy.sparse.load_npz(
        their_test_bow_path).toarray().astype(np.int8)
    print('their_test_bow:', their_test_bow.shape)

    print('calculating manhattan distance')
    manhattan_distance = pairwise_manhattan_distance_chunked(
        'test', their_test_bow, my_test_bow)

    np.savez(os.path.join(my_AG, 'test_matching.npz'),
             manhattan_distance=manhattan_distance)
