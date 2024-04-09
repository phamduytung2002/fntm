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


def pairwise_manhattan_distance_chunked(X, Y, chunk_size_o=400, chunk_size_i=400):
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
    distances = np.zeros((n_samples_X, n_samples_Y))

    for i in tqdm(range(0, n_samples_X, chunk_size_o)):
        X_chunk = X[i:i + chunk_size_o]
        for j in tqdm(range(0, n_samples_Y, chunk_size_i)):
            Y_chunk = Y[j:j + chunk_size_i]
            distances[i:i + chunk_size_o, j:j + chunk_size_i] = np.sum(
                np.abs(X_chunk[:, np.newaxis] - Y_chunk), axis=2)

    return distances


my_AG = os.path.join('datasets', 'AGNews')
their_AG = os.path.join('data', 'AGNews')

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

    # vocab_intersection = list(set(my_vocab_list) & set(their_vocab_list))
    # print('Vocab Intersection:', len(vocab_intersection))




    # my_train_bow_path = os.path.join(my_AG, 'train_bow.npz')
    # my_train_bow = scipy.sparse.load_npz(
    #     my_train_bow_path).toarray().astype('float32')

    # print('my_train_bow:', my_train_bow.shape)

    # their_train_bow_path = os.path.join(their_AG, 'train_bow.npz')
    # their_train_bow = scipy.sparse.load_npz(
    #     their_train_bow_path).toarray().astype('float32')

    # print('their_train_bow:', their_train_bow.shape)

    # manhattan_distance = pairwise_manhattan_distance_chunked(
    #     their_train_bow, my_train_bow)

    # np.savez(os.path.join(my_AG, 'train_matching.npz'),
    #          manhattan_distance=manhattan_distance)




    my_test_bow_path = os.path.join(my_AG, 'test_bow.npz')
    my_test_bow = scipy.sparse.load_npz(
        my_test_bow_path).toarray().astype('float32')

    print('my_test_bow:', my_test_bow.shape)

    their_test_bow_path = os.path.join(their_AG, 'test_bow.npz')
    their_test_bow = scipy.sparse.load_npz(
        their_test_bow_path).toarray().astype('float32')

    print('their_test_bow:', their_test_bow.shape)

    manhattan_distance = pairwise_manhattan_distance_chunked(
        their_test_bow, my_test_bow)

    np.savez(os.path.join(my_AG, 'test_matching.npz'),
             manhattan_distance=manhattan_distance)

