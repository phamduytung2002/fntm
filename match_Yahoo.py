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
import threading
import time

vocab_intersection_set = set()
semaphore = threading.Semaphore(3)
cnt = 0


def process_line(i, line, to_save):
    semaphore.acquire()
    ans = ' '.join(word for word in line.split()
                   if word in vocab_intersection_set)
    semaphore.release()
    to_save[i] = ans


def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    ans = [None] * len(lines)
    process_list = []
    for i in range(len(lines)):
        process_list.append(threading.Thread(
            target=process_line, args=(i, lines[i], ans)))
    for i in range(len(lines)):
        process_list[i].start()
    for i in range(len(lines)):
        process_list[i].join()
    return ans


def process(name, distances, X, Y, i, j, chunk_size_i, chunk_size_o):
    semaphore.acquire()
    # print('sem remain: ', semaphore._value)
    begin_time = time.time()
    X_chunk = X[i:i + chunk_size_o]
    Y_chunk = Y[j:j + chunk_size_i]
    if os.path.isfile(os.path.join(name, f'distances_{i}_{j}.npz')):
        x = np.load(os.path.join(name, f'distances_{i}_{j}.npz'))['arr_0']
    else:
        x = np.sum(
            np.abs(X_chunk[:, np.newaxis] - Y_chunk), axis=2).astype(np.int16)
    distances[0][i:i + chunk_size_o, j:j + chunk_size_i] = x
    global cnt
    cnt += 1
    print('n thread done: ', cnt, '- time: ', time.time() - begin_time)
    if not os.path.isfile(os.path.join(name, f'distances_{i}_{j}.npz')):
        np.savez(os.path.join(name, f'distances_{i}_{j}.npz'), x)
    semaphore.release()


def pairwise_manhattan_distance_chunked(name, X, Y, chunk_size_o=100, chunk_size_i=5000, n_threads=None):
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
    distances = np.zeros((n_samples_X, n_samples_Y), dtype=np.int16)

    begin_time = time.time()
    if n_threads is None:
        for i in tqdm(range(0, n_samples_X, chunk_size_o)):
            X_chunk = X[i:i + chunk_size_o]
            for j in tqdm(range(0, n_samples_Y, chunk_size_i)):
                Y_chunk = Y[j:j + chunk_size_i]
                # thread_list.append(threading.Thread(target=process, args=(distances, X, Y, i, j, chunk_size_i, chunk_size_o)))
                distances[i:i + chunk_size_o, j:j + chunk_size_i] = np.sum(
                    np.abs(X_chunk[:, np.newaxis] - Y_chunk), axis=2)
            # np.savez(os.path.join(name, f'distances_{i}.npz'), distances)
    else:
        thread_list = []
        for i in range(0, n_samples_X, chunk_size_o):
            for j in range(0, n_samples_Y, chunk_size_i):
                thread_list.append(threading.Thread(target=process, args=(
                    name, [distances], X, Y, i, j, chunk_size_i, chunk_size_o)))
        print('total thread: ', len(thread_list))
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()

    print('time: ', time.time() - begin_time)

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
    # my_test_bow = vectorizer.fit_transform(
    #     my_test_text).toarray().astype(np.int16)

    # scipy.sparse.save_npz(os.path.join(
    #     'datasets', 'YahooAnswer', 'my_test_bow_16.npz'), scipy.sparse.csr_matrix(my_test_bow))
    # print('my_test_bow:', my_test_bow.shape)

    # print('bowing their test')
    # their_test_text_path = os.path.join(their_AG, 'test_texts.txt')
    # their_test_text = process_file(their_test_text_path)
    # their_test_bow = vectorizer.fit_transform(
    #     their_test_text).toarray().astype(np.int16)
    # scipy.sparse.save_npz(os.path.join(
    #     'datasets', 'YahooAnswer', 'their_test_bow_16.npz'), scipy.sparse.csr_matrix(their_test_bow))
    # print('their_test_bow:', their_test_bow.shape)

    # # train texts
    # print('bowing my train')
    # my_train_text_path = os.path.join(my_AG, 'train_texts.txt')
    # my_train_text = process_file(my_train_text_path)
    # my_train_bow = vectorizer.fit_transform(
    #     my_train_text).toarray().astype(np.int16)
    # scipy.sparse.save_npz(os.path.join(
    #     'datasets', 'YahooAnswer', 'my_train_bow_16.npz'), scipy.sparse.csr_matrix(my_train_bow))
    # print('my_train_bow:', my_train_bow.shape)

    # print('bowing their train')
    # their_train_text_path = os.path.join(their_AG, 'train_texts.txt')
    # their_train_text = process_file(their_train_text_path)
    # their_train_bow = vectorizer.fit_transform(
    #     their_train_text).toarray().astype(np.int16)
    # scipy.sparse.save_npz(os.path.join(
    #     'datasets', 'YahooAnswer', 'their_train_bow_16.npz'), scipy.sparse.csr_matrix(their_train_bow))
    # print('their_train_bow:', their_train_bow.shape)

    # # compute manhattan distance
    # print('load my test bow')
    # my_test_bow_path = os.path.join(my_AG, 'my_test_bow.npz')
    # my_test_bow = scipy.sparse.load_npz(
    #     my_test_bow_path).toarray().astype(np.int16)
    # print('my_test_bow:', my_test_bow.shape)

    # print('load their test bow')
    # their_test_bow_path = os.path.join(my_AG, 'their_test_bow.npz')
    # their_test_bow = scipy.sparse.load_npz(
    #     their_test_bow_path).toarray().astype(np.int16)
    # print('their_test_bow:', their_test_bow.shape)

    # print('calculating manhattan distance')
    # manhattan_distance = pairwise_manhattan_distance_chunked(
    #     'test', their_test_bow, my_test_bow, n_threads=4)

    # np.savez(os.path.join(my_AG, 'test_matching.npz'),
    #          manhattan_distance=manhattan_distance)

    cnt = 0

    print('load my train bow')
    my_train_bow_path = os.path.join(my_AG, 'my_train_bow.npz')
    my_train_bow = scipy.sparse.load_npz(
        my_train_bow_path).toarray().astype(np.int16)

    print('my_train_bow:', my_train_bow.shape)

    print('load their train bow')
    their_train_bow_path = os.path.join(my_AG, 'their_train_bow.npz')
    their_train_bow = scipy.sparse.load_npz(
        their_train_bow_path).toarray().astype(np.int16)

    print('their_train_bow:', their_train_bow.shape)

    print('calculating manhattan distance')
    manhattan_distance = pairwise_manhattan_distance_chunked(
        'train', their_train_bow, my_train_bow, n_threads=4)

    np.savez(os.path.join(my_AG, 'train_matching.npz'),
             manhattan_distance=manhattan_distance)
