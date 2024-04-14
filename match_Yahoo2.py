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

"""
    This file calculate the inidice of the minimum value in the manhattan
    distance matrix
"""


if __name__ == "__main__":
    match_file = os.path.join('datasets', 'YahooAnswer', 'test_matching.npz')
    match_data = np.load(match_file)['manhattan_distance']
    match_argmin = np.argmin(match_data, axis=1)
    for i in range(len(match_argmin)):
        print(match_data[i, match_argmin[i]])
    np.savez(os.path.join('datasets', 'YahooAnswer',
             'test_matching_argmin.npz'), argmin=match_argmin)

    match_file = os.path.join('datasets', 'YahooAnswer', 'train_matching.npz')
    match_data = np.load(match_file)['manhattan_distance']
    match_argmin = np.argmin(match_data, axis=1)
    for i in range(len(match_argmin)):
        print(match_data[i, match_argmin[i]])
    np.savez(os.path.join('datasets', 'YahooAnswer',
             'train_matching_argmin.npz'), argmin=match_argmin)


