from datetime import datetime
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_current_datetime():
    # Get the current date and time
    current_datetime = datetime.now()

    # Convert it to a string
    datetime_string = current_datetime.strftime(
        "%Y-%m-%d_%H-%M-%S")  # Format as YYYY-MM-DD HH:MM:SS
    return datetime_string


def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Folder created:", folder_path)
    else:
        print("Folder already exists:", folder_path)


def tsne_viz(word_embedding, topic_embedding, save_path):
    tsne = TSNE(n_components=2, random_state=0)
    word_c = np.ones(word_embedding.shape[0])
    topic_c = np.zeros(topic_embedding.shape[0])
    wt_c = np.concatenate([word_c, topic_c], axis=0)
    word_and_topic_emb = np.concatenate(
        [word_embedding, topic_embedding], axis=0)
    wt_tsne = tsne.fit_transform(word_and_topic_emb)

    plt.figure(figsize=(10, 5))
    plt.scatter(wt_tsne[:, 0], wt_tsne[:, 1], c=wt_c)
    for i, txt in enumerate(topic_c):
        plt.annotate(
            txt, (wt_tsne[word_c.shape[0] + i, 0], wt_tsne[word_c.shape[0] + i, 1]))
    plt.title('Word and Topic Embeddings')
    plt.savefig(save_path)
