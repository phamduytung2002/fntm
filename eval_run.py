import numpy as np
import argparse
import topmost
from topmost.utils import log, config, static_utils
import os

if __name__ == "__main__":
    parser = topmost.utils.config.new_parser()
    parser.add_argument('--dir_path', type=str)
    args = parser.parse_args()

    dir = args.dir_path

    config_args = config.load_config(os.path.join(dir, 'config.txt'))
    logger = log.setup_logger(
        'main', os.path.join(dir, 'main.log'))

    if config_args.dataset in ['20NG', 'IMDB', 'Rakuten_Amazon',
                               'NYT', 'ECNews', 'Amazon_Review']:
        read_labels = True
    else:
        read_labels = False

    # load a preprocessed dataset
    dataset = topmost.data.BasicDatasetHandler(
        "./data/" + config_args.dataset, device=config_args.device, read_labels=read_labels,
        as_tensor=True)

    beta = np.load(os.path.join(dir, 'beta.npy')).reshape(config_args.num_topics, -1)
    train_theta = np.load(os.path.join(dir, 'train_theta.npy')).reshape(-1, config_args.num_topics)
    test_theta = np.load(os.path.join(dir, 'test_theta.npy')).reshape(-1, config_args.num_topics)
    
    print('beta shape: ', beta.size)

    top_words = static_utils.print_topic_words(
        beta, dataset.vocab, config_args.num_top_word)

    # model evaluation
    # TD
    TD = topmost.evaluations.compute_topic_diversity(top_words, _type="TD")
    print(f"TD: {TD:.5f}")
    logger.info(f"TD: {TD:.5f}")

    # evaluating clustering
    if read_labels:
        clustering_results = topmost.evaluations.evaluate_clustering(
            test_theta, dataset.test_labels)
        print(f"NMI: ", clustering_results['NMI'])
        print(f'Purity: ', clustering_results['Purity'])
        logger.info(f"NMI: {clustering_results['NMI']}")
        logger.info(f"Purity: {clustering_results['Purity']}")

    # TC
    _, TC = topmost.evaluations.topic_coherence.C_V_on_wikipedia(
        os.path.join(dir, 'top_words.txt'))
    print(f"TC: {TC:.5f}")
    logger.info(f"TC: {TC:.5f}")
