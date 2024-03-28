from topmost.utils import config, log, miscellaneous, seed
import topmost
import wandb
import os
import numpy as np
import scipy

RESULT_DIR = 'results'
DATA_DIR = 'data'

if __name__ == "__main__":

    parser = config.new_parser()
    config.add_dataset_argument(parser)
    config.add_model_argument(parser)
    config.add_training_argument(parser)
    args = parser.parse_args()

    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR, current_time)
    miscellaneous.create_folder_if_not_exist(current_run_dir)

    config.save_config(args, os.path.join(current_run_dir, 'config.txt'))
    seed.seedEverything(args.seed)
    print(args)

    logger = log.setup_logger(
        'main', os.path.join(current_run_dir, 'main.log'))
    wandb.init(project='ntm', config=args)

    if args.dataset in ['20NG', 'IMDB', 'Rakuten_Amazon',
                        'NYT', 'ECNews', 'Amazon_Review']:
        read_labels = True
    else:
        read_labels = False

    # load a preprocessed dataset
    dataset = topmost.data.BasicDatasetHandler(
        os.path.join(DATA_DIR, args.dataset), device=args.device, read_labels=read_labels,
        as_tensor=True)

    # create a model
    if args.use_pretrainWE:
        if args.model in ['ETM', 'ECRTM']:
            pretrainWE = scipy.sparse.load_npz(os.path.join(
                DATA_DIR, args.dataset, "word_embeddings.npz")).toarray()
            model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                          num_topics=args.num_topics,
                                                          dropout=args.dropout,
                                                          pretrained_WE=pretrainWE)
        else:
            raise ValueError(
                "Pretrained word embeddings are not supported for this model")
    else:
        model = topmost.models.MODEL_DICT[args.model](vocab_size=dataset.vocab_size,
                                                      num_topics=args.num_topics,
                                                      dropout=args.dropout)
    model = model.to(args.device)

    # create a trainer
    trainer = topmost.trainers.BasicTrainer(model, epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size)

    # train the model
    trainer.train(dataset)

    # save beta, theta and top words
    beta = trainer.save_beta(current_run_dir)
    train_theta, test_theta = trainer.save_theta(dataset, current_run_dir)
    top_words = trainer.save_top_words(
        dataset.vocab, args.num_top_word, current_run_dir)

    # save word embeddings and topic embeddings
    if args.model in ['ETM', 'ECRTM']:
        trainer.save_embeddings(current_run_dir)
        miscellaneous.tsne_viz(model.word_embeddings.detach().cpu().numpy(),
                               model.topic_embeddings.detach().cpu().numpy(),
                               os.path.join(current_run_dir, 'tsne.png'))

    # model evaluation
    # TD
    TD = topmost.evaluations.compute_topic_diversity(top_words, _type="TD")
    print(f"TD: {TD:.5f}")
    wandb.log({"TD": TD})
    logger.info(f"TD: {TD:.5f}")

    # evaluating clustering
    if read_labels:
        clustering_results = topmost.evaluations.evaluate_clustering(
            test_theta, dataset.test_labels)
        print(f"NMI: ", clustering_results['NMI'])
        print(f'Purity: ', clustering_results['Purity'])
        wandb.log({"NMI": clustering_results['NMI']})
        wandb.log({"Purity": clustering_results['Purity']})
        logger.info(f"NMI: {clustering_results['NMI']}")
        logger.info(f"Purity: {clustering_results['Purity']}")

    # TC
    TC_list, TC = topmost.evaluations.topic_coherence.C_V_on_wikipedia(
        os.path.join(current_run_dir, 'top_words.txt'))
    print(f"TC: {TC:.5f}")
    wandb.log({"TC": TC})
    logger.info(f"TC: {TC:.5f}")
    logger.info(f'TC list: {TC_list}')

    wandb.finish()
