from topmost.utils import config, log, miscellaneous, seed
import topmost
import wandb
import os

RESULT_DIR = 'results'

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

    # load a preprocessed dataset
    dataset = topmost.data.BasicDatasetHandler(
        "./data/" + args.dataset, device=args.device, read_labels=False, as_tensor=True)
    # create a model
    model = topmost.models.ProdLDA(dataset.vocab_size)
    model = model.to(args.device)

    # create a trainer
    trainer = topmost.trainers.BasicTrainer(model, epochs=args.epochs, 
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size)

    # train the model
    trainer.train(dataset)

    # get theta (doc-topic distributions)
    train_theta, test_theta = trainer.export_theta(dataset)
    # get top words of topics
    topic_top_words = trainer.export_top_words(dataset.vocab)
    
    wandb.finish()
