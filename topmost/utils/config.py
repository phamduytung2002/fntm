import argparse


def new_parser(name=None):
    return argparse.ArgumentParser(prog=name)


def add_dataset_argument(parser):
    parser.add_argument('--dataset', type=str,
                        help='dataset name, currently support datasets are: \
                            20NG, ACL, Amazon_Review, ECNews, IMDB, NeurIPS, \
                            NYT, Rakuten_Amazon, Wikitext-103')


def add_model_argument(parser):
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--num_topics', type=int, default=50)
    parser.add_argument('--num_top_word', type=int, default=15)


def add_training_argument(parser):
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run the model, cuda or cpu')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--lr_scheduler', type=str,
                        help='learning rate scheduler, dont use if not needed, \
                            currently support: step')
    parser.add_argument('--lr_step_size', type=int, default=125,
                        help='step size for learning rate scheduler')

def save_config(args, path):
    with open(path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
