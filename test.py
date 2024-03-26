from topmost.data import download_20ng
from topmost.data import download_dataset
from topmost.preprocessing import Preprocessing
import topmost
from topmost.utils.seed import seedEverything

# download_20ng.download_save(output_dir='./datasets/20NG')

seedEverything(2710)

device = "cuda"  # or "cpu"

datasets = ['20NG', 'ACL', 'Amazon_Review', 'ECNews', 'dict', 'IMDB',
            'NeurIPS', 'NYT', 'Rakuten_Amazon', 'stopwords', 'Wikitext-103']

for dataset in datasets:
    download_dataset(dataset, cache_path='data')

# load a preprocessed dataset
dataset = topmost.data.BasicDatasetHandler(
    "./data/ACL", device=device, read_labels=False, as_tensor=True)
# create a model
model = topmost.models.ProdLDA(dataset.vocab_size)
model = model.to(device)

# create a trainer
trainer = topmost.trainers.BasicTrainer(model, epochs=10)

# train the model
trainer.train(dataset)

# get theta (doc-topic distributions)
train_theta, test_theta = trainer.export_theta(dataset)
# get top words of topics
topic_top_words = trainer.export_top_words(dataset.vocab)

# evaluate topic diversity
TD = topmost.evaluations.compute_topic_diversity(topic_top_words)
print('TD: ', TD)
