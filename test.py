from topmost.data import download_20ng
from topmost.data import download_dataset
from topmost.preprocessing import Preprocessing
import topmost

# download_20ng.download_save(output_dir='./datasets/20NG')

device = "cuda" # or "cpu"

# load a preprocessed dataset
dataset = topmost.data.BasicDatasetHandler("./data/ACL", device=device, read_labels=False, as_tensor=True)
# create a model
model = topmost.models.ProdLDA(dataset.vocab_size)
model = model.to(device)

# create a trainer
trainer = topmost.trainers.BasicTrainer(model)

# train the model
trainer.train(dataset)

# get theta (doc-topic distributions)
train_theta, test_theta = trainer.export_theta(dataset)
# get top words of topics
topic_top_words = trainer.export_top_words(dataset.vocab)

# evaluate topic diversity
TD = topmost.evaluations.compute_topic_diversity(topic_top_words)
print('TD: ', TD)
