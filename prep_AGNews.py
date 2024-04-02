import topmost
import preprocess
import topmost.utils.file_handling as fh
import pandas
from tqdm import tqdm
from topmost.preprocessing import Preprocessing


train_link = "datasets\\AGNews\\raw\\train.csv"
test_link = "datasets\\AGNews\\raw\\test.csv"


def to_jsonlist(link, save_dir):
    df = pandas.read_csv(link)
    df = df.fillna(' ')
    df['text'] = df.iloc[:, -
                            2:].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    df['label'] = df['Class Index']

    df = df[['label', 'text']]
    data = []
    for index, row in tqdm(df.iterrows()):
        data.append({'label': row['label'], 'text': row['text']})
    fh.write_jsonlist(data, save_dir)


to_jsonlist(train_link, 'datasets\\AGNews\\train.jsonlist')
to_jsonlist(test_link, 'datasets\\AGNews\\test.jsonlist')

prep = Preprocessing(test_sample_size=2500, vocab_size=5000, stopwords='data\\stopwords\\snowball_stopwords.txt')
rst = prep.preprocess_jsonlist('datasets\\AGNews', label_name='label')
prep.save('datasets\\AGNews', **rst)