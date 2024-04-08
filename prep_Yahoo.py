import topmost
import preprocess
import topmost.utils.file_handling as fh
import pandas
from tqdm import tqdm
from topmost.preprocessing import Preprocessing


train_link = "datasets\\Yahoo_Answers\\raw\\train.csv"
test_link = "datasets\\Yahoo_Answers\\raw\\test.csv"


def to_jsonlist(link, save_dir):
    df = pandas.read_csv(link, names=['label', 'title', 'question', 'answer'])
    df = df.fillna(' ')
    df['text'] = df.iloc[:, -
                            3:].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

    df = df[['label', 'text']]
    data = []
    for index, row in tqdm(df.iterrows()):
        data.append({'label': row['label'], 'text': row['text']})
    fh.write_jsonlist(data, save_dir)


to_jsonlist(train_link, 'datasets\\Yahoo_Answers\\train.jsonlist')
to_jsonlist(test_link, 'datasets\\Yahoo_Answers\\test.jsonlist')

prep = Preprocessing(vocab_size=5000, stopwords='data\\stopwords\\snowball_stopwords.txt')
rst = prep.preprocess_jsonlist('datasets\\Yahoo_Answers', label_name='label')
prep.save('datasets\\Yahoo_Answers', **rst)
