import topmost
import preprocess
import topmost.utils.file_handling as fh
import pandas

train_link = "datasets\\YahooAnswers\\raw\\train.csv"
test_link = "datasets\\YahooAnswers\\raw\\test.csv"


def to_jsonlist(link, save_dir):
    df = pandas.read_csv(link, names=['label', 'title', 'question', 'answer'])
    df = df.fillna(' ')
    df['text'] = df.iloc[:, -
                            3:].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

    df = df[['label', 'text']]
    data = []
    for index, row in df.iterrows():
        data.append({'label': row['label'], 'text': row['text']})
    fh.write_jsonlist(data, save_dir)


to_jsonlist(train_link, 'datasets\\YahooAnswers\\train.jsonlist')
to_jsonlist(test_link, 'datasets\\YahooAnswers\\test.jsonlist')
