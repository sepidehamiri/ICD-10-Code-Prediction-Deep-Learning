from time import strftime
import pandas as pd


def data_load(dataset_filename):
    print('Data loaded at: {}'.format(strftime('%H:%M:%S')))
    data = pd.read_csv(dataset_filename).astype(str)
    data_text = data['text'].values.tolist()
    labels = data['label'].values.tolist()
    data_label = [label.split() for label in labels]
    return data_text, data_label
