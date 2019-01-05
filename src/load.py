from preprocessing import preprocess
import pandas as pd


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}


#load formatted id, text and (eventually) label from file in path
def load_data(path, training):
    data = pd.read_csv(path, encoding='utf-8', sep='\t')
    text = data[['turn1', 'turn2', 'turn3']].apply(lambda x: ' '.join(x), axis=1)
    if not training:
        return data['id'], text
    else:
        return data['id'], text, data['label']


#load data and preprocess them with ekphrasis
def load_preprocessed_data(path, training=True):
    if not training:
        id, text = load_data(path, training)
        t = text.apply(lambda x: preprocess(x))
        return id.values.tolist(), t.values.tolist()
    else:
        id, text, label = load_data(path, training)
        t = text.apply(lambda x: preprocess(x))
        l = label.apply(lambda x: emotion2label[x])
        return id.values.tolist(), t.values.tolist(), l.values.tolist()
