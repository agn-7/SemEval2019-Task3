from preprocessing import preprocess
import pandas as pd


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}


def load_data(path, training):
    data = pd.read_csv(path, encoding='utf-8', sep='\t')
    if not training:
        return data['id'], data['turn1'], data['turn2'], data['turn3']
    else:
        return data['id'], data['turn1'], data['turn2'], data['turn3'], data['label']


def load_preprocessed_data(path, training=True):
    if not training:
        id, turn1, turn2, turn3 = load_data(path, training)
        t1 = turn1.apply(lambda x: preprocess(x))
        t2 = turn2.apply(lambda x: preprocess(x))
        t3 = turn3.apply(lambda x: preprocess(x))
        return id.values, t1.values, t2.values, t3.values
    else:
        id, turn1, turn2, turn3, label = load_data(path, training)
        t1 = turn1.apply(lambda x: preprocess(x))
        t2 = turn2.apply(lambda x: preprocess(x))
        t3 = turn3.apply(lambda x: preprocess(x))
        l = label.apply(lambda x: emotion2label[x])
        return id.values, t1.values, t2.values, t3.values, l.values