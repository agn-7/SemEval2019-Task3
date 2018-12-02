from load import load_preprocessed_data, load_preprocessed_external_data, label2emotion
import numpy as np
import json
import io
import os
import argparse
from keras.models import Model, load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding, LSTM, Concatenate, Reshape, GRU, Bidirectional
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold


externalDataPath = ""
trainDataPath = ""
testDataPath = ""
solutionPath = ""
gloveDir = ""

NUM_FOLDS = None
NUM_CLASSES = None
MAX_NB_WORDS = None
MAX_SEQUENCE_LENGTH = None
EMBEDDING_DIM = None
BATCH_SIZE = None
LSTM_DIM = None
DROPOUT = None
NUM_EPOCHS = None


def load_config(configPath):
    with open(configPath) as configfile:
        config = json.load(configfile)

    global externalDataPath, trainDataPath, testDataPath, solutionPath, gloveDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE

    externalDataPath = config["external_data_path"]
    trainDataPath = config["train_data_path"]
    testDataPath = config["test_data_path"]
    solutionPath = config["solution_path"]
    gloveDir = config["glove_dir"]

    NUM_FOLDS = config["num_folds"]
    NUM_CLASSES = config["num_classes"]
    MAX_NB_WORDS = config["max_nb_words"]
    MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
    EMBEDDING_DIM = config["embedding_dim"]
    BATCH_SIZE = config["batch_size"]
    LSTM_DIM = config["lstm_dim"]
    DROPOUT = config["dropout"]
    LEARNING_RATE = config["learning_rate"]
    NUM_EPOCHS = config["num_epochs"]


def create_solution_file(model, u1_testSequences, u2_testSequences, u3_testSequences):
    u1_testData, u2_testData, u3_testData = pad_sequences(u1_testSequences, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(u2_testSequences, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(u3_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict([u1_testData, u2_testData, u3_testData], batch_size=BATCH_SIZE)
    predictions = predictions.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d"
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))
    return


def create_solution_file_external(model,u_testSequences):
    u_testData = pad_sequences(u_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(u_testData, batch_size=BATCH_SIZE)
    predictions = predictions.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d"
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))
    return


def getEmbeddingMatrix(wordIndex):
    embeddingsIndex = {}
    with io.open(os.path.join(gloveDir, 'glove.840B.300d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embeddingVector = np.array([float(val) for val in values[1:]])
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            embeddingMatrix[i] = embeddingVector

    return embeddingMatrix


def model1(embeddingMatrix):
    x1 = Input(shape=(100,), dtype='int32', name='main_input1')
    x2 = Input(shape=(100,), dtype='int32', name='main_input2')
    x3 = Input(shape=(100,), dtype='int32', name='main_input3')

    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    emb1 = embeddingLayer(x1)
    emb2 = embeddingLayer(x2)
    emb3 = embeddingLayer(x3)

    lstm = Bidirectional(LSTM(LSTM_DIM, dropout=DROPOUT))

    lstm1 = lstm(emb1)
    lstm2 = lstm(emb2)
    lstm3 = lstm(emb3)

    inp = Concatenate(axis=-1)([lstm1, lstm2, lstm3])

    inp = Reshape((3, 2*LSTM_DIM, )) (inp)

    lstm_up = LSTM(LSTM_DIM, dropout=DROPOUT)

    out = lstm_up(inp)

    out = Dense(NUM_CLASSES, activation='softmax')(out)

    adam = optimizers.adam(lr=LEARNING_RATE)
    model = Model([x1,x2,x3],out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    print(model.summary())
    return model


def model1_external(embeddingMatrix):
    x1 = Input(shape=(100,), dtype='int32', name='main_input1')

    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    emb1 = embeddingLayer(x1)

    lstm = Bidirectional(LSTM(LSTM_DIM, dropout=DROPOUT))

    lstm1 = lstm(emb1)

    lstm_up = LSTM(LSTM_DIM, dropout=DROPOUT)

    out = lstm_up(lstm1)

    out = Dense(NUM_CLASSES, activation='softmax')(out)

    adam = optimizers.adam(lr=LEARNING_RATE)
    model = Model(x1,out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=[keras_metrics.f1_score])
    print(model.summary())
    return model


def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True)
    args = parser.parse_args()

    load_config(args.config)

    print("Processing training data...")
    trainIndices, u1_train, u2_train, u3_train, labels = load_preprocessed_data(trainDataPath)
    print("Processing test data...")
    _, u1_test, u2_test, u3_test = load_preprocessed_data(testDataPath, training=False)
    #print("Processing external data...")
    #text_external, labels_external = load_preprocessed_external_data(externalDataPath)
    #trainIndices = [i for i in range(len(labels_external.values))]

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    zipped_train = list(zip(u1_train, u2_train, u3_train))
    tokenizer.fit_on_texts([' '.join(w) for w in zipped_train])
    #tokenizer.fit_on_texts(text_external)
    u1_trainSequences, u2_trainSequences, u3_trainSequences = tokenizer.texts_to_sequences(u1_train), tokenizer.texts_to_sequences(u2_train), tokenizer.texts_to_sequences(u3_train)
    u1_testSequences, u2_testSequences, u3_testSequences = tokenizer.texts_to_sequences(u1_test), tokenizer.texts_to_sequences(u2_test), tokenizer.texts_to_sequences(u3_test)
    #u_trainSequences = tokenizer.texts_to_sequences(text_external)
    #zipped_test = list(zip(u1_test, u2_test, u3_test))
    #u_testSequences = tokenizer.texts_to_sequences([' '.join(w) for w in zipped_test])

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    embeddingMatrix = getEmbeddingMatrix(wordIndex)

    u1_data = pad_sequences(u1_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_data = pad_sequences(u2_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_data = pad_sequences(u3_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    #u_data = pad_sequences(u_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    #labels = to_categorical(np.asarray(labels_external))
    print("Shape of training data tensor: ", u1_data.shape)
    print("Shape of label tensor: ", labels.shape)

    np.random.shuffle(trainIndices)

    u1_data = u1_data[trainIndices]
    u2_data = u2_data[trainIndices]
    u3_data = u3_data[trainIndices]
    #u_data = u_data[trainIndices]

    labels = labels[trainIndices]

    print("Building model...")
    checkpoint = ModelCheckpoint('./model1.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='auto')
    model = model1(embeddingMatrix)
    model.fit([u1_data, u2_data, u3_data], labels, validation_split=0.1, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint])
    #model = model1_external(embeddingMatrix)
    #model.fit(u_data, labels, validation_split=0.1, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint])
    model = load_model('./model1.h5')
    print("Creating solution file...")
    create_solution_file(model, u1_testSequences, u2_testSequences, u3_testSequences)
    #create_solution_file_external(model, u_testSequences)


if __name__ == '__main__':
    main()
