from load import load_preprocessed_data, label2emotion
import numpy as np
import json
import io
import os
import argparse
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
from keras.layers import Input, Dense, Embedding, LSTM, Concatenate, Reshape, GRU, Bidirectional, Dropout, Conv1D, Flatten, MaxPool1D
from keras.models import Sequential
from keras import optimizers


np.random.seed(7)


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
LEARNING_RATE = None
NUM_EPOCHS = None


def load_config(configPath):
    with open(configPath) as configfile:
        config = json.load(configfile)

    global trainDataPath, testDataPath, solutionPath, gloveDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE, MAX_SEQUENCE_LENGTH

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


def create_solution_file(model,u_testSequences):
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
        else:
            oov = [np.random.normal(size = EMBEDDING_DIM)]
            oov /= np.linalg.norm(oov)
            embeddingMatrix[i] = oov

    return embeddingMatrix


def model1(embeddingMatrix):
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(Dropout(DROPOUT))
    model.add(Bidirectional(LSTM(LSTM_DIM)))
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    adam = optimizers.adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    model.summary()
    return model


def model2(embeddingMatrix):
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(Dropout(DROPOUT))
    model.add(GRU(128))
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_CLASSES * 8, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    adam = optimizers.adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    model.summary()
    return model


def model3(embeddingMatrix):
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(Dropout(DROPOUT))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(Conv1D(32, 3, padding='same'))
    model.add(Conv1D(16, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(DROPOUT))
    model.add(Dense(180, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    adam = optimizers.adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    model.summary()
    return model


def model4(embeddingMatrix):
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(2))
    model.add(LSTM(LSTM_DIM))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    adam = optimizers.adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    model.summary()
    return model


def model5(embeddingMatrix):
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(LSTM(LSTM_DIM, return_sequences=True))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(2))
    model.add(Flatten())
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    adam = optimizers.adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    model.summary()
    return model


def model6(embeddingMatrix):
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(2))
    model.add(Dropout(DROPOUT))
    model.add(Bidirectional(LSTM(LSTM_DIM)))
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    adam = optimizers.adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    model.summary()
    return model


def model7(embeddingMatrix):
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(Bidirectional(LSTM(LSTM_DIM, return_sequences=True)))
    model.add(Dropout(DROPOUT))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(2))
    model.add(Flatten())
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    adam = optimizers.adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    model.summary()
    return model
    

def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True)
    args = parser.parse_args()

    load_config(args.config)

    print("Processing training data...")
    trainIndices, text_train, labels = load_preprocessed_data(trainDataPath)
    print("Processing test data...")
    _, text_test = load_preprocessed_data(testDataPath, training=False)

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(text_train)
    u_trainSequences = tokenizer.texts_to_sequences(text_train)
    u_testSequences = tokenizer.texts_to_sequences(text_test)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    embeddingMatrix = getEmbeddingMatrix(wordIndex)
    u_data = pad_sequences(u_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))

    np.random.shuffle(trainIndices)
    u_data = u_data[trainIndices]
    labels = labels[trainIndices]

    print("Building model...")
    cbks = [ModelCheckpoint('./model1.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto'),
            EarlyStopping(monitor='val_loss', patience=2)]
    model = model7(embeddingMatrix)
    model.fit(u_data, labels, validation_split=0.1, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=cbks)
    model = load_model('./model1.h5')
    print("Creating solution file...")
    create_solution_file(model, u_testSequences)


if __name__ == '__main__':
    main()
