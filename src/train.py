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
from keras.layers import Input, Dense, Embedding, LSTM, Concatenate, Reshape, GRU, Bidirectional, Dropout, Conv1D, Flatten, MaxPool1D, TimeDistributed
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
np.random.seed(7)


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        r = self.model.predict(x)
        getMetrics(r, y)


validationDataPath = ""
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

    global validationDataPath, trainDataPath, testDataPath, solutionPath, gloveDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE, MAX_SEQUENCE_LENGTH
    
    validationDataPath = config["validation_data_path"]
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


def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))
    
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    
    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()    
    
    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)
    
    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    
    return accuracy, microPrecision, microRecall, microF1


def model1(embeddingMatrix):
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
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
                                trainable=True)
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


def model8(embeddingMatrix):
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(Bidirectional(LSTM(LSTM_DIM, dropout=0.6)))
    model.add(Dropout(0.9))
    model.add(Dense(100, activation='tanh'))
    model.add(Dropout(0.9))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['acc'])
    model.summary()
    return model


def model9(embeddingMatrix):
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(2))
    model.add(GRU(LSTM_DIM))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    adam = optimizers.adam(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    model.summary()
    return model


def model10(embeddingMatrix):
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(GRU(LSTM_DIM, return_sequences=True))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPool1D(2))
    model.add(Flatten())
    model.add(Dropout(0.4))
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
    print("Processing validation data...")
    _, X_validation, y_validation = load_preprocessed_data(validationDataPath)

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(text_train)
    u_trainSequences = tokenizer.texts_to_sequences(text_train)
    u_testSequences = tokenizer.texts_to_sequences(text_test)
    u_validationSequences = tokenizer.texts_to_sequences(X_validation)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    embeddingMatrix = getEmbeddingMatrix(wordIndex)
    u_data = pad_sequences(u_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    u_validation = pad_sequences(u_validationSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels_validation = to_categorical(np.asarray(y_validation))

    np.random.shuffle(trainIndices)
    u_data = u_data[trainIndices]
    labels = labels[trainIndices]

    print("Building model...")
    cbks = [ModelCheckpoint('./model1.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto'),
            EarlyStopping(monitor='val_loss', patience=2),
            TestCallback((u_validation, labels_validation))]
    model = model10(embeddingMatrix)
    model.fit(u_data, labels, validation_data=(u_validation, labels_validation), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, callbacks=cbks)
    model = load_model('./model1.h5')
    print("Creating solution file...")
    create_solution_file(model, u_testSequences)


if __name__ == '__main__':
    main()
