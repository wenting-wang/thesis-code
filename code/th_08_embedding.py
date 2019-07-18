# Reference
# https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/word_embedding/
# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
# https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py

from __future__ import print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import GRU
from keras.models import Model
from keras.initializers import Constant

BASE_DIR = '/home/wenting/PycharmProjects/thesis/'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')  # glove.6B
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'data/prepared_data_respondent/')  # Change
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100  # 300
VALIDATION_SPLIT = 0.35

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:  # glove.6B.100d
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            # if fname.isdigit():
            fpath = os.path.join(path, fname)
            args = {} if sys.version_info < (3,) else {'encoding': 'utf-8'}
            with open(fpath, **args) as f:
                t = f.read()
                # i = t.find('\n\n')  # skip header
                # if 0 < i:
                #     t = t[i:]
                texts.append(t)
            labels.append(label_id)

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# labels = to_categorical(np.asarray(labels))
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

x = Conv1D(32, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(32, 5, activation='relu')(x)
x = GRU(32, dropout=0.1, recurrent_dropout=0.5)(x)
preds = Dense(1, activation='sigmoid')(x)
# preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)

# callbacks setting
# $ mkdir /home/wenting/PycharmProjects/thesis/log_dir
# $ conda activate nlp
# $ rm /home/wenting/PycharmProjects/thesis/log_dir/*
# $ tensorboard --logdir=/home/wenting/PycharmProjects/thesis/log_dir

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=BASE_DIR + 'model/' + 'model.h5',
        monitor='val_loss',
        save_best_only=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=10,
    ),
    keras.callbacks.TensorBoard(
        log_dir=BASE_DIR + 'log_dir',
        histogram_freq=1,
    )
]

# compile and fit model

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=50,
                    callbacks=callbacks_list,
                    validation_data=(x_val, y_val))

print('plot the loss and accuracy graph')


# draw graph
def draw_graph():
    plt.clf()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.figure()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.show()


draw_graph()