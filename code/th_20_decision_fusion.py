from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM, GRU, Input, Bidirectional, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.initializers import Constant
from keras import initializers
from keras.engine.topology import Layer, InputSpec
from keras import backend as K
from keras.layers import concatenate
import sys
sys.path.append('/home/wenting/PycharmProjects/thesis/code')
import os
import th_17_mix_data_justice
import th_16_model_justice
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# load model
model_info = load_model('/home/wenting/PycharmProjects/thesis/model/mixed_model_justice/best_model_info.h5')
model_text = load_model('/home/wenting/PycharmProjects/thesis/model/mixed_model_justice/best_model_text.h5')
model_audio = load_model('/home/wenting/PycharmProjects/thesis/model/mixed_model_justice/best_model_audio.h5')



# remove the last layer of each model
# freeze all layers before the last layer
# concatenate three model
# add new decision layer, and train with aligned data

for layer in model_info.layers[:-1]:
    layer.trainable = False
for layer in model_text.layers[:-2]:
    layer.trainable = False
for layer in model_audio.layers[:-2]:
    layer.trainable = False

# change layer name
for i, layer in enumerate(model_info.layers):
    layer.name = layer.name + '_info'
for i, layer in enumerate(model_text.layers):
    layer.name = layer.name + '_text'
for i, layer in enumerate(model_audio.layers):
    layer.name = layer.name + '_audio'

# show models
for i, layer in enumerate(model_info.layers):
    print(i, layer.name, layer.trainable)
for i, layer in enumerate(model_text.layers):
    print(i, layer.name, layer.trainable)
for i, layer in enumerate(model_audio.layers):
    print(i, layer.name, layer.trainable)


# load and prepare data
# embedding settings
MAX_NUM_WORDS = 20000
MAX_LENGTH = 400  # change
EMBEDDING_DIM = 100
BASE_DIR = '/home/wenting/PycharmProjects/thesis/'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')  # glove.6B

# load the saved model
# from keras.models import load_model
# saved_model = load_model('best_model.h5')


# load data
info_file = '/home/wenting/PycharmProjects/thesis/data/mixed_data_justice/case_info_justice_filtered.csv'
arguments_file_dir = '/home/wenting/PycharmProjects/thesis/data/mixed_data_justice/text_data_justice_filtered'
audio_file = '/home/wenting/PycharmProjects/thesis/data/mixed_data_justice/audio_filtered.csv'

print("[INFO] loading cases attributes csv...")
info_processed = th_17_mix_data_justice.load_structured_data(info_file)

print("[INFO] loading audio pitch...")
audio = th_17_mix_data_justice.load_audio_data(audio_file)
info_and_audio = pd.concat([info_processed, audio], axis=1)

print("[INFO] loading oral arguments text...")
oral_data = th_17_mix_data_justice.load_arguments_text(info_file, arguments_file_dir, MAX_NUM_WORDS, MAX_LENGTH)
texts_pad, word_index = oral_data


# GloVe embedding preparing
# indexing word vectors
print("[INFO] preparing embedding matrix...")
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:  # glove.6B.100d
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

# embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# split data
# train, val, test: 0.79, 0.11, 0.1
print("[INFO] processing data...")
trainAttrX, testAttrX, trainTextX, testTextX = train_test_split(info_and_audio, texts_pad, test_size=0.1, random_state=1)
trainAttrX, valAttrX, trainTextX, valTextX = train_test_split(trainAttrX, trainTextX, test_size=0.1, random_state=1)

# text
trainTextX_pe = trainTextX[:, :, 0]
trainTextX_re = trainTextX[:, :, 1]
testTextX_pe = testTextX[:, :, 0]
testTextX_re = testTextX[:, :, 1]
valTextX_pe = valTextX[:, :, 0]
valTextX_re = valTextX[:, :, 1]

# target
trainY = trainAttrX['petitioner_vote']
testY = testAttrX['petitioner_vote']
valY = valAttrX['petitioner_vote']

# audio
trainAudioX_pe = trainAttrX['petitioner_pitch']
trainAudioX_re = trainAttrX['respondent_pitch']
testAudioX_pe = testAttrX['petitioner_pitch']
testAudioX_re = testAttrX['respondent_pitch']
valAudioX_pe = valAttrX['petitioner_pitch']
valAudioX_re = valAttrX['respondent_pitch']

# structured data
trainAttrX = trainAttrX.drop(['petitioner_vote', 'petitioner_pitch', 'respondent_pitch'], axis=1)
testAttrX = testAttrX.drop(['petitioner_vote', 'petitioner_pitch', 'respondent_pitch'], axis=1)
valAttrX = valAttrX.drop(['petitioner_vote', 'petitioner_pitch', 'respondent_pitch'], axis=1)


def info_text():
    combinedInput = concatenate([model_info.output, model_text.output])

    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(4, activation='relu')(combinedInput)
    x = Dense(1, activation='sigmoid')(x)

    # build model
    model = Model(inputs=[model_info.input, model_text.input[0], model_text.input[1]], outputs=x)

    # compile the model using mean absolute percentage error as our loss
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print("[INFO] model summary...")
    print(model.summary())

    # simple early stopping
    best_model = '/home/wenting/PycharmProjects/thesis/model/mixed_model_justice/best_model_fine.h5'
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(best_model, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    # fit model
    print("[INFO] training model...")
    history = model.fit(
        [trainAttrX, trainTextX_pe, trainTextX_re], trainY,
        validation_data=([valAttrX, valTextX_pe, valTextX_re], valY),
        epochs=5, batch_size=16, verbose=0, callbacks=[es, mc])

    # evaluate the model
    _, train_acc = model.evaluate([trainAttrX, trainTextX_pe, trainTextX_re], trainY, verbose=0)
    _, val_acc = model.evaluate([valAttrX, valTextX_pe, valTextX_re], valY, verbose=0)
    _, test_acc = model.evaluate([testAttrX, testTextX_pe, testTextX_re], testY, verbose=0)

    print('Train: %.3f, Validation: %.3f, Test: %.3f' % (train_acc, val_acc, test_acc))


def info_audio():
    combinedInput = concatenate([model_info.output, model_audio.output])

    x = Dense(4, activation='relu')(combinedInput)
    x = Dense(1, activation='sigmoid')(x)

    # build model
    model = Model(inputs=[model_info.input, model_audio.input[0], model_audio.input[1]], outputs=x)

    # compile the model using mean absolute percentage error as our loss
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print("[INFO] model summary...")
    print(model.summary())

    # simple early stopping
    best_model = '/home/wenting/PycharmProjects/thesis/model/mixed_model_justice/best_model.h5'
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(best_model, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    # fit model
    print("[INFO] training model...")
    history = model.fit(
        [trainAttrX, trainAudioX_pe, trainAudioX_re], trainY,
        validation_data=([valAttrX, valAudioX_pe, valAudioX_re], valY),
        epochs=5, batch_size=16, verbose=0, callbacks=[es, mc])

    # evaluate the model
    _, train_acc = model.evaluate([trainAttrX, trainAudioX_pe, trainAudioX_re], trainY, verbose=0)
    _, val_acc = model.evaluate([valAttrX, valAudioX_pe, valAudioX_re], valY, verbose=0)
    _, test_acc = model.evaluate([testAttrX, testAudioX_pe, testAudioX_re], testY, verbose=0)

    print('Train: %.3f, Validation: %.3f, Test: %.3f' % (train_acc, val_acc, test_acc))


def text_audio():
    combinedInput = concatenate([model_text.output, model_audio.output])

    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(4, activation='relu')(combinedInput)
    x = Dense(1, activation='sigmoid')(x)

    # build model
    model = Model(inputs=[model_text.input[0], model_text.input[1], model_audio.input[0], model_audio.input[1]], outputs=x)

    # compile the model using mean absolute percentage error as our loss
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print("[INFO] model summary...")
    print(model.summary())

    # simple early stopping
    best_model = '/home/wenting/PycharmProjects/thesis/model/mixed_model_justice/best_model.h5'
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(best_model, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    # fit model
    print("[INFO] training model...")
    history = model.fit(
        [trainTextX_pe, trainTextX_re, trainAudioX_pe, trainAudioX_re], trainY,
        validation_data=([valTextX_pe, valTextX_re, valAudioX_pe, valAudioX_re], valY),
        epochs=5, batch_size=16, verbose=0, callbacks=[es, mc])

    # evaluate the model
    _, train_acc = model.evaluate([trainTextX_pe, trainTextX_re, trainAudioX_pe, trainAudioX_re], trainY, verbose=0)
    _, val_acc = model.evaluate([valTextX_pe, valTextX_re, valAudioX_pe, valAudioX_re], valY, verbose=0)
    _, test_acc = model.evaluate([testTextX_pe, testTextX_re, testAudioX_pe, testAudioX_re], testY, verbose=0)

    print('Train: %.3f, Validation: %.3f, Test: %.3f' % (train_acc, val_acc, test_acc))


def info_text_audio():
    combinedInput = concatenate([model_info.output, model_text.output, model_audio.output])

    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(4, activation='relu')(combinedInput)
    x = Dense(1, activation='sigmoid')(x)

    # build model
    model = Model(inputs=[model_info.input, model_text.input[0], model_text.input[1], model_audio.input[0], model_audio.input[1]], outputs=x)

    # compile the model using mean absolute percentage error as our loss
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print("[INFO] model summary...")
    print(model.summary())

    # simple early stopping
    best_model = '/home/wenting/PycharmProjects/thesis/model/mixed_model_justice/best_model.h5'
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(best_model, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    # fit model
    print("[INFO] training model...")
    history = model.fit(
        [trainAttrX, trainTextX_pe, trainTextX_re, trainAudioX_pe, trainAudioX_re], trainY,
        validation_data=([valAttrX, valTextX_pe, valTextX_re, valAudioX_pe, valAudioX_re], valY),
        epochs=5, batch_size=16, verbose=0, callbacks=[es, mc])

    # evaluate the model
    _, train_acc = model.evaluate([trainAttrX, trainTextX_pe, trainTextX_re, trainAudioX_pe, trainAudioX_re], trainY, verbose=0)
    _, val_acc = model.evaluate([valAttrX, valTextX_pe, valTextX_re, valAudioX_pe, valAudioX_re], valY, verbose=0)
    _, test_acc = model.evaluate([testAttrX, testTextX_pe, testTextX_re, testAudioX_pe, testAudioX_re], testY, verbose=0)

    print('Train: %.3f, Validation: %.3f, Test: %.3f' % (train_acc, val_acc, test_acc))

info_text_audio()
