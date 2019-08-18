import sys

sys.path.append('/home/wenting/PycharmProjects/thesis/code')
import os
import th_13_mix_data
import th_12_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# embedding settings
MAX_NUM_WORDS = 20000
MAX_LENGTH = 400  # change
EMBEDDING_DIM = 100
BASE_DIR = '/home/wenting/PycharmProjects/thesis/'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')  # glove.6B

# load the saved model
# from keras.models import load_model
# saved_model = load_model('best_model.h5')

"""decision fusion"""
# info_file = '/home/wenting/PycharmProjects/thesis/data/mixed_data/caseinfo_original.csv'



# load data
info_file = '/home/wenting/PycharmProjects/thesis/data/mixed_data/caseinfo_filtered.csv'
arguments_file_dir = '/home/wenting/PycharmProjects/thesis/data/mixed_data/text_data_filtered'
audio_file = '/home/wenting/PycharmProjects/thesis/data/mixed_data/audio_filtered.csv'

print("[INFO] loading cases attributes csv...")
info_processed = th_13_mix_data.load_structured_data(info_file)

print("[INFO] loading audio pitch...")
audio = th_13_mix_data.load_audio_data(audio_file)
info_and_audio = pd.concat([info_processed, audio], axis=1)

print("[INFO] loading oral arguments text...")
oral_data = th_13_mix_data.load_arguments_text(info_file, arguments_file_dir, MAX_NUM_WORDS, MAX_LENGTH)
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
# train, val, test: 0.64, 0.16, 0.2
print("[INFO] processing data...")
trainAttrX, testAttrX, trainTextX, testTextX = train_test_split(info_and_audio, texts_pad, test_size=0.2,
                                                                random_state=1)
trainAttrX, valAttrX, trainTextX, valTextX = train_test_split(trainAttrX, trainTextX, test_size=0.2, random_state=1)

# text
trainTextX_pe = trainTextX[:, :, 0]
trainTextX_re = trainTextX[:, :, 1]
testTextX_pe = testTextX[:, :, 0]
testTextX_re = testTextX[:, :, 1]
valTextX_pe = valTextX[:, :, 0]
valTextX_re = valTextX[:, :, 1]

# target
trainY = trainAttrX['partyWinning'].astype(int)
testY = testAttrX['partyWinning'].astype(int)
valY = valAttrX['partyWinning'].astype(int)

# audio
trainAudioX_pe = trainAttrX['petitioner_pitch']
trainAudioX_re = trainAttrX['respondent_pitch']
testAudioX_pe = testAttrX['petitioner_pitch']
testAudioX_re = testAttrX['respondent_pitch']
valAudioX_pe = valAttrX['petitioner_pitch']
valAudioX_re = valAttrX['respondent_pitch']

# structured data
trainAttrX = trainAttrX.drop(['partyWinning', 'petitioner_pitch', 'respondent_pitch'], axis=1)
testAttrX = testAttrX.drop(['partyWinning', 'petitioner_pitch', 'respondent_pitch'], axis=1)
valAttrX = valAttrX.drop(['partyWinning', 'petitioner_pitch', 'respondent_pitch'], axis=1)


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


def run_info():
    print("[INFO] building model...")
    mlp = th_12_model.create_mlp(trainAttrX.shape[1], regress=True)
    model = Model(inputs=mlp.input, outputs=mlp.output)

    # compile the model using mean absolute percentage error as our loss
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print("[INFO] model summary...")
    print(model.summary())

    # simple early stopping
    best_model = '/home/wenting/PycharmProjects/thesis/model/mixed_model/best_model_info.h5'
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(best_model, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    # fit model
    print("[INFO] training model...")
    history = model.fit(
        trainAttrX, trainY,
        validation_data=(valAttrX, valY),
        epochs=5, batch_size=16, verbose=0, callbacks=[es, mc])

    # evaluate the model
    _, train_acc = model.evaluate(trainAttrX, trainY, verbose=0)
    _, val_acc = model.evaluate(valAttrX, valY, verbose=0)
    _, test_acc = model.evaluate(testAttrX, testY, verbose=0)

    print('Train: %.3f, Validation: %.3f, Test: %.3f' % (train_acc, val_acc, test_acc))


def run_text():
    # create the MLP and CNN models
    print("[INFO] building model...")
    cnn_pe = th_12_model.create_cnn(num_words=num_words, max_length=MAX_LENGTH,
                                    embedding_dim=EMBEDDING_DIM, embedding_matrix=embedding_matrix, regress=False)
    cnn_re = th_12_model.create_cnn(num_words=num_words, max_length=MAX_LENGTH,
                                    embedding_dim=EMBEDDING_DIM, embedding_matrix=embedding_matrix, regress=False)

    # create the input to our final set of layers as the *output* of both
    # the MLP and CNN
    combinedInput = concatenate([cnn_pe.output, cnn_re.output])

    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(4, activation='relu')(combinedInput)
    x = Dense(1, activation='sigmoid')(x)

    # build model
    model = Model(inputs=[cnn_pe.input, cnn_re.input], outputs=x)

    # compile the model using mean absolute percentage error as our loss
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print("[INFO] model summary...")
    print(model.summary())

    # simple early stopping
    best_model = '/home/wenting/PycharmProjects/thesis/model/mixed_model/best_model_text.h5'
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(best_model, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    # fit model
    print("[INFO] training model...")
    history = model.fit(
        [trainTextX_pe, trainTextX_re], trainY,
        validation_data=([valTextX_pe, valTextX_re], valY),
        epochs=5, batch_size=16, verbose=0, callbacks=[es, mc])

    # evaluate the model
    _, train_acc = model.evaluate([trainTextX_pe, trainTextX_re], trainY, verbose=0)
    _, val_acc = model.evaluate([valTextX_pe, valTextX_re], valY, verbose=0)
    _, test_acc = model.evaluate([testTextX_pe, testTextX_re], testY, verbose=0)

    print('Train: %.3f, Validation: %.3f, Test: %.3f' % (train_acc, val_acc, test_acc))


def run_audio():
    # create the MLP
    print("[INFO] building model...")
    mlp_pe = th_12_model.create_mlp_audio(1, regress=False)
    mlp_re = th_12_model.create_mlp_audio(1, regress=False)

    combinedInput = concatenate([mlp_pe.output, mlp_re.output])

    x = Dense(2, activation='relu')(combinedInput)
    x = Dense(1, activation='sigmoid')(x)

    # build model
    model = Model(inputs=[mlp_pe.input, mlp_re.input], outputs=x)

    # compile the model using mean absolute percentage error as our loss
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print("[INFO] model summary...")
    print(model.summary())

    # simple early stopping
    best_model = '/home/wenting/PycharmProjects/thesis/model/mixed_model/best_model_audio.h5'
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint(best_model, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    # fit model
    print("[INFO] training model...")
    history = model.fit(
        [trainAudioX_pe, trainAudioX_re], trainY,
        validation_data=([valAudioX_pe, valAudioX_re], valY),
        epochs=5, batch_size=16, verbose=0, callbacks=[es, mc])

    # evaluate the model
    _, train_acc = model.evaluate([trainAudioX_pe, trainAudioX_re], trainY, verbose=0)
    _, val_acc = model.evaluate([valAudioX_pe, valAudioX_re], valY, verbose=0)
    _, test_acc = model.evaluate([testAudioX_pe, testAudioX_re], testY, verbose=0)

    print('Train: %.3f, Validation: %.3f, Test: %.3f' % (train_acc, val_acc, test_acc))


def run_info_text():
    # create the MLP and CNN models
    print("[INFO] building model...")
    mlp = th_12_model.create_mlp(trainAttrX.shape[1], regress=False)
    cnn_pe = th_12_model.create_cnn(num_words=num_words, max_length=MAX_LENGTH,
                                    embedding_dim=EMBEDDING_DIM, embedding_matrix=embedding_matrix, regress=False)
    cnn_re = th_12_model.create_cnn(num_words=num_words, max_length=MAX_LENGTH,
                                    embedding_dim=EMBEDDING_DIM, embedding_matrix=embedding_matrix, regress=False)

    # create the input to our final set of layers as the *output* of both
    # the MLP and CNN
    combinedInput = concatenate([mlp.output, cnn_pe.output, cnn_re.output])

    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(4, activation='relu')(combinedInput)
    x = Dense(1, activation='sigmoid')(x)

    # build model
    model = Model(inputs=[mlp.input, cnn_pe.input, cnn_re.input], outputs=x)

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
        [trainAttrX, trainTextX_pe, trainTextX_re], trainY,
        validation_data=([valAttrX, valTextX_pe, valTextX_re], valY),
        epochs=5, batch_size=16, verbose=0, callbacks=[es, mc])

    # evaluate the model
    _, train_acc = model.evaluate([trainAttrX, trainTextX_pe, trainTextX_re], trainY, verbose=0)
    _, val_acc = model.evaluate([valAttrX, valTextX_pe, valTextX_re], valY, verbose=0)
    _, test_acc = model.evaluate([testAttrX, testTextX_pe, testTextX_re], testY, verbose=0)

    print('Train: %.3f, Validation: %.3f, Test: %.3f' % (train_acc, val_acc, test_acc))


def run_info_audio():
    print("[INFO] building model...")
    mlp = th_12_model.create_mlp(trainAttrX.shape[1], regress=False)
    mlp_pe = th_12_model.create_mlp_audio(1, regress=False)
    mlp_re = th_12_model.create_mlp_audio(1, regress=False)

    combinedInput = concatenate([mlp.output, mlp_pe.output, mlp_re.output])

    x = Dense(4, activation='relu')(combinedInput)
    x = Dense(1, activation='sigmoid')(x)

    # build model
    model = Model(inputs=[mlp.input, mlp_pe.input, mlp_re.input], outputs=x)

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


def run_text_audio():
    # create the MLP and CNN models
    print("[INFO] building model...")
    cnn_pe = th_12_model.create_cnn(num_words=num_words, max_length=MAX_LENGTH,
                                    embedding_dim=EMBEDDING_DIM, embedding_matrix=embedding_matrix, regress=False)
    cnn_re = th_12_model.create_cnn(num_words=num_words, max_length=MAX_LENGTH,
                                    embedding_dim=EMBEDDING_DIM, embedding_matrix=embedding_matrix, regress=False)
    mlp_pe = th_12_model.create_mlp_audio(1, regress=False)
    mlp_re = th_12_model.create_mlp_audio(1, regress=False)

    # create the input to our final set of layers as the *output* of both
    # the MLP and CNN
    combinedInput = concatenate([cnn_pe.output, cnn_re.output, mlp_pe.output, mlp_re.output])

    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(4, activation='relu')(combinedInput)
    x = Dense(1, activation='sigmoid')(x)

    # build model
    model = Model(inputs=[cnn_pe.input, cnn_re.input, mlp_pe.input, mlp_re.input], outputs=x)

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


def run_info_text_audio():
    # create the MLP and CNN models
    print("[INFO] building model...")
    mlp = th_12_model.create_mlp(trainAttrX.shape[1], regress=False)
    cnn_pe = th_12_model.create_cnn(num_words=num_words, max_length=MAX_LENGTH,
                                            embedding_dim=EMBEDDING_DIM, embedding_matrix=embedding_matrix,
                                            regress=False)
    cnn_re = th_12_model.create_cnn(num_words=num_words, max_length=MAX_LENGTH,
                                            embedding_dim=EMBEDDING_DIM, embedding_matrix=embedding_matrix,
                                            regress=False)
    mlp_pe = th_12_model.create_mlp_audio(1, regress=False)
    mlp_re = th_12_model.create_mlp_audio(1, regress=False)

    # create the input to our final set of layers as the *output* of both
    # the MLP and CNN
    combinedInput = concatenate([mlp.output, cnn_pe.output, cnn_re.output, mlp_pe.output, mlp_re.output])

    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(4, activation='relu')(combinedInput)
    x = Dense(1, activation='sigmoid')(x)

    # build model
    model = Model(inputs=[mlp.input, cnn_pe.input, cnn_re.input, mlp_pe.input, mlp_re.input], outputs=x)

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
    _, train_acc = model.evaluate([trainAttrX, trainTextX_pe, trainTextX_re, trainAudioX_pe, trainAudioX_re], trainY,
                                  verbose=0)
    _, val_acc = model.evaluate([valAttrX, valTextX_pe, valTextX_re, valAudioX_pe, valAudioX_re], valY, verbose=0)
    _, test_acc = model.evaluate([testAttrX, testTextX_pe, testTextX_re, testAudioX_pe, testAudioX_re], testY,
                                 verbose=0)

    print('Train: %.3f, Validation: %.3f, Test: %.3f' % (train_acc, val_acc, test_acc))

print(texts_pad.shape)

run_audio()


