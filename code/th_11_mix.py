import sys
sys.path.append('/home/wenting/PycharmProjects/thesis/code')
import os
import th_13_mix_data
import th_12_model
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# embedding settings
MAX_NUM_WORDS = 5000
MAX_LENGTH = 500
EMBEDDING_DIM = 100
BASE_DIR = '/home/wenting/PycharmProjects/thesis/'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')  # glove.6B

# load data
info_file = '/home/wenting/PycharmProjects/thesis/data/mixed_data/caseinfo.csv'
arguments_file_dir = '/home/wenting/PycharmProjects/thesis/data/mixed_data/text_data'

print("[INFO] loading cases attributes csv...")
info_processed = th_13_mix_data.load_structured_data(info_file)

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
print("[INFO] processing data...")
split = train_test_split(info_processed, texts_pad, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainTextX, testTextX) = split

trainTextX_pe = trainTextX[:, :, 0]
trainTextX_re = trainTextX[:, :, 1]
testTextX_pe = testTextX[:, :, 0]
testTextX_re = testTextX[:, :, 1]

# target
trainY = trainAttrX['partyWinning']
testY = testAttrX['partyWinning']

# structured data
trainAttrX = trainAttrX.drop(['partyWinning'], axis=1)
testAttrX = testAttrX.drop(['partyWinning'], axis=1)

# create the MLP and CNN models
print("[INFO] building model...")
mlp = th_12_model.create_mlp(trainAttrX.shape[1], regress=False)
cnn_pe = th_12_model.create_cnn(max_num_words=MAX_NUM_WORDS, max_length=MAX_LENGTH,
                                embedding_dim=EMBEDDING_DIM, embedding_matrix=embedding_matrix, regress=False)
cnn_re = th_12_model.create_cnn(max_num_words=MAX_NUM_WORDS, max_length=MAX_LENGTH,
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
best_model = '/home/wenting/PycharmProjects/thesis/model/mixed_model/best_model.h5'
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint(best_model, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
# fit model
print("[INFO] training model...")
history = model.fit(
    [trainAttrX, trainTextX_pe, trainTextX_re], trainY,
    validation_data=([testAttrX, testTextX_pe, testTextX_re], testY),
    epochs=5, batch_size=16, verbose=0, callbacks=[es, mc])

# evaluate the model
_, train_acc = model.evaluate(trainAttrX, trainTextX_pe, trainTextX_re, verbose=0)
_, test_acc = model.evaluate(testAttrX, testTextX_pe, testTextX_re, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# load the saved model
# from keras.models import load_model
# saved_model = load_model('best_model.h5')
