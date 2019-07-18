from __future__ import print_function
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32

# print('Loading data...')
# max_features = 20000
# # save np.load
# np_load_old = np.load
# # modify the default parameters of np.load
# np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
# # call load_data with allow_pickle implicitly set to true
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# # restore np.load for future normal usage
# np.load = np_load_old




# try a small sample
project_path = '/home/wenting/PycharmProjects/thesis/'
dataset = pd.read_csv(project_path + 'test/case_centered_dataset.csv')
data = dataset[['partyWinning', 'petitioner', 'respondent']]
data.partyWinning = data.partyWinning.astype(int)
sample = data.sample(n=1000, random_state=1)


sample = sample.dropna()
X = sample.petitioner
y = sample.partyWinning



from sklearn.model_selection import train_test_split
X_experiment, X_test, y_experiment, y_test = train_test_split(X, y, train_size=0.90)

from sklearn.feature_extraction.text import CountVectorizer

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(X_experiment)
X_experiment = ngram_vectorizer.transform(X_experiment)
X_test = ngram_vectorizer.transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_experiment, y_experiment, train_size=0.70)



print(len(X_train), 'train sequences')
print(len(X_val), 'val sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)