from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM, GRU, Input, Bidirectional, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.initializers import Constant
from keras.models import Model
from keras import initializers
from keras.engine.topology import Layer, InputSpec
from keras import backend as K


def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(128, input_dim=dim, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))

    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="sigmoid"))

    # return our model
    return model


def create_cnn(num_words, max_length, embedding_dim, embedding_matrix, regress=False):
    # load pre-trained embedding
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_length,
                                trainable=False)

    sequence_input = Input(shape=(max_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(embedded_sequences)
    x = MaxPooling1D(pool_size=5)(x)
    x = GRU(32, dropout=0.1, recurrent_dropout=0.5)(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="sigmoid")(x)

    model = Model(sequence_input, x)
    # return the CNN
    return model

