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


def compute_output_shape(input_shape):
    return input_shape[0], input_shape[-1]


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output


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


def create_attention(num_words, max_length, embedding_dim, embedding_matrix, regress=False):
    # load pre-trained embedding
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_length,
                                trainable=False)

    sequence_input = Input(shape=(max_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    x = TimeDistributed(Dense(200))(x)
    x = AttLayer()(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="sigmoid")(x)

    model = Model(sequence_input, x)
    # return the CNN
    return model
