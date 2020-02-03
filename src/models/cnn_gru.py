from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, LSTM, Bidirectional
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD, RMSprop
from utills.util import ctc_lambda_func

class CnnGRU():
    def __init__(self, conv_filters, kernel_size, pool_size, img_w, img_h, time_dense_size, rnn_size, model_path):
        self.conv_filters =conv_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.input_shape = (img_w, img_h, 1)
        self.img_w = img_w
        self.img_h = img_h
        self.time_dense_size = time_dense_size
        self.rnn_size = rnn_size
        self.model_path = model_path

    def setup(self):
        act = 'relu'
        input_data = Input(name='the_input', shape=self.input_shape, dtype='float32')
        inner = Conv2D(self.conv_filters, self.kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv1')(input_data)
        inner = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='max1')(inner)
        inner = Conv2D(self.conv_filters, self.kernel_size, padding='same',
                       activation=act, kernel_initializer='he_normal',
                       name='conv2')(inner)
        inner = MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name='max2')(inner)

        conv_to_rnn_dims = (self.img_w // (self.pool_size ** 2),
                            (self.img_h // (self.pool_size ** 2)) * self.conv_filters)
        inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

        # cuts down input size going into RNN:
        inner = Dense(self.time_dense_size, activation=act, name='dense1')(inner)

        # Two layers of bidirectional GRUs
        # GRU seems to work as well, if not better than LSTM:
        lstm_1 = LSTM(self.rnn_size, return_sequences=True,
                      kernel_initializer='he_normal', name='lstm1')(inner)
        lstm_1b = LSTM(self.rnn_size, return_sequences=True,
                       go_backwards=True, kernel_initializer='he_normal',
                       name='lstm1_b')(inner)
        lstm1_merged = add([lstm_1, lstm_1b])
        lstm_2 = LSTM(self.rnn_size, return_sequences=True,
                      kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
        lstm_2b = LSTM(self.rnn_size, return_sequences=True, go_backwards=True,
                       kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)

        # transforms RNN output to character activations:
        inner = Dense(66, kernel_initializer='he_normal',
                      name='dense2')(concatenate([lstm_2, lstm_2b]))
        y_pred = Activation('softmax', name='softmax')(inner)
        # Model(inputs=input_data, outputs=y_pred).summary()

        labels = Input(name='the_labels', shape=[37], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(
            ctc_lambda_func, output_shape=(1,),
            name='ctc')([y_pred, labels, input_length, label_length])

        # clipnorm seems to speeds up convergence
        # sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        rms_prop = RMSprop(lr=0.0001)
        model = Model(inputs=[input_data, labels, input_length, label_length],
                      outputs=loss_out)
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=rms_prop)
        model.load_weights(self.model_path)
        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])

        model_p = Model(inputs=input_data, outputs=y_pred)
        return model_p

