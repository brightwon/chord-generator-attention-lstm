from keras_preprocessing import sequence
from keras.layers import *
from keras.models import Model
import numpy as np
import time
import os


def get_model(seq_length, input_dim, output_dim, units):
    """Create the neural net."""
    # make the keras tensor
    _input = Input(shape=(seq_length, input_dim), dtype='float32')
    input_layer = TimeDistributed(Dense(input_dim))(_input)
    activations = LSTM(units, return_sequences=True)(input_layer)

    # compute importance for each step
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(units)(attention)
    attention = Permute([2, 1])(attention)

    # apply the attention
    sent_representation = multiply([activations, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

    probabilities = Dense(output_dim, activation='softmax')(sent_representation)

    model = Model(inputs=_input, outputs=probabilities)
    return model


def train():
    # zero padding
    input_vec = sequence.pad_sequences(np.load('dataset/input_vector.npy'))
    target_vec = np.load('dataset/target_vector.npy')

    input_dim = input_vec.shape[2]
    output_dim = target_vec.shape[1]
    input_sequence_length = input_vec.shape[1]

    # hyper params
    num_epochs = 1
    batch_size = 512
    units = 128

    # get model
    model = get_model(input_sequence_length, input_dim, output_dim, units)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(input_vec, target_vec, batch_size=batch_size, epochs=num_epochs)

    # save weights file
    weights_dir = 'model_weights/'
    if not os.path.isdir(weights_dir):
        os.mkdir(weights_dir)
    weights_file = '%sepochs_%s' % (num_epochs, time.strftime("%Y%m%d_%H_%M.h5"))
    weights_path = '%s%s' % (weights_dir, weights_file)
    model.save_weights(weights_path)

    # save model
    json_string = model.to_json()
    model_dir = 'model_json/'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    model_file = '%sepochs_%s' % (num_epochs, time.strftime("%Y%m%d_%H_%M.json"))
    model_path = '%s%s' % (model_dir, model_file)
    open(model_path, 'w').write(json_string)

    print("Done!")


if __name__ == '__main__':
    train()
