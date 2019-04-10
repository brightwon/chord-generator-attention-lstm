import glob
import ntpath
import numpy as np
from os import listdir
from keras.models import model_from_json
from keras_preprocessing import sequence


def load_model():
    # load model file
    model_dir = 'model_json/'
    model_files = listdir(model_dir)
    for i, file in enumerate(model_files):
        print(str(i) + " : " + file)
    file_number_model = int(input('Choose the model:'))
    model_file = model_files[file_number_model]
    model_path = '%s%s' % (model_dir, model_file)

    # load weights file
    weights_dir = 'model_weights/'
    weights_files = listdir(weights_dir)
    for i, file in enumerate(weights_files):
        print(str(i) + " : " + file)
    file_number_weights = int(input('Choose the weights:'))
    weights_file = weights_files[file_number_weights]
    weights_path = '%s%s' % (weights_dir, weights_file)

    # load the model
    model = model_from_json(open(model_path).read())
    model.load_weights(weights_path)

    return model


def predict():
    chord_dictionary = ['C:maj', 'C:min',
                        'C#:maj', 'C#:min',
                        'D:maj', 'D:min',
                        'D#:maj', 'D#:min',
                        'E:maj', 'E:min',
                        'F:maj', 'F:min',
                        'F#:maj', 'F#:min',
                        'G:maj', 'G:min',
                        'G#:maj', 'G#:min',
                        'A:maj', 'A:min',
                        'A#:maj', 'A#:min',
                        'B:maj', 'B:min']

    model = load_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    file_path = 'dataset/test_npy/*.npy'
    npy_files = glob.glob(file_path)
    for song in npy_files:
        note_sequence = sequence.pad_sequences(np.load(song), maxlen=32)

        # predict
        prediction_list = []
        net_output = model.predict(note_sequence)
        for chord_index in net_output.argmax(axis=1):
            prediction_list.append(chord_dictionary[chord_index])

        print(ntpath.basename(song), prediction_list)


if __name__ == '__main__':
    predict()
