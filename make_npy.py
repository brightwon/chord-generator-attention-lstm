import glob
import csv
import os
import ntpath
import numpy as np


def one_hot_encoding(length, one_index):
    """Return the one hot vector."""
    vectors = [0] * length
    vectors[one_index] = 1
    return vectors


def make_test_npys(file_name, song_sequence):
    """Create npy file for each song in the test set."""
    file_path = "dataset/test_npy"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    np.save('%s/%s.npy' % (file_path, file_name.split('.')[0]), np.array(song_sequence))


def main():
    np.set_printoptions(threshold=np.inf)
    note_dictionary = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
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

    print("1. Train set\n2. Test set")
    _input = input('Choose dataset to make npy file :')
    if _input == '1':
        file_path = 'dataset/new_train/*.csv'
    elif _input == '2':
        file_path = 'dataset/new_test/*.csv'
    else:
        print("input error")
        return None

    csv_files = glob.glob(file_path)
    note_dict_len = len(note_dictionary)
    chord_dict_len = len(chord_dictionary)

    # list for final input/target vector
    result_input_matrix = []
    result_target_matrix = []

    # make the matrix from csv data
    for csv_path in csv_files:
        csv_ins = open(csv_path, 'r', encoding='utf-8')
        next(csv_ins)  # skip first line
        reader = csv.reader(csv_ins)

        note_sequence = []
        song_sequence = []  # list for each song(each npy file) in the test set
        pre_measure = None
        for line in reader:
            measure = int(line[0])
            chord = line[1]
            note = line[2]

            # find one hot index
            chord_index = chord_dictionary.index(chord)
            note_index = note_dictionary.index(note)

            one_hot_note_vec = one_hot_encoding(note_dict_len, note_index)
            one_hot_chord_vec = one_hot_encoding(chord_dict_len, chord_index)

            if pre_measure is None:  # case : first line
                note_sequence.append(one_hot_note_vec)
                result_target_matrix.append(one_hot_chord_vec)

            elif pre_measure == measure:  # case : same measure note
                note_sequence.append(one_hot_note_vec)

            else:  # case : next measure note
                song_sequence.append(note_sequence)
                result_input_matrix.append(note_sequence)
                note_sequence = [one_hot_note_vec]
                result_target_matrix.append(one_hot_chord_vec)
            pre_measure = measure
        result_input_matrix.append(note_sequence)  # case : last measure note

        if _input == '2':
            # make npy file for each song
            make_test_npys(ntpath.basename(csv_path), song_sequence)

    if _input == '1':
        np.save('dataset/input_vector.npy', np.array(result_input_matrix))
        np.save('dataset/target_vector.npy', np.array(result_target_matrix))
    elif _input == '2':
        np.save('dataset/test_vector.npy', np.array(result_input_matrix))


if __name__ == '__main__':
    main()
