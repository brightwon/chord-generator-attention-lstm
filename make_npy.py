import glob
import csv
import numpy as np


def one_hot_encoding(length, one_index):
    """Return the one hot vector."""
    vectors = [0] * length
    vectors[one_index] = 1
    return vectors


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
        exit()
    csv_files = glob.glob(file_path)
    note_dict_len = len(note_dictionary)
    chord_dict_len = len(chord_dictionary)

    result_input_matrix = []
    result_target_matrix = []

    # make the matrix from csv data.
    for csv_path in csv_files:
        csv_ins = open(csv_path, 'r', encoding='utf-8')
        next(csv_ins)  # skip first line
        reader = csv.reader(csv_ins)

        note_sequence = []
        pre_measure = None
        for line in reader:
            measure = int(line[0])
            chord = line[1]
            note = line[2]

            # find one hot index
            chord_index = chord_dictionary.index(chord)
            note_index = note_dictionary.index(note)

            if pre_measure is None:  # case : first line
                note_sequence.append(one_hot_encoding(note_dict_len, note_index))
                result_target_matrix.append(one_hot_encoding(chord_dict_len, chord_index))

            elif pre_measure == measure:  # case : same measure note
                note_sequence.append(one_hot_encoding(note_dict_len, note_index))

            else:  # case : next measure note
                result_input_matrix.append(note_sequence)
                note_sequence = [one_hot_encoding(note_dict_len, note_index)]
                result_target_matrix.append(one_hot_encoding(chord_dict_len, chord_index))
            pre_measure = measure
        result_input_matrix.append(note_sequence)  # case : last measure note

    if _input == '1':
        np.save('dataset/input_vector.npy', np.array(result_input_matrix))
        np.save('dataset/target_vector.npy', np.array(result_target_matrix))
    elif _input == '2':
        np.save('dataset/test_vector.npy', np.array(result_input_matrix))


if __name__ == '__main__':
    main()
