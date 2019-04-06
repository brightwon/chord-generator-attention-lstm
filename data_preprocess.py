import glob
import ntpath
import csv
import os
from pandas import DataFrame


def convert_chord_type(chord):
    """Change all the chord to the major or minor chord."""
    return {'maj': 'maj',
            'major': 'maj',
            'major-sixth': 'maj',
            'major-seventh': 'maj',
            'maj7': 'maj',
            'major-ninth': 'maj',
            'maj69': 'maj',
            'maj9': 'maj',
            'major-minor': 'maj',
            'minor': 'min',
            'min': 'min',
            'minor-sixth': 'min',
            'minor-seventh': 'min',
            'min7': 'min',
            'minor-ninth': 'min',
            'minor-11th': 'min',
            'minor-13th': 'min',
            'minor-major': 'min',
            'minMaj7': 'min',
            '6': 'maj',
            '7': 'maj',
            '9': 'maj',
            'dominant': 'maj',
            'dominant-seventh': 'maj',
            'dominant-ninth': 'maj',
            'dominant-11th': 'maj',
            'dominant-13th': 'maj',
            'augmented': 'maj',
            'aug': 'maj',
            'augmented-seventh': 'maj',
            'augmented-ninth': 'maj',
            'dim': 'min',
            'diminished': 'min',
            'diminished-seventh': 'min',
            'half-diminished': 'min',
            'm7b5': 'min',
            'dim7': 'min',
            ' dim7': 'min',
            'suspended-second': 'maj',
            'suspended-fourth': 'maj',
            'sus47': 'maj',
            'power': 'maj'}.get(chord, 'nan')


def index_to_scale(index):
    """Convert the integer to scale."""
    return {0: 'C0',
            1: 'Db',
            2: 'D0',
            3: 'Eb',
            4: 'E0',
            5: 'F0',
            6: 'Gb',
            7: 'G0',
            8: 'Ab',
            9: 'A0',
            10: 'Bb',
            11: 'B0'}.get(index, 'nan')


def scale_to_index(root):
    """Convert the scale to integer."""
    return {'B#': 0, 'C0': 0,
            'Db': 1, 'C#': 1,
            'D0': 2,
            'Eb': 3, 'D#': 3,
            'E0': 4, 'Fb': 4,
            'F0': 5, 'E#': 5,
            'Gb': 6, 'F#': 6,
            'G0': 7,
            'Ab': 8, 'G#': 8,
            'A0': 9,
            'Bb': 10, 'A#': 10,
            'Cb': 11, 'B0': 11}.get(root, 'nan')


def transpose(root, key, root_dict):
    """Transpose the key of the chords."""
    # get interval to transpose
    calculate_num = trans_calculator(key)

    # find the index of chord root dictionary
    root_index = scale_to_index(root)  # result : 0 ~ 11

    # transpose
    transposed_index = root_index + calculate_num
    if transposed_index >= 12:
        transposed_index = transposed_index - 12

    return root_dict[transposed_index]


def trans_calculator(key):
    """Calculate the interval to transpose."""
    return {'-5': -1,
            '-4': 4,
            '-3': -3,
            '-2': 2,
            '-1': -5,
            '1': 5,
            '2': -2,
            '3': 3,
            '4': -4,
            '5': 1}.get(key, 'nan')


def main():
    root_dictionary_all = ['C0', 'Db', 'D0', 'Eb', 'E0', 'F0', 'Gb', 'G0', 'Ab', 'A0', 'Bb', 'B0',
                           'B#', 'C#', 'D#', 'Fb', 'E#', 'F#', 'G#', 'A#', 'Cb']

    root_dictionary = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    print("1. Train set\n2. Test set")
    _input = input('Choose dataset to preprocess :')
    if _input == '1':
        file_path = 'dataset/new_train/*.csv'
        new_csv_path = 'dataset/new_train/'
    elif _input == '2':
        file_path = 'dataset/new_test/*.csv'
        new_csv_path = 'dataset/new_test/'
    else:
        print("input error")
        exit()

    csv_files = glob.glob(file_path)

    for i, csv_path in enumerate(csv_files):
        file_name = ntpath.basename(csv_path)

        csv_ins = open(csv_path, 'r', encoding='utf-8')
        next(csv_ins)  # skip first line
        reader = csv.reader(csv_ins)

        measure_list = []
        chord_list = []
        note_list = []

        # get all rows from csv file
        for line in reader:
            measure = line[1]
            key_fifths = line[2]
            chord_root = line[4]
            chord_type = line[5]
            note_root = line[6]

            # change the chord type to major or minor
            result_chord_type = convert_chord_type(chord_type)

            # ignore unknown root and chord type
            if note_root not in root_dictionary_all or chord_root not in root_dictionary_all or result_chord_type == 'nan':
                continue

            # if the key is C key
            if key_fifths == '0':
                chord_root_index = scale_to_index(chord_root)
                note_root_index = scale_to_index(note_root)

                result_chord = root_dictionary[chord_root_index]
                result_note = root_dictionary[note_root_index]

            # if the key is not C key
            else:
                result_chord = transpose(chord_root, key_fifths, root_dictionary)
                result_note = transpose(note_root, key_fifths, root_dictionary)

            measure_list.append(measure)
            chord_list.append(result_chord + ':' + result_chord_type)
            note_list.append(result_note)

        # make the new csv file
        collected_data = {'measure': measure_list,
                          'chord': chord_list,
                          'note': note_list}
        df = DataFrame(collected_data)

        if not os.path.isdir(new_csv_path):
            os.mkdir(new_csv_path)
        df.to_csv(new_csv_path + file_name, encoding='utf-8', index=False)


if __name__ == '__main__':
    main()
