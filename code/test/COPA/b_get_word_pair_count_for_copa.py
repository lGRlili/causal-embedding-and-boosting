import numpy as np
import pickle
from collections import defaultdict
from datetime import datetime

starts = datetime.now()


def print_time():
    ends = datetime.now()
    detla_time = ends - starts
    seconds = detla_time.seconds
    second = seconds % 60
    minitus = int(seconds / 60)
    minitus = minitus % 60
    hours = int(seconds / 3600)
    print("已经经过:", hours, "H ", minitus, "M ", second, "S")


def get_pairandword():
    cause_effect_pair_path = 'cause_effect_pair.npy'
    cause_effect_pairs = np.load(cause_effect_pair_path, allow_pickle=True)
    cause_effect_pairs = set(list(cause_effect_pairs))
    word_lists = set()
    print(len(cause_effect_pairs))
    for pair in cause_effect_pairs:
        word1, word2 = pair.split('_')
        word_lists.add(word1)
        word_lists.add(word2)
    return cause_effect_pairs, word_lists


def cal_count():
    with open(path_fir_word_count, 'r') as file:
        while True:
            word_count = file.readline()
            if word_count == '':
                break
            temp = word_count.split(' ')
            # print(temp)
            word, count = temp[0], float(temp[1])
            word = word.lower()
            if word in word_list:
                fir_word_count[word] += count
            fir_word_count['total_word_count'] += count
    with open(path_las_word_count, 'r') as file:
        while True:
            word_count = file.readline()
            if word_count == '':
                break
            temp = word_count.split(' ')
            # print(temp)
            word, count = temp[0], float(temp[1])
            word = word.lower()
            if word in word_list:
                las_word_count[word] += count
            las_word_count['total_word_count'] += count

    with open(path_total_word_count, 'r') as file:
        while True:
            word_count = file.readline()
            if word_count == '':
                break
            temp = word_count.split(' ')
            word, count = temp[0], float(temp[1])
            word = word.lower()
            if word in word_list:
                total_word_count[word] += count
            total_word_count['total_word_count'] += count

    id_number = 0
    with open(path_pair_count, 'r') as file:
        while True:
            id_number += 1
            word_count = file.readline()
            if word_count == '':
                break
            temp = word_count.split(' ')
            # print(temp)
            cause, effect, count = temp[0], temp[1], float(temp[2])
            cause_effect = cause + '_' + effect
            if id_number % 1000000 == 0:
                print(id_number, cause, effect, count)
            if cause_effect in cause_effect_pair:
                pair_count[cause_effect] += count
            pair_count['total_word_count'] += count
    print(id_number)


if __name__ == '__main__':
    total_count = defaultdict(float)

    pair_count = defaultdict(float)
    fir_word_count = defaultdict(float)
    las_word_count = defaultdict(float)
    total_word_count = defaultdict(float)
    cause_effect_pair, word_list = get_pairandword()

    data_path = '../../../data/'
    # data_path = '../../../../data_disk/ray/data/'
    evidence_choose = 0
    evidence_kinds = ['advcl', 'conj', 'inter']
    evidence = evidence_kinds[evidence_choose]
    data_choose = 1

    input_file = 'data_pair_count/'
    data_list = ['icw/', 'bok/', 'gut/']
    input_path = data_path + input_file + data_list[data_choose]
    print(evidence, data_list[data_choose])
    output_pair_count = 'data/' + data_list[data_choose] + evidence + '_output_pair_count.txt'
    output_fir_word_count = 'data/' + data_list[data_choose] + evidence + '_output_fir_word_count.txt'
    output_las_word_count = 'data/' + data_list[data_choose] + evidence + '_output_las_word_count.txt'
    output_total_word_count = 'data/' + data_list[data_choose] + evidence + '_output_total_word_count.txt'

    path_pair_count = input_path + evidence + '_output_pair_count.txt'
    path_fir_word_count = input_path + evidence + '_output_fir_word_count.txt'
    path_las_word_count = input_path + evidence + '_output_las_word_count.txt'
    path_total_word_count = input_path + evidence + '_output_total_word_count.txt'

    if evidence == 'inter':
        for i in range(5):
            path_pair_count = input_path + evidence + '_output_pair_count_' + str(i) + '.txt'
            path_fir_word_count = input_path + evidence + '_output_fir_word_count_' + str(i) + '.txt'
            path_las_word_count = input_path + evidence + '_output_las_word_count_' + str(i) + '.txt'
            path_total_word_count = input_path + evidence + '_output_total_word_count_' + str(i) + '.txt'
            print(path_pair_count)
            cal_count()
            print(len(pair_count), len(fir_word_count), len(las_word_count), len(total_word_count))
    else:
        cal_count()

    print(total_count)
    print_time()

    with open(output_pair_count, 'wb') as file:
        pickle.dump(pair_count, file)
    with open(output_fir_word_count, 'wb') as file:
        pickle.dump(fir_word_count, file)
    with open(output_las_word_count, 'wb') as file:
        pickle.dump(las_word_count, file)
    with open(output_total_word_count, 'wb') as file:
        pickle.dump(total_word_count, file)
    print('finish')
