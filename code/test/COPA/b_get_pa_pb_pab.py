import numpy as np
import pickle
from collections import defaultdict
from datetime import datetime

starts = datetime.now()


def print_tiem():
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
    with open(path_cause_word_count, 'r') as file:
        while True:
            word_count = file.readline()
            if word_count == '':
                break
            temp = word_count.split(' ')
            # print(temp)
            word, count = temp[0], float(temp[1])
            word = word.lower()
            if word in word_list:
                cause_count[word] += count
            total_count['total_cause'] += count

    with open(path_effect_word_count, 'r') as file:
        while True:
            word_count = file.readline()
            if word_count == '':
                break
            temp = word_count.split(' ')
            word, count = temp[0], float(temp[1])
            word = word.lower()
            if word in word_list:
                effect_count[word] += count
            total_count['total_effect'] += count

    print(len(cause_count), len(effect_count))
    id_number = 0
    with open(path_cause_effect_pair_count, 'r') as file:
        while True:
            id_number += 1
            word_count = file.readline()
            if word_count == '':
                break
            temp = word_count.split(' ')
            # print(temp)
            cause, effect, count = temp[0], temp[1], float(temp[2])
            cause_effect = cause + '_' + effect
            print(id_number, cause, effect, count)
            if cause_effect in cause_effect_pair:
                cause_effect_count[cause_effect] += count
            total_count['total_cause_effect'] += count


if __name__ == '__main__':
    total_count = {'total_cause': 0, 'total_effect': 0, 'total_cause_effect': 0}

    cause_effect_count = defaultdict(float)
    cause_count = defaultdict(float)
    effect_count = defaultdict(float)

    data_path = '../../../data'
    data_choose = ['', 'icw/', 'bok/', 'gut/']
    evidence_list = ['', 'cause_', 'time_', 'to_', 'fonction_classify_to']
    out_spec = ['cause_label_', '']
    out_spec_choose = 0
    choose_id = 3
    evidence_id = 1
    path_cause_effect_pair_count = data_path + '/data_pair_count/' + data_choose[choose_id] + evidence_list[evidence_id] +'cause_effect_pair_count.txt'
    path_cause_word_count = data_path + '/data_pair_count/' + data_choose[choose_id] + evidence_list[evidence_id] +'cause_word_count.txt'
    path_effect_word_count = data_path + '/data_pair_count/' + data_choose[choose_id] + evidence_list[evidence_id] +'effect_word_count.txt'
    out_put_cause_effect_pair_count = 'data/' + data_choose[choose_id] + evidence_list[evidence_id] +'count_cause_effect_pair.file'
    out_put_cause_word_count = 'data/' + data_choose[choose_id] + evidence_list[evidence_id] +'count_cause_word.file'
    out_put_effect_word_count = 'data/' + data_choose[choose_id] + evidence_list[evidence_id] +'count_effect_word.file'

    # path_cause_effect_pair_count = data_path + '/to_cause_effect_with_label/' + data_choose[choose_id] + out_spec[out_spec_choose] + 'to_cause_effect_pair_count.txt'
    # path_cause_word_count = data_path + '/to_cause_effect_with_label/' + data_choose[choose_id] + out_spec[out_spec_choose] + 'to_cause_word_count.txt'
    # path_effect_word_count = data_path + '/to_cause_effect_with_label/' + data_choose[choose_id] + out_spec[out_spec_choose] + 'to_effect_word_count.txt'
    # out_put_cause_effect_pair_count = 'data_direction_test/' + data_choose[choose_id] + out_spec[out_spec_choose] + 'to_count_cause_effect_pair.file'
    # out_put_cause_word_count = 'data_direction_test/' + data_choose[choose_id] + out_spec[out_spec_choose] + 'to_count_cause_word.file'
    # out_put_effect_word_count = 'data_direction_test/' + data_choose[choose_id] + out_spec[out_spec_choose] + 'to_count_effect_word.file'

    cause_effect_pair, word_list = get_pairandword()
    cal_count()

    print(total_count)
    print(len(cause_count), len(effect_count), len(cause_effect_count))
    print_tiem()

    with open(out_put_cause_effect_pair_count, 'wb') as file:
        pickle.dump(cause_effect_count, file)
    with open(out_put_cause_word_count, 'wb') as file:
        pickle.dump(cause_count, file)
    with open(out_put_effect_word_count, 'wb') as file:
        pickle.dump(effect_count, file)
    print('finish')
