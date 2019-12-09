import numpy as np
import pickle
from collections import defaultdict
from datetime import datetime
import math
starts = datetime.now()
"""
计算并保存所有元素的PMI值
"""

def print_tiem():
    ends = datetime.now()
    detla_time = ends - starts
    seconds = detla_time.seconds
    second = seconds % 60
    minitus = int(seconds / 60)
    minitus = minitus % 60
    hours = int(seconds / 3600)
    print("已经经过:", hours, "H ", minitus, "M ", second, "S")


def cal_count():
    with open(path_cause_word_count, 'r') as file:
        while True:
            word_count = file.readline()
            if word_count == '':
                break
            word_count = word_count.strip()
            temp = word_count.split(' ')
            # print(temp)
            word, count = temp[0], float(temp[1])
            if word != word.lower():
                print('大小写不统一')
            word = word.lower()
            cause_count[word] += count
            total_count_temp['total_cause'] += count

    with open(path_effect_word_count, 'r') as file:
        while True:
            word_count = file.readline()
            if word_count == '':
                break
            temp = word_count.split(' ')
            word, count = temp[0], float(temp[1])
            if word != word.lower():
                print('大小写不统一')
            word = word.lower()
            effect_count[word] += count
            total_count_temp['total_effect'] += count

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
            if effect != effect.lower():
                if cause != cause.lower():
                    print('大小写不统一')
            cause, effect = cause.lower(), effect.lower()
            print(id_number, cause, effect, count)
            cause_effect_count[cause][effect] += count
            total_count_temp['total_cause_effect'] += count


def cal_pmi():
    min_count = 1000000000
    max_count = 0
    for cause in cause_effect_count:
        for effect in cause_effect_count[cause]:
            probability_cause_effect_pair = math.log(float(cause_effect_count[cause][effect]) / total_cause_effect)
            probability_cause = math.log(float(cause_count[cause]) / total_cause)
            probability_effect = math.log(float(effect_count[effect]) / total_effect)
            pmi_word_pair = probability_cause_effect_pair - probability_cause - probability_effect
            min_count = min(min_count, pmi_word_pair)
            max_count = max(max_count, pmi_word_pair)

            PMI_word_pair_merge_cause_time[cause][effect] = pmi_word_pair

    print("Min:", min_count)
    print("Max:", max_count)
    # print(P_A_B)


"""
通过统计得到的cause effect 和count 计算PMI
"""
if __name__ == '__main__':
    total_count_temp = {'total_cause': 0, 'total_effect': 0, 'total_cause_effect': 0}
    total_count = {'total_cause': 14617361, 'total_effect': 14617361, 'total_cause_effect': 14617361}
    total_cause_effect = total_count['total_cause_effect']
    total_cause = total_count['total_cause']
    total_effect = total_count['total_effect']
    cause_effect_count = defaultdict(lambda: defaultdict(int))
    cause_count = defaultdict(float)
    effect_count = defaultdict(float)
    PMI_word_pair_merge_cause_time = defaultdict(lambda: defaultdict(int))

    data_path = '../../data'
    evidence_list = ['', 'cause_', 'time_', 'to_']
    choose_id = 0
    evidence_id = 3
    # path_cause_effect_pair_count = data_path + '/data_pair_count1/' + evidence_list[1] + evidence_list[2] + 'cause_effect_pair_count.txt'
    # path_cause_word_count = data_path + '/data_pair_count1/' + evidence_list[1] + evidence_list[2] + 'cause_word_count.txt'
    # path_effect_word_count = data_path + '/data_pair_count1/' + evidence_list[1] + evidence_list[2] + 'effect_word_count.txt'
    #
    # output_PMI_word_pair_merge_cause_time = data_path + '/direction_classifier/' + 'PMI_word_pair_merge_cause_time.txt'

    path_cause_effect_pair_count = data_path + '/data_pair_count1/' + evidence_list[1] + 'cause_effect_pair_count.txt'
    path_cause_word_count = data_path + '/data_pair_count1/' + evidence_list[1] + 'cause_word_count.txt'
    path_effect_word_count = data_path + '/data_pair_count1/' + evidence_list[1] + 'effect_word_count.txt'

    output_PMI_word_pair_merge_cause_time = data_path + '/direction_classifier/' + 'PMI_word_pair_cause.txt'
    # 读取数据
    cal_count()
    print('获得原始数据')
    cal_pmi()
    print(total_count)
    print(total_count_temp)
    print(len(cause_count), len(effect_count), len(cause_effect_count))

    with open(output_PMI_word_pair_merge_cause_time, 'w') as file:
        for cause in PMI_word_pair_merge_cause_time:
            for effect in PMI_word_pair_merge_cause_time[cause]:
                file.write('{} {} {} \n'.format(cause, effect, PMI_word_pair_merge_cause_time[cause][effect]))

    print_tiem()

    print('finish')

