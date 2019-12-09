
import pickle
import numpy as np
from glob import glob
import os
from datetime import datetime
from nltk.corpus import stopwords
from collections import defaultdict
import functools
starts = datetime.now()
"""
判断样本的正负方向
"""


def cmp(x, y):
    if x[3] > y[3]:
        return 1
    if x[3] < y[3]:
        return -1
    else:
        return 0


def print_time():
    # 用来打印时间
    ends = datetime.now()
    detla_time = ends - starts
    seconds = detla_time.seconds
    second = seconds % 60
    minitus = int(seconds / 60)
    minitus = minitus % 60
    hours = int(seconds / 3600)
    print("已经经过:", hours, "H ", minitus, "M ", second, "S")


def filter_pos_other(word):
    return word[1] not in pos_filter


def filter_stop_word(word):
    return word[0] not in eng_stop_word


def filter_isalpha(word):
    return word[0].encode('UTF-8').isalpha()


def filter_pos_function(word):
    return word[1] not in pos_function


def filter_pos_not_function_or_content(word):
    return word[1] not in pos_other


def decide_derection(evidence, out_name):
    file_list = list(sorted(glob(os.path.join(input_path, evidence + '*.npy'))))
    print(file_list)
    for cause_effect_pair_path in file_list:
        specific_name = cause_effect_pair_path.split('/')[-1]
        output_to_pair_path = output_path + '/' + out_name + specific_name
        print(cause_effect_pair_path)
        clause_to_pair = []
        cause_effect_list = np.load(cause_effect_pair_path, allow_pickle=True)
        # print(cause_effect_list)
        for cause_effect_pair in cause_effect_list:
            cause_clause, effect_clause = cause_effect_pair

            if filter_pos_other_flag:
                cause_clause = filter(filter_pos_other, cause_clause)
                effect_clause = filter(filter_pos_other, effect_clause)
                cause_clause, effect_clause = list(cause_clause), list(effect_clause)
            # 过滤停用词
            if filter_stop_word_flag:
                cause_clause = filter(filter_stop_word, cause_clause)
                effect_clause = filter(filter_stop_word, effect_clause)
                cause_clause, effect_clause = list(cause_clause), list(effect_clause)
            # 　过滤含有其他字符的词
            if filter_isalpha_flag:
                cause_clause = filter(filter_isalpha, cause_clause)
                effect_clause = filter(filter_isalpha, effect_clause)
                cause_clause, effect_clause = list(cause_clause), list(effect_clause)
            # 过滤第三类成分词
            if filter_pos_not_function_or_content_flag:
                cause_clause = filter(filter_pos_not_function_or_content, cause_clause)
                effect_clause = filter(filter_pos_not_function_or_content, effect_clause)
                cause_clause, effect_clause = list(cause_clause), list(effect_clause)
            # 过滤功能词
            if filter_pos_function_flag:
                cause_clause = filter(filter_pos_function, cause_clause)
                effect_clause = filter(filter_pos_function, effect_clause)
                cause_clause, effect_clause = list(cause_clause), list(effect_clause)
            pmi_score_left_to_right = 0
            pmi_score_right_to_left = 0
            for cause_word in cause_clause:
                for effect_word in effect_clause:
                    cause_w = cause_word[0].lower()
                    effect_w = effect_word[0].lower()
                    pmi_score_left_to_right += PMI_word_pair_merge_cause_time[cause_w][effect_w]
                    pmi_score_right_to_left += PMI_word_pair_merge_cause_time[effect_w][cause_w]
            if (len(cause_clause) * len(effect_clause)) > 0:
                pmi_score_left_to_right /= (len(cause_clause) * len(effect_clause))
                pmi_score_right_to_left /= (len(cause_clause) * len(effect_clause))
            if pmi_score_left_to_right > pmi_score_right_to_left:
                label = 1 # 此时方向为正向 就是说,主句在前,从句在后
            elif pmi_score_left_to_right < pmi_score_right_to_left:
                label = -1
            else:
                label = 0
            max_pmi = round(max(pmi_score_left_to_right, pmi_score_right_to_left), 2)
            max_pmi_list[max_pmi] += 1
            clause_to_pair.append([cause_effect_pair[0], cause_effect_pair[1], label, pmi_score_left_to_right, pmi_score_right_to_left])

        np.save(output_to_pair_path, clause_to_pair)
        print(output_to_pair_path)


"""
计算每个子句之间的PMI值--PMI代表因果性的强弱
并且获得子句的方向
"""
if __name__ == '__main__':
    data_path = '../../data/'
    PMI_word_pair_merge_cause_time = defaultdict(lambda: defaultdict(int))
    # imput_PMI_word_pair_merge_cause_time = data_path + '/direction_classifier/' + 'PMI_word_pair_merge_cause_time_copy.txt'
    out_spec = ['cause_label_']
    imput_PMI_word_pair_merge_cause_time = data_path + '/direction_classifier/' + 'PMI_word_pair_cause.txt'
    id_number = 0
    with open(imput_PMI_word_pair_merge_cause_time, 'r') as file:
        while True:
            id_number += 1
            word_count = file.readline()
            if word_count == '':
                break
            temp = word_count.split(' ')
            # print(temp)
            cause, effect, count = temp[0], temp[1], float(temp[2])
            # print(id_number, cause, effect, count)
            PMI_word_pair_merge_cause_time[cause][effect] += count
    filter_pos_other_flag = 1
    filter_stop_word_flag = 1
    filter_isalpha_flag = 1
    filter_pos_not_function_or_content_flag = 1
    filter_pos_function_flag = 1
    pos_filter = {'X', 'NUM', 'PRON', 'SYM', 'SPACE', 'DET', 'PUNCT'}
    pos_function = {'PART', 'ADP', 'CCONJ'}
    pos_content = {'VERB', 'ADJ', 'NOUN'}
    pos_other = {'INTJ', 'ADV', 'PROPN', }
    with open('stop_word.txt') as file:
        eng_stop_word = file.readlines()
        eng_stop_word = set([word.strip() for word in eng_stop_word])
    print(eng_stop_word)

    evidence_list = ['to_']
    input_list = ['data_cause_effect/icw', 'data_cause_effect/bok', 'data_cause_effect/gut']
    output_list = ['to_cause_effect_with_label/icw', 'to_cause_effect_with_label/bok', 'to_cause_effect_with_label/gut']

    evidence_choose = 0
    max_pmi_list = defaultdict(int)
    for i in range(3):
        data_choose = i
        input_path = data_path + input_list[data_choose]
        output_path = data_path + output_list[data_choose]
        decide_derection(evidence_list[evidence_choose], out_spec[0])


