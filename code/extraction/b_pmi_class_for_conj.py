import argparse
import codecs
import multiprocessing
import pickle
import pandas as pd
import string
import numpy as np
from glob import glob
import os
import re
from datetime import datetime
import sys

sys.path.extend(['../'])
from API.extraction import *
from collections import defaultdict

starts = datetime.now()
"""
抽取出因果/时间和to作为连词的子句关系
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


def get_word_id(txt):
    for word_id in range(len(txt)):
        if bool(re.search('[a-z]', txt[word_id])):
            return word_id


def extract_text_conj():
    print('extract_text_clause')
    file_list = list(sorted(glob(os.path.join(input_path, '*'))))
    print(file_list)
    for file_name in file_list:

        spacy_list = list(sorted(glob(os.path.join(file_name, evidence + '*.npy'))))
        for number, clause_path in enumerate(spacy_list):
            print('cause_effect_clause_pair', len(cause_effect_clause_pair), clause_path)
            date = np.load(clause_path, allow_pickle=True)

            for clause_pair in date:
                zhuju, ziju = clause_pair
                for word in zhuju:
                    word[0] = word[0].lower()
                for word in ziju:
                    word[0] = word[0].lower()
                zhuju_original, ziju_original = zhuju, ziju

                if zhuju[len(zhuju) - 1][0] == 'ziju':
                    zhuju = zhuju[:-1]
                    # 调整语序关系,让从句都在后面
                    # 这些都去除
                    continue
                else:
                    ziju = ziju[:-1]
                root_id = ziju[0][3]
                zhuju1 = sorted(zhuju, key=functools.cmp_to_key(cmp))
                ziju1 = sorted(ziju, key=functools.cmp_to_key(cmp))
                if filter_pos_other_flag:
                    zhuju = filter(filter_pos_other, zhuju)
                    ziju = filter(filter_pos_other, ziju)
                    zhuju, ziju = list(zhuju), list(ziju)
                # 过滤停用词
                if filter_stop_word_flag:
                    zhuju = filter(filter_stop_word, zhuju)
                    ziju = filter(filter_stop_word, ziju)
                    zhuju, ziju = list(zhuju), list(ziju)
                # 　过滤含有其他字符的词
                if filter_isalpha_flag:
                    zhuju = filter(filter_isalpha, zhuju)
                    ziju = filter(filter_isalpha, ziju)
                    zhuju, ziju = list(zhuju), list(ziju)
                # 过滤第三类成分词
                if filter_pos_not_function_or_content_flag:
                    zhuju = filter(filter_pos_not_function_or_content, zhuju)
                    ziju = filter(filter_pos_not_function_or_content, ziju)
                    zhuju, ziju = list(zhuju), list(ziju)
                # 过滤功能词
                if filter_pos_function_flag:
                    zhuju = filter(filter_pos_function, zhuju)
                    ziju = filter(filter_pos_function, ziju)
                    zhuju, ziju = list(zhuju), list(ziju)
                pmi_score_l_2_r = 0
                pmi_score_r_2_l = 0
                for cause_word in zhuju:
                    for effect_word in ziju:
                        cause_w = cause_word[0].lower()
                        effect_w = effect_word[0].lower()
                        pmi_score_l_2_r += PMI_word_pair_merge_cause_time[cause_w][effect_w]
                        pmi_score_r_2_l += PMI_word_pair_merge_cause_time[effect_w][cause_w]
                if (len(zhuju) * len(ziju)) > 0:
                    pmi_score_l_2_r /= (len(zhuju) * len(ziju))
                    pmi_score_r_2_l /= (len(zhuju) * len(ziju))
                if pmi_score_l_2_r > pmi_score_r_2_l:
                    label = 1  # 此时方向为正向 就是说,主句在前,从句在后
                elif pmi_score_l_2_r < pmi_score_r_2_l:
                    label = -1
                else:
                    label = 0
                pmi_score_cha = pmi_score_l_2_r - pmi_score_r_2_l
                pmi_socre_max = max(pmi_score_l_2_r, pmi_score_r_2_l)
                pmi_score_cha_list[round(pmi_score_cha, 2)] += 1
                pmi_score_max_list[round(pmi_socre_max, 2)] += 1

                # print(' '.join([temp[0] for temp in zhuju1]))
                # print(' '.join([temp[0] for temp in ziju1]))
                # print('---')
                # for num_id in range(len(zhuju)):
                #     word = zhuju[num_id]
                #     word_dict[word[0]] += 1
                #     word_pos = word[0] + '_' + word[2] + '_' + word[1]
                #     word_pos_dict[word_pos] += 1
                # for num_id in range(len(ziju)):
                #     word = ziju[num_id]
                #     word_dict[word[0]] += 1
                #     word_pos = word[0] + '_' + word[2] + '_' + word[1]
                #     word_pos_dict[word_pos] += 1
                cause_effect_clause_pair.append([zhuju_original, ziju_original, label, pmi_score_l_2_r,
                                                 pmi_score_r_2_l])
                if len(cause_effect_clause_pair) >= 10000000:
                    count1[0] += 1
                    save_path = output_path + '/' + 'phrase_and' + str(count1[0]) + '.npy'
                    np.save(save_path, cause_effect_clause_pair)
                    print(save_path)
                    del cause_effect_clause_pair[:]
                    print_time()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('data', type=int, help="choose data")
    # parser.add_argument('evidence', type=int, help="choose evidence kind")
    # args = parser.parse_args()
    # data_choose = args.data
    # evidence_choose = args.evidence
    cause_effect_clause_pair = []
    count1 = [0]
    word_dict = defaultdict(int)
    word_pos_dict = defaultdict(int)
    pmi_score_cha_list = defaultdict(int)
    pmi_score_max_list = defaultdict(int)
    # 过滤的一些参数
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

    evidence_choose = 1
    evidence_kinds = ['advcl', 'conj', 'inter']
    evidence = evidence_kinds[evidence_choose]

    data_choose = 1
    data_path = '../../data/'
    input_list = ['data_clause/icw', 'data_clause/bok', 'data_clause/gut']
    input_path = data_path + input_list[data_choose]
    output_list = ['to_cause_effect_with_label/icw', 'to_cause_effect_with_label/bok', 'to_cause_effect_with_label/gut']
    output_path = data_path + output_list[data_choose]

    PMI_word_pair_merge_cause_time = defaultdict(lambda: defaultdict(int))
    imput_PMI_word_pair_merge_cause_time = data_path + 'direction_classifier/' + 'cause_' + 'PMI_word_pair_cause.txt'
    with open(imput_PMI_word_pair_merge_cause_time, 'r') as file:
        while True:
            word_count = file.readline()
            if word_count == '':
                break
            temp = word_count.split(' ')
            cause, effect, count = temp[0], temp[1], float(temp[2])
            # print(id_number, cause, effect, count)
            PMI_word_pair_merge_cause_time[cause][effect] += count

    extract_text_conj()

    word_dict1 = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
    print(len(word_dict1), word_dict1)
    word_pos_dict1 = sorted(word_pos_dict.items(), key=lambda d: d[1], reverse=True)
    print(len(word_pos_dict1), word_pos_dict1)
    pmi_score_cha_list = sorted(pmi_score_cha_list.items(), key=lambda d: d[1], reverse=True)
    print(len(pmi_score_cha_list), pmi_score_cha_list)
    pmi_score_max_list = sorted(pmi_score_max_list.items(), key=lambda d: d[1], reverse=True)
    print(len(pmi_score_max_list), pmi_score_max_list)

    print_time()


# mark 基本都是ADP
# advmod 基本都是ADV
# advcl 基本都是VERB
# aux 除了to 都是VERB
# nsubj 都是代词


# 本实验用于抽取
# 1.因果中包含因果次的部分
# 2.to引导的部分
# 2.连接词中 表示先后顺序的部分
# 3. advmod中ly副词引导的部分
# 4.动词引导的部分
