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


def get_word_id(txt):
    for word_id in range(len(txt)):
        if bool(re.search('[a-z]', txt[word_id])):
            return word_id


def extract_text_clause():
    print('extract_text_clause')
    file_list = list(sorted(glob(os.path.join(input_path, '*'))))
    print(file_list)
    for file_name in file_list:

        spacy_list = list(sorted(glob(os.path.join(file_name, 'advcl*.npy'))))
        for number, clause_path in enumerate(spacy_list):
            print('cause_effect_clause_pair', len(cause_effect_clause_pair))
            date = np.load(clause_path, allow_pickle=True)
            for clause_pair in date:
                zhuju, ziju = clause_pair
                for word in zhuju:
                    word[0] = word[0].lower()
                for word in ziju:
                    word[0] = word[0].lower()
                # 此时标记,从句在句子的后向
                flag_for_order_2 = 1

                # 调整语序关系,此时从句在语言中出现在前面
                if zhuju[len(zhuju) - 1][0] == 'ziju':
                    zhuju = zhuju[:-1]
                    zhuju, ziju = ziju, zhuju
                    # 此时表示从句在原始顺序中在主句之前出现
                    flag_for_order_2 = 0
                else:
                    ziju = ziju[:-1]
                root_id = ziju[0][3]
                zhuju1 = sorted(zhuju, key=functools.cmp_to_key(cmp))
                ziju1 = sorted(ziju, key=functools.cmp_to_key(cmp))
                # zhuju_txt = ' '.join([temp[0] for temp in zhuju])
                # ziju_txt = ' '.join([temp[0] for temp in ziju])
                for word in ziju:
                    break
                    if (word[4] == root_id) & (word[2] == 'mark'):
                        # print(len(cause_effect_clause_pair))
                        word_dict[word[0]] += 1
                        word_pos = word[0] + '_' + word[1]
                        word_pos_dict[word_pos] += 1

                        if word[0] in causal_cause:
                            cause_effect_clause_pair.append([ziju, zhuju])
                        if word[0] in causal_effect_behind:
                            cause_effect_clause_pair.append([zhuju, ziju])
                        if word[0] in causal_cause_if:
                            cause_effect_clause_pair.append([ziju, zhuju])
                        break
                for num_id in range(len(ziju1)):
                    word = ziju1[num_id]
                    break

                    if (word[4] == root_id) & (word[3] < root_id):
                        if (word[2] == 'mark') | (word[2] == 'advmod'):

                            # if (word[1] == 'ADV'):
                            #     print(word[0], word[1])
                            #
                            #     word_dict[word[0]] += 1
                            #     word_pos = word[0] + '_' + word[2]
                            #     word_pos_dict[word_pos] += 1

                            if (word[0] in causal_cause):
                                # print('----')
                                cause_effect_clause_pair.append([ziju, zhuju])
                                break
                            if (word[0] in causal_effect_behind):
                                # print('----')
                                cause_effect_clause_pair.append([zhuju, ziju])
                                break
                            if (word[0] in causal_cause_if):
                                # print('----')
                                cause_effect_clause_pair.append([zhuju, ziju])
                                break
                for num_id in range(len(ziju1)):
                    word = ziju1[num_id]
                    # break

                    if (word[4] == root_id) & (word[3] < root_id):
                        if (word[2] == 'mark') | (word[2] == 'advmod'):

                            # if (word[1] == 'ADV'):
                            #     print(word[0], word[1])
                            #
                            #     word_dict[word[0]] += 1
                            #     word_pos = word[0] + '_' + word[2]
                            #     word_pos_dict[word_pos] += 1

                            if (word[0] in causal_cause) & (zhuju1[0][0] not in causal_cause):
                                # print(' '.join([temp[0] for temp in zhuju1]))
                                # print(' '.join([temp[0] for temp in ziju1]))
                                # # print(word[2], word[1], zhuju1[0][0])
                                # print('----')
                                cause_effect_clause_pair.append([ziju, zhuju])
                                break

                            if (word[0] in causal_effect_behind) & (flag_for_order_2 == 1):
                                if num_id > 0:
                                    if ziju1[num_id - 1][0] in causal_effect_behind_not:
                                        break
                                if num_id < len(ziju1) - 1:
                                    if ziju1[num_id + 1][0] in causal_effect_behind_not:
                                        break
                                # print(' '.join([temp[0] for temp in zhuju1]))
                                # print(' '.join([temp[0] for temp in ziju1]))
                                # print('----')
                                cause_effect_clause_pair.append([zhuju, ziju])
                                break
                            if (flag_for_order_2 == 0) & (word[0] in causal_cause_fir) & (
                                    zhuju1[0][0] not in causal_cause_fir_not):
                                # print(' '.join([temp[0] for temp in ziju1]))
                                # print(' '.join([temp[0] for temp in zhuju1]))
                                #
                                # print('----')
                                cause_effect_clause_pair.append([ziju, zhuju])
                                break

                if len(cause_effect_clause_pair) >= 100000:
                    count1[0] += 1
                    save_path = output_path + '/' + 'cause_cue_' + str(count1[0]) + '.npy'
                    np.save(save_path, cause_effect_clause_pair)
                    print(save_path)
                    del cause_effect_clause_pair[:]
                    print_time()

            print(clause_path)
    count1[0] += 1
    save_path = output_path + '/' + 'cause_cue_' + str(count1[0]) + '.npy'
    np.save(save_path, cause_effect_clause_pair)
    print(save_path)
    del cause_effect_clause_pair[:]
    print_time()


def extract_text_time():
    print('extract_text_time')
    file_list = list(sorted(glob(os.path.join(input_path, '*'))))
    print(file_list)
    for file_name in file_list:
        spacy_list = list(sorted(glob(os.path.join(file_name, 'advcl*.npy'))))

        for number, clause_path in enumerate(spacy_list):
            print(len(time_before_after_pair))
            date = np.load(clause_path, allow_pickle=True)
            for clause_pair in date:
                zhuju, ziju = clause_pair

                # 调整语序关系,让从句都在后面
                if zhuju[len(zhuju) - 1][0] == 'ziju':
                    zhuju = zhuju[:-1]
                    zhuju, ziju = ziju, zhuju
                else:
                    ziju = ziju[:-1]
                root_id = ziju[0][3]
                # zhuju = sorted(zhuju, key=functools.cmp_to_key(cmp))
                # ziju = sorted(ziju, key=functools.cmp_to_key(cmp))
                # zhuju_txt = ' '.join([temp[0] for temp in zhuju])
                # ziju_txt = ' '.join([temp[0] for temp in ziju])

                for word in ziju:
                    if (word[4] == root_id) & (word[2] == 'mark') & (word[3] < root_id):
                        # print(len(cause_effect_clause_pair))
                        word_dict[word[0]] += 1
                        word_pos = word[0] + '_' + word[1]
                        word_pos_dict[word_pos] += 1

                        if word[0] in time_before_after:
                            time_before_after_pair.append([ziju, zhuju])
                        if word[0] in time_after_before:
                            time_before_after_pair.append([zhuju, ziju])
                        break
                if len(time_before_after_pair) >= 100000:
                    count1[0] += 1
                    save_path = output_path + '/' + 'time_cue_' + str(count1[0]) + '.npy'
                    np.save(save_path, time_before_after_pair)
                    print(save_path)
                    del time_before_after_pair[:]
                    print_time()

            print(clause_path)
    count1[0] += 1
    save_path = output_path + '/' + 'time_cue_' + str(count1[0]) + '.npy'
    np.save(save_path, time_before_after_pair)
    print(save_path)
    del time_before_after_pair[:]
    print_time()


def extract_aux_to():
    print('extract_aux_to')
    file_list = list(sorted(glob(os.path.join(input_path, '*'))))
    print(file_list)
    for file_name in file_list:
        spacy_list = list(sorted(glob(os.path.join(file_name, 'advcl*.npy'))))

        for number, clause_path in enumerate(spacy_list):
            print(len(clause_to_pair))
            date = np.load(clause_path, allow_pickle=True)
            for clause_pair in date:
                zhuju, ziju = clause_pair

                # 调整语序关系,让从句都在后面
                if zhuju[len(zhuju) - 1][0] == 'ziju':
                    zhuju = zhuju[:-1]
                    # root_id = ziju[0][3]
                    zhuju, ziju = ziju, zhuju
                else:
                    ziju = ziju[:-1]
                root_id = ziju[0][3]
                zhuju = sorted(zhuju, key=functools.cmp_to_key(cmp))
                ziju = sorted(ziju, key=functools.cmp_to_key(cmp))
                zhuju_txt = ' '.join([temp[0] for temp in zhuju])
                ziju_txt = ' '.join([temp[0] for temp in ziju])

                # for word in ziju:
                for word in zhuju:
                    if (word[4] == root_id) & (word[2] == 'aux'):
                        # print(len(cause_effect_clause_pair))
                        word_dict[word[0]] += 1
                        word_pos = word[0] + '_' + word[1]
                        word_pos_dict[word_pos] += 1
                        print(word)
                        print(zhuju_txt)
                        print(ziju_txt)
                        print('---' * 45)

                        if word[0] == 'to':
                            clause_to_pair.append([ziju, zhuju])
                        break
                if len(clause_to_pair) >= 100000:
                    count1[0] += 1
                    save_path = output_path + '/' + 'to_cue_' + str(count1[0]) + '.npy'
                    np.save(save_path, clause_to_pair)
                    print(save_path)
                    del clause_to_pair[:]
                    print_time()

            print(clause_path)
    count1[0] += 1
    save_path = output_path + '/' + 'to_cue_' + str(count1[0]) + '.npy'
    np.save(save_path, clause_to_pair)
    print(save_path)
    del clause_to_pair[:]
    print_time()


# since:[3442, 10226]  ---因为;自从  从句先发生
# once[2191, 2078]   从句先发生
# before 316545  从句后发生
# after 194708   从句先发生
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('data', type=int, help="choose data")
    # parser.add_argument('evidence', type=int, help="choose evidence kind")
    # args = parser.parse_args()
    # data_choose = args.data
    # evidence_choose = args.evidence
    cause_effect_clause_pair = []
    time_before_after_pair = []
    clause_to_pair = []
    count1 = [0]
    word_dict = defaultdict(int)
    word_pos_dict = defaultdict(int)
    # 抽取因果句:主句中有so,such等,连词是that
    causal_cause = {'because', 'cause', 'cuz', "'cause"}
    causal_cause_if = {'if', 'unless'}
    causal_cause_fir = {'since'}
    causal_cause_fir_not = {'but'}
    causal_effect_behind = {'so'}
    causal_effect_behind_not = {'as', 'even', 'if'}

    time_before_after = {'since', 'once', 'after'}
    time_after_before = {'before'}
    # until

    evidence_choose = 0
    evidence_kinds = ['advcl', 'conj', 'inter']
    evidence = evidence_kinds[evidence_choose]

    data_choose = 2
    # data_path = '../../data/'
    data_path = '../../../../data_disk/ray/data/'
    input_list = ['data_clause/icw', 'data_clause/bok', 'data_clause/gut']
    input_path = data_path + input_list[data_choose]
    output_list = ['data_cause_effect/icw', 'data_cause_effect/bok', 'data_cause_effect/gut']
    output_path = data_path + output_list[data_choose]

    extract_text_clause()
    # extract_text_time()
    # extract_aux_to()

    # for i in range(len(file_list)):
    #     extract_text_clause(file_list[i])

    word_dict1 = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
    print(len(word_dict1), word_dict1)
    word_pos_dict1 = sorted(word_pos_dict.items(), key=lambda d: d[1], reverse=True)
    print(len(word_pos_dict1), word_pos_dict1)

    print_time()

    # lock = multiprocessing.Lock()
    # # 创建多个进程
    # thread_list = []
    # len_file_list = len(file_list)
    # for choose_file_list in range(1):
    #     choose = file_list[choose_file_list]
    #     sthread = multiprocessing.Process(target=extract_text_clause, args=choose)
    #     thread_list.append(sthread)
    # for th in thread_list:
    #     th.start()
    # for th in thread_list:
    #     th.join()

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
