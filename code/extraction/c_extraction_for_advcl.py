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
from tqdm import tqdm

sys.path.extend(['../'])
from API.extraction import *
from reference.base import FilterData
from collections import defaultdict

starts = datetime.now()
"""
抽取出因果/时间和to作为连词的子句关系
"""


def cmp(x, y):
    if x[6] > y[6]:
        return 1
    if x[6] < y[6]:
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


def get_sentence_tense(verb_word, clause):
    tense_present_past = {'VBD': 1, 'VBN': 1, 'VB': 3, 'VBP': 3, 'VBZ': 3, 'VBG': 3, 'MD': 4}
    if verb_word[3] == 'VERB':
        clause_tense = tense_present_past[verb_word[4]]
        for word in clause:

            if word[6] < verb_word[6] and word[6] in verb_word[8] and word[3] == 'VERB':
                # tense_list[word[0]] += 1
                if word[0] in {'’ve', 'is', 'are', '’m', '’s', 'have', 'has', "'ve", 'do', 'does', 'be'}:
                    clause_tense = 3
                    break
                if word[0] in {'’ll', 'will', "'ll", 'shall'}:
                    clause_tense = 4
                    break
                if word[0] in {'would'}:
                    clause_tense = 2
                    break
                if word[0] in {'was', 'did', 'had', 'were', 'been'}:
                    clause_tense = 1
                    break
                if word[0] in {'might', 'could', 'should'}:
                    clause_tense = 0
                    break
                pass
    else:
        clause_tense = 0
    return clause_tense


class CalAndSave(object):
    def __init__(self, output_pair_count, output_fir_word_count, output_las_word_count, output_total_word_count):
        self.pair_count = defaultdict(lambda: defaultdict(float))
        self.fir_word_count = defaultdict(float)
        self.las_word_count = defaultdict(float)
        self.total_word_count = defaultdict(float)
        self.output_pair_count = output_pair_count
        self.output_fir_word_count = output_fir_word_count
        self.output_las_word_count = output_las_word_count
        self.output_total_word_count = output_total_word_count
        self.count = 0

    def cal_for_count(self, fir_claus, last_claus, filter_data):
        # print(len(fir_claus), len(last_claus))

        fir_claus, last_claus = filter_data.filter_now(
            fir_claus, last_claus, filter_data.filter_stop_word)
        fir_clause, last_clause = filter_data.filter_now(
            fir_claus, last_claus, filter_data.filter_isalpha)

        # print(len(fir_clause), len(last_clause))
        # print('---')
        if len(fir_clause) < 1 or len(last_clause) < 1 or (len(fir_clause) == 1 and fir_clause[0][3] != 'VERB') or \
                (len(last_clause) == 1 and last_clause[0][3] != 'VERB'):
            # aa[8] += 1
            self.count += 1
        else:
            len_pair = len(fir_clause) * len(last_clause)
            fir_clause = [word[0] for word in fir_clause]
            last_clause = [word[0] for word in last_clause]
            len_pair = len(fir_clause) * len(last_clause)
            for fir_word in fir_clause:
                for las_word in last_clause:
                    self.pair_count[fir_word][las_word] += 1 / len_pair
            for fir_word in fir_clause:
                self.fir_word_count[fir_word] += 1 / len(fir_clause)
                self.total_word_count[fir_word] += 1
            for las_word in last_clause:
                self.las_word_count[las_word] += 1 / len(last_clause)
                self.total_word_count[las_word] += 1

    def embedding_txt(self, fir_claus, last_claus, filter_data):
        # print(len(fir_claus), len(last_claus))
        for word in fir_claus:
            word[1], word[3] = word[3], word[1]
        for word in last_claus:
            word[1], word[3] = word[3], word[1]

        fir_claus, last_claus = filter_data.filter_now(
            fir_claus, last_claus, filter_data.filter_stop_word)
        fir_claus, last_claus = filter_data.filter_now(
            fir_claus, last_claus, filter_data.filter_isalpha)
        fir_claus, last_claus = filter_data.filter_now(
            fir_claus, last_claus, filter_data.filter_pos_other)
        fir_claus, last_claus = filter_data.filter_now(
            fir_claus, last_claus, filter_data.filter_pos_function)
        fir_clause, last_clause = filter_data.filter_now(
            fir_claus, last_claus, filter_data.filter_pos_not_function_or_content)
        # print(len(fir_clause), len(last_clause))
        # print('---')
        if len(fir_clause) < 1 or len(last_clause) < 1 or (len(fir_clause) == 1 and fir_clause[0][1] != 'VERB') or \
                (len(last_clause) == 1 and last_clause[0][1] != 'VERB'):
            # aa[8] += 1
            self.count += 1
            return '\n'
        else:
            txt = ' '.join([word[0] for word in fir_clause]) + '----' + ' '.join([word[0] for word in last_clause]) + '\n'
            return txt

    def save_date(self):
        with open(self.output_pair_count, 'w') as file:
            for cause in self.pair_count:
                for effect in self.pair_count[cause]:
                    file.write('{} {} {} \n'.format(cause, effect, self.pair_count[cause][effect]))
        with open(self.output_fir_word_count, 'w') as file:
            for cause in self.fir_word_count:
                file.write('{} {}  \n'.format(cause, self.fir_word_count[cause]))
        with open(self.output_las_word_count, 'w') as file:
            for cause in self.las_word_count:
                file.write('{} {}  \n'.format(cause, self.las_word_count[cause]))
        with open(self.output_total_word_count, 'w') as file:
            for cause in self.total_word_count:
                file.write('{} {}  \n'.format(cause, self.total_word_count[cause]))
        print('finish save')


class Extract_Advcl(object):

    def __init__(self):
        self.causal_cause = {'because', 'cause', 'cuz', "'cause"}
        self.causal_cause_if = {'if', 'unless'}
        self.causal_cause_fir = {'since'}
        self.causal_cause_fir_not = {'but'}
        self.causal_effect_behind = {'so'}
        self.causal_effect_behind_not = {'as', 'even', 'if'}

        self.time_before_after = {'since', 'once', 'after'}
        self.time_after_before = {'before'}

        # 从句在前
        self.fir = {'whenever', 'when', 'whilst', 'while'}  # 'with',
        self.fir_cause = {'as', 'already', 'for', 'after', 'once', 'because', 'cause', 'cuz',
                          "'cause", 'if', 'unless', 'since', '’cause'}  #
        self.las = {'until', 'til', 'till', }
        self.las_cause = {'later', 'to', 'before', 'then', 'lest', 'so', 'thus', 'thereby'}
        # 句子原有顺序
        self.fir_last = {'now', 'anytime', 'that', 'somehow', 'how', }

        # 没有因果: 直接去除
        self.filter_fonction = {
            'instead', 'soon', 'than', 'otherwise', 'like', 'whether', 'where', 'wherever', 'anywhere',
            'why', 'except', 'which', 'although', 'though', 'whereas', 'however', 'even'}
        # 从句在前
        # fir = {'as', 'for', 'after', 'once', 'with', 'already', 'when', 'whenever', 'while', 'whilst', }
        # fir_cause = {'because', 'cause', 'cuz', "'cause", 'if', 'unless', 'since'}
        # las = {'to', 'before', 'then', 'lest', 'later', 'until', 'til', 'till', }
        # las_cause = {'so', 'thus', 'thereby'}
        # # 因果:
        # causal_not = {'but', 'even'}
        # # 句子原有顺序
        # fir_last = {'now', 'anytime', 'that', 'somehow', 'how', }
        #
        # # 没有因果: 直接去除
        # filter_fonction = {
        #     'instead', 'soon', 'than', 'otherwise', 'like', 'whether', 'where', 'wherever', 'anywhere', 'why',
        #     'except', 'which', 'although', 'though', 'whereas', 'however', 'even if', 'as if', 'that_nsubj', }
        # # # 并非真实引导词
        # # {'still', 'often', 'sometimes', 'of', 'right', 'long', 'always', 'only', 'almost', 'first', 'ever',
        # #  'much', 'well', 'also', 'again', 'ago', 'more', 'perhaps', 'usually', 'just', 'even', }

    @staticmethod
    def pprint(word, main_clause1, child_clause1, causal_pair):
        print(word)
        print(main_clause1)
        print(child_clause1)
        print(' '.join([temp[0] for temp in main_clause1]))
        print(' '.join([temp[0] for temp in child_clause1]))
        print(' '.join([temp[0] for temp in causal_pair]))
        print('---')

    def extract_text_clause(self):
        print('extract_text_clause')
        file_list = list(sorted(glob(os.path.join(input_path, '*'))))
        print(file_list)
        for file_name in file_list:

            spacy_list = list(sorted(glob(os.path.join(file_name, 'advcl*.npy'))))
            print(spacy_list)
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

                    # 调整语序关系,让从句都在后面
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
                    flag_specific = [0, 0]
                    for word in ziju:
                        if (word[4] == root_id) & (word[2] == 'mark'):
                            # print(len(cause_effect_clause_pair))
                            word_dict[word[0]] += 1
                            word_pos = word[0] + '_' + word[1]
                            word_pos_dict[word_pos] += 1

                            if word[0] in self.causal_cause:
                                cause_effect_clause_pair.append([ziju, zhuju])
                                flag_specific[0] = 1
                            if word[0] in self.causal_effect_behind:
                                cause_effect_clause_pair.append([zhuju, ziju])
                                flag_specific[0] = 1
                            if word[0] in self.causal_cause_if:
                                cause_effect_clause_pair.append([ziju, zhuju])
                                flag_specific[0] = 1
                            break
                    for num_id in range(len(ziju1)):
                        word = ziju1[num_id]

                        if (word[4] == root_id) & (word[3] < root_id):
                            if (word[2] == 'mark') | (word[2] == 'advmod'):

                                # if (word[1] == 'ADV'):
                                #     print(word[0], word[1])
                                #
                                #     word_dict[word[0]] += 1
                                #     word_pos = word[0] + '_' + word[2]
                                #     word_pos_dict[word_pos] += 1

                                if (word[0] in self.causal_cause) & (zhuju1[0][0] not in self.causal_cause):
                                    # print(' '.join([temp[0] for temp in zhuju1]))
                                    # print(' '.join([temp[0] for temp in ziju1]))
                                    # # print(word[2], word[1], zhuju1[0][0])
                                    # print('----')
                                    cause_effect_clause_pair.append([ziju, zhuju])
                                    break

                                if (word[0] in self.causal_effect_behind) & (flag_for_order_2 == 1):
                                    if num_id > 0:
                                        if ziju1[num_id - 1][0] in self.causal_effect_behind_not:
                                            break
                                    if num_id < len(ziju1) - 1:
                                        if ziju1[num_id + 1][0] in self.causal_effect_behind_not:
                                            break
                                    # print(' '.join([temp[0] for temp in zhuju1]))
                                    # print(' '.join([temp[0] for temp in ziju1]))
                                    # print('----')
                                    cause_effect_clause_pair.append([zhuju, ziju])
                                    break
                                if (flag_for_order_2 == 0) & (word[0] in self.causal_cause_fir) & (
                                        zhuju1[0][0] not in self.causal_cause_fir_not):
                                    # print(' '.join([temp[0] for temp in ziju1]))
                                    # print(' '.join([temp[0] for temp in zhuju1]))
                                    #
                                    # print('----')
                                    cause_effect_clause_pair.append([ziju, zhuju])
                                    break
                    if flag_specific == [0, 1]:
                        print(' '.join([temp[0] for temp in zhuju1]))
                        print(' '.join([temp[0] for temp in ziju1]))
                        print('---')
                    if len(cause_effect_clause_pair) >= 100000:
                        count1[0] += 1
                        save_path = output_path + '/' + 'cause_cue_' + str(count1[0]) + '.npy'
                        np.save(save_path, cause_effect_clause_pair)
                        print(save_path)
                        del cause_effect_clause_pair[:]
                        print_time()
        #
        #         print(clause_path)
        # count1[0] += 1
        # save_path = output_path + '/' + 'cause_cue_' + str(count1[0]) + '.npy'
        # np.save(save_path, cause_effect_clause_pair)
        # print(save_path)
        # del cause_effect_clause_pair[:]
        # print_time()

    def extract_text_time(self):
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

                            if word[0] in self.time_before_after:
                                time_before_after_pair.append([ziju, zhuju])
                            if word[0] in self.time_after_before:
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

    @staticmethod
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

    def extract_filter_advcl(self):
        filter_data = FilterData()
        print('extract_filter_advcl')
        file_list = list(sorted(glob(os.path.join(input_path, '*'))))
        print(file_list)
        for file_name in file_list:

            spacy_list = list(sorted(glob(os.path.join(file_name, 'advcl*.npy'))))
            # print(len(spacy_list), spacy_list[0])
            # for number_file, clause_path in enumerate(spacy_list):
            for number_file in tqdm(range(len(spacy_list))):
                clause_path = spacy_list[number_file]
                date = np.load(clause_path, allow_pickle=True)
                for clause_pair in date:
                    main_clause, child_clause = clause_pair
                    # 调整语序关系,让从句都在后面
                    if main_clause[len(main_clause) - 1][0] == 'ziju':
                        main_clause = main_clause[:-1]
                        main_clause, child_clause = child_clause, main_clause
                    else:
                        child_clause = child_clause[:-1]
                    child_root_id = child_clause[0][6]
                    main_root_id = main_clause[0][6]


                    causal_pair = main_clause + child_clause
                    causal_pair = sorted(causal_pair, key=functools.cmp_to_key(cmp))
                    main_clause1 = sorted(main_clause, key=functools.cmp_to_key(cmp))
                    child_clause1 = sorted(child_clause, key=functools.cmp_to_key(cmp))
                    child_fir_id_id = child_clause1[0][6]
                    str_sent1 = ' '.join([word[0] for word in main_clause1])
                    str_sent2 = ' '.join([word[0] for word in child_clause1])
                    if 'ust for seplit' in str_sent1 or 'ust for seplit' in str_sent2:
                        continue

                    aa[0] += 1
                    flag = 1
                    for num_id in range(len(child_clause1)):
                        word = child_clause1[num_id]
                        if word[7] == child_root_id and word[6] < child_root_id:
                            if word[0] == 'as':
                                flag = 0
                                if child_clause1[num_id + 1][0] == 'if':
                                    aa[2] += 1
                                    break
                                else:
                                    aa[1] += 1
                                    fir_clause, last_clause = child_clause1, main_clause1
                                    cal_and_save.cal_for_count(fir_clause, last_clause, filter_data)
                                    break
                            elif word[0] in self.fir_cause:
                                flag = 0
                                aa[1] += 1
                                fir_clause, last_clause = child_clause1, main_clause1
                                cal_and_save.cal_for_count(fir_clause, last_clause, filter_data)
                                break
                            elif word[0] in self.fir:
                                flag = 0
                                aa[3] += 1
                                fir_clause, last_clause = child_clause1, main_clause1
                                cal_and_save.cal_for_count(fir_clause, last_clause, filter_data)
                                break
                            elif word[0] in self.las:
                                flag = 0
                                aa[4] += 1
                                fir_clause, last_clause = main_clause1, child_clause1
                                cal_and_save.cal_for_count(fir_clause, last_clause, filter_data)
                                break
                            elif word[0] in self.las_cause:
                                flag = 0
                                aa[1] += 1
                                fir_clause, last_clause = main_clause1, child_clause1
                                cal_and_save.cal_for_count(fir_clause, last_clause, filter_data)
                                break
                            elif word[0] in self.filter_fonction:
                                flag = 0
                                aa[2] += 1
                                break

                        if word[6] < child_root_id:
                            if word[0] in self.filter_fonction:
                                flag = 0
                                aa[2] += 1
                                break
                                # Extract_Advcl.pprint(word, main_clause1, child_clause1, causal_pair)
                            if word[0] in self.fir_cause or word[0] in self.fir:
                                pass

                            if word[0] in self.las or word[0] in self.las_cause:
                                pass

                    if flag:
                        aa[5] += 1
                        if child_root_id > main_root_id:
                            #
                            fir_clause, last_clause = main_clause1, child_clause1
                            cal_and_save.cal_for_count(fir_clause, last_clause, filter_data)
                            pass
                        else:
                            # aa[4] += 1
                            fir_clause, last_clause = child_clause1, main_clause1
                            cal_and_save.cal_for_count(fir_clause, last_clause, filter_data)
                            # Extract_Advcl.pprint(word, main_clause1, child_clause1, causal_pair)

            # print('len_pair_count:', len(cal_and_save.pair_count))

    def extract_get_word_embedding_text(self):
        file = open('advcl_for_embedding.txt', 'w')

        filter_data = FilterData()
        print('extract_filter_advcl')
        file_list = list(sorted(glob(os.path.join(input_path, '*'))))
        print(file_list)
        for file_name in file_list:

            spacy_list = list(sorted(glob(os.path.join(file_name, 'advcl*.npy'))))
            # print(len(spacy_list), spacy_list[0])
            # for number_file, clause_path in enumerate(spacy_list):
            for number_file in tqdm(range(len(spacy_list))):
                clause_path = spacy_list[number_file]
                date = np.load(clause_path, allow_pickle=True)
                for clause_pair in date:
                    main_clause, child_clause = clause_pair
                    # 调整语序关系,让从句都在后面
                    if main_clause[len(main_clause) - 1][0] == 'ziju':
                        main_clause = main_clause[:-1]
                        main_clause, child_clause = child_clause, main_clause
                    else:
                        child_clause = child_clause[:-1]
                    child_root_id = child_clause[0][6]
                    main_root_id = main_clause[0][6]

                    causal_pair = main_clause + child_clause
                    causal_pair = sorted(causal_pair, key=functools.cmp_to_key(cmp))
                    main_clause1 = sorted(main_clause, key=functools.cmp_to_key(cmp))
                    child_clause1 = sorted(child_clause, key=functools.cmp_to_key(cmp))
                    child_fir_id_id = child_clause1[0][6]
                    str_sent1 = ' '.join([word[0] for word in main_clause1])
                    str_sent2 = ' '.join([word[0] for word in child_clause1])
                    if 'ust for seplit' in str_sent1 or 'ust for seplit' in str_sent2:
                        print(str_sent1, str_sent2)
                        continue

                    aa[0] += 1
                    flag = 1
                    for num_id in range(len(child_clause1)):
                        word = child_clause1[num_id]
                        if word[7] == child_root_id and word[6] < child_root_id:
                            if word[0] == 'as':
                                flag = 0
                                if child_clause1[num_id + 1][0] == 'if':
                                    aa[2] += 1
                                    break
                                else:
                                    aa[1] += 1
                                    fir_clause, last_clause = child_clause1, main_clause1
                                    txt_pair = cal_and_save.embedding_txt(fir_clause, last_clause, filter_data)
                                    file.write(txt_pair)
                                    break
                            elif word[0] in self.fir_cause:
                                flag = 0
                                aa[1] += 1
                                fir_clause, last_clause = child_clause1, main_clause1
                                txt_pair = cal_and_save.embedding_txt(fir_clause, last_clause, filter_data)
                                file.write(txt_pair)
                                break
                            elif word[0] in self.fir:
                                flag = 0
                                aa[3] += 1
                                fir_clause, last_clause = child_clause1, main_clause1
                                txt_pair = cal_and_save.embedding_txt(fir_clause, last_clause, filter_data)
                                file.write(txt_pair)
                                break
                            elif word[0] in self.las:
                                flag = 0
                                aa[4] += 1
                                fir_clause, last_clause = main_clause1, child_clause1
                                txt_pair = cal_and_save.embedding_txt(fir_clause, last_clause, filter_data)
                                file.write(txt_pair)

                                break
                            elif word[0] in self.las_cause:
                                flag = 0
                                aa[1] += 1
                                fir_clause, last_clause = main_clause1, child_clause1
                                txt_pair = cal_and_save.embedding_txt(fir_clause, last_clause, filter_data)

                                file.write(txt_pair)
                                break
                            elif word[0] in self.filter_fonction:
                                flag = 0
                                aa[2] += 1
                                break

                        if word[6] < child_root_id:
                            if word[0] in self.filter_fonction:
                                flag = 0
                                aa[2] += 1
                                break
                                # Extract_Advcl.pprint(word, main_clause1, child_clause1, causal_pair)
                            if word[0] in self.fir_cause or word[0] in self.fir:
                                pass

                            if word[0] in self.las or word[0] in self.las_cause:
                                pass

                    if flag:
                        aa[5] += 1
                        if child_root_id > main_root_id:
                            #
                            fir_clause, last_clause = main_clause1, child_clause1
                            txt_pair = cal_and_save.embedding_txt(fir_clause, last_clause, filter_data)

                            file.write(txt_pair)
                            pass
                        else:
                            # aa[4] += 1
                            fir_clause, last_clause = child_clause1, main_clause1
                            txt_pair = cal_and_save.embedding_txt(fir_clause, last_clause, filter_data)

                            file.write(txt_pair)

            # print('len_pair_count:', len(cal_and_save.pair_count))

        file.close()


class Extract_Conj(object):
    def __init__(self):
        self.conj_not = {'but', 'or', 'yet'}

    def extract_filter_conj(self):
        """
        规则1:如果连接词是but or,基本不会存在因果性,去除
        规则2:如果子句主句前后的时态不一致,按照时态规则进行排序
        规则3:默认主句和子句的先后顺序是,主句在前,子句在后
        :return:
        """
        print('extract_filter_conj')
        filter_data = FilterData()
        file_list = list(sorted(glob(os.path.join(input_path, '*'))))
        print(file_list)
        for file_name in file_list:
            spacy_list = list(sorted(glob(os.path.join(file_name, 'conj*.npy'))))
            # print(len(spacy_list), spacy_list[0])
            for number_file in tqdm(range(len(spacy_list))):
                clause_path = spacy_list[number_file]
                # for number, clause_path in enumerate(spacy_list):
                date = np.load(clause_path, allow_pickle=True)
                for clause_pair in date:
                    main_clause, child_clause = clause_pair
                    # 调整语序关系,让从句都在后面
                    if main_clause[len(main_clause) - 1][0] == 'ziju':
                        main_clause = main_clause[:-1]
                        main_clause, child_clause = child_clause, main_clause
                    else:
                        child_clause = child_clause[:-1]
                    child_root_id = child_clause[0][6]
                    main_root_id = main_clause[0][6]

                    causal_pair = main_clause + child_clause
                    causal_pair = sorted(causal_pair, key=functools.cmp_to_key(cmp))
                    main_clause1 = sorted(main_clause, key=functools.cmp_to_key(cmp))
                    child_clause1 = sorted(child_clause, key=functools.cmp_to_key(cmp))
                    child_fir_id_id = child_clause1[0][6]
                    str_sent1 = ' '.join([word[0] for word in main_clause1])
                    str_sent2 = ' '.join([word[0] for word in child_clause1])
                    if 'ust for seplit' in str_sent1 or 'ust for seplit' in str_sent2:
                        continue
                    aa[0] += 1

                    flag = 1
                    for word in main_clause1:
                        if (child_fir_id_id - word[6] > 0) and (child_fir_id_id - word[6] < 5) and (
                                word[0] in self.conj_not):
                            flag = 0
                            aa[2] += 1
                            break

                    if flag:
                        aa[1] += 1
                        if child_root_id > main_root_id:
                            aa[3] += 1
                            fir_clause, last_clause = main_clause1, child_clause1
                        else:
                            aa[4] += 1
                            fir_clause, last_clause = child_clause1, main_clause1

                        cal_and_save.cal_for_count(fir_clause, last_clause, filter_data)

                        main_clause_tense = get_sentence_tense(main_clause[0], main_clause1)

                        child_clause_tense = get_sentence_tense(child_clause[0], child_clause1)

                        if main_clause_tense != 0 and child_clause_tense != 0 and main_clause_tense != child_clause_tense:
                            if main_clause_tense > child_clause_tense:
                                aa[5] += 1
                                # print(main_clause_tense, child_clause_tense)
                                # print(main_clause1)
                                # print(child_clause1)
                                # print(' '.join([temp[0] for temp in main_clause1]))
                                # print(' '.join([temp[0] for temp in child_clause1]))
                                # print(' '.join([temp[0] for temp in causal_pair]))
                                # print('---')
                                main_clause, child_clause = child_clause, main_clause

            print('len_pair_count:', len(cal_and_save.pair_count))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('data', type=int, help="choose data")
    # # parser.add_argument('evidence', type=int, help="choose evidence kind")
    # args = parser.parse_args()
    # data_choose = args.data
    data_choose = 1
    # evidence_choose = args.evidence
    cause_effect_clause_pair = []
    time_before_after_pair = []
    clause_to_pair = []
    count1 = [0]
    word_dict = defaultdict(int)
    word_pos_dict = defaultdict(int)
    tense_list = defaultdict(int)

    aa = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    evidence_choose = 0
    evidence_kinds = ['advcl', 'conj', 'inter']
    evidence = evidence_kinds[evidence_choose]

    data_path = '../../data/'
    # data_path = '../../../../data_disk/ray/data/'
    input_file = 'data_clause/'
    output_file = 'data_pair_count/'
    data_list = ['icw/', 'bok/', 'gut/']
    input_path = data_path + input_file + data_list[data_choose]
    output_path = data_path + output_file + data_list[data_choose]
    print(evidence, data_list[data_choose])
    output_pair_count = output_path + evidence + '_output_pair_count.txt'
    output_fir_word_count = output_path + evidence + '_output_fir_word_count.txt'
    output_las_word_count = output_path + evidence + '_output_las_word_count.txt'
    output_total_word_count = output_path + evidence + '_output_total_word_count.txt'
    print(output_pair_count)

    cal_and_save = CalAndSave(output_pair_count, output_fir_word_count, output_las_word_count, output_total_word_count)

    if evidence == 'advcl':
        extract_advcl = Extract_Advcl()
        # Extract_Advcl.extract_text_clause()
        # Extract_Advcl.extract_text_time()
        # Extract_Advcl.extract_aux_to()
        # extract_advcl.extract_filter_advcl()
        extract_advcl.extract_get_word_embedding_text()
    elif evidence == 'conj':
        extract_conj = Extract_Conj()
        extract_conj.extract_filter_conj()

    # cal_and_save.save_date()

    # for i in range(len(file_list)):
    #     extract_text_clause(file_list[i])

    word_dict1 = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
    print(len(word_dict1), word_dict1)
    word_pos_dict1 = sorted(word_pos_dict.items(), key=lambda d: d[1], reverse=True)
    print(len(word_pos_dict1), word_pos_dict1)
    print(aa)

    print_time()
    print(tense_list)

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
# aux 除了to 都是VERB
# nsubj 都是代词
# advcl 基本都是VERB
#     dep_list = {'mark', 'aux', 'advmod'}
#     pos_list = {'PART', 'ADP', 'CCONJ', 'ADV'}

# {'NUM', 'X', 'PART', 'INTJ', 'PRON', 'ADV', 'SYM', 'ADP', 'SPACE', 'CCONJ', 'VERB', 'PROPN', 'DET', 'ADJ', 'PUNCT','NOUN'}
#  数字   其他 粒子-介词  感叹    代词    副词     符号   介词    空格       连接词     动词     专有名词   确定器  形容词   标点符号  名词 过滤: 'X',
# 'NUM', PRON, SYM SPACE DET PUNCT 功能词: PART, ADP CCONJ 内容词: INTJ, ADV VERB PROPN ADJ, 'NOUN

#
# [('which_nsubj', 10635), ('that_nsubj', 7373), ('to_aux', 175507), ('when_advmod', 154204), ('as_mark', 120624),
#  ('if_mark', 89488), ('because_mark', 82301), ('while_mark', 41121), ('so_mark', 33849), ('since_mark', 29909),
#  ('before_mark', 26739), ('like_mark', 24392), ('for_mark', 20176), ('until_mark', 19282), ('even_advmod', 17094),
#  ('after_mark', 17008), ('than_mark', 16202), ('just_advmod', 12725), ('where_advmod', 10033), ('although_mark', 9922),
#  ('though_mark', 9434), ('once_mark', 8137), ('so_advmod', 7446), ('only_advmod', 3986), ('unless_mark', 3526),
#  ('whenever_advmod', 2902), ('that_mark', 2118), ('then_advmod', 2117), ('cause_mark', 1857), ('till_mark', 1763),
#  ('still_advmod', 1216), ('whether_mark', 1121), ('why_advmod', 1063), ('once_advmod', 981), ('ever_advmod', 940),
#  ('cuz_mark', 933), ('with_mark', 911), ('how_advmod', 895), ('now_advmod', 833), ('except_mark', 699),
#  ("'cause_mark", 654), ('whilst_mark', 616), ('almost_advmod', 584), ('whereas_mark', 582), ('wherever_advmod', 567),
#  ('right_advmod', 518), ('soon_advmod', 508), ('later_advmod', 493), ('thus_advmod', 490), ('long_advmod', 475),
#  ('also_advmod', 471), ('always_advmod', 457), ('however_advmod', 408), ('already_advmod', 297), ('of_advmod', 282),
#  ('sometimes_advmod', 277), ('again_advmod', 243), ('instead_advmod', 241), ('often_advmod', 217),
#  ('usually_advmod', 206), ('much_advmod', 204), ('lest_mark', 203), ('perhaps_advmod', 199), ('in_mark', 198),
#  ('well_advmod', 190), ('more_advmod', 180), ('first_advmod', 176), ('til_mark', 172), ('as_advmod', 170),
#  ('thereby_advmod', 167), ('ago_advmod', 143), ('far_advmod', 141), ('somehow_advmod', 127), ('otherwise_advmod', 126),
#  ('anywhere_advmod', 117), ('anytime_mark', 107), ('at_mark', 100)]
