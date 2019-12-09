import argparse
import codecs
import multiprocessing
import pickle
import pandas as pd
import numpy as np
from glob import glob
import os
import re
import string
import sys
from datetime import datetime
from API.extraction import *
from collections import defaultdict
starts = datetime.now()
"""
观察分析数据
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


def extract_text_clause(file_name):
    print('extract_text_phrase')

    mark_list = set()
    advmod_list = set()
    mark_count = 0
    advmod_count = 0

    spacy_list = list(sorted(glob(os.path.join(file_name, 'advcl*.npy'))))
    for number, clause_path in enumerate(spacy_list):

        advcl_clause = []
        conj_clause = []

        date = np.load(clause_path, allow_pickle=True)
        for clause_pair in date:
            zhuju, ziju = clause_pair
            zhuju = sorted(zhuju, key=functools.cmp_to_key(cmp))
            ziju = sorted(ziju, key=functools.cmp_to_key(cmp))

            zhu_txt = [word[0]for word in zhuju]
            zhu_pos = [word[1] for word in zhuju]
            zhu_dep = [word[2] for word in zhuju]
            zi_txt = [word[0]for word in ziju]
            zi_pos = [word[1] for word in ziju]
            zi_dep = [word[2] for word in ziju]
            if zhuju[len(zhuju)-1][0] == 'ziju':

                word_id = get_word_id(zhu_txt)
                zhu_txt = zhu_txt[word_id:-1]
                zhu_dep = zhu_dep[word_id:-1]

                if len(zhu_txt) < 1:
                    continue
                print('zhu_txt:', ' '.join(zhu_txt[:3]))
                print('zhu_dep:', ' '.join(zhu_dep[:3]))
                # print(' '.join(zhu_txt), ':', ' '.join(zi_txt))
                word, dep = ''.join(re.split(r'[^a-z]', zhu_txt[0])), zhu_dep[0]
                print(word, dep)
                if dep == 'det':
                    dep_dict[dep] += 1
                    word_dict[word] += 1
                    word_dep_dict[word + ' ' + dep] += 1
            else:

                word_id = get_word_id(zi_txt)
                zi_txt = zi_txt[word_id:-1]
                zi_dep = zi_dep[word_id:-1]

                if len(zi_txt) < 1:
                    continue
                print('zi_txt:', ' '.join(zi_txt))
                print('zi_dep:', ' '.join(zi_dep))
                # print(' '.join(zhu_txt), ':', ' '.join(zi_txt))
                word, dep = ''.join(re.split(r'[^a-z]', zi_txt[0])), zi_dep[0]
                print(word, dep)
                if dep == 'det':
                    dep_dict[dep] += 1
                    word_dict[word] += 1
                    word_dep_dict[word + ' ' + dep] += 1

            # print(' '.join(zhu_txt))
            # print(' '.join(zhu_pos))
            # print(' '.join(zi_txt))
            # print(' '.join(zi_pos))
            # print('---' * 45)

        print(clause_path)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('data', type=int, help="choose data")
    # parser.add_argument('evidence', type=int, help="choose evidence kind")
    # args = parser.parse_args()
    # data_choose = args.data
    # evidence_choose = args.evidence
    count1 = [0, 0]
    trantab = str.maketrans({key: None for key in string.punctuation})
    data_path = '../../data/'
    input_list = ['data_clause/icw', 'data_clause/bok', 'data_clause/gut']
    evidence_kinds = ['advcl', 'conj', 'inter']
    dep_dict = defaultdict(int)
    word_dict = defaultdict(int)
    word_dep_dict = defaultdict(int)

    data_choose = 0
    evidence_choose = 0
    input_path = data_path + input_list[data_choose]
    evidence = evidence_kinds[evidence_choose]
    file_list = list(sorted(glob(os.path.join(input_path, '*'))))
    print(file_list)
    for i in range(len(file_list)):
        extract_text_clause(file_list[i])
    word_dict1 = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
    pos_dict1 = sorted(dep_dict.items(), key=lambda d: d[1], reverse=True)
    word_pos_dict1 = sorted(word_dep_dict.items(), key=lambda d: d[1], reverse=True)
    print(len(word_dict1), word_dict1)
    print(len(pos_dict1), pos_dict1)
    print(len(word_pos_dict1), word_pos_dict1)
    print(count1)

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


# {'NUM', 'X', 'PART', 'INTJ', 'PRON', 'ADV', 'SYM', 'ADP', 'SPACE', 'CCONJ', 'VERB', 'PROPN', 'DET', 'ADJ', 'PUNCT', 'NOUN'}
#   数字   其他 粒子-介词  感叹    代词    副词     符号   介词    空格       连接词     动词     专有名词   确定器  形容词   标点符号  名词
# 过滤: 'X','NUM', PRON, SYM SPACE DET PUNCT
# 功能词: PART, ADP CCONJ
# 内容词: INTJ, ADV VERB PROPN ADJ

# ADV---其中有because等

# 因果关系可能引导的:其中有动词, adp, adv
# ADP  ADV