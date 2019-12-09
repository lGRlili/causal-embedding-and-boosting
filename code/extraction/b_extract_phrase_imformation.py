import argparse
import codecs
import multiprocessing
import pickle
import pandas as pd
import numpy as np
from glob import glob
import os
import sys
from tqdm import tqdm

sys.path.extend(['../'])
from datetime import datetime
from API.extraction import *
from collections import defaultdict
starts = datetime.now()
"""
对spacy抽取得到的句子进行解析---得到advcl和conj的子句关系
"""


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


def extract_text_phrase(file_name, output_dir):
    print('extract_text_phrase')
    file_id = file_name.split('/')[-1]
    output_id = output_dir + '/' + file_id
    isExists = os.path.exists(output_id)
    if isExists:
        print('路径', output_id, '已经存在')
    else:
        os.mkdir(output_id)
        print('路径', output_id, '创建成功')
    spacy_list = list(sorted(glob(os.path.join(file_name, '*.pkl'))))
    for number in tqdm(range(len(spacy_list))):
        spacy_path = spacy_list[number]
    # for number, spacy_path in enumerate(spacy_list):
        book_name = spacy_path.split('/')
        book_name = book_name[-1]
        book_name = book_name[:len(book_name) - 4]
        advcl_out_put_path = output_id + '/advcl_' + book_name + ".npy"
        conj_out_put_path = output_id + '/conj_' + book_name + ".npy"
        advcl_clause = []
        conj_clause = []
        print(spacy_path)
        date = pd.read_pickle(spacy_path)
        doc = date['doc']

        if 'icwc' in file_name:
            for book in doc:
                # 此处是一个文本
                last_setence = []
                last_congju = []
                for sen in book:

                    print('解析当前句子')
                    print(' '.join([word.text for word in sen]))

                    total_clause, total_clause_level, len_total_congju = parse_setence(sen)
                    advcl_clause_list = get_intra_advcl(sen, total_clause_level)
                    conj_clause_list = get_intra_conj(sen, total_clause_level)
                    for clause in advcl_clause_list:
                        advcl_clause.append(clause)
                    for clause in conj_clause_list:
                        conj_clause.append(clause)

                    # count_inter, inter_level_p_a_b_pos, last_setence, last_congju = \
                    #     cal_p_a_b_inter_num_weight_dis_stop_with(
                    #         sen, total_clause, inter_level_p_a_b_pos, str_id_word_count, last_setence, last_congju,
                    #         count_inter, stopkey)
                    # count_inter, inter_p_a_b_pos_weight, last_setence, last_congju = \
                    #     cal_p_a_b_inter_stop_with(
                    #         sen, total_clause, inter_p_a_b_pos_weight, str_id_word_count, last_setence, last_congju,
                    #         count_inter, stopkey)
        else:
            last_setence = []
            last_congju = []
            for sen_with2 in doc:
                sen = [sentence[0] for sentence in sen_with2]
                print('解析当前句子')
                print(' '.join([word.text for word in sen]))
                total_clause, total_clause_level, len_total_congju = parse_setence(sen)
                advcl_clause_list = get_intra_advcl(sen, total_clause_level)
                conj_clause_list = get_intra_conj(sen, total_clause_level)
                for clause in advcl_clause_list:
                    advcl_clause.append(clause)
                for clause in conj_clause_list:
                    conj_clause.append(clause)

        np.save(advcl_out_put_path, advcl_clause)
        np.save(conj_out_put_path, conj_clause)
        # sys.stderr.write("\rFinished %6d / %6d.\n" % (number, len(spacy_list)))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('data', type=int, help="choose data")
    # parser.add_argument('evidence', type=int, help="choose evidence kind")
    # args = parser.parse_args()
    # data_choose = args.data
    # evidence_choose = args.evidence
    verb_dep_list = defaultdict(int)

    data_path = '../../data/'
    # data_path = '../../../../data_disk/ray/data/'
    input_list = ['data_spacy/icw', 'data_spacy/bok', 'data_spacy/gut']

    output_list = ['data_clause/icw', 'data_clause/bok', 'data_clause/gut']
    input_list = ['data_repickle/icw', 'data_repickle/bok', 'data_repickle/gut']

    evidence_kinds = ['advcl', 'conj', 'inter']

    data_choose = 0
    evidence_choose = 0
    input_path = data_path + input_list[data_choose]
    output_path = data_path + output_list[data_choose]
    evidence = evidence_kinds[evidence_choose]
    file_list = list(sorted(glob(os.path.join(input_path, '*'))))
    # file_list = file_list[:1]
    print(file_list)
    lock = multiprocessing.Lock()
    # 创建多个进程
    thread_list = []
    len_file_list = len(file_list)
    for choose_file_list in range(len_file_list):
        sthread = multiprocessing.Process(target=extract_text_phrase, args=(file_list[choose_file_list], output_path))
        thread_list.append(sthread)
    for th in thread_list:
        th.start()
    for th in thread_list:
        th.join()


# {'NUM', 'X', 'PART', 'INTJ', 'PRON', 'ADV', 'SYM', 'ADP', 'SPACE', 'CCONJ', 'VERB', 'PROPN', 'DET', 'ADJ', 'PUNCT', 'NOUN'}
#   数字   其他 粒子-介词  感叹    代词    副词     符号   介词    空格       连接词     动词     专有名词   确定器  形容词   标点符号  名词
# 过滤: 'X','NUM', PRON, SYM SPACE DET PUNCT
# 功能词: PART, ADP CCONJ
# 内容词: INTJ, ADV VERB PROPN ADJ, 'NOUN

# ADV---其中有because等

# 因果关系可能引导的:其中有动词, adp, adv
# ADP  ADV

# tense = {}
#
# tense['present'] = ['VB', 'VBP', 'VBZ', 'VBG']
# # VBG 看前一个词,
# tense['future'] = ['MD'] & 'will' 'shall'
# tense['pase'] = ['VBD', 'VBN']
# tense['past_future'] = ['MD']& 'would' ''

# 找到主要的动词,然后查找他的child,
# 如果存在will 就是将来,表示4,如果有would,那就是过去将来,是2,
# 如果有had was,就是过去1, 有have is are 就是现在3, 其他的就是root本身的词性

# VBN:被动
# :进行时 ing　　
# MD:情态动词
#　

# 连词存在问题,