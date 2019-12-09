import argparse
import codecs
import multiprocessing
import pickle
import pandas as pd
import numpy as np
from glob import glob
import os
import sys

sys.path.extend(['../'])
from reference.base import FilterData
from extraction.c_extraction_for_advcl import CalAndSave
from extraction.c_extraction_for_advcl import get_sentence_tense
from datetime import datetime
from API.extraction import *
from reference.base import FilterData
from collections import defaultdict
from tqdm import tqdm

starts = datetime.now()
"""
对spacy抽取得到的句子进行解析---得到advcl和conj的子句关系
"""


class parse(object):
    # 保存数据的数据结构
    def __init__(self, text, norm, lemma_, pos_, tag_, dep_, head, id, child, left, right, ancestor):
        self.text = text
        self.norm = norm
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.tag_ = tag_
        self.dep_ = dep_
        self.head = head
        self.id = id
        self.child = child
        self.left = left
        self.right = right
        self.ancestor = ancestor


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


# def get_sentence_tense(verb_word, clause):
#     tense_present_past = {'VBD': 1, 'VBN': 1, 'VB': 3, 'VBP': 3, 'VBZ': 3, 'VBG': 3, 'MD': 4}
#     if verb_word.pos_ == 'VERB':
#         clause_tense = tense_present_past[verb_word.tag_]
#         for word in clause:
#
#             if word.id < verb_word.id and word.id in verb_word.child and word.pos_ == 'VERB':
#                 # tense_list[word[0]] += 1
#                 if word.text in {'’ve', 'is', 'are', '’m', '’s', 'have', 'has', "'ve", 'do', 'does', 'be'}:
#                     clause_tense = 3
#                     break
#                 if word.text in {'’ll', 'will', "'ll", 'shall'}:
#                     clause_tense = 4
#                     break
#                 if word.text in {'would'}:
#                     clause_tense = 2
#                     break
#                 if word.text in {'was', 'did', 'had', 'were', 'been'}:
#                     clause_tense = 1
#                     break
#                 if word.text in {'might', 'could', 'should'}:
#                     clause_tense = 0
#                     break
#                 pass
#     else:
#         clause_tense = 0
#     return clause_tense

class Extract_Inter(object):

    def __init__(self):
        pass

    def extract_text_phrase(self, file_name, output_dir, choose_file):
        aa = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        tense_list = defaultdict(int)
        output_pair_count = output_dir + '_output_pair_count_' + str(choose_file) + '.txt'
        output_fir_word_count = output_dir + '_output_fir_word_count_' + str(choose_file) + '.txt'
        output_las_word_count = output_dir + '_output_las_word_count_' + str(choose_file) + '.txt'
        output_total_word_count = output_dir + '_output_total_word_count_' + str(choose_file) + '.txt'

        cal_and_save = CalAndSave(output_pair_count, output_fir_word_count, output_las_word_count, output_total_word_count)

        print('extract_filter_inter')
        filter_data = FilterData()
        spacy_list = list(sorted(glob(os.path.join(file_name, '*.pkl'))))
        print('len_spacy_list:', len(spacy_list))
        for number in tqdm(range(len(spacy_list[:10]))):
            spacy_path = spacy_list[number]

            print(spacy_path)
            date = pd.read_pickle(spacy_path)
            doc = date['doc']

            last_setence = []
            last_sentence_tense = 0
            for sen in doc:
                sen = sen[0]
                # sen = [sentence[0] for sentence in sen]
                aa[0] += 1
                # print('解析当前句子')
                str_sent = ' '.join([word.text for word in sen])
                print(str_sent)
                if 'ust for seplit' in str_sent:
                    last_setence = []
                    last_sentence_tense = 0
                    # print(str_sent)
                    continue
                # print()
                now_sentence = []
                for temp_node in sen:
                    # 之前没有最小化,后续应该最小化单词
                    word_text = temp_node.text.lower()
                    word_lemma_ = temp_node.lemma_
                    word_tag_ = temp_node.tag_
                    word_norm = temp_node.norm
                    word_pos = temp_node.pos_
                    word_dep = temp_node.dep_
                    word_id = temp_node.id
                    word_head_id = temp_node.head
                    word_child_id = temp_node.child
                    word = [word_text, word_norm, word_lemma_, word_pos, word_tag_, word_dep, word_id, word_head_id,
                            word_child_id]
                    now_sentence.append(word)

                for token in now_sentence:
                    if token[5] == 'ROOT':
                        root_node = token

                now_sentence_tense = get_sentence_tense(root_node, now_sentence)

                if len(last_setence):
                    fir_setence, las_sentence = last_setence, now_sentence
                    if last_sentence_tense != 0 and now_sentence_tense != 0 and last_sentence_tense != now_sentence_tense:
                        if last_sentence_tense > now_sentence_tense:
                            aa[1] += 1
                            tense_list[str(last_sentence_tense)+str(now_sentence_tense)] += 1
                            # print(last_sentence_tense, now_sentence_tense)
                            # print(last_setence)
                            # print(now_sentence)
                            # print(' '.join([temp[0] for temp in last_setence]))
                            # print(' '.join([temp[0] for temp in now_sentence]))
                            # print('---')
                            fir_setence, las_sentence = now_sentence, last_setence
                        else:
                            tense_list[str(last_sentence_tense) + str(now_sentence_tense)] += 1
                            aa[2] += 1
                            fir_setence, las_sentence = last_setence, now_sentence
                    else:

                        aa[3] += 1
                        fir_setence, las_sentence = last_setence, now_sentence

                    cal_and_save.cal_for_count(fir_setence, las_sentence, filter_data)

                last_setence = now_sentence
                last_sentence_tense = now_sentence_tense

            # sys.stderr.write("\rFinished %6d / %6d.\n" % (number, len(spacy_list)))

        cal_and_save.save_date()
        print(cal_and_save.count)
        print(aa)
        print(tense_list)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('data', type=int, help="choose data")
    # parser.add_argument('evidence', type=int, help="choose evidence kind")
    # args = parser.parse_args()
    # data_choose = args.data
    # evidence_choose = args.evidence

    extract_inter = Extract_Inter()
    evidence_choose = 2
    evidence_kinds = ['advcl', 'conj', 'inter']
    evidence = evidence_kinds[evidence_choose]
    print(evidence)
    data_choose = 0
    data_path = '../../data/'
    # data_path = '../../../../data_disk/ray/data/'
    input_file = 'data_repickle/'
    output_file = 'data_pair_count/'
    data_list = ['icw/', 'bok/', 'gut/']

    input_path = data_path + input_file + data_list[data_choose]
    output_path = data_path + output_file + data_list[data_choose]

    file_list = list(sorted(glob(os.path.join(input_path, '*'))))
    print(file_list)
    thread_list = []
    len_file_list = len(file_list)
    for choose_file_list in range(len_file_list):
        sthread = multiprocessing.Process(target=extract_inter.extract_text_phrase,
                                          args=(file_list[choose_file_list], output_path + evidence, choose_file_list))
        thread_list.append(sthread)
    for th in thread_list:
        th.start()
    for th in thread_list:
        th.join()

