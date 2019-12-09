# lstm:思路如下
# 构建全局的字典,将单词转换成为id
# 首先:生成负样本:生成方法如下:用在正例中随机采样句子,采样得到的非正确的都是负例
# 使用lstm 或者cnn后者fasttext 将句子编码成为向量,然后cause effect两个向量做cos内积,后面接一个全连接层,输出label, 观察模型的acc是多少

import numpy as np
import tensorflow as tf
import pickle
import math
import pickle
import numpy as np
from glob import glob
import os
import random
from datetime import datetime
from collections import defaultdict
from nltk.corpus import stopwords
from collections import defaultdict
import functools

starts = datetime.now()


"""
针对内容分类器;
用于生成因果分类器的正负样例
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


# 过滤: 'X','NUM', PRON, SYM SPACE DET PUNCT
# 功能词: PART, ADP CCONJ
# 内容词: INTJ, ADV VERB PROPN ADJ

class Generate_Data(object):

    def __init__(self):
        self.filter_pos_other_flag = 1
        self.filter_stop_word_flag = 1
        self.filter_isalpha_flag = 1
        self.filter_pos_not_function_or_content_flag = 1
        self.filter_pos_function_flag = 1
        self.pos_filter = {'X', 'NUM', 'PRON', 'SYM', 'SPACE', 'DET', 'PUNCT'}
        self.pos_function = {'PART', 'ADP', 'CCONJ'}
        self.pos_content = {'VERB', 'ADJ', 'NOUN'}
        self.pos_other = {'INTJ', 'ADV', 'PROPN', }
        with open('stop_word.txt') as file:
            eng_stop_word = file.readlines()
            self.eng_stop_word = set([word.strip() for word in eng_stop_word])
        print(eng_stop_word)

        self.evidence_list = ['', 'cause_', 'time_', 'to_']
        self.input_list = ['data_cause_effect/icw', 'data_cause_effect/bok', 'data_cause_effect/gut']
        self.output_list = ['classifier_data/content_classify/icw', 'classifier_data/content_classify/bok', 'classifier_data/content_classify/gut']

        self.evidence_choose = 1
        self.data_choose = 1

        self.data_path = '../../data/'
        str_id_path = self.data_path + 'extraction/' + 'str_id_word' + '.file'
        with open(str_id_path, 'rb') as file:
            self.str_id_word = pickle.load(file)
        self.pos_function_dict = defaultdict(int)

    def generate_data(self, evidence, input_path, output_path):
        def cmp(x, y):
            if x[3] > y[3]:
                return 1
            if x[3] < y[3]:
                return -1
            else:
                return 0

        def filter_pos_other(word):
            return word[1] not in self.pos_filter

        def filter_stop_word(word):
            return word[0] not in self.eng_stop_word

        def filter_isalpha(word):
            return word[0].encode('UTF-8').isalpha()

        def filter_pos_function(word):
            return word[1] not in self.pos_function

        def filter_pos_not_function_or_content(word):
            return word[1] not in self.pos_other

        file_list = list(sorted(glob(os.path.join(input_path, evidence + '*.npy'))))
        print(file_list)
        for cause_effect_pair_path in file_list:
            now_file_name = cause_effect_pair_path.split('/')[-1]
            True_data = []
            False_data = []
            print(cause_effect_pair_path)
            cause_effect_list = np.load(cause_effect_pair_path, allow_pickle=True)
            # print(cause_effect_list)
            for cause_effect_pair in cause_effect_list:
                cause_clause, effect_clause = cause_effect_pair
                cause_clause = sorted(cause_clause, key=functools.cmp_to_key(cmp))
                effect_clause = sorted(effect_clause, key=functools.cmp_to_key(cmp))

                if self.filter_pos_other_flag:
                    cause_clause = filter(filter_pos_other, cause_clause)
                    effect_clause = filter(filter_pos_other, effect_clause)
                    cause_clause, effect_clause = list(cause_clause), list(effect_clause)
                if self.filter_stop_word_flag:
                    cause_clause = filter(filter_stop_word, cause_clause)
                    effect_clause = filter(filter_stop_word, effect_clause)
                    cause_clause, effect_clause = list(cause_clause), list(effect_clause)
                if self.filter_isalpha_flag:
                    cause_clause = filter(filter_isalpha, cause_clause)
                    effect_clause = filter(filter_isalpha, effect_clause)
                    cause_clause, effect_clause = list(cause_clause), list(effect_clause)
                if self.filter_pos_not_function_or_content_flag:
                    cause_clause = filter(filter_pos_not_function_or_content, cause_clause)
                    effect_clause = filter(filter_pos_not_function_or_content, effect_clause)
                    cause_clause, effect_clause = list(cause_clause), list(effect_clause)
                for word in cause_clause:
                    if word[1] in self.pos_function:
                        self.pos_function_dict[word[0].low()] += 1
                for word in effect_clause:
                    if word[1] in self.pos_function:
                        self.pos_function_dict[word[0].low()] += 1
                if self.filter_pos_function_flag:
                    cause_clause = filter(filter_pos_function, cause_clause)
                    effect_clause = filter(filter_pos_function, effect_clause)
                    cause_clause, effect_clause = list(cause_clause), list(effect_clause)
                cause_clause = [word[0] for word in cause_clause]
                effect_clause = [word[0] for word in effect_clause]
                True_data.append([cause_clause, effect_clause])
            len_true_data = len(True_data)
            for cause_id in range(len_true_data):
                effect_id = random.randint(0, len_true_data-1)
                false_cause = True_data[cause_id][0]
                false_effect = True_data[effect_id][1]
                False_data.append([false_cause, false_effect])
            save_path = output_path + '/' + 'True_data' + now_file_name
            np.save(save_path, True_data)
            save_path = output_path + '/' + 'False_data' + now_file_name
            np.save(save_path, False_data)
            print('保存完成')

    def main(self):
        for i in range(3):
            data_choose = i
            input_path = self.data_path + self.input_list[data_choose]
            output_path = self.data_path + self.output_list[data_choose]

            # self.generate_data(self.evidence_list[1], input_path, output_path)
            # self.generate_data(self.evidence_list[2], input_path, output_path)
            self.generate_data(self.evidence_list[3], input_path, output_path)
            print(self.pos_function_dict)
            print(len(self.pos_function_dict))
            print_time()


Generate_Data().main()

