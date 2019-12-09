import pickle
import numpy as np
from glob import glob
import os
from datetime import datetime
from nltk.corpus import stopwords
from collections import defaultdict
import functools

starts = datetime.now()


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


def filter_now(cause_clause, effect_clause, filter_kind):
    cause_clause = filter(filter_kind, cause_clause)
    effect_clause = filter(filter_kind, effect_clause)
    cause_clause, effect_clause = list(cause_clause), list(effect_clause)
    return cause_clause, effect_clause


def save_date():
    with open(output_cause_effect_pair_count, 'w') as file:
        for cause in cause_effect_pair_count:
            for effect in cause_effect_pair_count[cause]:
                file.write('{} {} {} \n'.format(cause, effect, cause_effect_pair_count[cause][effect]))
    with open(output_cause_word_count, 'w') as file:
        for cause in cause_word_count:
            file.write('{} {}  \n'.format(cause, cause_word_count[cause]))
    with open(output_effect_word_count, 'w') as file:
        for effect in effect_word_count:
            file.write('{} {}  \n'.format(effect, effect_word_count[effect]))
    print(len(cause_word_count), len(effect_word_count))
    print('finish')


def cal_count(evidence):
    file_list = list(sorted(glob(os.path.join(input_path, evidence + '*.npy'))))
    print(file_list)
    for cause_effect_pair_path in file_list:
        print(cause_effect_pair_path)
        cause_effect_list = np.load(cause_effect_pair_path, allow_pickle=True)
        # print(cause_effect_list)
        for cause_effect_pair in cause_effect_list:
            cause_clause, effect_clause = cause_effect_pair

            # print(len(cause_clause), len(effect_clause))
            cause_clause = sorted(cause_clause, key=functools.cmp_to_key(cmp))
            effect_clause = sorted(effect_clause, key=functools.cmp_to_key(cmp))

            print(' '.join([word[0] for word in cause_clause]))
            print(' '.join([word[0] for word in effect_clause]))
            # 过滤其他类型
            if filter_pos_other_flag:
                cause_clause, effect_clause = filter_now(cause_clause, effect_clause, filter_pos_other)

            # 过滤停用词
            if filter_stop_word_flag:
                cause_clause, effect_clause = filter_now(cause_clause, effect_clause, filter_stop_word)

            # 　过滤含有其他字符的词
            if filter_isalpha_flag:
                cause_clause, effect_clause = filter_now(cause_clause, effect_clause, filter_isalpha)

            # 过滤第三类成分词
            if filter_pos_not_function_or_content_flag:
                cause_clause, effect_clause = filter_now(cause_clause, effect_clause, filter_pos_not_function_or_content)

            # 过滤功能词
            if filter_pos_function_flag:
                cause_clause, effect_clause = filter_now(cause_clause, effect_clause, filter_pos_function)
            for word in cause_clause:
                if word[1] in pos_other:
                    # print(word[0])
                    word_list.add(word[0])
            for word in effect_clause:
                if word[1] in pos_other:
                    # print(word[0])
                    word_list.add(word[0])
                    print(word_list)
                    print(len(word_list))

            for cause_word in cause_clause:
                for effect_word in effect_clause:
                    cause_w = cause_word[0].lower()
                    effect_w = effect_word[0].lower()
                    cause_effect_pair_count[cause_w][effect_w] += 1 / (len(cause_clause) * len(effect_clause))
            # cause_clause = sorted(cause_clause, key=functools.cmp_to_key(cmp))
            # effect_clause = sorted(effect_clause, key=functools.cmp_to_key(cmp))
            print('过滤')
            print(' '.join([word[0] for word in cause_clause]))
            print(' '.join([word[0] for word in effect_clause]))
            print('---')

            total_count['total_cause_effect'] += 1
            for cause_word in cause_clause:
                cause_w = cause_word[0].lower()
                cause_word_count[cause_w] += 1 / len(cause_clause)
            total_count['total_cause'] += 1
            for effect_word in effect_clause:
                effect_w = effect_word[0].lower()
                effect_word_count[effect_w] += 1 / len(effect_clause)
            total_count['total_effect'] += 1

        print(len(cause_word_count), len(effect_word_count))


# 过滤: 'X','NUM', PRON, SYM SPACE DET PUNCT
# 功能词: PART, ADP CCONJ
# 内容词: INTJ, ADV VERB PROPN ADJ
# 其中:过滤掉其他词
# 过滤掉包含其他字符的词语
# 过滤掉停用词
# 划分出功能词和内容词

"""
此处保存的是,只用内容词计算得到的PMI
"""
if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('data', type=int, help="choose data")
    # parser.add_argument('evidence', type=int, help="choose evidence kind")
    # args = parser.parse_args()
    # data_choose = args.data
    # evidence_choose = args.evidence
    total_count = {'total_cause': 0, 'total_effect': 0, 'total_cause_effect': 0}
    word_list = set()
    filter_pos_other_flag = 1
    filter_stop_word_flag = 1
    filter_isalpha_flag = 1
    filter_pos_not_function_or_content_flag = 1
    filter_pos_function_flag = 1
    merge_flag = 1
    pos_filter = {'X', 'NUM', 'PRON', 'SYM', 'SPACE', 'DET', 'PUNCT'}
    pos_function = {'PART', 'ADP', 'CCONJ'}
    pos_content = {'VERB', 'ADJ', 'NOUN'}
    pos_other = {'INTJ', 'ADV', 'PROPN', }

    with open('stop_word.txt') as file:
        eng_stop_word = file.readlines()
        eng_stop_word = set([word.strip() for word in eng_stop_word])
    print(eng_stop_word)

    # eng_stop_word = stopwords.words('english')
    # eng_stop_word = set(eng_stop_word)
    # with open('stop_word.txt', 'w') as file:
    #     for word in eng_stop_word:
    #         file.write(word + '\n')
    cause_effect_pair_count = defaultdict(lambda: defaultdict(int))
    cause_word_count = defaultdict(int)
    effect_word_count = defaultdict(int)
    count1 = [0]
    word_dict = defaultdict(int)
    evidence_list = ['', 'cause_', 'time_', 'to_']
    input_list = ['data_cause_effect/icw', 'data_cause_effect/bok', 'data_cause_effect/gut']
    output_list = ['data_pair_count1/icw', 'data_pair_count1/bok', 'data_pair_count1/gut']

    evidence_choose = 1
    data_choose = 1

    data_path = '../../data/'
    str_id_path = data_path + 'extraction/' + 'str_id_word' + '.file'

    with open(str_id_path, 'rb') as f:
        str_id_word = pickle.load(f)
    if merge_flag:
        for i in range(3):
            data_choose = i

            input_path = data_path + input_list[data_choose]
            cal_count(evidence_list[1])
            # cal_count(evidence_list[2])
        output_cause_effect_pair_count = data_path + '/data_pair_count1/' + evidence_list[1] + evidence_list[2] + 'cause_effect_pair_count.txt'
        output_cause_word_count = data_path + '/data_pair_count1/' + evidence_list[1] + evidence_list[2] + 'cause_word_count.txt'
        output_effect_word_count = data_path + '/data_pair_count1/' + evidence_list[1] + evidence_list[2] + 'effect_word_count.txt'
        # for i in range(3):
        #     data_choose = i
        #
        #     input_path = data_path + input_list[data_choose]
        #     cal_count(evidence_list[evidence_choose])
        # output_cause_effect_pair_count = data_path + '/data_pair_count1/' + evidence_list[
        #     evidence_choose] + 'cause_effect_pair_count.txt'
        # output_cause_word_count = data_path + '/data_pair_count1/' + evidence_list[
        #     evidence_choose] + 'cause_word_count.txt'
        # output_effect_word_count = data_path + '/data_pair_count1/' + evidence_list[
        #     evidence_choose] + 'effect_word_count.txt'
    else:
        input_path = data_path + input_list[data_choose]
        output_path = data_path + output_list[data_choose]
        cal_count(evidence_list[evidence_choose])
        output_cause_effect_pair_count = output_path + '/' + evidence_list[
            evidence_choose] + 'cause_effect_pair_count.txt'
        output_cause_word_count = output_path + '/' + evidence_list[evidence_choose] + 'cause_word_count.txt'
        output_effect_word_count = output_path + '/' + evidence_list[evidence_choose] + 'effect_word_count.txt'

    # save_date()
    print(total_count)
    print_time()

# icw 共 13727556
# bok 共 16742066
# gut 共 46462020
# 总计 64340264

# 数据大小:
# cause:2.0G + 2.5G + 5.5G   time:0.7G + 0.9G + 1.6G   to: 1.5G + 2.2G + 6.8G
#             10G                       3.2G                   10.5G