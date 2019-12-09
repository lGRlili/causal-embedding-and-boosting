import pickle
import math
import copy
from datetime import datetime
from xml.dom.minidom import parse
import xml.dom.minidom
import codecs
from collections import defaultdict
import numpy as np
import functools
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from reference.test_for_COPA import Cruve_PR
from reference.base import FilterData
import json
import time


starts = datetime.now()


def cmp(x, y):
    # 用来调整顺序
    if x[0] > y[0]:
        return -1
    if x[0] < y[0]:
        return 1
    return 0


def print_tiem():
    # 用来打印时间
    ends = datetime.now()
    detla_time = ends - starts
    seconds = detla_time.seconds
    second = seconds % 60
    minitus = int(seconds / 60)
    minitus = minitus % 60
    hours = int(seconds / 3600)
    print("已经经过:", hours, "H ", minitus, "M ", second, "S")


def filter_isalpha(word):
    return word.encode('UTF-8').isalpha()


def filter_stop_word(word):
    return word not in eng_stop_word


def get_result(temp1, temp2, files, count):
    if temp1 > temp2:
        result = 0
    else:
        result = 1

    line = files.readline()
    line = line.strip()
    lines = line.split(" ")

    if int(lines[1]) == 1:
        label = 0
    else:
        label = 1

    if result == label:
        result = 'right'
        count = count + 1
    else:
        result = 'error'
    return count, result


def calculate_clause_score(cause_phrase, effect_phrase):
    score_phrase = 0
    len_count = 0
    for w1 in cause_phrase:
        for w2 in effect_phrase:
            w1_w2 = w1 + "_" + w2
            w2_w1 = w2 + "_" + w1
            if pmi_result[w1_w2] <= 0:
                len_count = len_count + 1
            # elif pmi_result[w1_w2] - pmi_result[w2_w1] < -2 or abs(pmi_result[w1_w2] - pmi_result[w2_w1]) <= 0.01:
            #     len_count = len_count + 1
            elif w1 == w2:
                len_count += 1
            else:
                score_phrase += pmi_result[w1_w2] #* (pmi_result[w1_w2] - pmi_result[w2_w1])
            # if w1 != w2:
            #     w1_w2 = w1 + "_" + w2
            #     # w1_w2 = w2 + "_" + w1
            #     if pmi_result[w1_w2] == 0:
            #         len_count = len_count + 1
            #     else:
            #         score_phrase += pmi_result[w1_w2]
            # else:
            #     len_count = len_count + 1

    return score_phrase, len_count


def cal_pmi(pair_count, fir_word_count, las_word_count):
    print(pair_count['total_word_count'], fir_word_count['total_word_count'], las_word_count['total_word_count'])

    pmi_copa = defaultdict(float)
    min_count = 1000000000
    max_count = 0
    for word_pair in cause_effect_pair:
        # print(w)
        w = word_pair.split("_")
        cause = w[0]
        effect = w[1]
        if pair_count[word_pair] != 0:
            probability_cause_effect_pair = math.log(float(pair_count[word_pair]) / pair_count['total_word_count'])
            # total_word_count fir_word_count las_word_count
            probability_cause = math.log(float(fir_word_count[cause]) / fir_word_count['total_word_count'])
            probability_effect = math.log(float(las_word_count[effect]) / las_word_count['total_word_count'])
            pmi_word_pair = probability_cause_effect_pair - probability_cause - probability_effect
            min_count = min(min_count, pmi_word_pair)
            max_count = max(max_count, pmi_word_pair)

            pmi_copa[word_pair] = pmi_word_pair
            aaaaa[0] += 1
        else:
            pmi_copa[word_pair] = 0

    print("Min:", min_count)
    print("Max:", max_count)
    # print(P_A_B)

    return pmi_copa


def get_premise_choice1_choice2(train_data, filter_stop_word_flag=1, filter_isalpha_flag=1):
    phrase_fir = train_data['premise']
    phrase_choose1 = train_data['choice1']
    phrase_choice2 = train_data['choice2']
    phrase_fir = phrase_fir.lower()
    phrase_choose1 = phrase_choose1.lower()
    phrase_choice2 = phrase_choice2.lower()

    # 对数据进行分词
    # 分词
    phrase_fir = WordPunctTokenizer().tokenize(phrase_fir)
    phrase_choose1 = WordPunctTokenizer().tokenize(phrase_choose1)
    phrase_choice2 = WordPunctTokenizer().tokenize(phrase_choice2)
    if filter_stop_word_flag:
        phrase_fir = filter(filter_stop_word, phrase_fir)
        phrase_choose1 = filter(filter_stop_word, phrase_choose1)
        phrase_choice2 = filter(filter_stop_word, phrase_choice2)
        phrase_fir, phrase_choose1, phrase_choice2 = list(phrase_fir), list(phrase_choose1), list(phrase_choice2)
    if filter_isalpha_flag:
        phrase_fir = filter(filter_isalpha, phrase_fir)
        phrase_choose1 = filter(filter_isalpha, phrase_choose1)
        phrase_choice2 = filter(filter_isalpha, phrase_choice2)
        phrase_fir, phrase_choose1, phrase_choice2 = list(phrase_fir), list(phrase_choose1), list(phrase_choice2)
    return phrase_fir, phrase_choose1, phrase_choice2


def get_acc(file_name, pr_save_path):
    acc = 0
    result_list = []
    show_list = []
    with open(file_name, 'r') as data_file:
        while True:
            str_data = data_file.readline()
            if str_data == '':
                break
            train_data = json.loads(str_data)
            # print(train_data)
            # print(train_data['idx'])
            flag = train_data['question']
            label = train_data['label']
            phrase_fir, phrase_choose1, phrase_choice2 = get_premise_choice1_choice2(train_data)

            if flag == 'effect':
                score_phrase_1, len_count_1 = calculate_clause_score(phrase_fir, phrase_choose1)
                score_phrase_2, len_count_2 = calculate_clause_score(phrase_fir, phrase_choice2)
            else:
                score_phrase_1, len_count_1 = calculate_clause_score(phrase_choose1, phrase_fir)
                score_phrase_2, len_count_2 = calculate_clause_score(phrase_choice2, phrase_fir)

            len_p_a1 = (len(phrase_fir) * len(phrase_choose1)) #- len_count_1
            len_p_a2 = (len(phrase_fir) * len(phrase_choice2)) #- len_count_2

            if score_phrase_1 != 0:
                score_phrase_1 = score_phrase_1 / len_p_a1
            if score_phrase_2 != 0:
                score_phrase_2 = score_phrase_2 / len_p_a2

            if label == 0:
                result_list.append([score_phrase_1, 1])
                result_list.append([score_phrase_2, 0])
                show_list.append([score_phrase_1, phrase_fir, phrase_choose1, 1])
                show_list.append([score_phrase_2, phrase_fir, phrase_choice2, 0])
            elif label == 1:
                result_list.append([score_phrase_1, 0])
                result_list.append([score_phrase_2, 1])
                show_list.append([score_phrase_1, phrase_fir, phrase_choose1, 0])
                show_list.append([score_phrase_2, phrase_fir, phrase_choice2, 1])

            if score_phrase_1 == score_phrase_2:# == 0
                id_number_for_space_phrase[0] += 1

            score_compare = (score_phrase_1 < score_phrase_2)

            if score_compare == label:
                acc += 1

    total_congju = sorted(result_list, key=functools.cmp_to_key(cmp))
    show_list = sorted(show_list, key=functools.cmp_to_key(cmp))

    max_points = Cruve_PR.get_pr_points(total_congju, relevant_label=1)
    Cruve_PR.save_points(max_points, pr_save_path)

    flag_id = 0
    for i in show_list:
        flag_id += 1
        # print(flag_id, i)
        if flag_id > 10000:
            break
    total_congjus = total_congju

    count_map = 0
    num_true = 0
    num_total = 0
    for i in total_congjus:
        num_total += 1
        if i[1] == 1:
            num_true += 1
            count_map += float(num_true / num_total)
        # if i[0] < 1:
        #     break
        #     pass

    print(count_map, num_true, num_total)
    map_acc = float(count_map) / num_true
    print(map_acc)

    return acc/(500 - id_number_for_space_phrase[0]), acc/500


if __name__ == '__main__':

    # pair_count = defaultdict(float)
    # fir_word_count = defaultdict(float)
    # las_word_count = defaultdict(float)
    # total_word_count = defaultdict(float)
    aaaaa = [0]
    data_path = '../../../data/'
    # data_path = '../../../../data_disk/ray/data/'
    evidence_choose = 1
    evidence_kinds = ['advcl', 'conj', 'inter']
    evidence = evidence_kinds[evidence_choose]
    data_choose = 0

    input_file = 'data_pair_count/'
    data_list = ['icw/', 'bok/', 'gut/']
    input_path = data_path + input_file + data_list[data_choose]
    print(evidence, data_list[data_choose])
    path_pair_count = 'data/' + data_list[data_choose] + evidence + '_output_pair_count.txt'
    path_fir_word_count = 'data/' + data_list[data_choose] + evidence + '_output_fir_word_count.txt'
    path_las_word_count = 'data/' + data_list[data_choose] + evidence + '_output_las_word_count.txt'
    path_total_word_count = 'data/' + data_list[data_choose] + evidence + '_output_total_word_count.txt'
    # path_fir_word_count = 'data/' + data_list[data_choose] + 'inter' + '_output_fir_word_count.txt'
    # path_las_word_count = 'data/' + data_list[data_choose] + 'inter' + '_output_las_word_count.txt'
    # path_total_word_count = 'data/' + data_list[data_choose] + 'inter' + '_output_total_word_count.txt'

    filter_stop_word_flag = 1
    filter_isalpha_flag = 1
    id_number_for_space_phrase = [0]

    filter_data = FilterData()
    eng_stop_word = filter_data.eng_stop_word

    with open(path_pair_count, 'rb') as file:
        pair_count = pickle.load(file)
    with open(path_fir_word_count, 'rb') as file:
        fir_word_count = pickle.load(file)
    with open(path_las_word_count, 'rb') as file:
        las_word_count = pickle.load(file)
    with open(path_total_word_count, 'rb') as file:
        total_word_count = pickle.load(file)

    cause_effect_pair_path = 'cause_effect_pair.npy'
    cause_effect_pair = np.load(cause_effect_pair_path, allow_pickle=True)
    cause_effect_pair = set(list(cause_effect_pair))

    print(len(pair_count), len(fir_word_count), len(las_word_count), len(total_word_count), len(cause_effect_pair), )

    # pmi_result = cal_pmi(pair_count, fir_word_count, las_word_count)
    pmi_result = cal_pmi(pair_count, total_word_count, total_word_count)
    print(len(pmi_result))
    pmi_result2 = sorted(pmi_result.items(), key=lambda d: d[1], reverse=True)

    print(len(pmi_result2))
    print(pmi_result2[:2000])
    print(pmi_result2[18000:])

    output_cause_effect_pair_count = data_path + 'data_pair_count/cause_effect_pair_count.txt'
    train_file_name = data_path + 'test/COPA/COPA/train.jsonl'
    test_file_name = data_path + 'test/COPA/COPA/test.jsonl'
    print(train_file_name)
    pr_save_path = 'PRCurve_copa' + '_dev' + '_PMI_dir.txt'
    dev_acc = get_acc(train_file_name, pr_save_path)
    print(test_file_name)
    pr_save_path = 'PRCurve_copa' + '_test' + '_PMI_dir.txt'
    test_acc = get_acc(test_file_name, pr_save_path)
    print(len(cause_effect_pair))
    print("dev_acc:", dev_acc, 'test_acc:', test_acc)
    print('---' * 45)
    print(id_number_for_space_phrase[0])
    print(aaaaa)
    time.sleep(5)


