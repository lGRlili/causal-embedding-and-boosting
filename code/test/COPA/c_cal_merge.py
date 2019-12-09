import pickle
import math
import copy
from datetime import datetime
from xml.dom.minidom import parse
import xml.dom.minidom
import codecs
from collections import defaultdict
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import json


starts = datetime.now()


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
            # w1_w2 = w2 + "_" + w1
            if pmi_result[w1_w2] == 0:
                len_count = len_count + 1
            else:
                score_phrase += pmi_result[w1_w2]
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


def cal_pmi():
    pmi_copa = defaultdict(float)
    min_count = 1000000000
    max_count = 0
    for word_pair in cause_effect_count:
        # print(w)
        w = word_pair.split("_")
        cause = w[0]
        effect = w[1]
        probability_cause_effect_pair = math.log(float(cause_effect_count[word_pair]) / total_cause_effect)
        probability_cause = math.log(float(cause_count[cause]) / total_cause)
        probability_effect = math.log(float(effect_count[effect]) / total_effect)
        pmi_word_pair = probability_cause_effect_pair - probability_cause - probability_effect
        min_count = min(min_count, pmi_word_pair)
        max_count = max(max_count, pmi_word_pair)

        pmi_copa[word_pair] = pmi_word_pair

    print("Min:", min_count)
    print("Max:", max_count)
    # print(P_A_B)

    return pmi_copa


def get_premise_choice1_choice2(train_data):
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


def get_acc(file_name):
    acc = 0
    with open(file_name, 'r') as data_file:
        while True:
            str_data = data_file.readline()
            if str_data == '':
                break
            train_data = json.loads(str_data)
            print(train_data)
            print(train_data['idx'])
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
            if score_phrase_1 == score_phrase_2 == 0:
                id_number_for_space_phrase[0] += 1

            score_compare = (score_phrase_1 < score_phrase_2)

            if score_compare == label:
                acc += 1
    return acc/500


# icw {'total_cause': 2280150.0000013346, 'total_effect': 2299059.0000013523, 'total_cause_effect': 2228038.0000061374}
# bok {'total_cause': 2836251.0000008834, 'total_effect': 2910776.0000011376, 'total_cause_effect': 2744667.000043098}
# gut {'total_cause': 4955098.999998843, 'total_effect': 4880675.999998131, 'total_cause_effect': 4774645.000474625}
# all {'total_cause': 10071500.000011887, 'total_effect': 10090511.000009857, 'total_cause_effect': 9747350.000860112}
if __name__ == '__main__':

    evidence_list = ['', 'cause_', 'time_', 'to_', 'fonction_classify_to']
    evidence_id_1 = 1
    evidence_id_2 = 2
    out_spec = ['cause_label_', '']
    out_spec_choose = 1
    data_choose = ['', 'icw/', 'bok/', 'gut/']
    choose_id = 0
    total_cause_list = [[10437651, 3769510, 12843401, 6648841], [2334181, 903972, 2114379, 1249973],
                        [2977571, 1159764, 3185289, 1951999], [5125898, 1705771, 7543732, 3446868]]
    total_effect_list = [[10415464, 3769605, 12793093, 6697493], [2314968, 905235, 2106843, 1257066],
                         [2899547, 1158888, 3154802, 1970608], [5200948, 1705479, 7531447, 3469818]]
    total_cause_effect_list = [[10086237, 3704187, 12722542, 6641311], [2262179, 887109, 2094798, 1248711],
                               [2806805, 1136740, 3137186, 1950333], [5017253, 1680336, 7490557, 3442265]]

    # total_cause_list_test = [[12549957, 12583683], [2070475, 2076210], [3096109, 3108546], [7383371, 7398925]]
    # total_effect_list_test = [[12549957, 12583683], [2070475, 2076210], [3096109, 3108546], [7383371, 7398925]]
    # total_cause_effect_list_test = [[12549957, 12583683], [2070475, 2076210], [3096109, 3108546], [7383371, 7398925]]

    total_cause = total_cause_list[choose_id][evidence_id_1-1] + total_cause_list[choose_id][evidence_id_2-1]
    total_effect = total_effect_list[choose_id][evidence_id_1-1] + total_cause_list[choose_id][evidence_id_2-1]
    total_cause_effect = total_cause_effect_list[choose_id][evidence_id_1-1] + total_cause_list[choose_id][evidence_id_2-1]
    # total_effect = total_effect_list[choose_id][evidence_id_1-1] + total_cause_effect_list[choose_id][evidence_id_2-1]
    # total_cause_effect = total_cause_effect_list[choose_id][evidence_id_1-1] + total_effect_list[choose_id][evidence_id_2-1]

    in_put_cause_effect_pair_count_1 = 'data/' + data_choose[choose_id] + evidence_list[evidence_id_1] +'count_cause_effect_pair.file'
    in_put_cause_word_count_1 = 'data/' + data_choose[choose_id] + evidence_list[evidence_id_1] +'count_cause_word.file'
    in_put_effect_word_count_1 = 'data/' + data_choose[choose_id] + evidence_list[evidence_id_1] +'count_effect_word.file'

    # in_put_cause_effect_pair_count_2 = 'data_direction_test/' + data_choose[choose_id] + out_spec[out_spec_choose] + 'to_count_cause_effect_pair.file'
    # in_put_cause_word_count_2 = 'data_direction_test/' + data_choose[choose_id] + out_spec[out_spec_choose] + 'to_count_cause_word.file'
    # in_put_effect_word_count_2 = 'data_direction_test/' + data_choose[choose_id] + out_spec[out_spec_choose] + 'to_count_effect_word.file'

    in_put_cause_effect_pair_count_2 = 'data/' + data_choose[choose_id] + evidence_list[evidence_id_2] +'count_cause_effect_pair.file'
    in_put_cause_word_count_2 = 'data/' + data_choose[choose_id] + evidence_list[evidence_id_2] +'count_cause_word.file'
    in_put_effect_word_count_2 = 'data/' + data_choose[choose_id] + evidence_list[evidence_id_2] +'count_effect_word.file'

    filter_stop_word_flag = 1
    filter_isalpha_flag = 1
    id_number_for_space_phrase = [0]
    eng_stop_word = stopwords.words('english')
    eng_stop_word = set(eng_stop_word)
    print(eng_stop_word)
    print(len(eng_stop_word))

    cause_effect_pair_path = 'cause_effect_pair.npy'
    cause_effect_pair = np.load(cause_effect_pair_path, allow_pickle=True)
    cause_effect_pair = set(list(cause_effect_pair))

    cause_effect_count = defaultdict(float)
    cause_count = defaultdict(float)
    effect_count = defaultdict(float)

    with open(in_put_cause_effect_pair_count_1, 'rb') as file:
        cause_effect_count_1 = pickle.load(file)
    with open(in_put_cause_word_count_1, 'rb') as file:
        cause_count_1 = pickle.load(file)
    with open(in_put_effect_word_count_1, 'rb') as file:
        effect_count_1 = pickle.load(file)

    with open(in_put_cause_effect_pair_count_2, 'rb') as file:
        cause_effect_count_2 = pickle.load(file)
    with open(in_put_cause_word_count_2, 'rb') as file:
        cause_count_2 = pickle.load(file)
    with open(in_put_effect_word_count_2, 'rb') as file:
        effect_count_2 = pickle.load(file)

    for i in cause_effect_count_1:
        cause_effect_count[i] += cause_effect_count_1[i]
    for i in cause_count_1:
        cause_count[i] += cause_count_1[i]
    for i in effect_count_1:
        effect_count[i] += effect_count_1[i]

    for i in cause_effect_count_2:
        i1, i2 = i.split('_')
        ii = i2 + '_' + i1
        if ii in cause_effect_count_2:
            cause_effect_count[i] += cause_effect_count_2[ii]
    for i in cause_count_2:
        effect_count[i] += cause_count_2[i]
    for i in effect_count_2:
        cause_count[i] += effect_count_2[i]

    for i in cause_effect_count_2:
        cause_effect_count[i] += cause_effect_count_2[i]
    for i in cause_count_2:
        cause_count[i] += cause_count_2[i]
    for i in effect_count_2:
        effect_count[i] += effect_count_2[i]

    pmi_result = cal_pmi()

    data_path = '../../../data/'
    output_cause_effect_pair_count = data_path + 'data_pair_count/cause_effect_pair_count.txt'
    train_file_name = data_path + 'test/COPA/COPA/train.jsonl'
    test_file_name = data_path + 'test/COPA/COPA/test.jsonl'
    print(train_file_name)
    dev_acc = get_acc(train_file_name)
    print(test_file_name)
    test_acc = get_acc(test_file_name)
    print(len(cause_effect_pair))
    print(len(cause_effect_count))
    print("dev_acc:", dev_acc, 'test_acc:', test_acc)
    print('---' * 45)
    print(id_number_for_space_phrase[0])


