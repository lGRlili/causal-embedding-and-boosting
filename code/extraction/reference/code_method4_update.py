import math
import pickle
from xml.dom.minidom import parse
import xml.dom.minidom
from nltk.tokenize import WordPunctTokenizer
import codecs
import copy
import pandas as pd
from datetime import datetime, time
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np

starts = datetime.now()


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


def get_data(num):
    # 获得文本
    # 使用minidom解析器打开 XML 文档
    # /home/keweb/PycharmProjects/gutenberg/COPA
    if num == 0:
        path = "COPA-resources/datasets/copa-all.xml"

    elif num == 1:
        path = "COPA-resources/datasets/copa-dev.xml"

    else:
        # 此处存储文本的内容
        path = "COPA-resources/datasets/copa-test.xml"
    # print(path)

    DOMTree = xml.dom.minidom.parse(path)
    collection = DOMTree.documentElement
    # 在集合中获取数据
    patterns = collection.getElementsByTagName("item")
    len_pattern = len(patterns)
    return patterns, len_pattern


def get_result_file(num):
    # 获得答案
    # /home/keweb/PycharmProjects/gutenberg/COPA/
    if num == 0:
        files = open("COPA-resources/results/gold.all", "r")
    elif num == 1:
        files = open("COPA-resources/results/gold.dev", "r")
    else:
        files = open("COPA-resources/results/gold.test", "r")
    return files


def compare_result(ratio, files, count):
    if ratio > 1:
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


# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def get_p_a_lemma(patten, stopkey, sw_flag):
    # 每次处理一个句子
    flag = patten.getAttribute("asks-for")
    type_1 = patten.getElementsByTagName('p')[0]
    p = type_1.childNodes[0].data
    format_1 = patten.getElementsByTagName('a1')[0]
    a1 = format_1.childNodes[0].data
    rating = patten.getElementsByTagName('a2')[0]
    a2 = rating.childNodes[0].data

    wordnet_lemmatizer = WordNetLemmatizer()
    p = p.lower()
    a1 = a1.lower()
    a2 = a2.lower()
    # 对数据进行分词
    seg_p = WordPunctTokenizer().tokenize(p)
    seg_a1 = WordPunctTokenizer().tokenize(a1)
    seg_a2 = WordPunctTokenizer().tokenize(a2)
    l_p = []
    l_a1 = []
    l_a2 = []
    for word in seg_p:
        pos_word = nltk.pos_tag([word])
        word_pos = get_wordnet_pos(pos_word[0][1]) or wordnet.NOUN
        # print(pos_word[0][1])
        i = wordnet_lemmatizer.lemmatize(word, pos=word_pos)
        if sw_flag == '':
            # print('youxiao')
            if i not in stopkey:  # 去停用词 + 词性筛选
                l_p.append(i)
        else:
            l_p.append(i)

    for word in seg_a2:
        pos_word = nltk.pos_tag([word])
        word_pos = get_wordnet_pos(pos_word[0][1]) or wordnet.NOUN
        # print(pos_word[0][1])
        i = wordnet_lemmatizer.lemmatize(word, pos=word_pos)
        if sw_flag == '':
            if i not in stopkey:  # 去停用词 + 词性筛选
                l_a2.append(i)
        else:
            l_a2.append(i)

    for word in seg_a1:
        pos_word = nltk.pos_tag([word])
        word_pos = get_wordnet_pos(pos_word[0][1]) or wordnet.NOUN
        # print(pos_word[0][1])
        i = wordnet_lemmatizer.lemmatize(word, pos=word_pos)
        if sw_flag == '':
            if i not in stopkey:  # 去停用词 + 词性筛选
                l_a1.append(i)
        else:
            l_a1.append(i)

    return flag, l_p, l_a1, l_a2


def get_p_a(patten, stopkey, sw_flag):
    flag = patten.getAttribute("asks-for")
    type_1 = patten.getElementsByTagName('p')[0]
    p = type_1.childNodes[0].data
    format_1 = patten.getElementsByTagName('a1')[0]
    a1 = format_1.childNodes[0].data
    rating = patten.getElementsByTagName('a2')[0]
    a2 = rating.childNodes[0].data

    p = p.lower()
    a1 = a1.lower()
    a2 = a2.lower()
    # 对数据进行分词
    seg_p = WordPunctTokenizer().tokenize(p)
    seg_a1 = WordPunctTokenizer().tokenize(a1)
    seg_a2 = WordPunctTokenizer().tokenize(a2)
    l_p = []
    l_a1 = []
    l_a2 = []
    for i in seg_p:
        if sw_flag == '':
            # print('youxiao')
            if i not in stopkey:  # 去停用词 + 词性筛选
                l_p.append(i)
        else:
            l_p.append(i)

    for i in seg_a2:
        if sw_flag == '':
            if i not in stopkey:  # 去停用词 + 词性筛选
                l_a2.append(i)
        else:
            l_a2.append(i)

    for i in seg_a1:
        if sw_flag == '':
            if i not in stopkey:  # 去停用词 + 词性筛选
                l_a1.append(i)
        else:
            l_a1.append(i)

    return flag, l_p, l_a1, l_a2


def get_flag_p_a1_a2(patterns, sw_flag, islemma):
    stopkey = [w.strip() for w in codecs.open('stop_word.txt', 'r', encoding='utf-8').readlines()]
    patterns_list = []
    for patten in patterns:
        # 依次处理数据
        # 用于判断该数据是由因到果还是由果到因
        if islemma == 'lemma':
            flag, l_p, l_a1, l_a2 = get_p_a_lemma(patten, stopkey, sw_flag)
        else:
            flag, l_p, l_a1, l_a2 = get_p_a(patten, stopkey, sw_flag)
        temp_list = [flag, l_p, l_a1, l_a2]
        patterns_list.append(temp_list)
    return patterns_list


def calculate_sentence_pair_likelihood_ratio(sent_X, sent_Y, COPA_word_pair_class_likelihood_ratios, method="pairwise"):
    score_matrix = np.zeros(shape=(len(sent_X), len(sent_Y)))
    # print(len(sent_X), len(sent_Y))
    # 构建一个全连接矩阵
    row = 0
    for word_x in sent_X:
        col = 0
        for word_y in sent_Y:
            word_pair = word_x + "_" + word_y
            score_matrix[row, col] = COPA_word_pair_class_likelihood_ratios[word_pair]
            col += 1
        row += 1
    temp = 1.
    len_count = 0
    if method == "pairwise":
        for word_x in sent_X:
            for word_y in sent_Y:
                if word_x != word_y:
                    word_pair = word_x + "_" + word_y
                    if COPA_word_pair_class_likelihood_ratios[word_pair] == 0:
                        len_count = len_count + 1
                    else:
                        temp *= COPA_word_pair_class_likelihood_ratios[word_pair]
        # for row in range(len(sent_X)):
        #     for col in range(len(sent_Y)):
        #         if score_matrix[row, col] == 0:
        #             len_count = len_count + 1
        #         else:
        #             temp *= score_matrix[row, col]

    elif method == "singleton":
        if len(sent_X) == 0 or len(sent_Y) == 0:
            return 1.0, len_count
        row_max = np.max(score_matrix, axis=0)
        col_max = np.max(score_matrix, axis=1)
        for row in range(len(row_max)):
            if row_max[row] != 0:
                temp *= row_max[row]
            else:
                len_count += 1
        for col in range(len(col_max)):
            if col_max[col] != 0:
                temp *= col_max[col]
            else:
                len_count += 1
    # temp *= class_prior_ratio
    return temp, len_count


def get_COPA_problem_class_posterior_ratio(flag, sent_A, sent_B1, sent_B2, COPA_word_pair_class_likelihood_ratios,
                                           class_priors):

    if flag == 'effect':  # cause
        class_likelihood_ratio_1_0, len_count_1_0 = calculate_sentence_pair_likelihood_ratio(
            sent_A, sent_B1, COPA_word_pair_class_likelihood_ratios[0])
        class_likelihood_ratio_2_0, len_count_2_0 = calculate_sentence_pair_likelihood_ratio(
            sent_A, sent_B2, COPA_word_pair_class_likelihood_ratios[0])
        class_likelihood_ratio_1_1, len_count_1_1 = calculate_sentence_pair_likelihood_ratio(
            sent_A, sent_B1, COPA_word_pair_class_likelihood_ratios[1])
        class_likelihood_ratio_2_1, len_count_2_1 = calculate_sentence_pair_likelihood_ratio(
            sent_A, sent_B2, COPA_word_pair_class_likelihood_ratios[1])
    else:
        class_likelihood_ratio_1_0, len_count_1_0 = calculate_sentence_pair_likelihood_ratio(
            sent_B1, sent_A, COPA_word_pair_class_likelihood_ratios[0])
        class_likelihood_ratio_2_0, len_count_2_0 = calculate_sentence_pair_likelihood_ratio(
            sent_B2, sent_A, COPA_word_pair_class_likelihood_ratios[0])
        class_likelihood_ratio_1_1, len_count_1_1 = calculate_sentence_pair_likelihood_ratio(
            sent_B1, sent_A, COPA_word_pair_class_likelihood_ratios[1])
        class_likelihood_ratio_2_1, len_count_2_1 = calculate_sentence_pair_likelihood_ratio(
            sent_B2, sent_A, COPA_word_pair_class_likelihood_ratios[1])
    # print('len_count_1:', len_count_1, 'len_count_2', len_count_2)
    class_likelihood_ratio_1 = class_likelihood_ratio_1_0 / class_likelihood_ratio_1_1
    class_likelihood_ratio_2 = class_likelihood_ratio_2_0 / class_likelihood_ratio_2_1
    class_prior_ratio = class_priors[0] / class_priors[1]
    class_posterior_1 = 1 / (1 + class_likelihood_ratio_1 * class_prior_ratio)
    class_posterior_2 = 1 / (1 + class_likelihood_ratio_2 * class_prior_ratio)

    probability_XY_D_1 = class_likelihood_ratio_1_0 * class_priors[0] + class_likelihood_ratio_1_1 * class_priors[1]
    probability_XY_D_2 = class_likelihood_ratio_2_0 * class_priors[0] + class_likelihood_ratio_2_1 * class_priors[1]

    # print(class_posterior_1, class_posterior_2)
    return class_posterior_1, class_posterior_2, probability_XY_D_1, probability_XY_D_2


def calculate_result(patterns_list, files, COPA_word_pair_class_likelihood_ratios, class_priors):
    # 此处获得的是结果
    # 此处记录的是正确题目的数目

    count = 0
    stopkey = [w.strip() for w in codecs.open('stop_word.txt', 'r', encoding='utf-8').readlines()]
    # print(sorted(advcl_P_A_B_pairs.values()))
    len_pattern = len(patterns_list)
    for temp_list in patterns_list:
        # 这里针对的是每个句子判断对错
        flag, sent_A, sent_B1, sent_B2 = temp_list
        class_posterior_1, class_posterior_2, probability_XY_D_1, probability_XY_D_2 = \
            get_COPA_problem_class_posterior_ratio(
                flag, sent_A, sent_B1, sent_B2, COPA_word_pair_class_likelihood_ratios, class_priors)
        ratio = class_posterior_1 / class_posterior_2
        count, result = compare_result(ratio, files, count)
        # print(temp1, temp2)
    acc = float(count) / len_pattern

    # print("accuracy:", acc)
    return acc


def calculate_result_merge(patterns_list, files, COPA_word_pair_class_likelihood_ratios_fir, class_priors_fir,
                           COPA_word_pair_class_likelihood_ratios_las, class_priors_las, weight_grid):
    # 此处获得的是结果
    # 此处记录的是正确题目的数目

    count = 0
    stopkey = [w.strip() for w in codecs.open('stop_word.txt', 'r', encoding='utf-8').readlines()]
    # print(sorted(advcl_P_A_B_pairs.values()))
    len_pattern = len(patterns_list)
    for temp_list in patterns_list:
        # 这里针对的是每个句子判断对错
        flag, sent_A, sent_B1, sent_B2 = temp_list
        class_posterior_1_fir, class_posterior_2_fir, probability_XY_D_1_fir, probability_XY_D_2_fir = \
            get_COPA_problem_class_posterior_ratio(
                flag, sent_A, sent_B1, sent_B2, COPA_word_pair_class_likelihood_ratios_fir, class_priors_fir)
        class_posterior_1_las, class_posterior_2_las, probability_XY_D_1_las, probability_XY_D_2_las = \
            get_COPA_problem_class_posterior_ratio(
                flag, sent_A, sent_B1, sent_B2, COPA_word_pair_class_likelihood_ratios_las, class_priors_las)
        sum_XY = probability_XY_D_1_fir * (1 - weight_grid) + probability_XY_D_1_las * weight_grid
        probability_XY_D_1_fir /= sum_XY
        probability_XY_D_1_las /= sum_XY
        sum_XY = probability_XY_D_2_fir * (1 - weight_grid) + probability_XY_D_2_las * weight_grid
        probability_XY_D_2_fir /= sum_XY
        probability_XY_D_2_las /= sum_XY
        class_posterior_1 = (1 - weight_grid) * probability_XY_D_1_fir * class_posterior_1_fir + weight_grid * probability_XY_D_1_las * class_posterior_1_las
        class_posterior_2 = (1 - weight_grid) * probability_XY_D_2_fir * class_posterior_2_fir + weight_grid * probability_XY_D_2_las * class_posterior_2_las

        # class_posterior_1 = (1 - weight_grid) * class_posterior_1_fir + weight_grid * class_posterior_1_las
        # class_posterior_2 = (1 - weight_grid) * class_posterior_2_fir + weight_grid * class_posterior_2_las

        ratio = class_posterior_1 / class_posterior_2
        count, result = compare_result(ratio, files, count)
        # print(temp1, temp2)
    acc = float(count) / len_pattern

    # print("accuracy:", acc)
    return acc


def calculate_result_merge_three(
        patterns_list, files, COPA_word_pair_class_likelihood_ratios_fir, class_priors_fir,
        COPA_word_pair_class_likelihood_ratios_mid, class_priors_mid,
        COPA_word_pair_class_likelihood_ratios_las, class_priors_las, weight_grid_12, weight_grid_23):
    # 此处获得的是结果
    # 此处记录的是正确题目的数目

    count = 0
    stopkey = [w.strip() for w in codecs.open('stop_word.txt', 'r', encoding='utf-8').readlines()]
    # print(sorted(advcl_P_A_B_pairs.values()))
    len_pattern = len(patterns_list)
    for temp_list in patterns_list:
        # 这里针对的是每个句子判断对错
        flag, sent_A, sent_B1, sent_B2 = temp_list
        class_posterior_1_fir, class_posterior_2_fir = get_COPA_problem_class_posterior_ratio(
            flag, sent_A, sent_B1, sent_B2, COPA_word_pair_class_likelihood_ratios_fir, class_priors_fir)

        class_posterior_1_mid, class_posterior_2_mid = get_COPA_problem_class_posterior_ratio(
            flag, sent_A, sent_B1, sent_B2, COPA_word_pair_class_likelihood_ratios_mid, class_priors_mid)

        class_posterior_1_las, class_posterior_2_las = get_COPA_problem_class_posterior_ratio(
            flag, sent_A, sent_B1, sent_B2, COPA_word_pair_class_likelihood_ratios_las, class_priors_las)
        class_posterior_1 = (1 - weight_grid_12) * class_posterior_1_fir + weight_grid_12 * class_posterior_1_mid
        class_posterior_2 = (1 - weight_grid_12) * class_posterior_2_fir + weight_grid_12 * class_posterior_2_mid
        class_posterior_1 = (1 - weight_grid_23) * class_posterior_1 + weight_grid_23 * class_posterior_1_las
        class_posterior_2 = (1 - weight_grid_23) * class_posterior_2 + weight_grid_23 * class_posterior_2_las
        ratio = class_posterior_1 / class_posterior_2
        count, result = compare_result(ratio, files, count)
        # print(temp1, temp2)
    acc = float(count) / len_pattern

    # print("accuracy:", acc)
    return acc


def cal_total_count(p_a_pos, p_b_pos, db_p_a_b_pos):
    count_p_a_pos = 0
    count_p_b_pos = 0
    count_db_p_a_b_pos = 0
    range_len = 50000
    num_p_a_pos = 0
    num_p_b_pos = 0
    num_db_p_a_b_pos = 0
    # 统计词频
    count_p_a_pos = np.sum(p_a_pos)
    count_p_b_pos = np.sum(p_b_pos)
    count_db_p_a_b_pos = db_p_a_b_pos['count']  # total counter of all word pairs
    # count_db_p_a_b_pos = np.sum(db_p_a_b_pos)
    num_p_a_pos = np.sum(p_a_pos != 0)
    num_p_b_pos = np.sum(p_b_pos != 0)
    num_db_p_a_b_pos = db_p_a_b_pos['sum']
    # num_db_p_a_b_pos = np.sum(db_p_a_b_pos != 0)
    # print('count_p_a_pos:', count_p_a_pos, 'count_p_b_pos:', count_p_b_pos, 'count_db_p_a_b_pos:', count_db_p_a_b_pos)
    # print('num_p_a_pos:', num_p_a_pos, 'num_p_b_pos:', num_p_b_pos, 'num_db_p_a_b_pos:', num_db_p_a_b_pos)
    return count_p_a_pos, count_p_b_pos, count_db_p_a_b_pos


def cal_pos_neg_zero(PMI_dict):
    count_pos = 0
    count_neg = 0
    count_zero = 0
    for key in PMI_dict:
        if PMI_dict[key] > 0:
            count_pos += 1
        elif PMI_dict[key] < 0:
            count_neg += 1
        else:
            count_zero += 1
    # print('count_pos:', count_pos, 'count_neg:', count_neg, 'count_zero:', count_zero)


# 获得最初需要的参数---对应不同数据集
def get_data_for_grid(doc_name, name_str):
    with open('database_original/temp' + doc_name + '/str_id_' + 'word_count_delete.file', 'rb') as f:
        str_id_word_count_icw = pickle.load(f)
    P_word_id = np.load('database_original/temp' + doc_name + '/P_word_id.npy')
    advcl_p_a_pos_wei_icw = np.load('database_original/temp' + doc_name + '/advcl_p_a_pos_wei' + name_str + '.npy')
    advcl_p_b_pos_wei_icw = np.load('database_original/temp' + doc_name + '/advcl_p_b_pos_wei' + name_str + '.npy')
    # 获得conj
    conj_p_a_pos_wei_icw = np.load('database_original/temp' + doc_name + '/conj_p_a_pos_wei' + name_str + '.npy')
    conj_p_b_pos_wei_icw = np.load('database_original/temp' + doc_name + '/conj_p_b_pos_wei' + name_str + '.npy')
    # 获得inter
    inter_level_p_a_pos_icw = np.load('database_original/temp' + doc_name + '/inter_level_p_a_pos' + name_str + '.npy')
    inter_level_p_b_pos_icw = np.load('database_original/temp' + doc_name + '/inter_level_p_b_pos' + name_str + '.npy')
    with open('database_original/temp' + doc_name + '/' + 'advcl_pair_pos_copa' + '.file', 'rb') as f:
        advcl_pair_pos_copa_icw = pickle.load(f)
    with open('database_original/temp' + doc_name + '/' + 'conj_pair_pos_copa' + '.file', 'rb') as f:
        conj_pair_pos_copa_icw = pickle.load(f)
    # with open('database_original/temp' + doc_name + '/' + 'inter_pair_pos_copa' + '.file', 'rb') as f:
    #     inter_pair_pos_copa_icw = pickle.load(f)
    with open('database_original/temp' + doc_name + '/' + 'inter_noclause_pair_pos_copa' + '.file', 'rb') as f:
        inter_pair_pos_copa_icw = pickle.load(f)
    return str_id_word_count_icw, advcl_p_a_pos_wei_icw, advcl_p_b_pos_wei_icw, conj_p_a_pos_wei_icw, \
           conj_p_b_pos_wei_icw, inter_level_p_a_pos_icw, inter_level_p_b_pos_icw, advcl_pair_pos_copa_icw, \
           conj_pair_pos_copa_icw, inter_pair_pos_copa_icw, P_word_id


def cal_evidence_priors(COPA_word_pairs, dict_word2id, word_id_counter, COPA_advcl_word_pair_counter,
                        COPA_conj_word_pair_counter, COPA_inter_word_pair_counter, prior_prob_other):
    # the total counter of all words in the first position, the total counter of all words in the second position,
    # the total counter of all word pairs
    advcl_a_total_count, advcl_b_total_count, advcl_ab_total_count = cal_total_count(word_id_counter, word_id_counter,
                                                                                     COPA_advcl_word_pair_counter)
    conj_a_total_count, conj_b_total_count, conj_ab_total_count = cal_total_count(word_id_counter, word_id_counter,
                                                                                  COPA_conj_word_pair_counter)
    inter_a_total_count, inter_b_total_count, inter_ab_total_count = cal_total_count(word_id_counter, word_id_counter,
                                                                                     COPA_inter_word_pair_counter)
    other_a_total_count, other_b_total_count = advcl_a_total_count, advcl_b_total_count
    # cal_total_count(word_id_counter, word_id_counter, COPA_advcl_word_pair_counter)
    # #advcl_a_total_count, advcl_b_total_count

    p_advcl = advcl_ab_total_count / (advcl_ab_total_count + conj_ab_total_count + inter_ab_total_count)
    p_conj = conj_ab_total_count / (advcl_ab_total_count + conj_ab_total_count + inter_ab_total_count)
    p_inter = inter_ab_total_count / (advcl_ab_total_count + conj_ab_total_count + inter_ab_total_count)

    # prior probabilities of the four evidence types
    # prior_prob_other = 0.95
    prior_prob_advcl = p_advcl * (1 - prior_prob_other)
    prior_prob_conj = p_conj * (1 - prior_prob_other)
    prior_prob_inter = p_inter * (1 - prior_prob_other)

    return [prior_prob_advcl, prior_prob_conj, prior_prob_inter, prior_prob_other], [advcl_ab_total_count,
                                                                                     conj_ab_total_count,
                                                                                     inter_ab_total_count,
                                                                                     other_a_total_count ** 2]


def cal_all_word_pair_likelihood_ratios(COPA_word_pairs, dict_word2id, word_id_counter, COPA_advcl_word_pair_counter,
                                        COPA_conj_word_pair_counter, COPA_inter_word_pair_counter, evidence_priors,
                                        evidence_counts, class_1_evidence_probs, class_0_evidence_probs):
    COPA_word_pair_class_likelihood_ratios_1 = {}
    COPA_word_pair_class_likelihood_ratios_0 = {}
    Min = 1000000000
    Max = 0
    COPA_words = []
    for wp in COPA_word_pairs:
        # print(w)
        a = wp.split("_")
        word_a = a[0]
        word_b = a[1]
        # 将str转换成id
        id_a = dict_word2id[word_a]
        id_b = dict_word2id[word_b]
        COPA_words.extend([word_a, word_b])

        w1w2 = wp
        w2w1 = a[1] + '_' + a[0]

        advcl_likelihood = (COPA_advcl_word_pair_counter[w1w2] + 1e-4) / evidence_counts[0]  # 1e-4
        conj_likelihood = (COPA_conj_word_pair_counter[w1w2] + 1e-4) / evidence_counts[1]  # 1e-4
        inter_likelihood = (COPA_inter_word_pair_counter[w1w2] + 2e-4) / evidence_counts[2]  # 2e-4
        other_likelihood = (word_id_counter[id_a] * word_id_counter[id_b] + 1e5) / evidence_counts[3]  # 1e5

        # advcl_likelihood = (COPA_advcl_word_pair_counter[w1w2]) / evidence_counts[0]  # 1e-4
        # conj_likelihood = (COPA_conj_word_pair_counter[w1w2]) / evidence_counts[1]  # 1e-4
        # inter_likelihood = (COPA_inter_word_pair_counter[w1w2]) / evidence_counts[2]  # 2e-4
        # other_likelihood = (word_id_counter[id_a] * word_id_counter[id_b] + 1e5) / evidence_counts[3]  # 1e5

        evidence_likelihoods = [advcl_likelihood, conj_likelihood, inter_likelihood, other_likelihood]

        numerator, denominator = 0, 0
        for i in range(4):
            numerator += class_0_evidence_probs[i] * evidence_likelihoods[i]
            denominator += class_1_evidence_probs[i] * evidence_likelihoods[i]

        COPA_word_pair_class_likelihood_ratios_0[w1w2] = numerator
        COPA_word_pair_class_likelihood_ratios_1[w1w2] = denominator
    COPA_word_pair_class_likelihood_ratios = \
        [COPA_word_pair_class_likelihood_ratios_0, COPA_word_pair_class_likelihood_ratios_1]

    return COPA_word_pair_class_likelihood_ratios


def get_used_parameter(evidence_class_1_probs, evidence_priors):
    advcl_class_0_prob = 1 - evidence_class_1_probs[0]
    conj_class_0_prob = 1 - evidence_class_1_probs[1]
    inter_class_0_prob = 1 - evidence_class_1_probs[2]
    other_class_0_prob = 1 - evidence_class_1_probs[3]
    evidence_class_0_probs = [advcl_class_0_prob, conj_class_0_prob, inter_class_0_prob,
                              other_class_0_prob]
    class_1_prob = 0.
    class_0_prob = 0.
    class_1_evidence_probs, class_0_evidence_probs = [], []
    for i in range(4):
        class_0_prob += evidence_class_0_probs[i] * evidence_priors[i]
        class_1_prob += evidence_class_1_probs[i] * evidence_priors[i]
    for i in range(4):
        class_0_evidence_probs.append(
            evidence_class_0_probs[i] * evidence_priors[i] / class_0_prob)
        class_1_evidence_probs.append(
            evidence_class_1_probs[i] * evidence_priors[i] / class_1_prob)

    class_priors = [class_0_prob, class_1_prob]
    return class_1_evidence_probs, class_0_evidence_probs, class_priors


def grid_tuning_five(data_id, islemma, doc_name):
    print(doc_name)
    print("grid_tuning_five")
    sw_flag = ''
    choose_list = ['', '_no', '_stop']
    name_str = choose_list[2]
    print('获得基础数据')

    # dict_word2id: a dict mapping word to id.
    # word_id_counter: a list recording the counts of words (ids)
    # COPA_advcl_word_pair_counter:
    # COPA_conj_word_pair_counter:
    # COPA_inter_word_pair_counter:
    dict_word2id, COPA_advcl_word_x_counter, COPA_advcl_word_y_counter, COPA_conj_word_x_counter, \
    COPA_conj_word_y_counter, COPA_inter_word_x_counter, COPA_inter_word_y_counter, COPA_advcl_word_pair_counter, \
    COPA_conj_word_pair_counter, COPA_inter_word_pair_counter, word_id_counter \
        = get_data_for_grid(doc_name, name_str)

    # load the list of word pairs in COPA into P_A_B_pair
    if islemma == 'lemma':
        print("dict_date_lemma/P_A_B_total" + ".file")
        with open("dict_date_lemma/P_A_B_total" + ".file", "rb") as fi:
            COPA_word_pairs = pickle.load(fi)
    else:
        # print("dict_date/P_A_B_total" + ".file")
        with open("dict_date/P_A_B_total" + ".file", "rb") as fi:
            COPA_word_pairs = pickle.load(fi)

    print("读取数据完毕")
    max_num = 0
    min_num = 1
    max_list = []
    max_2_list = []
    max_2_num = 0
    # patterns, len_pattern = get_data(data_id)
    # 2, 4, 8, 16,
    # advcl_class_1_prob_grids = [i / 100 for i in
    #                             [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80]]  # range(15, 60, 5)]
    # conj_class_1_prob_grids = [i / 100 for i in [5, 10, 15, 20, 25, 30]]  # range(2, 12, 2)]
    # inter_class_1_prob_grids = [i / 100 for i in
    #                             [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80]]  # range(40, 90, 5)]
    advcl_class_1_prob_grids = [i / 100 for i in range(40, 90, 2)]  # range(15, 60, 5)]
    conj_class_1_prob_grids = [i / 100 for i in range(1, 21, 1)]  # range(2, 12, 2)]
    inter_class_1_prob_grids = [i / 100 for i in range(20, 90, 2)]  # range(40, 90, 5)]
    other_class_1_prob_grids = [0.0005]  # 0.05, 0.01
    prior_prob_other = 0.98
    # 此处分别得到4中证据类型的比例和
    evidence_priors, evidence_counts = cal_evidence_priors(COPA_word_pairs, dict_word2id, word_id_counter,
                                                           COPA_advcl_word_pair_counter, COPA_conj_word_pair_counter,
                                                           COPA_inter_word_pair_counter, prior_prob_other)
    print(prior_prob_other)
    acc_list = []

    for abcde in range(1):
        for other_class_1_prob in other_class_1_prob_grids:
            other_class_0_prob = 1 - other_class_1_prob
            for advcl_class_1_prob in advcl_class_1_prob_grids:
                advcl_class_0_prob = 1 - advcl_class_1_prob
                for conj_class_1_prob in conj_class_1_prob_grids:
                    conj_class_0_prob = 1 - conj_class_1_prob
                    for inter_class_1_prob in inter_class_1_prob_grids:
                        inter_class_0_prob = 1 - inter_class_1_prob
                        # for each configuration of hyperparameters:

                        if True:
                            evidence_class_0_probs = [advcl_class_0_prob, conj_class_0_prob, inter_class_0_prob,
                                                      other_class_0_prob]
                            evidence_class_1_probs = [advcl_class_1_prob, conj_class_1_prob, inter_class_1_prob,
                                                      other_class_1_prob]

                            class_1_prob = 0.
                            class_0_prob = 0.
                            class_1_evidence_probs, class_0_evidence_probs = [], []
                            for i in range(4):
                                class_0_prob += evidence_class_0_probs[i] * evidence_priors[i]
                                class_1_prob += evidence_class_1_probs[i] * evidence_priors[i]
                            for i in range(4):
                                class_0_evidence_probs.append(
                                    evidence_class_0_probs[i] * evidence_priors[i] / class_0_prob)
                                class_1_evidence_probs.append(
                                    evidence_class_1_probs[i] * evidence_priors[i] / class_1_prob)

                            class_priors = [class_0_prob, class_1_prob]

                            # print("The evidence class 0 probs:", evidence_class_0_probs)
                            # print("The evidence class 1 probs:", evidence_class_1_probs)
                            # print("The class priors:", class_priors)
                            # print("The evidence probs:", evidence_priors)

                            # 针对COPA中的所有词对，计算出它们的证据类型似然比
                            COPA_word_pair_class_likelihood_ratios = cal_all_word_pair_likelihood_ratios(
                                COPA_word_pairs, dict_word2id, word_id_counter,
                                COPA_advcl_word_pair_counter, COPA_conj_word_pair_counter, COPA_inter_word_pair_counter,
                                evidence_priors, evidence_counts, class_1_evidence_probs, class_0_evidence_probs)

                            # print(COPA_word_pair_class_likelihood_ratios)

                            print([advcl_class_1_prob, conj_class_1_prob, inter_class_1_prob,
                                   other_class_1_prob, prior_prob_other])
                            # files = get_result_file(data_id)
                            # acc = calculate_resule(patterns_list, len_pattern, files,
                            #                        advcl_P_A_B_pairs, conj_P_A_B_pairs, inter_P_A_B_pairs,
                            #                        advcl_pos_grid, conj_pos_grid, inter_pos_grid, p_advcl, p_conj,
                            #                        p_inter)
                            files = get_result_file(1)
                            patterns, len_pattern = get_data(1)
                            patterns_list = get_flag_p_a1_a2(patterns, sw_flag, islemma)

                            # 此时只考虑正向的因果
                            # list_of_word_pair_evidence_ratio = [COPA_word_pair_advcl_ratio, COPA_word_pair_conj_ratio, COPA_word_pair_inter_ratio, COPA_word_pair_other_ratio]
                            # list_of_evidence_prior = [prior_prob_advcl, prior_prob_conj, prior_prob_inter, prior_prob_other]
                            # list_of_hyperparameter = [advcl_pos_grid, conj_pos_grid, inter_pos_grid, other_pos_grid]

                            acc = calculate_result(patterns_list, files, COPA_word_pair_class_likelihood_ratios,
                                                   class_priors)

                            files = get_result_file(2)
                            patterns, len_pattern = get_data(2)
                            patterns_list = get_flag_p_a1_a2(patterns, sw_flag, islemma)

                            acc_test = calculate_result(patterns_list, files, COPA_word_pair_class_likelihood_ratios,
                                                        class_priors)

                            print("acc_dev:", acc, "     acc_test:", acc_test)
                            # acc = calculate_resule_pair_max(patterns_list, len_pattern, files, P_A_B_pair)
                            acc_list.append(acc)

                            if acc > max_num:
                                max_2_num = acc - 0.002
                                max_2_list = []
                                if max_2_num == max_num:
                                    max_2_list = max_list
                                max_num = acc
                                print('更新max:', max_num)
                                max_list = []
                            if acc == max_num:
                                max_list.append(
                                    [acc_test, [advcl_class_1_prob, conj_class_1_prob, inter_class_1_prob,
                                                other_class_1_prob, prior_prob_other]])
                            if acc == max_2_num:
                                max_2_list.append([
                                    acc_test, [advcl_class_1_prob, conj_class_1_prob, inter_class_1_prob,
                                               other_class_1_prob, prior_prob_other]])
                            if acc < min_num:
                                min_num = acc
                                print('更新min:', min_num)

                print('max:', max_num, max_list)
                print('max_2:', max_2_num, max_2_list)
                print('min:', min_num)

    return acc_list


def grid_tuning_merge_spe(data_id, islemma):
    print("grid_tuning_merge_spe")
    sw_flag = ''
    choose_list = ['', '_no', '_stop']
    name_str = choose_list[2]
    print('获得基础数据')
    with open("dict_date/P_A_B_total" + ".file", "rb") as f:
        P_A_B_pair = pickle.load(f)
    choose = ['', '_icw', '_gut', '_book_corpus']
    weight = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
    weight = [i / 100 for i in range(0, 25, 6)]
    # weight = [i / 1000 for i in range(4, 10)]# 4, 10
    # 0, 1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    weight23 = [0]
    # icw_grids_662 = [[0.7, 0.05, 0.45, 0.0001, 0.95], [0.8, 0.05, 0.5, 0.0001, 0.95]]  # 0.676 0.672
    # icw_grids_666 = [[0.7, 0.05, 0.3, 0.0001, 0.99], [0.7, 0.05, 0.35, 0.0001, 0.99], [0.8, 0.05, 0.35, 0.0001, 0.99],
    #                  [0.8, 0.05, 0.4, 0.0001, 0.99], [0.8, 0.05, 0.45, 0.0001, 0.99]]
    # icw_grids_668 = [[0.6, 0.05, 0.35, 0.0001, 0.98]]
    # bok_grids_644 = [[0.3, 0.01, 0.45, 0.0001, 0.95], [0.16, 0.01, 0.4, 0.0001, 0.95]]  # 0.676 0.672
    # #
    # # gut_grids_616 = []
    # # # 有约束
    #
    # icw_grids_664 = [[0.6, 0.08, 0.36, 0.0001, 0.98], [0.6, 0.08, 0.36, 0.0001, 0.98],
    #                  [0.58, 0.08, 0.34, 0.0001, 0.98], [0.5, 0.07, 0.3, 0.0001, 0.98], [0.5, 0.06, 0.34, 0.0001, 0.98]]
    #
    # icw_grids_672 = [[0.48, 0.02, 0.26, 0.0001, 0.98]]
    # icw_grids_672 = [[0.42, 0.02, 0.22, 0.0001, 0.98], [0.58, 0.03, 0.24, 0.0001, 0.98],
    #                  [0.48, 0.02, 0.26, 0.0001, 0.98]]
    #
    # bok_grids_646 = [[0.2, 0.03, 0.44, 0.0001, 0.9], [0.22, 0.04, 0.5, 0.0001, 0.9], [0.22, 0.05, 0.4, 0.0001, 0.9],
    #                  [0.22, 0.05, 0.42, 0.0001, 0.9], [0.24, 0.05, 0.42, 0.0001, 0.9], [0.24, 0.05, 0.44, 0.0001, 0.9]]
    # gut_grids_604 = [
    #     [0.64, 0.01, 0.08, 0.0001, 0.9], [0.86, 0.02, 0.02, 0.001, 0.85], [0.88, 0.02, 0.02, 0.001, 0.85],
    #     [0.24, 0.01, 0.04, 0.0001, 0.85], [0.48, 0.02, 0.08, 0.0001, 0.85], [0.7, 0.03, 0.12, 0.0001, 0.85]
    #     , [0.84, 0.02, 0.02, 0.0001, 0.85], [0.86, 0.02, 0.02, 0.0001, 0.85], [0.88, 0.02, 0.02, 0.0001, 0.85]]
    #
    # bok_grids_646 = [[0.32, 0.01, 0.52, 0.0005, 0.98], [0.16, 0.08, 0.56, 0.0005, 0.98], [0.2, 0.1, 0.72, 0.0005, 0.98]]
    # # max: 0.604 [[0.634, [0.64, 0.01, 0.08, 0.0001, 0.9]]]
    # # max: 0.604 [[0.636, [0.86, 0.02, 0.02, 0.001, 0.85]], [0.634, [0.88, 0.02, 0.02, 0.001, 0.85]]]
    # # max: 0.604 [[0.632, [0.24, 0.01, 0.04, 0.0001, 0.85]], [0.634, [0.48, 0.02, 0.08, 0.0001, 0.85]],
    # #             [0.636, [0.7, 0.03, 0.12, 0.0001, 0.85]], [0.63, [0.84, 0.02, 0.02, 0.0001, 0.85]],
    # #             [0.63, [0.86, 0.02, 0.02, 0.0001, 0.85]], [0.63, [0.88, 0.02, 0.02, 0.0001, 0.85]]]

    # bok_grids_646 = [[]]
    # # icw_grids_652 = [[]]
    #
    # # [0.678, ], [0.698, ], [0.686,]
    # # [0.1, 0.02, 0.25, 0.0001, 0.9],
    # bok_grids_648 = [[0.2, 0.04, 0.35, 0.0001, 0.9], [0.25, 0.04, 0.55, 0.0001, 0.9]]
    # # [[0.692, ], [0.688, ],
    # #  [0.686, ], [0.688, ],
    # #  [0.688, ], [0.684, ]]
    # icw_grids_668 = [[0.5, 0.04, 0.2, 0.0001, 0.98], [0.6, 0.04, 0.25, 0.0001, 0.98], [0.7, 0.04, 0.3, 0.0001, 0.98],
    #                  [0.75, 0.04, 0.3, 0.0001, 0.98], [0.85, 0.06, 0.35, 0.0001, 0.98], [0.95, 0.06, 0.4, 0.0001, 0.98]]
    # # [[0.694, ], [0.698, ],
    # #  [0.694, ], [0.696, ]]
    # icw_grids_664 = [[0.85, 0.02, 0.45, 0.0001, 0.99], [0.9, 0.02, 0.5, 0.0001, 0.99], [0.9, 0.04, 0.4, 0.0001, 0.99],
    #                  [0.95, 0.02, 0.55, 0.0001, 0.99]]
    #
    # gut_grids_604 = [[]]

    icw_grids_666 = [[0.3, 0.04, 0.2, 0.0001, 0.98]]
    bok_grids_642 = [[0.1, 0.04, 0.4, 0.0001, 0.9], [0.1, 0.04, 0.45, 0.0001, 0.9], [0.15, 0.04, 0.65, 0.0001, 0.9],
                     [0.15, 0.06, 0.65, 0.0001, 0.9], [0.2, 0.04, 0.75, 0.0001, 0.9], [0.2, 0.04, 0.8, 0.0001, 0.9],
                     [0.2, 0.04, 0.85, 0.0001, 0.9], [0.2, 0.06, 0.8, 0.0001, 0.9], [0.25, 0.06, 0.85, 0.0001, 0.9],
                     [0.25, 0.06, 0.9, 0.0001, 0.9]]
    gut_grids_604 = [[]]

    # 获得对应的数据和矩阵
    if islemma == 'lemma':
        print("dict_date_lemma/P_A_B_total" + ".file")
        with open("dict_date_lemma/P_A_B_total" + ".file", "rb") as fi:
            COPA_word_pairs = pickle.load(fi)
    else:
        # print("dict_date/P_A_B_total" + ".file")
        with open("dict_date/P_A_B_total" + ".file", "rb") as fi:
            COPA_word_pairs = pickle.load(fi)
    doc_name = choose[1]
    dict_word2id_icw, COPA_advcl_word_x_counter_icw, COPA_advcl_word_y_counter_icw, COPA_conj_word_x_counter_icw, \
    COPA_conj_word_y_counter_icw, COPA_inter_word_x_counter_icw, COPA_inter_word_y_counter_icw, \
    COPA_advcl_word_pair_counter_icw, COPA_conj_word_pair_counter_icw, COPA_inter_word_pair_counter_icw, \
    word_id_counter_icw = get_data_for_grid(doc_name, name_str)
    doc_name = choose[2]
    # dict_word2id_gut, COPA_advcl_word_x_counter_gut, COPA_advcl_word_y_counter_gut, COPA_conj_word_x_counter_gut, \
    # COPA_conj_word_y_counter_gut, COPA_inter_word_x_counter_gut, COPA_inter_word_y_counter_gut, \
    # COPA_advcl_word_pair_counter_gut, COPA_conj_word_pair_counter_gut, COPA_inter_word_pair_counter_gut, \
    # word_id_counter_gut = get_data_for_grid(doc_name, name_str)
    doc_name = choose[3]
    dict_word2id_bok, COPA_advcl_word_x_counter_bok, COPA_advcl_word_y_counter_bok, COPA_conj_word_x_counter_bok, \
    COPA_conj_word_y_counter_bok, COPA_inter_word_x_counter_bok, COPA_inter_word_y_counter_bok, \
    COPA_advcl_word_pair_counter_bok, COPA_conj_word_pair_counter_bok, COPA_inter_word_pair_counter_bok, \
    word_id_counter_bok = get_data_for_grid(doc_name, name_str)

    prior_prob_other = 0.95

    print("读取数据完毕")
    max_num = 0
    acc_list = []
    max_num = 0
    min_num = 1
    max_list = []
    max_2_list = []
    max_2_num = 0
    for weight_gird_23 in weight23:
        for weight_grid12 in weight:
            for evidence_class_1_probs_icw in icw_grids_666:
                for evidence_class_1_probs_bok in bok_grids_642:
                    for evidence_class_1_probs_gut in gut_grids_604:
                        prior_prob_other = evidence_class_1_probs_icw[4]
                        evidence_priors_icw, evidence_counts_icw = cal_evidence_priors(
                            COPA_word_pairs, dict_word2id_icw, word_id_counter_icw, COPA_advcl_word_pair_counter_icw,
                            COPA_conj_word_pair_counter_icw, COPA_inter_word_pair_counter_icw, prior_prob_other)
                        class_1_evidence_probs_icw, class_0_evidence_probs_icw, class_priors_icw = get_used_parameter(
                            evidence_class_1_probs_icw, evidence_priors_icw)
                        COPA_word_pair_class_likelihood_ratios_icw = cal_all_word_pair_likelihood_ratios(
                            COPA_word_pairs, dict_word2id_icw, word_id_counter_icw, COPA_advcl_word_pair_counter_icw,
                            COPA_conj_word_pair_counter_icw, COPA_inter_word_pair_counter_icw, evidence_priors_icw,
                            evidence_counts_icw, class_1_evidence_probs_icw, class_0_evidence_probs_icw)

                        prior_prob_other = evidence_class_1_probs_bok[4]
                        evidence_priors_bok, evidence_counts_bok = cal_evidence_priors(
                            COPA_word_pairs, dict_word2id_bok, word_id_counter_bok, COPA_advcl_word_pair_counter_bok,
                            COPA_conj_word_pair_counter_bok, COPA_inter_word_pair_counter_bok, prior_prob_other)
                        class_1_evidence_probs_bok, class_0_evidence_probs_bok, class_priors_bok = get_used_parameter(
                            evidence_class_1_probs_bok, evidence_priors_bok)
                        COPA_word_pair_class_likelihood_ratios_bok = cal_all_word_pair_likelihood_ratios(
                            COPA_word_pairs, dict_word2id_bok, word_id_counter_bok, COPA_advcl_word_pair_counter_bok,
                            COPA_conj_word_pair_counter_bok, COPA_inter_word_pair_counter_bok, evidence_priors_bok,
                            evidence_counts_bok, class_1_evidence_probs_bok, class_0_evidence_probs_bok)

                        # prior_prob_other = evidence_class_1_probs_gut[4]
                        # evidence_priors_gut, evidence_counts_gut = cal_evidence_priors(
                        #     COPA_word_pairs, dict_word2id_gut, word_id_counter_gut, COPA_advcl_word_pair_counter_gut,
                        #     COPA_conj_word_pair_counter_gut, COPA_inter_word_pair_counter_gut, prior_prob_other)
                        # class_1_evidence_probs_gut, class_0_evidence_probs_gut, class_priors_gut = get_used_parameter(
                        #     evidence_class_1_probs_gut, evidence_priors_gut)
                        # COPA_word_pair_class_likelihood_ratios_gut = cal_all_word_pair_likelihood_ratios(
                        #     COPA_word_pairs, dict_word2id_gut, word_id_counter_gut, COPA_advcl_word_pair_counter_gut,
                        #     COPA_conj_word_pair_counter_gut, COPA_inter_word_pair_counter_gut, evidence_priors_gut,
                        #     evidence_counts_gut, class_1_evidence_probs_gut, class_0_evidence_probs_gut)

                        files = get_result_file(1)
                        patterns, len_pattern = get_data(1)
                        patterns_list = get_flag_p_a1_a2(patterns, sw_flag, islemma)
                        acc = calculate_result_merge(
                            patterns_list, files, COPA_word_pair_class_likelihood_ratios_icw, class_priors_icw,
                            COPA_word_pair_class_likelihood_ratios_bok, class_priors_bok, weight_grid12)

                        files = get_result_file(2)
                        patterns, len_pattern = get_data(2)
                        patterns_list = get_flag_p_a1_a2(patterns, sw_flag, islemma)
                        acc_test = calculate_result_merge(
                            patterns_list, files, COPA_word_pair_class_likelihood_ratios_icw, class_priors_icw,
                            COPA_word_pair_class_likelihood_ratios_bok, class_priors_bok, weight_grid12)

                        # files = get_result_file(1)
                        # patterns, len_pattern = get_data(1)
                        # patterns_list = get_flag_p_a1_a2(patterns, sw_flag, islemma)
                        # acc = calculate_result_merge_three(
                        #     patterns_list, files, COPA_word_pair_class_likelihood_ratios_icw, class_priors_icw,
                        #     COPA_word_pair_class_likelihood_ratios_gut, class_priors_gut,
                        #     COPA_word_pair_class_likelihood_ratios_bok, class_priors_bok,
                        #     weight_grid12, weight_gird_23)
                        #
                        # files = get_result_file(2)
                        # patterns, len_pattern = get_data(2)
                        # patterns_list = get_flag_p_a1_a2(patterns, sw_flag, islemma)
                        # acc_test = calculate_result_merge_three(
                        #     patterns_list, files, COPA_word_pair_class_likelihood_ratios_icw, class_priors_icw,
                        #     COPA_word_pair_class_likelihood_ratios_gut, class_priors_gut,
                        #     COPA_word_pair_class_likelihood_ratios_bok, class_priors_bok,
                        #     weight_grid12, weight_gird_23)

                        acc_list.append(acc)
                        print(acc, acc_test, weight_gird_23, weight_grid12)
                        print(evidence_class_1_probs_icw, evidence_class_1_probs_bok, evidence_class_1_probs_gut)
                        if acc > max_num:
                            max_2_num = acc - 0.002
                            max_2_list = []
                            if max_2_num == max_num:
                                max_2_list = max_list
                            max_num = acc
                            print('更新max:', max_num, [
                                weight_gird_23, weight_grid12, evidence_class_1_probs_icw, evidence_class_1_probs_bok,
                                evidence_class_1_probs_gut])
                            max_list = []
                        if acc == max_num:
                            max_list.append(["acc_test:", acc_test, weight_gird_23, weight_grid12,
                                             evidence_class_1_probs_icw,
                                             evidence_class_1_probs_bok, evidence_class_1_probs_gut])
                        if acc == max_2_num:
                            max_2_list.append(["acc_test:", acc_test, weight_gird_23, weight_grid12,
                                               evidence_class_1_probs_icw,
                                               evidence_class_1_probs_bok, evidence_class_1_probs_gut])
                        if acc < min_num:
                            min_num = acc
                            print('更新min:', min_num)
            print('max:', max_num, max_list)
            print('max_2:', max_2_num, max_2_list)
            print('min:', min_num)
    return acc_list


def get_copa_dict():
    # 　 获取得到3Wpair对应的单词
    with open("dict_date/P_A_B_total" + ".file", "rb") as f:
        P_A_B_pair = pickle.load(f)
    with open('dict_date/str_id_' + 'word_count_delete.file', 'rb') as f:
        str_id_word_count = pickle.load(f)
    choose_list = ['', '_no', '_stop']
    name_str = choose_list[2]
    # 获得advcl
    advcl_p_a_pos_wei = np.load('database_original/advcl_p_a_pos_wei' + name_str + '1.npy')
    advcl_p_b_pos_wei = np.load('database_original/advcl_p_b_pos_wei' + name_str + '1.npy')
    # 获得conj
    conj_p_a_pos_wei = np.load('database_original/conj_p_a_pos_wei' + name_str + '1.npy')
    conj_p_b_pos_wei = np.load('database_original/conj_p_b_pos_wei' + name_str + '1.npy')
    # 获得inter
    inter_level_p_a_pos = np.load('database_original/inter_level_p_a_pos' + name_str + '1.npy')
    inter_level_p_b_pos = np.load('database_original/inter_level_p_b_pos' + name_str + '1.npy')
    # 获得advcl
    db_path = 'database_original/advcl_p_a_b_pos_wei' + name_str + '1.npy'
    print(db_path)
    advcl_p_a_b_pos_wei = np.load(db_path)
    # 获得conj
    db_path = 'database_original/conj_p_a_b_pos_wei' + name_str + '1.npy'
    print(db_path)
    conj_p_a_b_pos_wei = np.load(db_path)
    # 获得inter
    db_path = 'database_original/inter_level_p_a_b_pos' + name_str + '1.npy'
    print(db_path)
    inter_level_p_a_b_pos = np.load(db_path)

    advcl_pair_pos_copa = {}
    conj_pair_pos_copa = {}
    inter_pair_pos_copa = {}
    for pair in P_A_B_pair:
        a = pair.split("_")
        str_p_a = a[0]
        str_p_b = a[1]
        if (str_p_a in str_id_word_count) & (str_p_b in str_id_word_count):
            # 将str转换成id
            id_p_a = str_id_word_count[str_p_a]
            id_p_b = str_id_word_count[str_p_b]

            advcl_count = advcl_p_a_b_pos_wei[id_p_a][id_p_b]
            advcl_pair_pos_copa[pair] = advcl_count

            conj_count = conj_p_a_b_pos_wei[id_p_a][id_p_b]
            conj_pair_pos_copa[pair] = conj_count

            inter_count = inter_level_p_a_b_pos[id_p_a][id_p_b]
            inter_pair_pos_copa[pair] = inter_count
    advcl_sum = np.sum(advcl_p_a_b_pos_wei != 0)
    conj_sum = np.sum(conj_p_a_b_pos_wei != 0)
    inter_sum = np.sum(inter_level_p_a_b_pos != 0)
    advcl_pair_pos_copa['sum'] = advcl_sum
    conj_pair_pos_copa['sum'] = conj_sum
    inter_pair_pos_copa['sum'] = inter_sum
    advcl_count = np.sum(advcl_p_a_b_pos_wei)
    conj_count = np.sum(conj_p_a_b_pos_wei)
    inter_count = np.sum(inter_level_p_a_b_pos)
    advcl_pair_pos_copa['count'] = advcl_count
    conj_pair_pos_copa['count'] = conj_count
    inter_pair_pos_copa['count'] = inter_count

    with open('database_original/temp/' + 'advcl_pair_pos_copa' + '.file', 'wb') as f:
        pickle.dump(advcl_pair_pos_copa, f)
    with open('database_original/temp/' + 'conj_pair_pos_copa' + '.file', 'wb') as f:
        pickle.dump(conj_pair_pos_copa, f)
    with open('database_original/temp/' + 'inter_pair_pos_copa' + '.file', 'wb') as f:
        pickle.dump(inter_pair_pos_copa, f)
        # 获得advcl
    np.save('database_original/temp/advcl_p_a_pos_wei' + name_str + '.npy', advcl_p_a_pos_wei)
    np.save('database_original/temp/advcl_p_b_pos_wei' + name_str + '.npy', advcl_p_b_pos_wei)
    # 获得conj
    np.save('database_original/temp/conj_p_a_pos_wei' + name_str + '.npy', conj_p_a_pos_wei)
    np.save('database_original/temp/conj_p_b_pos_wei' + name_str + '.npy', conj_p_b_pos_wei)
    # 获得inter
    np.save('database_original/temp/inter_level_p_a_pos' + name_str + '.npy', inter_level_p_a_pos)
    np.save('database_original/temp/inter_level_p_b_pos' + name_str + '.npy', inter_level_p_b_pos)


def main(test_id):
    data_id = test_id
    choose = ['', '_icw', '_gut', '_book_corpus']

    acc_list = grid_tuning_five(data_id, '', choose[1])
    # acc_list = grid_tuning_five(data_id, '', choose[2])
    # acc_list = grid_tuning_five(data_id, '', choose[3])

    # acc_list = grid_tuning_merge(data_id, '')
    # acc_list = grid_tuning_merge_spe(data_id, '')

    return acc_list


if __name__ == '__main__':
    # get_copa_dict()
    # 次数1代表dev 2代表test
    dev_acc = main(1)
    print_time()

# 单个数据集:
# max: 0.666
# max: 0.668
# 0.98
# [[0.686, [0.6, 0.05, 0.3, 0.0001]]]
# max_2: 0.666
# [[0.688, [0.6, 0.05, 0.35, 0.0001]], [0.682,  [0.7, 0.05, 0.3, 0.0001]], [0.684, [0.7, 0.05, 0.35, 0.0001]],
#  [0.686,  [0.7, 0.05, 0.4, 0.0001]], [0.68, [0.8, 0.05, 0.35, 0.0001]]]
# 0.99
# [[0.686,  [0.7, 0.05, 0.3, 0.0001]], [0.686,  [0.7, 0.05, 0.35, 0.0001]] [0.686,  [0.8, 0.05, 0.35, 0.0001]]
#  [0.686,  [0.8, 0.05, 0.4, 0.0001]], [0.686,  [0.8, 0.05, 0.45, 0.0001]]
# 模型融合结果

# max: 0.674 [['acc_test:', 0.692, 0.01, [0.58, 0.03, 0.24, 0.0001, 0.98], [0.2, 0.03, 0.44, 0.0001, 0.9], []],
#             ['acc_test:', 0.692, 0.01, [0.58, 0.03, 0.24, 0.0001, 0.98], [0.22, 0.05, 0.4, 0.0001, 0.9], []],
#             ['acc_test:', 0.692, 0.01, [0.58, 0.03, 0.24, 0.0001, 0.98], [0.22, 0.05, 0.42, 0.0001, 0.9], []],
#             ['acc_test:', 0.692, 0.01, [0.58, 0.03, 0.24, 0.0001, 0.98], [0.24, 0.05, 0.44, 0.0001, 0.9], []],
#             ['acc_test:', 0.69, 0.02, [0.42, 0.02, 0.22, 0.0001, 0.98], [0.24, 0.05, 0.42, 0.0001, 0.9], []],
#             ['acc_test:', 0.696, 0.02, [0.58, 0.03, 0.24, 0.0001, 0.98], [0.2, 0.03, 0.44, 0.0001, 0.9], []],
#             ['acc_test:', 0.698, 0.02, [0.58, 0.03, 0.24, 0.0001, 0.98], [0.24, 0.05, 0.42, 0.0001, 0.9], []]]


# bok
#
#     bok_grids_646 = [[0.2, 0.03, 0.44, 0.0001, 0.9], [0.22, 0.04, 0.5, 0.0001, 0.9], [0.22, 0.05, 0.4, 0.0001, 0.9],
#              [0.22, 0.05, 0.42, 0.0001, 0.9], [0.24, 0.05, 0.42, 0.0001, 0.9], [0.24, 0.05, 0.44, 0.0001, 0.9]]

# gut
# max: 0.604 [[0.634, [0.64, 0.01, 0.08, 0.0001, 0.9]]]
# max: 0.604 [[0.636, [0.86, 0.02, 0.02, 0.001, 0.85]], [0.634, [0.88, 0.02, 0.02, 0.001, 0.85]]]
# max: 0.604 [[0.632, [0.24, 0.01, 0.04, 0.0001, 0.85]], [0.634, [0.48, 0.02, 0.08, 0.0001, 0.85]],
#             [0.636, [0.7, 0.03, 0.12, 0.0001, 0.85]], [0.63, [0.84, 0.02, 0.02, 0.0001, 0.85]],
#             [0.63, [0.86, 0.02, 0.02, 0.0001, 0.85]], [0.63, [0.88, 0.02, 0.02, 0.0001, 0.85]]]
# max_2: 0.602 [[0.634, [0.26, 0.01, 0.04, 0.0001, 0.85]], [0.638, [0.3, 0.01, 0.04, 0.0001, 0.85]],
#              [0.636, [0.5, 0.02, 0.08, 0.0001, 0.85]], [0.64, [0.6, 0.02, 0.08, 0.0001, 0.85]],
#               [0.636, [0.66, 0.03, 0.1, 0.0001, 0.85]], [0.638, [0.74, 0.02, 0.1, 0.0001, 0.85]]]
# max_2: 0.602 [[0.636, [0.5, 0.01, 0.06, 0.0001, 0.9]], [0.636, [0.6, 0.01, 0.08, 0.0001, 0.9]]]
# max: 0.602 [[0.64, [0.86, 0.01, 0.02, 0.0001, 0.98]]]
# max_2: 0.6 [[0.642, [0.82, 0.01, 0.02, 0.0001, 0.98]], [0.64, [0.84, 0.01, 0.02, 0.0001, 0.98]],
#             [0.64, [0.88, 0.01, 0.02, 0.0001, 0.98]]]
