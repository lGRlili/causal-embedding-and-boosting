import argparse
import codecs
import multiprocessing
import pickle
import pandas as pd
import numpy as np
from new_test_7_3 import extract_API
from new_test_7_3 import API_2
from datetime import datetime

starts = datetime.now()


# 此时不用在句子内部的全部语料
# 根据获得的pair队,构建dict---形成全局的dict
# 构建函数---分别计算p(a/b)p(a)p(b)


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


class parse(object):
    # 保存数据的数据结构
    def __init__(self, text, lemma_, pos_, dep_, head, id, child, left, right, ancestor):
        self.text = text
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.dep_ = dep_
        self.head = head
        self.id = id
        self.child = child
        self.left = left
        self.right = right
        self.ancestor = ancestor


def save_date(dict_list, dict_list_str, sw_flag):
    for i, dict_name in enumerate(dict_list):
        print(dict_list_str[i])
        print([i for i in dict_name.values()][:100])
        with open("dict_date2/" + dict_list_str[i] + sw_flag + ".file", "wb") as f:
            pickle.dump(dict_name, f)


def extract_text(start_id, end_id, type_kind):
    array_len = 50000
    print(type_kind)
    advcl_p_a_b_pos_wei = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    conj_p_a_b_pos_wei = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    inter_level_p_a_b_pos = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    if type_kind == 'advcl_p_a_b_pos_wei':
        advcl_p_a_b_pos_wei.fill(0)
    elif type_kind == 'conj_p_a_b_pos_wei':
        conj_p_a_b_pos_wei.fill(0)
    elif type_kind == 'inter_level_p_a_b_pos':
        inter_level_p_a_b_pos.fill(0)

    with open('dict_date/str_id_' + 'word_count.file', 'rb') as f:
        str_id_word_count = pickle.load(f)
    with open('dict_date/str_id_' + 'word_count_delete.file', 'rb') as f:
        str_id_word_count = pickle.load(f)
    print(len(str_id_word_count))
    # for word in str_id_word_count:
    #     print(word, str_id_word_count[word])

    # 定义需要存储的字典
    # 句内计算的变量
    count_advcl = 0
    count_conj = 0
    count_inter = 0
    for count in range(start_id, end_id):
        if count == 6214:
            continue
        doc_path = '../../icwsm09stories_date/spacy_data/' + str(count) + ".pkl"
        print(doc_path)
        date = pd.read_pickle(doc_path)
        doc = date['doc']
        for book in doc:
            # 此处是一个文本
            last_setence = []
            last_congju = []
            for sen in book:
                total_clause, total_clause_str, total_clause_level, total_clause_level_str, len_total_congju = \
                    extract_API.parse_setence(sen)

                # 句内关系
                if type_kind == 'advcl_p_a_b_pos_wei':
                    count_advcl, advcl_p_a_b_pos_wei = extract_API.cal_intra_advcl_wei(sen, total_clause_level,
                                                                                       advcl_p_a_b_pos_wei,
                                                                                       str_id_word_count, count_advcl)
                elif type_kind == 'conj_p_a_b_pos_wei':
                    count_conj, conj_p_a_b_pos_wei = extract_API.cal_intra_conj_wei(sen, total_clause_level,
                                                                                    conj_p_a_b_pos_wei,
                                                                                    str_id_word_count,
                                                                                    count_conj)
                elif type_kind == 'inter_level_p_a_b_pos':
                    count_inter, inter_level_p_a_b_pos, last_setence, last_congju = \
                        extract_API.cal_p_a_b_inter_num_weight_dis(sen, total_clause, inter_level_p_a_b_pos,
                                                                   str_id_word_count, last_setence, last_congju,
                                                                   count_inter)
        if count % 10 == 0:
            print(count, 'count_advcl:', count_advcl, 'count_conj:', count_conj, 'count_inter:', count_inter)
        if count % 1000 == 0:
            if type_kind == 'advcl_p_a_b_pos_wei':
                np.save("database/advcl_p_a_b_pos_wei" + str(start_id) + ".npy", advcl_p_a_b_pos_wei)
            elif type_kind == 'conj_p_a_b_pos_wei':
                np.save("database/conj_p_a_b_pos_wei" + str(start_id) + ".npy", conj_p_a_b_pos_wei)
            elif type_kind == 'inter_level_p_a_b_pos':
                np.save("database/inter_level_p_a_b_pos" + str(start_id) + ".npy", inter_level_p_a_b_pos)
            print_time()
            print(count, '安全保存')
            print(count, '---' * 45)

    if type_kind == 'advcl_p_a_b_pos_wei':
        np.save("database/advcl_p_a_b_pos_wei" + str(start_id) + ".npy", advcl_p_a_b_pos_wei)
    elif type_kind == 'conj_p_a_b_pos_wei':
        np.save("database/conj_p_a_b_pos_wei" + str(start_id) + ".npy", conj_p_a_b_pos_wei)
    elif type_kind == 'inter_level_p_a_b_pos':
        np.save("database/inter_level_p_a_b_pos" + str(start_id) + ".npy", inter_level_p_a_b_pos)
    print_time()
    print('安全保存')
    print('---' * 45)


def extract_text_up(start_id, end_id, type_kind):
    print('extract_text_up')
    stopkey = [w.strip() for w in codecs.open('stop_word.txt', 'r', encoding='utf-8').readlines()]
    # print(len(stopkey))
    stopkey = set(stopkey)
    array_len = 50000
    print(type_kind)
    advcl_p_a_b_pos_wei = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    conj_p_a_b_pos_wei = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    inter_level_p_a_b_pos = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    inter_level_p_a_b_pos_weight = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    if type_kind == 'advcl_p_a_b_pos_wei':
        advcl_p_a_b_pos_wei.fill(0)
    elif type_kind == 'conj_p_a_b_pos_wei':
        conj_p_a_b_pos_wei.fill(0)
    elif type_kind == 'inter_level_p_a_b_pos':
        inter_level_p_a_b_pos.fill(0)
    elif type_kind == 'inter_level_p_a_b_pos_weight':
        inter_level_p_a_b_pos_weight.fill(0)

    # with open('dict_date/str_id_' + 'word_count.file', 'rb') as f:
    #     str_id_word_count = pickle.load(f)
    with open('dict_date/str_id_' + 'word_count_delete.file', 'rb') as f:
        str_id_word_count = pickle.load(f)
    print(len(str_id_word_count))
    # for word in str_id_word_count:
    #     print(word, str_id_word_count[word])

    count_advcl = 0
    count_conj = 0
    count_inter = 0
    count_inter_weight = 0
    for count in range(start_id, end_id):
        if count == 6214:
            continue
        doc_path = '../../icwsm09stories_date/spacy_data/' + str(count) + ".pkl"
        print(doc_path)
        date = pd.read_pickle(doc_path)
        doc = date['doc']
        for book in doc:
            # 此处是一个文本
            last_setence = []
            last_congju = []
            for sen in book:
                total_clause, total_clause_str, total_clause_level, total_clause_level_str, len_total_congju = \
                    extract_API.parse_setence(sen)

                # 句内关系
                if type_kind == 'advcl_p_a_b_pos_wei':
                    count_advcl, advcl_p_a_b_pos_wei = extract_API.cal_intra_advcl_wei_up(sen, total_clause_level,
                                                                                          advcl_p_a_b_pos_wei,
                                                                                          str_id_word_count,
                                                                                          count_advcl, stopkey)
                elif type_kind == 'conj_p_a_b_pos_wei':
                    count_conj, conj_p_a_b_pos_wei = extract_API.cal_intra_conj_wei_up(sen, total_clause_level,
                                                                                       conj_p_a_b_pos_wei,
                                                                                       str_id_word_count,
                                                                                       count_conj, stopkey)
                elif type_kind == 'inter_level_p_a_b_pos':
                    count_inter, inter_level_p_a_b_pos, last_setence, last_congju = \
                        extract_API.cal_p_a_b_inter_num_weight_dis_up(sen, total_clause, inter_level_p_a_b_pos,
                                                                      str_id_word_count, last_setence, last_congju,
                                                                      count_inter, stopkey)
                elif type_kind == 'inter_level_p_a_b_pos_weight':
                    count_inter_weight, inter_level_p_a_b_pos_weight, last_setence, last_congju = \
                        extract_API.cal_p_a_b_inter_num_weight_dis_weight_up(sen, total_clause,
                                                                             inter_level_p_a_b_pos_weight,
                                                                             str_id_word_count, last_setence,
                                                                             last_congju,
                                                                             count_inter, stopkey)
        if count % 10 == 0:
            print(count, 'count_advcl:', count_advcl, 'count_conj:', count_conj, 'count_inter:', count_inter,
                  'count_inter_weight', count_inter_weight)
        if count % 1000 == 0:
            if type_kind == 'advcl_p_a_b_pos_wei':
                np.save("database/up_advcl_p_a_b_pos_wei" + str(start_id) + ".npy", advcl_p_a_b_pos_wei)
            elif type_kind == 'conj_p_a_b_pos_wei':
                np.save("database/up_conj_p_a_b_pos_wei" + str(start_id) + ".npy", conj_p_a_b_pos_wei)
            elif type_kind == 'inter_level_p_a_b_pos':
                np.save("database/up_inter_level_p_a_b_pos" + str(start_id) + ".npy", inter_level_p_a_b_pos)
            elif type_kind == 'inter_level_p_a_b_pos_weight':
                np.save("database/up_inter_level_p_a_b_pos_weight" + str(start_id) + ".npy",
                        inter_level_p_a_b_pos_weight)
            print_time()
            print(count, '安全保存')
            print(count, '---' * 45)

    if type_kind == 'advcl_p_a_b_pos_wei':
        np.save("database/up_advcl_p_a_b_pos_wei" + str(start_id) + ".npy", advcl_p_a_b_pos_wei)
    elif type_kind == 'conj_p_a_b_pos_wei':
        np.save("database/up_conj_p_a_b_pos_wei" + str(start_id) + ".npy", conj_p_a_b_pos_wei)
    elif type_kind == 'inter_level_p_a_b_pos':
        np.save("database/up_inter_level_p_a_b_pos" + str(start_id) + ".npy", inter_level_p_a_b_pos)
    elif type_kind == 'inter_level_p_a_b_pos_weight':
        np.save("database/up_inter_level_p_a_b_pos_weight" + str(start_id) + ".npy", inter_level_p_a_b_pos_weight)
    print_time()
    print('安全保存')
    print('---' * 45)


def extract_text_conj(start_id, end_id, type_kind):
    array_len = 50000
    print(type_kind)
    # 保存 conj pa pb pab
    conj_p_a_pos_wei = np.zeros(shape=array_len, dtype=np.float32)
    conj_p_b_pos_wei = np.zeros(shape=array_len, dtype=np.float32)
    conj_p_a_pos_wei.fill(0)
    conj_p_b_pos_wei.fill(0)
    conj_p_a_b_pos_wei = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    conj_p_a_b_pos_wei.fill(0)
    with open('dict_date/str_id_' + 'word_count.file', 'rb') as f:
        str_id_word_count = pickle.load(f)
    print(len(str_id_word_count))
    # for word in str_id_word_count:
    #     print(word, str_id_word_count[word])
    # 定义需要存储的字典
    # 句内计算的变量
    count_conj = 0
    for count in range(start_id, end_id):
        if count == 6214:
            continue
        doc_path = '../../icwsm09stories_date/spacy_data/' + str(count) + ".pkl"
        print(doc_path)
        date = pd.read_pickle(doc_path)
        doc = date['doc']
        for book in doc:
            # 此处是一个文本
            last_setence = []
            last_congju = []
            for sen in book:
                total_clause, total_clause_str, total_clause_level, total_clause_level_str, len_total_congju = \
                    extract_API.parse_setence(sen)
                conj_p_a_pos_wei, conj_p_b_pos_wei = \
                    extract_API.cal_intra_conj_wei_speci_ab(sen, total_clause_level, conj_p_a_pos_wei, conj_p_b_pos_wei,
                                                            str_id_word_count)

                count_conj, conj_p_a_b_pos_wei = extract_API.cal_intra_conj_wei_speci(sen, total_clause_level,
                                                                                      conj_p_a_b_pos_wei,
                                                                                      str_id_word_count, count_conj)
        if count % 10 == 0:
            print(count, 'count_conj:', count_conj)
        if count % 1000 == 0:
            np.save("database/conj_p_a_pos_wei_speci" + str(start_id) + ".npy", conj_p_a_pos_wei)
            np.save("database/conj_p_b_pos_wei_speci" + str(start_id) + ".npy", conj_p_b_pos_wei)
            np.save("database/conj_p_a_b_pos_wei_speci" + str(start_id) + ".npy", conj_p_a_b_pos_wei)
            print_time()
            print(count, '安全保存')
            print(count, '---' * 45)
    np.save("database/conj_p_a_pos_wei_speci" + str(start_id) + ".npy", conj_p_a_pos_wei)
    np.save("database/conj_p_b_pos_wei_speci" + str(start_id) + ".npy", conj_p_b_pos_wei)
    np.save("database/conj_p_a_b_pos_wei_speci" + str(start_id) + ".npy", conj_p_a_b_pos_wei)
    print_time()
    print('安全保存')
    print('---' * 45)


def extract_text_2(start_id, end_id, type_kind):
    stopkey = [w.strip() for w in codecs.open('stop_word.txt', 'r', encoding='utf-8').readlines()]
    # print(len(stopkey))
    stopkey = set(stopkey)
    array_len = 50000
    print(type_kind)
    advcl_p_a_b_pos_wei = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    conj_p_a_b_pos_wei = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    inter_level_p_a_b_pos = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    if type_kind == 'advcl_p_a_b_pos_wei':
        advcl_p_a_b_pos_wei.fill(0)
    elif type_kind == 'conj_p_a_b_pos_wei':
        conj_p_a_b_pos_wei.fill(0)
    elif type_kind == 'inter_level_p_a_b_pos':
        inter_level_p_a_b_pos.fill(0)

    with open('dict_date/str_id_' + 'word_count.file', 'rb') as f:
        str_id_word_count = pickle.load(f)
    with open('dict_date/str_id_' + 'word_count_delete.file', 'rb') as f:
        str_id_word_count = pickle.load(f)
    print(len(str_id_word_count))
    # for word in str_id_word_count:
    #     print(word, str_id_word_count[word])

    # 定义需要存储的字典
    # 句内计算的变量
    count_advcl = 0
    count_conj = 0
    count_inter = 0
    for count in range(start_id, end_id):
        if count == 6214:
            continue
        doc_path = '../../icwsm09stories_date/spacy_data/' + str(count) + ".pkl"
        print(doc_path)
        date = pd.read_pickle(doc_path)
        doc = date['doc']
        for book in doc:
            # 此处是一个文本
            last_setence = []
            last_congju = []
            for sen in book:
                total_clause, total_clause_str, total_clause_level, total_clause_level_str, len_total_congju = \
                    API_2.parse_setence(sen)

                # 句内关系
                if type_kind == 'advcl_p_a_b_pos_wei':
                    count_advcl, advcl_p_a_b_pos_wei = API_2.cal_intra_advcl_wei(sen, total_clause_level,
                                                                                 advcl_p_a_b_pos_wei,
                                                                                 str_id_word_count, count_advcl)
                elif type_kind == 'conj_p_a_b_pos_wei':
                    count_conj, conj_p_a_b_pos_wei = API_2.cal_intra_conj_wei(sen, total_clause_level,
                                                                              conj_p_a_b_pos_wei,
                                                                              str_id_word_count,
                                                                              count_conj)
                elif type_kind == 'inter_level_p_a_b_pos':
                    count_inter, inter_level_p_a_b_pos, last_setence, last_congju = \
                        API_2.cal_p_a_b_inter_num_weight_dis(sen, total_clause, inter_level_p_a_b_pos,
                                                             str_id_word_count, last_setence, last_congju,
                                                             count_inter)
        if count % 10 == 0:
            print(count, 'count_advcl:', count_advcl, 'count_conj:', count_conj, 'count_inter:', count_inter)
        if count % 1000 == 0:
            if type_kind == 'advcl_p_a_b_pos_wei':
                np.save("database_original/advcl_p_a_b_pos_wei" + str(start_id) + ".npy", advcl_p_a_b_pos_wei)
            elif type_kind == 'conj_p_a_b_pos_wei':
                np.save("database_original/conj_p_a_b_pos_wei" + str(start_id) + ".npy", conj_p_a_b_pos_wei)
            elif type_kind == 'inter_level_p_a_b_pos':
                np.save("database_original/inter_level_p_a_b_pos" + str(start_id) + ".npy", inter_level_p_a_b_pos)
            print_time()
            print(count, '安全保存')
            print(count, '---' * 45)

    if type_kind == 'advcl_p_a_b_pos_wei':
        np.save("database_original/advcl_p_a_b_pos_wei" + str(start_id) + ".npy", advcl_p_a_b_pos_wei)
    elif type_kind == 'conj_p_a_b_pos_wei':
        np.save("database_original/conj_p_a_b_pos_wei" + str(start_id) + ".npy", conj_p_a_b_pos_wei)
    elif type_kind == 'inter_level_p_a_b_pos':
        np.save("database_original/inter_level_p_a_b_pos" + str(start_id) + ".npy", inter_level_p_a_b_pos)
    print_time()
    print('安全保存')
    print('---' * 45)


def extract_text_stop(start_id, end_id, type_kind):
    print('extract_text_stop')
    stopkey = [w.strip() for w in codecs.open('stop_word.txt', 'r', encoding='utf-8').readlines()]
    # print(len(stopkey))
    stopkey = set(stopkey)
    array_len = 50000
    print(type_kind)
    advcl_p_a_b_pos_wei = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    conj_p_a_b_pos_wei = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    inter_level_p_a_b_pos = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    inter_p_a_b_pos_weight = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    if type_kind == 'advcl_p_a_b_pos_wei':
        advcl_p_a_b_pos_wei.fill(0)
    elif type_kind == 'conj_p_a_b_pos_wei':
        conj_p_a_b_pos_wei.fill(0)
    elif type_kind == 'inter_level_p_a_b_pos':
        inter_level_p_a_b_pos.fill(0)
    elif type_kind == 'inter_p_a_b_pos':
        inter_p_a_b_pos_weight.fill(0)

    with open('dict_date/str_id_' + 'word_count_delete.file', 'rb') as f:
        str_id_word_count = pickle.load(f)
    print(len(str_id_word_count))
    # for word in str_id_word_count:
    #     print(word, str_id_word_count[word])

    # 定义需要存储的字典
    # 句内计算的变量
    count_advcl = 0
    count_conj = 0
    count_inter = 0
    for count in range(start_id, end_id):
        if count == 6214:
            continue
        doc_path = '../../icwsm09stories_date/spacy_data/' + str(count) + ".pkl"
        print(doc_path)
        date = pd.read_pickle(doc_path)
        doc = date['doc']
        for book in doc:
            # 此处是一个文本
            last_setence = []
            last_congju = []
            for sen in book:
                total_clause, total_clause_str, total_clause_level, total_clause_level_str, len_total_congju = \
                    API_2.parse_setence(sen)

                # 句内关系
                if type_kind == 'advcl_p_a_b_pos_wei':
                    count_advcl, advcl_p_a_b_pos_wei = API_2.cal_intra_advcl_wei_stop_with(
                        sen, total_clause_level, advcl_p_a_b_pos_wei, str_id_word_count, count_advcl, stopkey)
                elif type_kind == 'conj_p_a_b_pos_wei':
                    count_conj, conj_p_a_b_pos_wei = API_2.cal_intra_conj_wei_stop_with(
                        sen, total_clause_level, conj_p_a_b_pos_wei, str_id_word_count, count_conj, stopkey)
                elif type_kind == 'inter_level_p_a_b_pos':
                    count_inter, inter_level_p_a_b_pos, last_setence, last_congju = \
                        API_2.cal_p_a_b_inter_num_weight_dis_stop_with(
                            sen, total_clause, inter_level_p_a_b_pos, str_id_word_count, last_setence, last_congju,
                            count_inter, stopkey)
                elif type_kind == 'inter_p_a_b_pos':
                    count_inter, inter_p_a_b_pos_weight, last_setence, last_congju = \
                        API_2.cal_p_a_b_inter_stop_with(
                            sen, total_clause, inter_p_a_b_pos_weight, str_id_word_count, last_setence, last_congju,
                            count_inter, stopkey)
        if count % 10 == 0:
            print(count, 'count_advcl:', count_advcl, 'count_conj:', count_conj, 'count_inter:', count_inter)
        if count % 1000 == 0:
            if type_kind == 'advcl_p_a_b_pos_wei':
                np.save("database_original/advcl_p_a_b_pos_wei_stop" + str(start_id) + ".npy", advcl_p_a_b_pos_wei)
            elif type_kind == 'conj_p_a_b_pos_wei':
                np.save("database_original/conj_p_a_b_pos_wei_stop" + str(start_id) + ".npy", conj_p_a_b_pos_wei)
            elif type_kind == 'inter_level_p_a_b_pos':
                np.save("database_original/inter_level_p_a_b_pos_stop" + str(start_id) + ".npy", inter_level_p_a_b_pos)
            elif type_kind == 'inter_p_a_b_pos':
                np.save("database_original/inter_p_a_b_pos_stop" + str(start_id) + ".npy", inter_p_a_b_pos_weight)
            print_time()
            print(count, '安全保存')
            print(count, '---' * 45)

    if type_kind == 'advcl_p_a_b_pos_wei':
        np.save("database_original/advcl_p_a_b_pos_wei_stop" + str(start_id) + ".npy", advcl_p_a_b_pos_wei)
    elif type_kind == 'conj_p_a_b_pos_wei':
        np.save("database_original/conj_p_a_b_pos_wei_stop" + str(start_id) + ".npy", conj_p_a_b_pos_wei)
    elif type_kind == 'inter_level_p_a_b_pos':
        np.save("database_original/inter_level_p_a_b_pos_stop" + str(start_id) + ".npy", inter_level_p_a_b_pos)
    elif type_kind == 'inter_p_a_b_pos':
        np.save("database_original/inter_p_a_b_pos_stop" + str(start_id) + ".npy", inter_p_a_b_pos_weight)
    print_time()
    print('安全保存')
    print('---' * 45)


def extract_text_stop_no(start_id, end_id, type_kind):
    print('extract_text_stop_no')
    stopkey = [w.strip() for w in codecs.open('stop_word.txt', 'r', encoding='utf-8').readlines()]
    # print(len(stopkey))
    stopkey = set(stopkey)
    array_len = 50000
    print(type_kind)
    advcl_p_a_b_pos_wei = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    conj_p_a_b_pos_wei = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    inter_level_p_a_b_pos = np.zeros(shape=(array_len, array_len), dtype=np.float32)
    if type_kind == 'advcl_p_a_b_pos_wei':
        advcl_p_a_b_pos_wei.fill(0)
    elif type_kind == 'conj_p_a_b_pos_wei':
        conj_p_a_b_pos_wei.fill(0)
    elif type_kind == 'inter_level_p_a_b_pos':
        inter_level_p_a_b_pos.fill(0)

    with open('dict_date/str_id_' + 'word_count.file', 'rb') as f:
        str_id_word_count = pickle.load(f)
    with open('dict_date/str_id_' + 'word_count_delete.file', 'rb') as f:
        str_id_word_count = pickle.load(f)
    print(len(str_id_word_count))
    # for word in str_id_word_count:
    #     print(word, str_id_word_count[word])

    # 定义需要存储的字典
    # 句内计算的变量
    count_advcl = 0
    count_conj = 0
    count_inter = 0
    for count in range(start_id, end_id):
        if count == 6214:
            continue
        doc_path = '../../icwsm09stories_date/spacy_data/' + str(count) + ".pkl"
        print(doc_path)
        date = pd.read_pickle(doc_path)
        doc = date['doc']
        for book in doc:
            # 此处是一个文本
            last_setence = []
            last_congju = []
            for sen in book:
                total_clause, total_clause_str, total_clause_level, total_clause_level_str, len_total_congju = \
                    API_2.parse_setence(sen)

                # 句内关系
                if type_kind == 'advcl_p_a_b_pos_wei':
                    count_advcl, advcl_p_a_b_pos_wei = API_2.cal_intra_advcl_wei_stop_no(sen, total_clause_level,
                                                                                         advcl_p_a_b_pos_wei,
                                                                                         str_id_word_count, count_advcl)
                elif type_kind == 'conj_p_a_b_pos_wei':
                    count_conj, conj_p_a_b_pos_wei = API_2.cal_intra_conj_wei_stop_no(sen, total_clause_level,
                                                                                      conj_p_a_b_pos_wei,
                                                                                      str_id_word_count,
                                                                                      count_conj)
                elif type_kind == 'inter_level_p_a_b_pos':
                    count_inter, inter_level_p_a_b_pos, last_setence, last_congju = \
                        API_2.cal_p_a_b_inter_num_weight_dis_stop_no(sen, total_clause, inter_level_p_a_b_pos,
                                                                     str_id_word_count, last_setence, last_congju,
                                                                     count_inter)
        if count % 10 == 0:
            print(count, 'count_advcl:', count_advcl, 'count_conj:', count_conj, 'count_inter:', count_inter)
        if count % 1000 == 0:
            if type_kind == 'advcl_p_a_b_pos_wei':
                np.save("database_original/advcl_p_a_b_pos_wei_no" + str(start_id) + ".npy", advcl_p_a_b_pos_wei)
            elif type_kind == 'conj_p_a_b_pos_wei':
                np.save("database_original/conj_p_a_b_pos_wei_no" + str(start_id) + ".npy", conj_p_a_b_pos_wei)
            elif type_kind == 'inter_level_p_a_b_pos':
                np.save("database_original/inter_level_p_a_b_pos_no" + str(start_id) + ".npy", inter_level_p_a_b_pos)
            print_time()
            print(count, '安全保存')
            print(count, '---' * 45)

    if type_kind == 'advcl_p_a_b_pos_wei':
        np.save("database_original/advcl_p_a_b_pos_wei_no" + str(start_id) + ".npy", advcl_p_a_b_pos_wei)
    elif type_kind == 'conj_p_a_b_pos_wei':
        np.save("database_original/conj_p_a_b_pos_wei_no" + str(start_id) + ".npy", conj_p_a_b_pos_wei)
    elif type_kind == 'inter_level_p_a_b_pos':
        np.save("database_original/inter_level_p_a_b_pos_no" + str(start_id) + ".npy", inter_level_p_a_b_pos)
    print_time()
    print('安全保存')
    print('---' * 45)


def main():
    sw_flags = ['', '_without_sw']
    lock = multiprocessing.Lock()
    start_end_list = [[1, 1700], [1700, 3400], [3400, 5100], [5100, 6800]]
    # 创建多个进程
    thread_list = []
    for i in range(1):
        sthread = multiprocessing.Process(target=extract_text,
                                          args=(start_end_list[i][0], start_end_list[i][1], sw_flags[0], lock))
        thread_list.append(sthread)
    for th in thread_list:
        th.start()
    for th in thread_list:
        th.join()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('x', type=int, help="the base")
    args = parser.parse_args()
    now_id = int(args.x)
    ty_kind = ['advcl_p_a_b_pos_wei', 'conj_p_a_b_pos_wei', 'inter_level_p_a_b_pos', 'inter_p_a_b_pos']
    # extract_text(1, 6800, ty_kind[now_id])
    # extract_text_2(1, 6800, ty_kind[now_id])
    extract_text_stop(2000, 4000, ty_kind[now_id])
    # extract_text_stop_no(1, 6800, ty_kind[now_id])
    # extract_text_up(1, 6800, ty_kind[now_id])
