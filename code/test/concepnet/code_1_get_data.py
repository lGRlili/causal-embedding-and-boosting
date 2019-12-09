import pickle
import sys
from collections import defaultdict

import numpy as np
# sys.path.append()
import codecs
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
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


def lemminaze(word):
    try:
        pos_word = nltk.pos_tag([word])
        word_pos = get_wordnet_pos(pos_word[0][1]) or wordnet.NOUN
        # print(pos_word[0][1])
        word = wordnet_lemmatizer.lemmatize(word, pos=word_pos)
    except Exception as e:
        print(e)
        print("出现异常")
    return word


def get_copa_dict():
    # 　 获取得到3Wpair对应的单词
    with open("data/cause_effect_pair" + ".file", "rb") as f:
        P_A_B_pair = pickle.load(f)
    with open("data/copa_word_lemma.file", "rb") as f:
        copa_word_lemma = pickle.load(f)
    with open("data/copa_lemma_word.file", "rb") as f:
        copa_lemma_word = pickle.load(f)
    with open("data/pair_lemma_list.file", "rb") as f:
        pair_lemma_list = pickle.load(f)

    with open('../dict_date/str_id_' + 'word_count_delete.file', 'rb') as f:
        str_id_word_count = pickle.load(f)

    choose_list = ['', '_no', '_stop']
    name_str = choose_list[2]

    # 获得advcl
    db_path = '../database_original/advcl_p_a_b_pos_wei' + name_str + '1.npy'
    print(db_path)
    advcl_p_a_b_pos_wei = np.load(db_path)
    # 获得conj
    db_path = '../database_original/conj_p_a_b_pos_wei' + name_str + '1.npy'
    print(db_path)
    conj_p_a_b_pos_wei = np.load(db_path)

    # 获得inter
    db_path = '../database_original/inter_level_p_a_b_pos' + name_str + '1.npy'
    print(db_path)
    inter_level_p_a_b_pos = np.load(db_path)

    advcl_pair_pos_copa = defaultdict(int)
    conj_pair_pos_copa = defaultdict(int)
    inter_pair_pos_copa = defaultdict(int)
    for lemma_pair in P_A_B_pair:
        if len(pair_lemma_list[lemma_pair]) == 0:
            advcl_pair_pos_copa[lemma_pair] = 0
            conj_pair_pos_copa[lemma_pair] = 0
            inter_pair_pos_copa[lemma_pair] = 0
        else:
            for no_lemma_pair in pair_lemma_list[lemma_pair]:

                a = no_lemma_pair.split("_")
                str_p_a = a[0]
                str_p_b = a[1]
                # 将str转换成id
                id_p_a = str_id_word_count[str_p_a]
                id_p_b = str_id_word_count[str_p_b]

                advcl_count = advcl_p_a_b_pos_wei[id_p_a][id_p_b]
                advcl_pair_pos_copa[lemma_pair] += advcl_count

                conj_count = conj_p_a_b_pos_wei[id_p_a][id_p_b]
                conj_pair_pos_copa[lemma_pair] += conj_count

                inter_count = inter_level_p_a_b_pos[id_p_a][id_p_b]
                inter_pair_pos_copa[lemma_pair] += inter_count

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

    with open('data/concept_advcl_pair_pos' + '.file', 'wb') as f:
        pickle.dump(advcl_pair_pos_copa, f)
    with open('data/concept_conj_pair_pos' + '.file', 'wb') as f:
        pickle.dump(conj_pair_pos_copa, f)
    with open('data/concept_inter_pair_pos' + '.file', 'wb') as f:
        pickle.dump(inter_pair_pos_copa, f)
    print('finish')


def get_copa_dict_inter():
    # 　 获取得到3Wpair对应的单词

    with open("data/cause_effect_pair" + ".file", "rb") as f:
        P_A_B_pair = pickle.load(f)
    with open("data/copa_word_lemma.file", "rb") as f:
        copa_word_lemma = pickle.load(f)
    with open("data/copa_lemma_word.file", "rb") as f:
        copa_lemma_word = pickle.load(f)
    with open("data/pair_lemma_list.file", "rb") as f:
        pair_lemma_list = pickle.load(f)

    with open('../dict_date/str_id_' + 'word_count_delete.file', 'rb') as f:
        str_id_word_count = pickle.load(f)

    choose_list = ['', '_no', '_stop']
    name_str = choose_list[2]

    # 获得advcl
    db_path = '../database_original/inter_p_a_b_pos' + name_str + '1.npy'
    print(db_path)
    inter_p_a_b_pos_1 = np.load(db_path)
    # 获得conj
    db_path = '../database_original/inter_p_a_b_pos' + name_str + '2000.npy'
    print(db_path)
    inter_p_a_b_pos_2000 = np.load(db_path)
    # 获得inter
    db_path = '../database_original/inter_p_a_b_pos' + name_str + '4000.npy'
    print(db_path)
    inter_p_a_b_pos_4000 = np.load(db_path)

    inter_p_a_b_pos = inter_p_a_b_pos_1 + inter_p_a_b_pos_2000 + inter_p_a_b_pos_4000

    inter_pair_pos_copa = defaultdict(int)
    for lemma_pair in P_A_B_pair:
        if len(pair_lemma_list[lemma_pair]) == 0:
            inter_pair_pos_copa[lemma_pair] = 0
        else:
            for no_lemma_pair in pair_lemma_list[lemma_pair]:
                a = no_lemma_pair.split("_")
                str_p_a = a[0]
                str_p_b = a[1]
                # 将str转换成id
                id_p_a = str_id_word_count[str_p_a]
                id_p_b = str_id_word_count[str_p_b]

                inter_count = inter_p_a_b_pos[id_p_a][id_p_b]
                inter_pair_pos_copa[lemma_pair] += inter_count

    inter_sum = np.sum(inter_p_a_b_pos != 0)
    inter_pair_pos_copa['sum'] = inter_sum
    inter_count = np.sum(inter_p_a_b_pos)

    inter_pair_pos_copa['count'] = inter_count
    db_path = '../database_original/inter_p_a_b_pos' + name_str + '_total.npy'
    np.save(db_path, inter_p_a_b_pos)
    print('保存成功')

    with open('data/concept_inter_pair_pos' + '.file', 'wb') as f:
        pickle.dump(inter_pair_pos_copa, f)
    print('finish')


def get_copa_dict_sep(pair_kind):
    # 　 获取得到3Wpair对应的单词

    with open("data/cause_effect_pair" + ".file", "rb") as f:
        P_A_B_pair = pickle.load(f)
    with open("data/copa_word_lemma.file", "rb") as f:
        copa_word_lemma = pickle.load(f)
    with open("data/copa_lemma_word.file", "rb") as f:
        copa_lemma_word = pickle.load(f)
    with open("data/pair_lemma_list.file", "rb") as f:
        pair_lemma_list = pickle.load(f)

    with open('../dict_date/str_id_' + 'word_count_delete.file', 'rb') as f:
        str_id_word_count = pickle.load(f)

    choose_list = ['', '_no', '_stop']
    name_str = choose_list[2]

    db_path = '../database_original/' + pair_kind + '_p_a_b_pos' + name_str + '_total.npy'
    print(db_path)
    inter_p_a_b_pos = np.load(db_path)

    inter_pair_pos_copa = defaultdict(int)
    for lemma_pair in P_A_B_pair:
        if len(pair_lemma_list[lemma_pair]) == 0:
            inter_pair_pos_copa[lemma_pair] = 0
        else:
            for no_lemma_pair in pair_lemma_list[lemma_pair]:
                a = no_lemma_pair.split("_")
                str_p_a = a[0]
                str_p_b = a[1]
                # 将str转换成id
                id_p_a = str_id_word_count[str_p_a]
                id_p_b = str_id_word_count[str_p_b]

                inter_count = inter_p_a_b_pos[id_p_a][id_p_b]
                inter_pair_pos_copa[lemma_pair] += inter_count

    inter_sum = np.sum(inter_p_a_b_pos != 0)
    inter_pair_pos_copa['sum'] = inter_sum
    inter_count = np.sum(inter_p_a_b_pos)
    inter_pair_pos_copa['count'] = inter_count

    with open('data/concept_' + pair_kind + '_pair_pos' + '.file', 'wb') as f:
        pickle.dump(inter_pair_pos_copa, f)
    print('finish')


def filter_word(phrase, stopkey):
    # 过滤,去掉停用词
    filter_words = []
    for word in phrase:
        if word not in stopkey:
            word = lemminaze(word)
            filter_words.append(word)
    return filter_words


def get_cause_effect(file_list, stopkey, now_id):
    # 从原始数据中提取出cause和effect
    now = file_list[now_id]
    now = now.replace('\n', '')
    now = now.split('|||')
    cause = now[0].split(' ')
    effect = now[1].split(' ')
    cause = filter_word(cause, stopkey)
    effect = filter_word(effect, stopkey)
    # print(now)
    # print(cause, effect)
    return [cause, effect]


def get_data():
    # 获取数据,从中抽取出相应需要的句子
    # 过滤停用词
    # 保存所有的pair对
    # 保存所有对应的句子
    with open("data/cause_effect_pair" + ".file", "rb") as f:
        cause_effect_pair = pickle.load(f)
    P_word_id = np.load('../dict_date/P_word_id.npy')
    with open('../dict_date/str_id_' + 'word_count_delete.file', 'rb') as f:
        str_id_word_count_icw = pickle.load(f)
    copa_word_lemma = defaultdict(str)
    copa_lemma_word = defaultdict(set)
    pair_lemma_list = defaultdict(set)
    P_word_lemma_num = defaultdict(int)
    # 对提取出的进行词形还原
    for word in str_id_word_count_icw:
        word_lemma = lemminaze(word)
        copa_word_lemma[word] = word_lemma
        copa_lemma_word[word_lemma].add(word)
    for word_lemma in copa_lemma_word:
        for word in copa_lemma_word[word_lemma]:
            P_word_lemma_num[word_lemma] += P_word_id[str_id_word_count_icw[word]]

    print(len(copa_word_lemma), len(copa_lemma_word), len(P_word_lemma_num))
    print(len(str_id_word_count_icw))

    for word_pair in cause_effect_pair:
        cause, effect = word_pair.split('_')
        cause_pair = copa_lemma_word[cause]
        effect_pair = copa_lemma_word[effect]
        for cause_sig in cause_pair:
            for effect_sig in effect_pair:
                lemminaze_pair = cause_sig + '_' + effect_sig
                pair_lemma_list[word_pair].add(lemminaze_pair)
        # print(len(pair_lemma_list[word_pair]))

    with open("data/P_word_lemma_num.file", "wb") as f:
        pickle.dump(P_word_lemma_num, f)
    with open("data/copa_word_lemma.file", "wb") as f:
        pickle.dump(copa_word_lemma, f)
    with open("data/copa_lemma_word.file", "wb") as f:
        pickle.dump(copa_lemma_word, f)
    with open("data/pair_lemma_list.file", "wb") as f:
        pickle.dump(pair_lemma_list, f)


if __name__ == '__main__':
    # 1461    1375
    choose = ['', '_icw', '_gut', '_book_corpus']
    doc_name = choose[1]
    print(doc_name)
    wordnet_lemmatizer = WordNetLemmatizer()
    get_data()
    get_copa_dict()
    get_copa_dict_inter()
    get_copa_dict_sep('inter')
