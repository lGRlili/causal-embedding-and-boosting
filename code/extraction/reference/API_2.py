import copy
import functools
import queue


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


def id_to_str(doc, total_congju, dis):
    # 通过顺序得到原句子
    total_congju_str = []

    for congju in total_congju:
        congju_str = []
        # print(congju)
        # congju = sorted(congju)
        # print(congju)
        for id in congju:
            congju_str.append(doc[id - dis].text)
        total_congju_str.append(congju_str)
    return total_congju_str


def id_to_str_level(doc, total_congju_level, dis):
    # 通过顺序得到原句子
    # 保存全部的句子
    total_congju_str = []

    for congju_pair in total_congju_level:
        # 从中依次获得句子
        congju_pair_str = []
        for congju in congju_pair:
            congju_str = []
            # print(congju)
            # congju = sorted(congju)
            # print(congju)
            for id in congju:
                congju_str.append(doc[id - dis].text)
            congju_pair_str.append(congju_str)
        total_congju_str.append(congju_pair_str)
    return total_congju_str


def cmp(x, y):
    # 用来调整顺序
    if x[0] > y[0]:
        return 1
    if x[0] < y[0]:
        return -1
    return 0


def cmp_level(x, y):
    # 用来调整顺序
    if x[0][0] > y[0][0]:
        return 1
    if x[0][0] < y[0][0]:
        return -1
    return 0


# 根据两组不同的结构解析句子
def get_clause(doc, root, total_clause, dis):
    # 用来解析从句--进行广度遍历
    clause_word = []
    q = queue.Queue()
    clause_word.append(root)
    q.put(root)
    while not q.empty():
        now = q.get()
        for child in doc[now - dis].child:
            if (doc[child - dis].pos_ == 'VERB') & (len(doc[child - dis].child) > 0) & (
                    (doc[child - dis].dep_ == 'conj') | (doc[child - dis].dep_ == 'advcl')):
                get_clause(doc, child, total_clause, dis)
            else:
                q.put(child)
                clause_word.append(child)
    total_clause.append(clause_word)


# 根据两组不同的结构解析句子
def get_clause_level(doc, root, total_clause_level, dis):
    # 在这里保存的都是id
    # 用来解析从句--进行广度遍历
    clause_word_level = []
    clause_zhuju = []
    q = queue.Queue()
    q_child = queue.Queue()
    clause_zhuju.append(root)
    q.put(root)
    while not q.empty():
        now = q.get()
        for child in doc[now - dis].child:
            if (doc[child - dis].pos_ == 'VERB') & (len(doc[child - dis].child) > 0) & (
                    (doc[child - dis].dep_ == 'conj') | (doc[child - dis].dep_ == 'advcl')):
                # print('get')
                q_child.put(child)
                get_clause_level(doc, child, total_clause_level, dis)
            else:
                q.put(child)
                clause_zhuju.append(child)
    clause_word_level.append(clause_zhuju)
    while not q_child.empty():
        clause_ziju = []
        now = q_child.get()
        clause_ziju.append(now)
        for token in doc:
            if now in token.ancestor:
                clause_ziju.append(token.id)
        clause_word_level.append(clause_ziju)
    total_clause_level.append(clause_word_level)


def parse_setence(doc):
    dis = doc[0].id
    # print("distance:", dis)

    for token in doc:
        # print(token.text.rjust(11), token.lemma_.rjust(11), token.pos_.rjust(11), token.dep_.rjust(11),
        #       str(token.head).rjust(11)
        #       , str(token.id).rjust(11), token.child, token.left, token.right, token.ancestor)
        if token.dep_ == 'ROOT':
            # 注意此时找到的是文本中的id并非此文中的id
            root = token.id

    # 此时total_congju为全局变量
    total_clause = []
    get_clause(doc, root, total_clause, dis)
    total_clause = sorted(total_clause, key=functools.cmp_to_key(cmp))
    total_clause_str = id_to_str(doc, total_clause, dis)

    total_clause_level = []
    get_clause_level(doc, root, total_clause_level, dis)
    # total_clause_level = sorted(total_clause_level, key=functools.cmp_to_key(cmp))
    total_clause_level_str = id_to_str_level(doc, total_clause_level, dis)

    return total_clause, total_clause_str, total_clause_level, total_clause_level_str, len(total_clause)


def get_str(doc, clause):
    str_list = []
    for id in clause:
        str_word = doc[id - doc[0].id].text
        str_list.append(str_word)
    return str_list


def get_str_id(doc, clause, str_id_word_count):
    id_list = []
    for id in clause:
        str_word = doc[id - doc[0].id].text
        if str_word in str_id_word_count:
            id_list.append(str_id_word_count[str_word])
    return id_list


def cal_intra_advcl_wei_ab(sen, total_congju, advcl_p_a_pos_wei, advcl_p_b_pos_wei, str_id_word_count):
    # 句内非从句---句内全连接
    # 此处保证,都是原因从句在前,结果从句在后
    # 从理论上来说,同一个句子,只可能有一个结果状语从句,所以在计算的时候,若有多个结果状语从句,需要除掉权重
    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            # 遍历所有的从句
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'advcl':
                    congju_a = get_str_id(sen, zhuju, str_id_word_count)
                    congju_b = get_str_id(sen, ziju, str_id_word_count)
                    len_clause_pair = len(congju_a) * len(congju_b)
                    # 调整句子的方法
                    if zhuju[0] > ziju[0]:
                        congju_a, congju_b = congju_b, congju_a
                    if len_clause_pair > 0:
                        for wordA in congju_a:
                            advcl_p_a_pos_wei[wordA] += round(len(congju_b) / len_clause_pair, 6)
                        for wordB in congju_b:
                            advcl_p_b_pos_wei[wordB] += round(len(congju_a) / len_clause_pair, 6)
    return advcl_p_a_pos_wei, advcl_p_b_pos_wei


def cal_intra_conj_wei_ab(sen, total_congju, conj_p_a_pos_wei, conj_p_b_pos_wei, str_id_word_count):

    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'conj':
                    congju_a = get_str_id(sen, zhuju, str_id_word_count)
                    congju_b = get_str_id(sen, ziju, str_id_word_count)
                    len_clause_pair = len(congju_a) * len(congju_b)
                    # 调整句子的方法
                    if zhuju[0] > ziju[0]:
                        congju_a, congju_b = congju_b, congju_a
                    if len_clause_pair > 0:
                        for wordA in congju_a:
                            conj_p_a_pos_wei[wordA] += round(len(congju_b) / len_clause_pair, 6)
                        for wordB in congju_b:
                            conj_p_b_pos_wei[wordB] += round(len(congju_a) / len_clause_pair, 6)

    return conj_p_a_pos_wei, conj_p_b_pos_wei


def cal_p_a_b_inter_num_weight_dis_ab(sen, total_congju, inter_level_p_a_pos, inter_level_p_b_pos,
                                      last_setence, last_congju, str_id_word_count):
    sen_now, sen_last = [], []
    now_len = len(total_congju)
    last_len = len(last_congju)
    clause__len = now_len * last_len

    if len(last_setence) > 0:
        # root句子
        for sens_last in last_congju:
            count_num_last = 1
            last_id = sens_last[0]
            while last_setence[last_id - last_setence[0].id].dep_ != 'ROOT':
                if last_setence[last_id - last_setence[0].id].pos_ == 'VERB':
                    count_num_last = count_num_last + 1
                last_id = last_setence[last_id - last_setence[0].id].head
            for sens_now in total_congju:
                count_num_now = 1
                now_id = sens_now[0]
                while sen[now_id - sen[0].id].dep_ != 'ROOT':
                    if sen[now_id - sen[0].id].pos_ == 'VERB':
                        count_num_now = count_num_now + 1
                    now_id = sen[now_id - sen[0].id].head
                # 这里开始我们对获得的从句进行计算
                # 先将得到的从句id转化为str
                clause_now = get_str_id(sen, sens_now, str_id_word_count)
                clause_lasts = get_str_id(last_setence, sens_last, str_id_word_count)
                len_clause_pair = len(clause_now) * len(clause_lasts)
                len_temp_level = count_num_last * count_num_now * len_clause_pair * clause__len
                if len_temp_level > 0:
                    for word_last in clause_lasts:
                        inter_level_p_a_pos[word_last] += round(len(clause_now) / len_temp_level, 6)
                    for word_now in clause_now:
                        inter_level_p_b_pos[word_now] += round(len(clause_lasts) / len_temp_level, 6)

    # 如果句子中出现verb,更新
    for token in sen:
        if token.pos_ == 'VERB':
            last_setence = sen
            last_congju = total_congju
    return inter_level_p_a_pos, inter_level_p_b_pos, last_setence, last_congju


def cal_intra_advcl_wei(sen, total_congju, advcl_p_a_b_pos_wei, str_id_word_count, count_advcl):
    # 句内非从句---句内全连接
    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            # 遍历所有的从句
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'advcl':
                    congju_a = get_str_id(sen, zhuju, str_id_word_count)
                    congju_b = get_str_id(sen, ziju, str_id_word_count)
                    len_clause_pair = len(congju_a) * len(congju_b)
                    # 调整句子的方法
                    if zhuju[0] > ziju[0]:
                        congju_a, congju_b = congju_b, congju_a
                    if len_clause_pair > 0:
                        for wordA in congju_a:
                            for wordB in congju_b:
                                if advcl_p_a_b_pos_wei[wordA][wordB] == 0:
                                    count_advcl += 1
                                advcl_p_a_b_pos_wei[wordA][wordB] += round(1/len_clause_pair, 6)
    return count_advcl, advcl_p_a_b_pos_wei


def cal_intra_conj_wei(sen, total_congju, conj_p_a_b_pos_wei, str_id_word_count, count_conj):

    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'conj':
                    congju_a = get_str_id(sen, zhuju, str_id_word_count)
                    congju_b = get_str_id(sen, ziju, str_id_word_count)
                    len_clause_pair = len(congju_a) * len(congju_b)
                    # 调整句子的方法
                    if zhuju[0] > ziju[0]:
                        congju_a, congju_b = congju_b, congju_a
                    if len_clause_pair > 0:
                        for wordA in congju_a:
                            for wordB in congju_b:
                                if conj_p_a_b_pos_wei[wordA][wordB] == 0:
                                    count_conj += 1
                                conj_p_a_b_pos_wei[wordA][wordB] += round(1/len_clause_pair, 6)

    return count_conj, conj_p_a_b_pos_wei


def cal_p_a_b_inter_num_weight_dis(sen, total_congju, inter_level_p_a_b_pos, str_id_word_count, last_setence, last_congju, count_inter):
    sen_now, sen_last = [], []
    now_len = len(total_congju)
    last_len = len(last_congju)
    clause__len = now_len * last_len

    if len(last_setence) > 0:
        # root句子
        for sens_last in last_congju:
            count_num_last = 1
            last_id = sens_last[0]
            while last_setence[last_id - last_setence[0].id].dep_ != 'ROOT':
                if last_setence[last_id - last_setence[0].id].pos_ == 'VERB':
                    count_num_last = count_num_last + 1
                last_id = last_setence[last_id - last_setence[0].id].head
            for sens_now in total_congju:
                count_num_now = 1
                now_id = sens_now[0]
                while sen[now_id - sen[0].id].dep_ != 'ROOT':
                    if sen[now_id - sen[0].id].pos_ == 'VERB':
                        count_num_now = count_num_now + 1
                    now_id = sen[now_id - sen[0].id].head
                # 这里开始我们对获得的从句进行计算
                # 先将得到的从句id转化为str
                clause_now = get_str_id(sen, sens_now, str_id_word_count)
                clause_lasts = get_str_id(last_setence, sens_last, str_id_word_count)
                len_clause_pair = len(clause_now) * len(clause_lasts)
                len_temp_level = count_num_last * count_num_now * len_clause_pair * clause__len
                if len_temp_level > 0:
                    for word_last in clause_lasts:
                        for word_now in clause_now:
                            if inter_level_p_a_b_pos[word_last][word_now] == 0:
                                count_inter += 1
                            inter_level_p_a_b_pos[word_last][word_now] += round(1 / len_temp_level, 6)

    # 如果句子中出现verb,更新
    for token in sen:
        if token.pos_ == 'VERB':
            last_setence = sen
            last_congju = total_congju
    return count_inter, inter_level_p_a_b_pos, last_setence, last_congju


def cal_intra_conj_wei_speci_ab(sen, total_congju, conj_p_a_pos_wei, conj_p_b_pos_wei, str_id_word_count):

    for clause in total_congju:
        # 每一个clause中包含0 是主句,后面的是从句
        if len(clause) > 1:
            dist_clause = [clause[0]]
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'conj':
                    dist_clause.append(ziju)
            for i in range(0, len(dist_clause)):
                for j in range(i + 1, len(dist_clause)):
                    congju_a = get_str_id(sen, dist_clause[i], str_id_word_count)
                    congju_b = get_str_id(sen, dist_clause[j], str_id_word_count)
                    len_clause_pair = len(congju_a) * len(congju_b)
                    if dist_clause[i] > dist_clause[j]:
                        congju_a, congju_b = congju_b, congju_a
                    if len_clause_pair > 0:
                        for wordA in congju_a:
                            conj_p_a_pos_wei[wordA] += len(congju_b) / len_clause_pair
                        for wordB in congju_b:
                            conj_p_b_pos_wei[wordB] += len(congju_a) / len_clause_pair

    return conj_p_a_pos_wei, conj_p_b_pos_wei


def cal_intra_conj_wei_speci(sen, total_congju, conj_p_a_b_pos_wei, str_id_word_count, count_conj):

    for clause in total_congju:
        if len(clause) > 1:
            dist_clause = [clause[0]]
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'conj':
                    dist_clause.append(ziju)
            for i in range(0, len(dist_clause)):
                for j in range(i + 1, len(dist_clause)):
                    congju_a = get_str_id(sen, dist_clause[i], str_id_word_count)
                    congju_b = get_str_id(sen, dist_clause[j], str_id_word_count)
                    len_clause_pair = len(congju_a) * len(congju_b)
                    if dist_clause[i] > dist_clause[j]:
                        congju_a, congju_b = congju_b, congju_a
                    for wordA in congju_a:
                        for wordB in congju_b:
                            if conj_p_a_b_pos_wei[wordA][wordB] == 0:
                                count_conj += 1
                            conj_p_a_b_pos_wei[wordA][wordB] += 1/len_clause_pair

    return count_conj, conj_p_a_b_pos_wei


# 方法2---这里用了stop改变了分母
def get_str_id_stop(doc, clause, str_id_word_count, stopkey):
    str_list = []
    id_list = []
    # 首先获得全部的str文本,然后判断这个单词是否在stopkey中
    for id in clause:
        str_word = doc[id - doc[0].id].text
        if str_word not in stopkey:
            str_list.append(str_word)
    for word in str_list:
        if word in str_id_word_count:
            id_list.append(str_id_word_count[word])
    len_str_list = len(str_list)
    return id_list, len_str_list


def get_str_id_stop_for_sen(doc, clause, str_id_word_count, stopkey):
    str_list = []
    id_list = []
    # 首先获得全部的str文本,然后判断这个单词是否在stopkey中
    for word_spacy in clause:
        str_word = doc[word_spacy.id - doc[0].id].text
        if str_word not in stopkey:
            str_list.append(str_word)
    for word in str_list:
        if word in str_id_word_count:
            id_list.append(str_id_word_count[word])
    len_str_list = len(str_list)
    return id_list, len_str_list


# 注意,此处为未改过的版本
def cal_intra_advcl_wei_ab_stop_with(sen, total_congju, advcl_p_a_pos_wei, advcl_p_b_pos_wei, str_id_word_count, stopkey):
    # 句内非从句---句内全连接
    # 此处保证,都是原因从句在前,结果从句在后
    # 从理论上来说,同一个句子,只可能有一个结果状语从句,所以在计算的时候,若有多个结果状语从句,需要除掉权重
    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            # 遍历所有的从句
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'advcl':
                    congju_a, len_stop_a = get_str_id_stop(sen, zhuju, str_id_word_count, stopkey)
                    congju_b, len_stop_b = get_str_id_stop(sen, ziju, str_id_word_count, stopkey)
                    len_clause_pair = len_stop_a * len_stop_b
                    # 调整句子的方法
                    if zhuju[0] > ziju[0]:
                        congju_a, congju_b = congju_b, congju_a
                    if len_clause_pair > 0:
                        for wordA in congju_a:
                            advcl_p_a_pos_wei[wordA] += round(len(congju_b) / len_clause_pair, 6)
                        for wordB in congju_b:
                            advcl_p_b_pos_wei[wordB] += round(len(congju_a) / len_clause_pair, 6)
    return advcl_p_a_pos_wei, advcl_p_b_pos_wei


def cal_intra_conj_wei_ab_stop_with(sen, total_congju, conj_p_a_pos_wei, conj_p_b_pos_wei, str_id_word_count, stopkey):

    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'conj':
                    congju_a, len_stop_a = get_str_id_stop(sen, zhuju, str_id_word_count, stopkey)
                    congju_b, len_stop_b = get_str_id_stop(sen, ziju, str_id_word_count, stopkey)
                    len_clause_pair = len_stop_a * len_stop_b
                    # 调整句子的方法
                    if zhuju[0] > ziju[0]:
                        congju_a, congju_b = congju_b, congju_a
                    if len_clause_pair > 0:
                        for wordA in congju_a:
                            conj_p_a_pos_wei[wordA] += round(len(congju_b) / len_clause_pair, 6)
                        for wordB in congju_b:
                            conj_p_b_pos_wei[wordB] += round(len(congju_a) / len_clause_pair, 6)

    return conj_p_a_pos_wei, conj_p_b_pos_wei


def cal_p_a_b_inter_num_weight_dis_ab_stop_with(sen, total_congju, inter_level_p_a_pos, inter_level_p_b_pos,
                                      last_setence, last_congju, str_id_word_count, stopkey):
    sen_now, sen_last = [], []
    now_len = len(total_congju)
    last_len = len(last_congju)
    clause__len = now_len * last_len

    if len(last_setence) > 0:
        # root句子
        for sens_last in last_congju:
            count_num_last = 1
            last_id = sens_last[0]
            while last_setence[last_id - last_setence[0].id].dep_ != 'ROOT':
                if last_setence[last_id - last_setence[0].id].pos_ == 'VERB':
                    count_num_last = count_num_last + 1
                last_id = last_setence[last_id - last_setence[0].id].head
            for sens_now in total_congju:
                count_num_now = 1
                now_id = sens_now[0]
                while sen[now_id - sen[0].id].dep_ != 'ROOT':
                    if sen[now_id - sen[0].id].pos_ == 'VERB':
                        count_num_now = count_num_now + 1
                    now_id = sen[now_id - sen[0].id].head
                # 这里开始我们对获得的从句进行计算
                # 先将得到的从句id转化为str
                clause_now, len_stop_a = get_str_id_stop(sen, sens_now, str_id_word_count, stopkey)
                clause_lasts, len_stop_b = get_str_id_stop(last_setence, sens_last, str_id_word_count, stopkey)
                len_clause_pair = len_stop_a * len_stop_b
                len_temp_level = count_num_last * count_num_now * len_clause_pair * clause__len
                if len_temp_level > 0:
                    for word_last in clause_lasts:
                        inter_level_p_a_pos[word_last] += round(len(clause_now) / len_temp_level, 6)
                    for word_now in clause_now:
                        inter_level_p_b_pos[word_now] += round(len(clause_lasts) / len_temp_level, 6)

    # 如果句子中出现verb,更新
    for token in sen:
        if token.pos_ == 'VERB':
            last_setence = sen
            last_congju = total_congju
    return inter_level_p_a_pos, inter_level_p_b_pos, last_setence, last_congju


def cal_intra_advcl_wei_stop_with(sen, total_congju, advcl_p_a_b_pos_wei, str_id_word_count, count_advcl, stopkey):
    # 句内非从句---句内全连接
    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            # 遍历所有的从句
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'advcl':
                    congju_a, len_stop_a = get_str_id_stop(sen, zhuju, str_id_word_count, stopkey)
                    congju_b, len_stop_b = get_str_id_stop(sen, ziju, str_id_word_count, stopkey)
                    len_clause_pair = len_stop_a * len_stop_b
                    # 调整句子的方法
                    if zhuju[0] > ziju[0]:
                        congju_a, congju_b = congju_b, congju_a
                    if len_clause_pair > 0:
                        for wordA in congju_a:
                            for wordB in congju_b:
                                if advcl_p_a_b_pos_wei[wordA][wordB] == 0:
                                    count_advcl += 1
                                advcl_p_a_b_pos_wei[wordA][wordB] += round(1/len_clause_pair, 6)
    return count_advcl, advcl_p_a_b_pos_wei


def cal_intra_conj_wei_stop_with(sen, total_congju, conj_p_a_b_pos_wei, str_id_word_count, count_conj, stopkey):

    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'conj':
                    congju_a, len_stop_a = get_str_id_stop(sen, zhuju, str_id_word_count, stopkey)
                    congju_b, len_stop_b = get_str_id_stop(sen, ziju, str_id_word_count, stopkey)
                    len_clause_pair = len_stop_a * len_stop_b
                    # 调整句子的方法
                    if zhuju[0] > ziju[0]:
                        congju_a, congju_b = congju_b, congju_a
                    if len_clause_pair > 0:
                        for wordA in congju_a:
                            for wordB in congju_b:
                                if conj_p_a_b_pos_wei[wordA][wordB] == 0:
                                    count_conj += 1
                                conj_p_a_b_pos_wei[wordA][wordB] += round(1/len_clause_pair, 6)

    return count_conj, conj_p_a_b_pos_wei


def cal_p_a_b_inter_stop_with(sen, total_congju, inter_level_p_a_b_pos, str_id_word_count, last_setence, last_congju, count_inter, stopkey):
    # sen_now, sen_last = [], []

    if len(last_setence) > 0:
        # root句子
        clause_now, len_stop_a = get_str_id_stop_for_sen(sen, sen, str_id_word_count, stopkey)
        clause_lasts, len_stop_b = get_str_id_stop_for_sen(last_setence, last_setence, str_id_word_count, stopkey)
        len_temp_level = len_stop_a * len_stop_b
        if len_temp_level > 0:
            for word_last in clause_lasts:
                for word_now in clause_now:
                    if inter_level_p_a_b_pos[word_last][word_now] == 0:
                        count_inter += 1
                    inter_level_p_a_b_pos[word_last][word_now] += round(1 / len_temp_level, 6)

    # 如果句子中出现verb,更新
    for token in sen:
        if token.pos_ == 'VERB':
            last_setence = sen
            last_congju = total_congju
    return count_inter, inter_level_p_a_b_pos, last_setence, last_congju


def cal_p_a_b_inter_num_weight_dis_stop_with(sen, total_congju, inter_level_p_a_b_pos, str_id_word_count, last_setence, last_congju, count_inter, stopkey):
    sen_now, sen_last = [], []
    now_len = len(total_congju)
    last_len = len(last_congju)
    clause__len = now_len * last_len

    if len(last_setence) > 0:
        # root句子
        for sens_last in last_congju:
            count_num_last = 1
            last_id = sens_last[0]
            while last_setence[last_id - last_setence[0].id].dep_ != 'ROOT':
                if last_setence[last_id - last_setence[0].id].pos_ == 'VERB':
                    count_num_last = count_num_last + 1
                last_id = last_setence[last_id - last_setence[0].id].head
            for sens_now in total_congju:
                count_num_now = 1
                now_id = sens_now[0]
                while sen[now_id - sen[0].id].dep_ != 'ROOT':
                    if sen[now_id - sen[0].id].pos_ == 'VERB':
                        count_num_now = count_num_now + 1
                    now_id = sen[now_id - sen[0].id].head
                # 这里开始我们对获得的从句进行计算
                # 先将得到的从句id转化为str
                clause_now, len_stop_a = get_str_id_stop(sen, sens_now, str_id_word_count, stopkey)
                clause_lasts, len_stop_b = get_str_id_stop(last_setence, sens_last, str_id_word_count, stopkey)
                len_clause_pair = len_stop_a * len_stop_b
                len_temp_level = count_num_last * count_num_now * len_clause_pair * clause__len
                if len_temp_level > 0:
                    for word_last in clause_lasts:
                        for word_now in clause_now:
                            if inter_level_p_a_b_pos[word_last][word_now] == 0:
                                count_inter += 1
                            inter_level_p_a_b_pos[word_last][word_now] += round(1 / len_temp_level, 6)

    # 如果句子中出现verb,更新
    for token in sen:
        if token.pos_ == 'VERB':
            last_setence = sen
            last_congju = total_congju
    return count_inter, inter_level_p_a_b_pos, last_setence, last_congju


# 方法3---选择的是没有用stop去掉的单词
def cal_intra_advcl_wei_ab_stop_no(sen, total_congju, advcl_p_a_pos_wei, advcl_p_b_pos_wei, str_id_word_count):
    # 句内非从句---句内全连接
    # 此处保证,都是原因从句在前,结果从句在后
    # 从理论上来说,同一个句子,只可能有一个结果状语从句,所以在计算的时候,若有多个结果状语从句,需要除掉权重
    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            # 遍历所有的从句
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'advcl':
                    congju_a = get_str_id(sen, zhuju, str_id_word_count)
                    congju_b = get_str_id(sen, ziju, str_id_word_count)
                    len_clause_pair = len(zhuju) * len(ziju)
                    # 调整句子的方法
                    if zhuju[0] > ziju[0]:
                        congju_a, congju_b = congju_b, congju_a
                    if len_clause_pair > 0:
                        for wordA in congju_a:
                            advcl_p_a_pos_wei[wordA] += round(len(congju_b) / len_clause_pair, 6)
                        for wordB in congju_b:
                            advcl_p_b_pos_wei[wordB] += round(len(congju_a) / len_clause_pair, 6)
    return advcl_p_a_pos_wei, advcl_p_b_pos_wei


def cal_intra_conj_wei_ab_stop_no(sen, total_congju, conj_p_a_pos_wei, conj_p_b_pos_wei, str_id_word_count):

    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'conj':
                    congju_a = get_str_id(sen, zhuju, str_id_word_count)
                    congju_b = get_str_id(sen, ziju, str_id_word_count)
                    len_clause_pair = len(zhuju) * len(ziju)
                    # 调整句子的方法
                    if zhuju[0] > ziju[0]:
                        congju_a, congju_b = congju_b, congju_a
                    if len_clause_pair > 0:
                        for wordA in congju_a:
                            conj_p_a_pos_wei[wordA] += round(len(congju_b) / len_clause_pair, 6)
                        for wordB in congju_b:
                            conj_p_b_pos_wei[wordB] += round(len(congju_a) / len_clause_pair, 6)

    return conj_p_a_pos_wei, conj_p_b_pos_wei


def cal_p_a_b_inter_num_weight_dis_ab_stop_no(sen, total_congju, inter_level_p_a_pos, inter_level_p_b_pos,
                                      last_setence, last_congju, str_id_word_count):
    sen_now, sen_last = [], []
    now_len = len(total_congju)
    last_len = len(last_congju)
    clause__len = now_len * last_len

    if len(last_setence) > 0:
        # root句子
        for sens_last in last_congju:
            count_num_last = 1
            last_id = sens_last[0]
            while last_setence[last_id - last_setence[0].id].dep_ != 'ROOT':
                if last_setence[last_id - last_setence[0].id].pos_ == 'VERB':
                    count_num_last = count_num_last + 1
                last_id = last_setence[last_id - last_setence[0].id].head
            for sens_now in total_congju:
                count_num_now = 1
                now_id = sens_now[0]
                while sen[now_id - sen[0].id].dep_ != 'ROOT':
                    if sen[now_id - sen[0].id].pos_ == 'VERB':
                        count_num_now = count_num_now + 1
                    now_id = sen[now_id - sen[0].id].head
                # 这里开始我们对获得的从句进行计算
                # 先将得到的从句id转化为str
                clause_now = get_str_id(sen, sens_now, str_id_word_count)
                clause_lasts = get_str_id(last_setence, sens_last, str_id_word_count)
                len_clause_pair = len(sens_now) * len(sens_last)
                len_temp_level = count_num_last * count_num_now * len_clause_pair * clause__len
                if len_temp_level > 0:
                    for word_last in clause_lasts:
                        inter_level_p_a_pos[word_last] += round(len(clause_now) / len_temp_level, 6)
                    for word_now in clause_now:
                        inter_level_p_b_pos[word_now] += round(len(clause_lasts) / len_temp_level, 6)

    # 如果句子中出现verb,更新
    for token in sen:
        if token.pos_ == 'VERB':
            last_setence = sen
            last_congju = total_congju
    return inter_level_p_a_pos, inter_level_p_b_pos, last_setence, last_congju


def cal_intra_advcl_wei_stop_no(sen, total_congju, advcl_p_a_b_pos_wei, str_id_word_count, count_advcl):
    # 句内非从句---句内全连接
    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            # 遍历所有的从句
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'advcl':
                    congju_a = get_str_id(sen, zhuju, str_id_word_count)
                    congju_b = get_str_id(sen, ziju, str_id_word_count)
                    len_clause_pair = len(zhuju) * len(ziju)
                    # 调整句子的方法
                    if zhuju[0] > ziju[0]:
                        congju_a, congju_b = congju_b, congju_a
                    if len_clause_pair > 0:
                        for wordA in congju_a:
                            for wordB in congju_b:
                                if advcl_p_a_b_pos_wei[wordA][wordB] == 0:
                                    count_advcl += 1
                                advcl_p_a_b_pos_wei[wordA][wordB] += round(1/len_clause_pair, 6)
    return count_advcl, advcl_p_a_b_pos_wei


def cal_intra_conj_wei_stop_no(sen, total_congju, conj_p_a_b_pos_wei, str_id_word_count, count_conj):

    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'conj':
                    congju_a = get_str_id(sen, zhuju, str_id_word_count)
                    congju_b = get_str_id(sen, ziju, str_id_word_count)
                    len_clause_pair = len(zhuju) * len(ziju)
                    # 调整句子的方法
                    if zhuju[0] > ziju[0]:
                        congju_a, congju_b = congju_b, congju_a
                    if len_clause_pair > 0:
                        for wordA in congju_a:
                            for wordB in congju_b:
                                if conj_p_a_b_pos_wei[wordA][wordB] == 0:
                                    count_conj += 1
                                conj_p_a_b_pos_wei[wordA][wordB] += round(1/len_clause_pair, 6)

    return count_conj, conj_p_a_b_pos_wei


def cal_p_a_b_inter_num_weight_dis_stop_no(sen, total_congju, inter_level_p_a_b_pos, str_id_word_count, last_setence, last_congju, count_inter):
    sen_now, sen_last = [], []
    now_len = len(total_congju)
    last_len = len(last_congju)
    clause__len = now_len * last_len

    if len(last_setence) > 0:
        # root句子
        for sens_last in last_congju:
            count_num_last = 1
            last_id = sens_last[0]
            while last_setence[last_id - last_setence[0].id].dep_ != 'ROOT':
                if last_setence[last_id - last_setence[0].id].pos_ == 'VERB':
                    count_num_last = count_num_last + 1
                last_id = last_setence[last_id - last_setence[0].id].head
            for sens_now in total_congju:
                count_num_now = 1
                now_id = sens_now[0]
                while sen[now_id - sen[0].id].dep_ != 'ROOT':
                    if sen[now_id - sen[0].id].pos_ == 'VERB':
                        count_num_now = count_num_now + 1
                    now_id = sen[now_id - sen[0].id].head
                # 这里开始我们对获得的从句进行计算
                # 先将得到的从句id转化为str
                clause_now = get_str_id(sen, sens_now, str_id_word_count)
                clause_lasts = get_str_id(last_setence, sens_last, str_id_word_count)
                len_clause_pair = len(sens_now) * len(sens_last)
                len_temp_level = count_num_last * count_num_now * len_clause_pair * clause__len
                if len_temp_level > 0:
                    for word_last in clause_lasts:
                        for word_now in clause_now:
                            if inter_level_p_a_b_pos[word_last][word_now] == 0:
                                count_inter += 1
                            inter_level_p_a_b_pos[word_last][word_now] += round(1 / len_temp_level, 6)

    # 如果句子中出现verb,更新
    for token in sen:
        if token.pos_ == 'VERB':
            last_setence = sen
            last_congju = total_congju
    return count_inter, inter_level_p_a_b_pos, last_setence, last_congju


