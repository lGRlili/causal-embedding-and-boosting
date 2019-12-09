import copy
import functools
import queue


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
    # q_child中保存了n个动词,都是子句的动词
    q_child = queue.Queue()
    clause_zhuju.append(root)
    q.put(root)
    while not q.empty():
        now = q.get()
        # 查找的是我的孩子节点中链接的conj或advcl的词
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
    for temp_id in doc:
        # print('{:>10s}, {:>10s}, {:>10s}, {:>5s}, {:>5s}, {:>5s}, {:>4d}, {:>4d}, {:>10s}'.format(
        #     temp_id.text, temp_id.lemma_, temp_id.norm, temp_id.pos_, temp_id.tag_,
        #     temp_id.dep_, temp_id.id, temp_id.head, str(temp_id.child)))
        pass
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
    # total_clause_str = id_to_str(doc, total_clause, dis)

    total_clause_level = []
    get_clause_level(doc, root, total_clause_level, dis)
    total_clause_level = sorted(total_clause_level, key=functools.cmp_to_key(cmp))
    # total_clause_level_str = id_to_str_level(doc, total_clause_level, dis)

    return total_clause, total_clause_level, len(total_clause)


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


def recon_sen(total_congju, sen):
    # 对conj进行坍塌操作处理,将conj都排到同一个层级中,在child中
    # print(total_congju)
    for clause in total_congju:
        node = sen[clause[0] - sen[0].id]
        temp_node = node
        while temp_node.dep_ == 'conj':
            # print(node.child)
            father_node_id = temp_node.head
            father_node = sen[father_node_id - sen[0].id]
            for child in father_node.child:
                if child not in node.child:
                    node.child.append(child)
            temp_node = father_node
            # print(node.child)
    # print('------')
    return sen


# 方法2---这里用了stop改变了分母
def get_word_text_pos_dep(doc, clause):
    str_list = []
    # text, lemma_, pos_, dep_, head, id, child, left, right, ancestor
    for temp_id in clause:
        pass
        # print('{:>10s}, {:>10s}, {:>10s}, {:>5d}, {:>5s}, {:>5s}, {:>10s}, {:>4d}, {:>10s}'.format(
        #     doc[temp_id - doc[0].id].text, doc[temp_id - doc[0].id].lemma_, doc[temp_id - doc[0].id].norm,
        #     temp_id, doc[temp_id - doc[0].id].pos_, doc[temp_id - doc[0].id].tag_,
        #     doc[temp_id - doc[0].id].dep_, doc[temp_id - doc[0].id].head, doc[
        #         doc[temp_id - doc[0].id].head - doc[0].id].text))

    # 首先获得全部的str文本,然后判断这个单词是否在stopkey中
    # clause = sorted(clause)
    clause_txt = [doc[temp_id - doc[0].id].text for temp_id in clause]
    print(' '.join(clause_txt))
    print('---' * 45)
    for temp_id in clause:
        # 之前没有最小化,后续应该最小化单词
        word_text = doc[temp_id - doc[0].id].text.lower()
        word_lemma_ = doc[temp_id - doc[0].id].lemma_
        word_tag_ = doc[temp_id - doc[0].id].tag_
        word_norm = doc[temp_id - doc[0].id].norm
        word_pos = doc[temp_id - doc[0].id].pos_
        word_dep = doc[temp_id - doc[0].id].dep_
        word_id = doc[temp_id - doc[0].id].id
        word_head_id = doc[temp_id - doc[0].id].head
        word_child_id = doc[temp_id - doc[0].id].child
        word = [word_text, word_norm, word_lemma_,  word_pos, word_tag_, word_dep, word_id, word_head_id, word_child_id]
        str_list.append(word)
    return str_list


def get_intra_advcl(sen, total_congju):
    # 句内非从句---句内全连接
    clause_list = []
    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            # 遍历所有的从句
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'advcl':
                    zhuju_str = get_word_text_pos_dep(sen, zhuju)
                    ziju_str = get_word_text_pos_dep(sen, ziju)
                    ziju_str.append(['ziju', 'ziju', 'ziju', 0, 0])
                    if zhuju[0] > ziju[0]:
                        zhuju_str, ziju_str = ziju_str, zhuju_str
                    clause_list.append([zhuju_str, ziju_str])
    return clause_list


def get_intra_conj(sen, total_congju):
    # 句内非从句---句内全连接
    clause_list = []
    for clause in total_congju:
        if len(clause) > 1:
            zhuju = clause[0]
            # 遍历所有的从句
            for i in range(1, len(clause)):
                ziju = clause[i]
                if sen[ziju[0] - sen[0].id].dep_ == 'conj':
                    zhuju_str = get_word_text_pos_dep(sen, zhuju)
                    ziju_str = get_word_text_pos_dep(sen, ziju)
                    ziju_str.append(['ziju', 'ziju', 'ziju', 0, 0])
                    if zhuju[0] > ziju[0]:
                        # 此时主句在后,所以调换顺序,按照语言顺序排列
                        zhuju_str, ziju_str = ziju_str, zhuju_str
                    clause_list.append([zhuju_str, ziju_str])
    return clause_list


def cal_p_a_b_inter_stop_with(sen, total_congju, inter_level_p_a_b_pos, str_id_word_count, last_setence, last_congju,
                              count_inter, stopkey):
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


def cal_p_a_b_inter_num_weight_dis_stop_with(sen, total_congju, inter_level_p_a_b_pos, str_id_word_count, last_setence,
                                             last_congju, count_inter, stopkey):
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
