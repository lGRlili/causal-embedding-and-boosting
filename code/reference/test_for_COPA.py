from reference.base import FilterData
import codecs
import json
from collections import defaultdict
from nltk.tokenize import WordPunctTokenizer
import functools
import numpy as np
import sys

sys.path.extend(['../'])
sys.path.extend(['../test/COPA/'])


class Cruve_PR(object):
    def __init__(self):
        pass

    @staticmethod
    def get_pr_points(sorted_score, relevant_label):
        """
        :param sorted_score: item in it is like: (score, label)
        :param relevant_label:
        :return:
        """
        numPair = len(sorted_score)
        assert numPair > 0
        a = [1 for s in sorted_score if s[1] == relevant_label]
        numRelevant = sum([1 for s in sorted_score if s[1] == relevant_label])
        curvePoints = []
        scores = sorted(list(set([s[0] for s in sorted_score])), reverse=True)
        groupedByScore = [(s, [item for item in sorted_score if item[0] == s]) for s in scores]
        for i in range(len(groupedByScore)):
            score, items = groupedByScore[i]
            numRelevantInGroup = sum([1 for item in items if item[1] == relevant_label])
            if numRelevantInGroup > 0:
                sliceGroup = groupedByScore[:i + 1]
                items_slice = [x for y in sliceGroup for x in y[1]]
                numRelevantInSlice = sum([1 for s in items_slice if s[1] == relevant_label])
                sliceRecall = numRelevantInSlice / float(numRelevant)
                slicePrecision = numRelevantInSlice / float(len(items_slice))
                curvePoints.append((sliceRecall, slicePrecision))
        return curvePoints

    @staticmethod
    def save_points(points, saved_path):
        fout = codecs.open(saved_path, 'w', 'utf-8')
        for recall, precision in points:
            fout.write('{} {}\n'.format(recall, precision))
        print('pr曲线保存完毕')


def cmp(x, y):
    # 用来调整顺序
    if x[0] > y[0]:
        return -1
    if x[0] < y[0]:
        return 1
    return 0


def filter_isalpha(word):
    return word.encode('UTF-8').isalpha()


def filter_stop_word(word):
    return word not in eng_stop_word


def get_premise_choice1_choice2(train_data, filter_stop_word_flag=1, filter_isalpha_flag=1):
    phrase_fir = train_data['premise']
    phrase_choose1 = train_data['choice1']
    phrase_choice2 = train_data['choice2']
    phrase_fir = phrase_fir.lower()
    phrase_choose1 = phrase_choose1.lower()
    phrase_choice2 = phrase_choice2.lower()

    # 对数据进行分词
    phrase_fir = WordPunctTokenizer().tokenize(phrase_fir)
    phrase_choose1 = WordPunctTokenizer().tokenize(phrase_choose1)
    phrase_choice2 = WordPunctTokenizer().tokenize(phrase_choice2)
    # print(phrase_fir, phrase_choose1, phrase_choice2)
    filter_data = FilterData()
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


def calculate_clause_score_old(left, right):
    """
    计算两个词向量之间的内积,然后做排序
    #　分别计算了两个词在句子中的排序和
    :param left: 原因句
    :param right: 结果句
    :return:
    """
    d = {}
    for l in left:
        for r in right:
            if l not in cause_embed or r not in effect_embed:
                # print(l, r)
                continue
            l_vec, r_vec = np.array(cause_embed[l], dtype=np.float64), np.array(effect_embed[r], dtype=np.float64)
            # print(len(l_vec), len(r_vec))
            d[' '.join([str(l), str(r)])] = l_vec.dot(r_vec.T)
    result = sorted(d.items(), key=lambda item: item[1], reverse=True)
    # print(result)
    return result


class Calculate_score(object):
    def __init__(self):
        self.cause_embed = defaultdict(list)
        self.effect_embed = defaultdict(list)
        pass

    def calculate_clause_score(self, left, right):
        left_list, right_list = [], []
        for word in left:
            if word in self.cause_embed:
                left_list.append(word)
        for word in right:
            if word in self.effect_embed:
                right_list.append(word)
        n_l, n_r = len(left_list), len(right_list)

        dot_list = [[float('-inf')] * n_r for i in range(n_l)]
        # 行n_l 列:n_r
        for l in range(n_l):
            for r in range(n_r):
                l_vec, r_vec = np.array(self.cause_embed[left_list[l]], dtype=np.float64), np.array(
                    self.effect_embed[right_list[r]], dtype=np.float64)
                dot_list[l][r] = l_vec.dot(r_vec.T)
                if left_list[l] == right_list[r]:
                    dot_list[l][r] = float('-inf')
                    pass
        dot_list = np.array(dot_list)
        return dot_list, n_l, n_r, left_list, right_list

    def get_cause_word_embedding(self, cause_path, effect_path):

        cause_embed_dict = defaultdict(list)
        effect_embed_dict = defaultdict(list)
        with codecs.open(cause_path, 'r', 'utf-8') as fcause, codecs.open(effect_path, 'r', 'utf-8') as feffect:
            fcause.readline()
            line = fcause.readline()
            while line:
                if line.strip() == '':
                    continue
                lists = line.strip().split(' ')
                cause_embed_dict[lists[0]] = lists[1:]
                line = fcause.readline()
                # for i in range(1, len(lists)):
                #     cause_embed_dict[lists[0]].append(lists[i])

            feffect.readline()
            line = feffect.readline()
            while line:
                if line.strip() == '':
                    continue
                lists = line.strip().split(' ')
                effect_embed_dict[lists[0]] = lists[1:]
                line = feffect.readline()
                # for i in range(1, len(lists)):
                #     effect_embed_dict[lists[0]].append(lists[i])
            print('读取完毕')
            self.cause_embed = cause_embed_dict
            self.effect_embed = effect_embed_dict
        return cause_embed_dict, effect_embed_dict

    @staticmethod
    def get_score(choose_kind, score_phrase_1, len_l_1, len_r_1, left_list_1, right_list_1, score_phrase_2, len_l_2,
                  len_r_2, left_list_2, right_list_2):
        # score_1, score_2 = 0, 0
        score_1, score_2 = float('-inf'), float('-inf')
        if choose_kind == "max_macth":
            count_for_id_choose = 0
            score_phrase_1_ = sorted(score_phrase_1.reshape(-1), reverse=True)
            score_phrase_2_ = sorted(score_phrase_2.reshape(-1), reverse=True)
            if len_l_1 * len_r_1 > 0:
                score_1 = score_phrase_1_[count_for_id_choose]
            if len_l_2 * len_r_2 > 0:
                score_2 = score_phrase_2_[count_for_id_choose]
            while score_1 == score_2:
                try:
                    count_for_id_choose += 1
                    score_1 = score_phrase_1_[count_for_id_choose]
                    score_2 = score_phrase_2_[count_for_id_choose]
                except IndexError as e:
                    score_1, score_2 = float('-inf'), float('-inf')
                    break
        elif choose_kind == "pair_wise_macth":
            if len_l_1 * len_r_1 > 0:
                score_1 = np.sum(score_phrase_1) / (len_l_1 * len_r_1)
            if len_l_2 * len_r_2 > 0:
                score_2 = np.sum(score_phrase_2) / (len_l_2 * len_r_2)
        elif choose_kind == "top_k_match":
            score_phrase_1 = sorted(score_phrase_1.reshape(-1), reverse=True)
            score_phrase_2 = sorted(score_phrase_2.reshape(-1), reverse=True)
            if len(score_phrase_1) > 0:
                max_len_p_a1 = max(len(score_phrase_1), 2)
                score_1 = sum(score_phrase_1[0:max_len_p_a1]) / max_len_p_a1
            if len(score_phrase_2) > 0:
                max_len_p_a2 = max(len(score_phrase_2), 2)
                score_2 = sum(score_phrase_2[0:max_len_p_a2]) / max_len_p_a2
        elif choose_kind == "attentive_match":
            if len_l_1 * len_r_1 > 0:
                score_1 = (np.sum(np.max(score_phrase_1, axis=0)) + np.sum(np.max(score_phrase_1, axis=1))) / (
                        len_l_1 + len_r_1)
            if len_l_2 * len_r_2 > 0:
                score_2 = (np.sum(np.max(score_phrase_2, axis=0)) + np.sum(np.max(score_phrase_2, axis=1))) / (
                        len_l_2 + len_r_2)
        if choose_kind == 'max_macth':
            if score_1 != score_2:
                if len_l_1 * len_r_1 > 0 and len_l_2 * len_r_2 > 0:
                    x, y = np.unravel_index(score_phrase_1.argmax(), score_phrase_1.shape)
                    # print(left_list_1[x], right_list_1[y], score_phrase_1_[count_for_id_choose])
                    x, y = np.unravel_index(score_phrase_2.argmax(), score_phrase_2.shape)
                    # print(left_list_2[x], right_list_2[y], score_phrase_2_[count_for_id_choose])
                    pass
        return score_1, score_2


def get_acc(file_name, pr_save_path, choose_kind="max_macth"):
    acc = 0
    id_number_for_space_phrase = 0
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
            # print(phrase_fir, phrase_choose1, phrase_choice2)

            if flag == 'effect':
                score_phrase_1, len_l_1, len_r_1, left_list_1, right_list_1 = calculate_score.calculate_clause_score(
                    phrase_fir, phrase_choose1)
                score_phrase_2, len_l_2, len_r_2, left_list_2, right_list_2 = calculate_score.calculate_clause_score(
                    phrase_fir, phrase_choice2)
            else:
                score_phrase_1, len_l_1, len_r_1, left_list_1, right_list_1 = calculate_score.calculate_clause_score(
                    phrase_choose1, phrase_fir)
                score_phrase_2, len_l_2, len_r_2, left_list_2, right_list_2 = calculate_score.calculate_clause_score(
                    phrase_choice2, phrase_fir)

            # print(score_phrase_1.shape, len_l_1, len_r_1)
            # print(score_phrase_2.shape, len_l_2, len_r_2)
            score_1, score_2 = Calculate_score.get_score(
                choose_kind, score_phrase_1, len_l_1, len_r_1, left_list_1, right_list_1, score_phrase_2, len_l_2,
                len_r_2, left_list_2, right_list_2)

            if label == 0:
                result_list.append([score_1, 1])
                result_list.append([score_2, 0])
                show_list.append([score_1, phrase_fir, phrase_choose1, 1])
                show_list.append([score_2, phrase_fir, phrase_choice2, 0])
            elif label == 1:
                result_list.append([score_1, 0])
                result_list.append([score_2, 1])
                show_list.append([score_1, phrase_fir, phrase_choose1, 0])
                show_list.append([score_2, phrase_fir, phrase_choice2, 1])

            if score_1 == score_2:
                id_number_for_space_phrase += 1
            else:
                score_compare = (score_1 < score_2)
                if score_compare == label:
                    acc += 1
                else:
                    # print(train_data)
                    pass
    print('id_number_for_space_phrase:', id_number_for_space_phrase)

    total_congju = sorted(result_list, key=functools.cmp_to_key(cmp))
    show_list = sorted(show_list, key=functools.cmp_to_key(cmp))

    max_points = Cruve_PR.get_pr_points(total_congju, relevant_label=1)
    Cruve_PR.save_points(max_points, pr_save_path)

    # flag_id = 0
    # for i in show_list:
    #     flag_id += 1
    #     print(flag_id, i)
    #     if flag_id > 10000:
    #         break
    total_congjus = total_congju

    count_map = 0
    num_true = 0
    num_total = 0
    for i in total_congjus:
        num_total += 1
        if i[1] == 1:
            num_true += 1
            count_map += float(num_true / num_total)
        if i[0] < 0:
            # break
            pass

    print(count_map, num_true, num_total)
    map_acc = float(count_map) / num_true
    print(map_acc)

    return round(acc / (500 - id_number_for_space_phrase), 4)


if __name__ == '__main__':
    pass
    with open('stop_word.txt') as file:
        eng_stop_word = file.readlines()
        eng_stop_word = set([word.strip() for word in eng_stop_word])
    data_path = '../../data/'
    train_file_name = data_path + 'test/COPA/COPA/train.jsonl'
    test_file_name = data_path + 'test/COPA/COPA/test.jsonl'
    step = 2
    cause_dev = ['claim', 'mercy', 'law', 'accident', 'explosion', 'earthquake', 'virus', 'storm', 'war']
    effect_dev = ['happy', 'heard', 'surprised', 'pollution', 'death', 'loss', 'failure', 'disease', 'illness', 'flood']

    cause_output_path = data_path + 'embedding' + '/cause_embedding_max_match_output_path'
    effect_output_path = data_path + 'embedding' + '/effect_embedding_max_match_output_path'

    # cause_output_path = data_path + 'embedding' + '/cause_embedding_top_k_match_output_path'
    # effect_output_path = data_path + 'embedding' + '/effect_embedding_top_k_match_output_path'
    #
    # cause_output_path = data_path + 'embedding' + '/cause_embedding_pair_wise_match_output_path'
    # effect_output_path = data_path + 'embedding' + '/effect_embedding_pair_wise_match_output_path'
    #
    # cause_output_path = data_path + 'embedding' + '/cause_embedding_attentive_match_output_path'
    # effect_output_path = data_path + 'embedding' + '/effect_embedding_attentive_match_output_path'

    print(cause_output_path, effect_output_path)

    for i in range(2, 20):
        step = i
        tail = '_' + str(step) + '.txt'
        cause_paths, effect_paths = cause_output_path + tail, effect_output_path + tail
        calculate_score = Calculate_score()
        cause_embed, effect_embed = calculate_score.get_cause_word_embedding(cause_paths, effect_paths)
        # print(cause_embed['to'])
        # print(effect_embed['to'])
        # print(type(cause_embed), type(effect_embed))
        # for word in cause_dev:
        #     print(word, cause_embed[word])
        # print('---')
        # for word in effect_dev:
        #     print(word, effect_embed[word])

        # print(len(cause_embed))
        # print(len(effect_embed))
        pr_save_paths = 'PRCurve_copa' + '_dev' + '_cause_effect_embedding.txt'
        dev_acc = get_acc(train_file_name, pr_save_paths, choose_kind="max_macth")
        pr_save_paths = 'PRCurve_copa' + '_test' + '_cause_effect_embedding.txt'
        test_acc = get_acc(test_file_name, pr_save_paths, choose_kind="max_macth")
        print(step, "max_macth:", ", dev_acc:", dev_acc, 'test_acc:', test_acc)

        # dev_acc = get_acc(train_file_name, choose_kind="pair_wise_macth")
        #
        # test_acc = get_acc(test_file_name, choose_kind="pair_wise_macth")
        # print(step, "pair_wise_macth", ", dev_acc:", dev_acc, 'test_acc:', test_acc)
        #
        # dev_acc = get_acc(train_file_name, choose_kind="top_k_match")
        # test_acc = get_acc(test_file_name, choose_kind="top_k_match")
        # print(step, "top_k_match", ", dev_acc:", dev_acc, 'test_acc:', test_acc)
        #
        # dev_acc = get_acc(train_file_name, choose_kind="attentive_match")
        # test_acc = get_acc(test_file_name, choose_kind="attentive_match")
        # print(step, "attentive_match", ", dev_acc:", dev_acc, 'test_acc:', test_acc)
