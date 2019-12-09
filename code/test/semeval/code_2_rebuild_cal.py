import pickle
import random
import sys
import numpy as np
# sys.path.append()
import codecs
# from new_test_7_3 import code_method4
import functools
from reference.test_for_COPA import Calculate_score
from reference.test_for_COPA import Cruve_PR


def cmp(x, y):
    # 用来调整顺序
    if x[0] > y[0]:
        return -1
    if x[0] < y[0]:
        return 1
    return 0


def filter_word(phrase, stopkey):
    # 过滤,去掉停用词
    filter_words = []
    for word in phrase:
        if word not in stopkey:
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


def get_pr_points(sorted_score, relevant_label):
    """
    :param sorted_score: item in it is like: (score, label)
    :param relevant_label:
    :return:
    """
    numPair = len(sorted_score)
    assert numPair > 0
    a= [1 for s in sorted_score if s[1] == relevant_label]
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


def save_points(points, saved_path):
    fout = codecs.open(saved_path, 'w', 'utf-8')
    for recall, precision in points:
        fout.write('{} {}\n'.format(recall, precision))


def calculate_result(data_list_lines1, data_list_lines2, pr_save_path, choose_kind='max_macth'):
    len_pattern = len(data_list_lines1)
    result_list = []
    show_list = []
    # 统计得到结果
    for i in range(len_pattern):
        sent_X_1, sent_Y_1, label_1 = data_list_lines1[i]
        sent_X_2, sent_Y_2, label_2 = data_list_lines2[i]
        sent_X_1, sent_Y_1, sent_X_2, sent_Y_2 = [sent_X_1], [sent_Y_1], [sent_X_2], [sent_Y_2]
        score_phrase_1, len_l_1, len_r_1, left_list_1, right_list_1 = calculate_score.calculate_clause_score(
            sent_X_1, sent_Y_1)

        score_phrase_2, len_l_2, len_r_2, left_list_2, right_list_2 = calculate_score.calculate_clause_score(
            sent_X_2, sent_Y_2)
        score_1, score_2 = Calculate_score.get_score(
            choose_kind, score_phrase_1, len_l_1, len_r_1, left_list_1, right_list_1, score_phrase_2, len_l_2,
            len_r_2, left_list_2, right_list_2)

        result_list.append([score_1, 1])
        result_list.append([score_2, 0])
        show_list.append([score_1, sent_X_1, sent_Y_1, 1])
        show_list.append([score_2, sent_X_2, sent_Y_2, 0])

    total_congju = sorted(result_list, key=functools.cmp_to_key(cmp))
    show_list = sorted(show_list, key=functools.cmp_to_key(cmp))
    # print(total_congju)

    max_points = Cruve_PR.get_pr_points(total_congju, relevant_label=1)
    Cruve_PR.save_points(max_points, pr_save_path)

    flag_id = 0
    # for total_congju_sig in show_list:
    #     print(flag_id, total_congju_sig)
    #     flag_id += 1
    #     if flag_id == 200:
    #         break

    count = 0
    num_true = 0
    num_total = 0
    for i in total_congju:
        num_total += 1
        if i[1] == 1:
            num_true += 1

            count += float(num_true / num_total)
            if i[0] < -0.5:
                # break
                pass

    print(int(count), num_true, len_pattern)
    acc = float(count) / num_true

    # print("accuracy:", acc)
    return acc


if __name__ == '__main__':

    cause_data = np.load("data_list_line" + ".npy", allow_pickle=True)
    not_cause_data = np.load("not_cause_data_list_line" + ".npy", allow_pickle=True)
    sample_not_cause_data = np.load("temp_sample_not_cause_data_list_line" + ".npy", allow_pickle=True)

    neg_data = random.sample(list(not_cause_data), len(cause_data))
    print(len(cause_data))
    print(len(not_cause_data))
    print(len(sample_not_cause_data))

    data_path = '../../../data/'
    doc_name = ''
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

    for i in range(3, 4):
        step = i
        tail = '_' + str(step) + '.txt'
        cause_paths, effect_paths = cause_output_path + tail, effect_output_path + tail
        calculate_score = Calculate_score()
        calculate_score.get_cause_word_embedding(cause_paths, effect_paths)
        pr_save_paths = 'PRCurve' + '_semeval' + '_PMI.txt'
        acc = calculate_result(cause_data, neg_data, pr_save_paths, choose_kind="max_macth")
        print(step, "max_macth:", ", acc:", acc)


