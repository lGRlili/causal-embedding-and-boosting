import pickle
import sys
import numpy as np
# sys.path.append()
import codecs
from new_test_7_3 import code_method4
import functools


def cmp(x, y):
    # 用来调整顺序
    if x[0] > y[0]:
        return -1
    if x[0] < y[0]:
        return 1
    return 0


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


def calculate_sentence_pair_likelihood_ratio(sent_X, sent_Y, COPA_word_pair_class_likelihood_ratios, choosed_pair,
                                             method="pairwise"):

    temp = 1.
    for word_x in sent_X:
        for word_y in sent_Y:
            word_pair = word_x + "_" + word_y
            if word_pair in COPA_word_pair_class_likelihood_ratios:
                temp *= COPA_word_pair_class_likelihood_ratios[word_pair]
    return temp


def calculate_result(data_list_lines1, data_list_lines2, COPA_word_pair_class_likelihood_ratios,
                     class_priors, choosed_pair, doc_name):
    class_prior_ratio = class_priors[0] / class_priors[1]
    count = 0
    len_pattern = len(data_list_lines1)
    true_pattern = len(data_list_lines1)
    result_list = []
    for i in range(len_pattern):
        sent_X, sent_Y = data_list_lines1[i]
        class_likelihood_ratio_1 = calculate_sentence_pair_likelihood_ratio(
            sent_X, sent_Y, COPA_word_pair_class_likelihood_ratios, choosed_pair)

        sent_X_2, sent_Y_2 = data_list_lines2[i]
        class_likelihood_ratio_2 = calculate_sentence_pair_likelihood_ratio(
            sent_X_2, sent_Y_2, COPA_word_pair_class_likelihood_ratios, choosed_pair)

        # class_posterior_1 = 1 / (1 + class_likelihood_ratio_1 * class_prior_ratio)
        # class_posterior_2 = 1 / (1 + class_likelihood_ratio_2 * class_prior_ratio)

        if class_likelihood_ratio_1 == 1:
            class_posterior_1 = 1e-10
        else:
            class_posterior_1 = 1 / (1 + class_likelihood_ratio_1 * class_prior_ratio)
        if class_likelihood_ratio_2 == 1:
            class_posterior_2 = 1e-10
        else:
            class_posterior_2 = 1 / (1 + class_likelihood_ratio_2 * class_prior_ratio)

        ratio = class_posterior_1 / class_posterior_2
        if ratio > 1:
            count = count + 1
        if ratio == 1:
            true_pattern -= 1
        result_list.append([class_posterior_1, 1])
        result_list.append([class_posterior_2, 0])
    print(count, true_pattern, len_pattern)
    print(count/true_pattern)
    print('二选一准确率', count / true_pattern)
    total_congju = sorted(result_list, key=functools.cmp_to_key(cmp))
    pr_save_path = 'PRCurve' + doc_name + '_ourmethods.txt'
    max_points = get_pr_points(total_congju, relevant_label=1)
    save_points(max_points, pr_save_path)
    flag_id = 0
    # for total_congju_sig in total_congju:
    #     print(flag_id, total_congju_sig)
    #     flag_id += 1
    count_map = 0
    num_true = 0
    num_total = 0
    for i in total_congju:
        num_total += 1
        if i[1] == 1:
            num_true += 1

            count_map += float(num_true / num_total)
    print(num_true, num_total)
    acc = float(count_map) / num_true
    print(acc)
    # print(count, true_pattern, len_pattern)
    # acc = float(count) / true_pattern

    # print("accuracy:", acc)
    return acc


def cal_all_word_pair_likelihood_ratios(COPA_word_pairs, dict_word2id, P_word_lemma_num, COPA_advcl_word_pair_counter,
                                        COPA_conj_word_pair_counter, COPA_inter_word_pair_counter, evidence_priors,
                                        evidence_counts, class_1_evidence_probs, class_0_evidence_probs):
    COPA_word_pair_class_likelihood_ratios = {}

    Min = 1000000000
    Max = 0
    COPA_words = []
    for wp in COPA_word_pairs:
        # print(w)
        a = wp.split("_")
        word_a = a[0]
        word_b = a[1]
        # 将str转换成id
        COPA_words.extend([word_a, word_b])

        w1w2 = wp
        w2w1 = a[1] + '_' + a[0]

        advcl_likelihood = (COPA_advcl_word_pair_counter[w1w2] + 1e-4) / evidence_counts[0]  # 1e-4
        conj_likelihood = (COPA_conj_word_pair_counter[w1w2] + 1e-4) / evidence_counts[1]  # 1e-4
        inter_likelihood = (COPA_inter_word_pair_counter[w1w2] + 2e-4) / evidence_counts[2]  # 2e-4
        other_likelihood = (P_word_lemma_num[word_a] * P_word_lemma_num[word_b] + 1e5) / evidence_counts[3]  # 1e5

        # advcl_likelihood = (COPA_advcl_word_pair_counter[w1w2]) / evidence_counts[0]  # 1e-4
        # conj_likelihood = (COPA_conj_word_pair_counter[w1w2]) / evidence_counts[1]  # 1e-4
        # inter_likelihood = (COPA_inter_word_pair_counter[w1w2]) / evidence_counts[2]  # 2e-4
        # other_likelihood = (word_id_counter[id_a] * word_id_counter[id_b] + 1e5) / evidence_counts[3]  # 1e5

        evidence_likelihoods = [advcl_likelihood, conj_likelihood, inter_likelihood, other_likelihood]

        numerator, denominator = 0, 0
        for i in range(4):
            numerator += class_0_evidence_probs[i] * evidence_likelihoods[i]
            denominator += class_1_evidence_probs[i] * evidence_likelihoods[i]

        COPA_word_pair_class_likelihood_ratios[w1w2] = numerator / denominator

    return COPA_word_pair_class_likelihood_ratios


def grid_tuning_five(islemma, doc_name, data_list_line1, data_list_line2, COPA_word_pairs, grid):
    print(doc_name)
    print("grid_tuning_five")
    print('获得基础数据')
    # dict_word2id: a dict mapping word to id.
    # word_id_counter: a list recording the counts of words (ids)
    # COPA_advcl_word_pair_counter:
    # COPA_conj_word_pair_counter:
    # COPA_inter_word_pair_counter:

    if doc_name == '_bok':
        doc_name_temp = '_book_corpus'
        with open('../database_original/temp' + doc_name_temp + '/str_id_' + 'word_count_delete.file', 'rb') as f:
            dict_word2id = pickle.load(f)
        print(len(dict_word2id))
        word_id_counter = np.load('../database_original/temp' + doc_name_temp + '/P_word_id.npy')
    else:
        with open('../database_original/temp' + doc_name + '/str_id_' + 'word_count_delete.file', 'rb') as f:
            dict_word2id = pickle.load(f)
        print(len(dict_word2id))
        word_id_counter = np.load('../database_original/temp' + doc_name + '/P_word_id.npy')
    with open("data/P_word_lemma_num.file", "rb") as f:
        P_word_lemma_num = pickle.load(f)
    with open('data/temp' + doc_name + '/concept_advcl_pair_pos' + '.file', 'rb') as f:
        COPA_advcl_word_pair_counter = pickle.load(f)
    with open('data/temp' + doc_name + '/concept_conj_pair_pos' + '.file', 'rb') as f:
        COPA_conj_word_pair_counter = pickle.load(f)
    with open('data/temp' + doc_name + '/concept_inter_pair_pos' + '.file', 'rb') as f:
        COPA_inter_word_pair_counter = pickle.load(f)

    choosed_pair = []
    print("读取数据完毕")

    prior_prob_other = grid[4]
    # 此处分别得到4中证据类型的比例和
    evidence_priors, evidence_counts = code_method4.cal_evidence_priors(COPA_word_pairs, dict_word2id, word_id_counter,
                                                           COPA_advcl_word_pair_counter, COPA_conj_word_pair_counter,
                                                           COPA_inter_word_pair_counter, prior_prob_other)
    advcl_class_1_prob = grid[0]
    conj_class_1_prob = grid[1]
    inter_class_1_prob = grid[2]
    other_class_1_prob = grid[3]

    other_class_0_prob = 1 - other_class_1_prob
    advcl_class_0_prob = 1 - advcl_class_1_prob
    conj_class_0_prob = 1 - conj_class_1_prob
    inter_class_0_prob = 1 - inter_class_1_prob

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
    COPA_word_pair_class_likelihood_ratios = \
        cal_all_word_pair_likelihood_ratios(
            COPA_word_pairs, dict_word2id, P_word_lemma_num, COPA_advcl_word_pair_counter,
            COPA_conj_word_pair_counter, COPA_inter_word_pair_counter, evidence_priors,
            evidence_counts, class_1_evidence_probs, class_0_evidence_probs)

    # print(COPA_word_pair_class_likelihood_ratios)
    print('COPA_word_pair_class_likelihood_ratios:', len(COPA_word_pair_class_likelihood_ratios))
    print([advcl_class_1_prob, conj_class_1_prob, inter_class_1_prob,
           other_class_1_prob, prior_prob_other])

    # 后面对具体的进行计算

    acc = calculate_result(
        data_list_line1, data_list_line2, COPA_word_pair_class_likelihood_ratios,
        class_priors, choosed_pair, doc_name)
    print("accuracy:", acc)


def filter_nparray(doc_name):
    with open("data/cause_effect_pair" + ".file", "rb") as f:
        P_A_B_pair = pickle.load(f)
    cause_effect_choose_pair = set()
    with open('data/temp' + doc_name + '/concept_advcl_pair_pos' + '.file', 'rb') as f:
        advcl_pair_pos_copa_icw = pickle.load(f)
    with open('data/temp' + doc_name + '/concept_conj_pair_pos' + '.file', 'rb') as f:
        conj_pair_pos_copa_icw = pickle.load(f)
    with open('data/temp' + doc_name + '/concept_inter_pair_pos' + '.file', 'rb') as f:
        inter_pair_pos_copa_icw = pickle.load(f)
    # with open('data/concept_advcl_pair_pos' + '.file', 'rb') as f:
    #     advcl_pair_pos_copa_icw = pickle.load(f)
    # with open('data/concept_conj_pair_pos' + '.file', 'rb') as f:
    #     conj_pair_pos_copa_icw = pickle.load(f)
    # with open('data/concept_inter_pair_pos' + '.file', 'rb') as f:
    #     inter_pair_pos_copa_icw = pickle.load(f)
    # inter_pair_pos_copa_icw, conj_pair_pos_copa_icw = advcl_pair_pos_copa_icw, advcl_pair_pos_copa_icw
    for lemma_pair in P_A_B_pair:
        if (advcl_pair_pos_copa_icw[lemma_pair] == 0) | (conj_pair_pos_copa_icw[lemma_pair] == 0) | (
                inter_pair_pos_copa_icw[lemma_pair] == 0):
            pass
        else:
            cause_effect_choose_pair.add(lemma_pair)
    print(len(P_A_B_pair), len(cause_effect_choose_pair))

    with open("data/cause_effect_choose_pair.file", "wb") as f:
        pickle.dump(cause_effect_choose_pair, f)
    return cause_effect_choose_pair


# 399 410----覆盖了覆盖了0.97
if __name__ == '__main__':

    data_list_line1 = np.load("data/data_list_line1" + ".npy", allow_pickle=True)
    data_list_line2 = np.load("data/data_list_line2" + ".npy", allow_pickle=True)
    # with open("data/cause_effect_pair" + ".file", "rb") as fi:
    #     COPA_word_pairs = pickle.load(fi)
    # print(len(COPA_word_pairs))
    # print(len(data_list_line2), len(data_list_line1))
    icw_grids_666 = [[0.15, 0.02, 0.1, 0.0001, 0.98], [0.3, 0.04, 0.2, 0.0001, 0.98]]
    bok_grids_642 = [[0.1, 0.04, 0.4, 0.0001, 0.9], [0.1, 0.04, 0.45, 0.0001, 0.9]]
    # 0.648
    gut_grids_606 = [[0.65, 0.06, 0.05, 0.0001, 0.95], [0.9, 0.08, 0.15, 0.0001, 0.85],
                     [0.95, 0.08, 0.15, 0.0001, 0.85]]
    grids = [[], [0.15, 0.02, 0.1, 0.0001, 0.98], [0.65, 0.06, 0.05, 0.0001, 0.95], [0.1, 0.04, 0.4, 0.0001, 0.9]]
    choose = ['', '_icw', '_gut', '_bok']
    for i in range(1, 4):
        cause_effect_choose_pair = filter_nparray(choose[i])
        print('cause_effect_choose_pair', len(cause_effect_choose_pair))
        grid_tuning_five('', choose[i], data_list_line1, data_list_line2, cause_effect_choose_pair, grids[i])
        code_method4.print_time()
        # 0.716
