import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from sklearn import preprocessing
import pickle
from glob import glob
import os
from collections import defaultdict
import functools
from datetime import datetime

# # 模拟数据点
# def generate(sample_size, mean, cov, diff):
#     samples_per_class = int(sample_size / 2)
#
#     X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
#     Y0 = np.zeros(samples_per_class)
#
#     for ci, d in enumerate(diff):
#         X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
#         Y1 = (ci + 1) * np.ones(samples_per_class)
#
#         X0 = np.concatenate((X0, X1))
#         Y0 = np.concatenate((Y0, Y1))
#
#     X, Y = shuffle(X0, Y0)
#
#     return X, Y
#
#
# input_dim = 2
# np.random.seed(10)
# num_classes = 2  # 2分类
# mean = np.random.randn(num_classes)
# cov = np.eye(num_classes)
# # dataset
# X, Y = generate(1000, mean, cov, [3.0])
# print(len(X), len(Y))
# colors = ['r' if l == 0 else 'b' for l in Y[:]]
# plt.scatter(X[:, 0], X[:, 1], c=colors)
# plt.xlabel("Scaled age (in yrs)")
# plt.ylabel("Tumor size (in cm)")
# plt.show()
starts = datetime.now()
"""
在to_cause_effect_with_label中保存最原始的数据
# 经过筛选得到100w+和100w-样本 保存在classifier_data/fonction_classify 中
将数据由str转成id,保存在classifier_data/fonction_str_id中
"""


def cmp(x, y):
    if x[3] > y[3]:
        return 1
    if x[3] < y[3]:
        return -1
    else:
        return 0


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


def get_str_id_word(now_data_path):
    str_id_path = now_data_path + 'extraction/' + 'fonction_word_str_id_word' + '.file'
    with open(str_id_path, 'rb') as file:
        str_id_word_temp = pickle.load(file)
    str_id_word = defaultdict(int)
    for word in str_id_word_temp:
        str_id_word[word] += str_id_word_temp[word]
    return str_id_word


class Generate_Data(object):

    def __init__(self):
        self.filter_pos_other_flag = 1
        self.filter_stop_word_flag = 1
        self.filter_isalpha_flag = 1
        self.filter_pos_not_function_or_content_flag = 1
        self.filter_pos_function_flag = 1
        self.pos_filter = {'X', 'NUM', 'PRON', 'SYM', 'SPACE', 'DET', 'PUNCT'}
        # self.pos_function = {'PART', 'ADP', 'CCONJ'}
        self.pos_function = {'PART', 'ADP', 'CCONJ', 'ADV'}  #
        self.pos_content = {'VERB', 'ADJ', 'NOUN'}
        self.pos_other = {'INTJ', 'ADV', 'PROPN', }
        with open('stop_word.txt') as file:
            eng_stop_word = file.readlines()
            self.eng_stop_word = set([word.strip() for word in eng_stop_word])
        # print(self.eng_stop_word)

        self.evidence_list = ['cause_label_to', 'phrase_and', 'cause_cue']
        # self.input_list = ['to_cause_effect_with_label/icw', 'to_cause_effect_with_label/bok',
        #                    'to_cause_effect_with_label/gut']
        self.input_list = ['data_cause_effect/icw', 'data_cause_effect/bok', 'data_cause_effect/gut']
        self.output_list = ['classifier_data/fonction_classify/icw', 'classifier_data/fonction_classify/bok',
                            'classifier_data/fonction_classify/gut']

        self.evidence_choose = 1
        self.data_choose = 1

        self.data_path = '../../data/'
        str_id_path = self.data_path + 'extraction/' + 'fonction_word_str_id_word' + '.file'
        with open(str_id_path, 'rb') as file:
            self.str_id_word = pickle.load(file)
        self.pos_function_dict = defaultdict(int)
        self.word_list = defaultdict(int)
        self.fonction_word_dict = {'towards', 'hastily', 'precisely', 'heavily', 'perpetually', 'ago', 'independently',
                                   'thankfully', 'decidedly', 'hardly', 'seldom', 'eternally', 'equally', 'undoubtedly',
                                   'vaguely', 'surely', 'whilst', 'loud', 'patiently', 'first', 'particularly', 'best',
                                   'longer', 'regardless', 'sometimes', 'per', 'lol', 'thither', 'comparatively',
                                   'late', 'financially', 'least', 'lest', 'namely', 'wherein', 'damn', 'repeatedly',
                                   'closer', 'astride', 'altogether', 'faithfully', 'backwards', 'minus', 'while',
                                   'mainly', 'inside', 'consequently', 'twice', 'closely', 'kind', 'eachother',
                                   'typically', 'low', 'carefully', 'unlike', 'automatically', 'therein', 'properly',
                                   'unfortunately', 'thereby', 'somewhat', 'live', 'beneath', 'continually', 'rarely',
                                   'individually', 'casually', 'rather', 'consistently', 'farther', 'shortly',
                                   'further', 'overly', 'spiritually', 'calmly', 'literally', 'personally', 'gradually',
                                   'intensely', 'subsequently', 'anymore', 'loosely', 'daily', 'mostly', 'likely',
                                   'thereof', 'potentially', 'brightly', 'sincerely', 'lightly', 'south', 'full',
                                   'nicely', 'conveniently', 'anyways', 'easy', 'violently', 'highly', 'peculiarly',
                                   'hitherto', 'indirectly', 'wanna', 'xd', 'course', 'reluctantly', 'second', 'aloud',
                                   'blindly', 'open', 'everyday', 'seemingly', 'sometime', 'good', 'invariably',
                                   'fairly', 'horribly', 'earlier', 'lily', 'super', 'publicly', 'solely', 'obviously',
                                   'ta', 'remarkably', 'forth', 'emotionally', 'totally', 'round', 'doubt', 'forever',
                                   'fast', 'inevitably', 'outside', 'afterwards', 'lower', 'principally', 'easily',
                                   'immensely', 'within', 'basically', 'nowhere', 'everywhere', 'relatively', 'behind',
                                   'ordinarily', 'promptly', 'intimately', 'worse', 'anyway', 'secretly', 'currently',
                                   'utterly', 'intentionally', 'whereof', 'honestly', 'beforehand', 'upon', 'barely',
                                   'profoundly', 'scarcely', 'legally', 'actively', 'oddly', 'voluntarily', 'anyhow',
                                   'wisely', 'anywhere', 'deep', 'presumably', 'vs', 'awhile', 'primarily',
                                   'previously', 'less', 'necessarily', 'steadily', 'n', 'finally', 'perfectly',
                                   'curiously', 'luckily', 'definitely', 'practically', 'hopefully', 'readily', 'sure',
                                   'beyond', 'wildly', 'effectually', 'among', 'eagerly', 'quicker', 'securely',
                                   'evidently', 'temporarily', 'amidst', 'entirely', 'someday', 'immediately',
                                   'wherever', 'unusually', 'physically', 'incredibly', 'wrong', 'gladly', 'nearest',
                                   'anti', 'across', 'last', 'strictly', 'henceforth', 'lately', 'effectively',
                                   'likewise', 'possibly', 'furthermore', 'gently', 'rapidly', 'accordingly', 'firmly',
                                   'secondly', 'exactly', 'short', 'positively', 'technically', 'somewhere', 'whereas',
                                   'til', 'importantly', 'openly', 'exceedingly', 'regularly', 'infinitely', 'cos',
                                   'quickly', 'formerly', 'beautifully', 'randomly', 'onto', 'overboard',
                                   'ridiculously', 'though', 'underneath', 'near', 'wholly', 'initially', 'abruptly',
                                   'upside', 'before', 'around', 'against', 'next', 'aside', 'freely', 'wonderfully',
                                   'unconsciously', 'inasmuch', 'naturally', 'sa', 'earnestly', 'online', 'sooner',
                                   'louder', 'cheerfully', 'otherwise', 'directly', 'without', 'specifically', 'truly',
                                   'strangely', 'hereafter', 'besides', 'widely', 'successfully', 'ashore', 'merely',
                                   'tightly', 'backward', 'sadly', 'differently', 'certainly', 'save', 'de',
                                   'downstairs', 'comfortably', 'slowly', 'again', 'continuously', 'firstly', 'tight',
                                   'sufficiently', 'alongside', 'verily', 'else', 'usually', 'dearly', 'boldly',
                                   'significantly', 'way', 'supposedly', 'upstairs', 'between', 'permanently', 'justly',
                                   'moreover', 'frequently', 'forward', 'safely', 'albeit', 'adequately', 'higher',
                                   'considerably', 'indeed', 'unless', 'apart', 'ever', 'fine', 'via', 'why', 'anytime',
                                   'silently', 'ill', 'accidentally', 'politely', 'whether', 'together', 'beside',
                                   'ultimately', 'william', 'socially', 'thru', 'nowadays', 'therefrom', 'seriously',
                                   'wouldst', 'thirdly', 'approximately', 'nearer', 'prior', 'fortunately', 'sort',
                                   'probably', 'keenly', 'once', 'somehow', 'correctly', 'toward', 'cruelly',
                                   'sexually', 'similarly', 'enough', 'largely', 'happily', 'cautiously', 'elsewhere',
                                   'badly', 'partly', 'nearly', 'surprisingly', 'speedily', 'occasionally', 'until',
                                   'gravely', 'despite', 'generally', 'neither', 'greatly', 'amid', 'newly', 'briefly',
                                   'later', 'terribly', 'deeper', 'poorly', 'during', 'purposely', 'below', 'except',
                                   'faster', 'roughly', 'doubly', 'partially', 'quietly', 'almost', 'close', 'north',
                                   'pleasantly', 'everytime', 'commonly', 'yet', 'therefore', 'better', 'smoothly',
                                   'whither', 'officially', 'straight', 'amongst', 'reasonably', 'bitterly', 'loudly',
                                   'fully', 'high', 'half', 'above', 'desperately', 'expressly', 'constantly', 'upward',
                                   'halfway', 'sharply', 'meanwhile', 'evenly', 'unto', 'alone', 'already', 'frankly',
                                   'real', 'sally', 'although', 'kindly', 'lastly', 'especially', 'nearby', 'along',
                                   'hard', 'swiftly', 'versus', 'distinctly', 'astray', 'morally', 'instead',
                                   'overnight', 'essentially', 'kinda', 'upwards', 'nay', 'due', 'thus', 'past',
                                   'originally', 'purely', 'privately', 'matter', 'whence', 'downward', 'exclusively',
                                   'nevertheless', 'harder', 'plainly', 'painfully', 'clearly', 'instinctively',
                                   'verses', 'emily', 'presently', 'willingly', 'throughout', 'away', 'eventually',
                                   'quick', 'aboard', 'whereby', 'wide', 'universally', 'through', 'slightly',
                                   'heartily', 'deeply', 'genuinely', 'whatsoever', 'strongly', 'hence', 'normally',
                                   'virtually', 'neatly', 'deliberately', 'chiefly', 'afterward', 'en', 'accurately',
                                   'softly', 'thereafter', 'alike', 'suddenly', 'unexpectedly', 'whenever', 'plus',
                                   'till', 'wherefore', 'either', 'instantly', 'logically', 'recently', 'early', 'ok',
                                   'absolutely', 'billy', 'separately', 'ahead', 'awfully', 'bout', 'notwithstanding',
                                   'fiercely', 'consciously', 'little', 'completely', 'mentally', 'rightly',
                                   'specially', 'increasingly', 'thereupon', 'actually', 'ere', 'rob', 'after',
                                   'severely', 'apparently', 'abroad', 'extremely', 'sideways', 'doubtless',
                                   'simultaneously', 'extra', 'opposite', 'thoroughly', 'under'}

        # self.fonction_word_dict = {'furthermore', 'consequently', 'twice', 'closely', }

    def cmp(self, x, y):
        if x[3] > y[3]:
            return 1
        if x[3] < y[3]:
            return -1
        else:
            return 0

    def filter_pos_other(self, word):
        return word[1] not in self.pos_filter

    def filter_left_word(self, word):
        return word[0] in self.fonction_word_dict

    def filter_stop_word(self, word):
        return word[0] not in self.eng_stop_word

    def filter_isalpha(self, word):
        return word[0].encode('UTF-8').isalpha()

    def filter_pos_function(self, word):
        return word[1] not in self.pos_function

    def get_pos_function(self, word):
        return word[1] in self.pos_function

    def filter_pos_not_function_or_content(self, word):
        return word[1] not in self.pos_other

    def generate_data(self, evidence, input_path, output_path):
        """
        原始数据中包含: 原因,结果,前后方向和score
        将其中的数据保存成只包含function词的部分
        :param evidence:
        :param input_path:
        :param output_path:
        :return:
        """

        file_list = list(sorted(glob(os.path.join(input_path, evidence + '*.npy'))))
        print(file_list)
        count1 = [0, 0, 0]
        content_data_true = []
        for cause_effect_pair_path in file_list:
            now_file_name = cause_effect_pair_path.split('/')[-1]

            print(cause_effect_pair_path)
            cause_effect_list = np.load(cause_effect_pair_path, allow_pickle=True)
            # print(cause_effect_list)
            for cause_effect_pair in cause_effect_list:
                cause_clause1, effect_clause1 = cause_effect_pair
                # print([word[0] for word in cause_clause])
                # print([word[0] for word in effect_clause])
                # print('---')

                if evidence == 'phrase_and':
                    effect_clause1 = effect_clause1[:-1]

                cause_clause1 = sorted(cause_clause1, key=functools.cmp_to_key(self.cmp))
                effect_clause1 = sorted(effect_clause1, key=functools.cmp_to_key(self.cmp))
                cause_clause, effect_clause = cause_clause1, effect_clause1
                if self.filter_pos_other_flag:
                    cause_clause = filter(self.filter_pos_other, cause_clause)
                    effect_clause = filter(self.filter_pos_other, effect_clause)
                    cause_clause, effect_clause = list(cause_clause), list(effect_clause)
                if self.filter_stop_word_flag:
                    cause_clause = filter(self.filter_stop_word, cause_clause)
                    effect_clause = filter(self.filter_stop_word, effect_clause)
                    cause_clause, effect_clause = list(cause_clause), list(effect_clause)
                if self.filter_isalpha_flag:
                    cause_clause = filter(self.filter_isalpha, cause_clause)
                    effect_clause = filter(self.filter_isalpha, effect_clause)
                    cause_clause, effect_clause = list(cause_clause), list(effect_clause)
                cause_clause = filter(self.get_pos_function, cause_clause)
                effect_clause = filter(self.get_pos_function, effect_clause)
                cause_clause, effect_clause = list(cause_clause), list(effect_clause)
                cause_clause = filter(self.filter_left_word, cause_clause)
                effect_clause = filter(self.filter_left_word, effect_clause)
                cause_clause, effect_clause = list(cause_clause), list(effect_clause)
                for word in cause_clause:
                    self.word_list[word[0]] += 1
                for word in effect_clause:
                    self.word_list[word[0]] += 1

                cause_clause = [word[0] for word in cause_clause]
                effect_clause = [word[0] for word in effect_clause]
                # pmi_cha_list[socre_cha] += 1

                # 过滤

                if len(cause_clause) + len(effect_clause) > 1:
                    # print(' '.join([word for word in cause_clause]))
                    # print(' '.join([word for word in effect_clause]))
                    # print(' '.join([word[0] for word in cause_clause1]))
                    # print(' '.join([word[0] for word in effect_clause1]))
                    data_label = 1
                    content_data_true.append([cause_clause, effect_clause, data_label])

                # print(fonction_word_pos_list)
                if len(content_data_true) >= 10000:
                    print(len(content_data_true))
                    content_data_false = []
                    L = len(content_data_true)
                    for i in range(5 * L):
                        # print(random.randint(0, L), random.randint(0, L))
                        left = content_data_true[random.randint(0, L - 1)][0]
                        right = content_data_true[random.randint(0, L - 1)][1]
                        data_label = 0
                        content_data_false.append([left, right, data_label])

                    count1[0] += 1
                    save_path = output_path + '/' + 'content_data_true' + str(count1[0]) + '.npy'
                    np.save(save_path, content_data_true)
                    save_path_false = output_path + '/' + 'content_data_false' + str(count1[0]) + '.npy'
                    np.save(save_path_false, content_data_false)
                    print(save_path, save_path_false)
                    del content_data_true[:]
                    print_time()
                    print('保存完成')

        count1[0] += 1
        count1[1] += 1
        count1[2] += 1
        content_data_false = []
        L = len(content_data_true)
        for i in range(5 * L):
            # print(random.randint(0, L), random.randint(0, L))
            left = content_data_true[random.randint(0, L - 1)][0]
            right = content_data_true[random.randint(0, L - 1)][1]
            data_label = 0
            content_data_false.append([left, right, data_label])
        save_path = output_path + '/' + 'content_data_true' + str(count1[0]) + '.npy'
        np.save(save_path, content_data_true)
        save_path_false = output_path + '/' + 'content_data_false' + str(count1[0]) + '.npy'
        np.save(save_path_false, content_data_false)
        print(save_path)
        print('保存完成')

    def main(self):
        for i in range(3):
            data_choose = i
            #     break
            # if True:
            #     data_choose = 1
            input_path = self.data_path + self.input_list[data_choose]
            output_path = self.data_path + self.output_list[data_choose]

            self.generate_data(self.evidence_list[2], input_path, output_path)
            print(self.pos_function_dict)
            print(len(self.pos_function_dict))
            print_time()
            print(self.word_list)
            print(len(self.word_list))


class Generate_Data_for_predict(Generate_Data):

    def generate_data_for_predict(self, input_pair_path):
        """
        这个函数的主要目的是,想要获得每个句子对应的predict socre
        :param input_pair_path:
        :param output_pair_path:
        :return:
        """

        full_data_with_predic = []
        print(input_pair_path)
        cause_effect_list = np.load(input_pair_path, allow_pickle=True)
        # print(cause_effect_list)
        for cause_effect_pair in cause_effect_list:

            cause_clause, effect_clause, direction_label, pmi_score_left_to_right, pmi_score_right_to_left = cause_effect_pair
            # print([word[0] for word in cause_clause])
            # print([word[0] for word in effect_clause])
            # print('---')
            # if direction_label == 0:
            #     # 过滤
            #     continue
            cause_clause_orig, effect_clause_orig = cause_clause, effect_clause
            cause_clause = sorted(cause_clause, key=functools.cmp_to_key(self.cmp))
            effect_clause = sorted(effect_clause, key=functools.cmp_to_key(self.cmp))

            if self.filter_pos_other_flag:
                cause_clause = filter(self.filter_pos_other, cause_clause)
                effect_clause = filter(self.filter_pos_other, effect_clause)
                cause_clause, effect_clause = list(cause_clause), list(effect_clause)
            if self.filter_stop_word_flag:
                cause_clause = filter(self.filter_stop_word, cause_clause)
                effect_clause = filter(self.filter_stop_word, effect_clause)
                cause_clause, effect_clause = list(cause_clause), list(effect_clause)
            if self.filter_isalpha_flag:
                cause_clause = filter(self.filter_isalpha, cause_clause)
                effect_clause = filter(self.filter_isalpha, effect_clause)
                cause_clause, effect_clause = list(cause_clause), list(effect_clause)
            cause_clause = filter(self.get_pos_function, cause_clause)
            effect_clause = filter(self.get_pos_function, effect_clause)
            cause_clause, effect_clause = list(cause_clause), list(effect_clause)
            cause_clause = [word[0] for word in cause_clause]
            effect_clause = [word[0] for word in effect_clause]

            # 过滤
            full_data_with_predic.append(
                [cause_clause_orig, effect_clause_orig, cause_clause, effect_clause])
            # if (len(cause_clause) > 1) & (len(effect_clause) > 0):
            #
            #     # if direction_label == -1:
            #     #     cause_clause, effect_clause = effect_clause, cause_clause
            #
            #     full_data_with_predic.append(
            #         [cause_clause_orig, effect_clause_orig, cause_clause, effect_clause, direction_label, score])
        return full_data_with_predic


def generate_merge_data():
    def generate_data(true_data):
        x_list = []
        for data in true_data:
            word_bags = [0] * (2 * len_str_id_word + 2)
            cause_list, effect_list, label = data
            for word in cause_list:
                word_bags[str_id_word[word]] += 1
            for word in effect_list:
                word_bags[len_str_id_word + 1 + str_id_word[word]] += 1
            word_bags = word_bags[:input_dim]
            word_bags = preprocessing.minmax_scale(word_bags)
            word_bags = list(word_bags)
            word_bags.append(label)
            x_list.append(word_bags)
            # print(cause_list, effect_list, word_bags)
            # x_list.append([word_bags, label])
        print(len(x_list))
        return x_list

    sample_batch_list = ['content_data_left_', 'content_data_right_', 'negtive_data_', 'content_data_true',
                         'content_data_false']

    input_list = ['classifier_data/fonction_classify/icw', 'classifier_data/fonction_classify/bok',
                  'classifier_data/fonction_classify/gut']
    output_list = ['classifier_data/fonction_str_id/icw', 'classifier_data/fonction_str_id/bok',
                   'classifier_data/fonction_str_id/gut']

    for data_choose in range(0, 3):
        input_path = data_path + input_list[data_choose]
        output_path = data_path + output_list[data_choose]
        Content_file_list = list(sorted(glob(os.path.join(input_path, sample_batch_list[3] + '*.npy'))))
        count = 0
        for evidence_pair_path_id in range(len(Content_file_list)):
            count += 1
            # if count < 66:
            #     continue
            file_name = Content_file_list[evidence_pair_path_id].split('/')[-1]
            output_file_name = output_path + '/' + file_name
            print(Content_file_list[evidence_pair_path_id])
            print(output_file_name)
            Content_data = np.load(Content_file_list[evidence_pair_path_id], allow_pickle=True)
            save_data = generate_data(Content_data)
            np.save(output_file_name, save_data)
        print(word_total_list)
        print(len(word_total_list))


def logistic_model(input_tensor, regularizer, net_choose='ligistic'):
    if net_choose == 'ligistic':
        weight = tf.Variable(initial_value=tf.random_normal([input_dim, num_labels]), name="weight")  # 2 x 1
        b = tf.Variable(initial_value=tf.random_normal([num_labels]), name="bias")  # 1
        logist = tf.matmul(input_tensor, weight) + b
    elif net_choose == 'tanhNN':
        print(net_choose)
        weight = tf.Variable(initial_value=tf.random_normal([input_dim, n_hidden]), name="weight1")  # 2 x 1
        d = tf.Variable(initial_value=tf.random_normal([n_hidden]), name="bias1")  # 1
        u = tf.Variable(initial_value=tf.random_normal([n_hidden, num_labels]), name="weight2")  # 2 x 1
        b = tf.Variable(initial_value=tf.random_normal([num_labels]), name="bias2")  # 1
        tanh = tf.nn.tanh(tf.matmul(input_tensor, weight) + d)
        logist = tf.matmul(tanh, u) + b
    tf.add_to_collection('weight', weight)
    tf.add_to_collection('bias', b)
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weight))
    # tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
    return logist


def train_model(net_choose='ligistic'):
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[None, input_dim])
    y = tf.placeholder(shape=[None, ], dtype=tf.int32, name="input_ids")
    one_hot_labels = tf.one_hot(y, depth=num_labels, dtype=tf.float32)
    y1 = tf.placeholder(tf.float32, shape=[None, 1])

    regularizer = tf.contrib.layers.l2_regularizer(l2_learning_rate)

    logist = logistic_model(x, regularizer, net_choose=net_choose)
    w = tf.get_collection('weight')
    w = w[0]
    b = tf.get_collection('bias')
    b = b[0]
    # log_probs = tf.nn.log_softmax(logist, axis=-1)
    # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    # loss = tf.reduce_mean(per_example_loss)
    # loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logist, labels=one_hot_labels))

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels, logits=logist))
    cost = loss + tf.add_n(tf.get_collection('losses'))
    predict = tf.greater(logist, 0.5)
    acc = tf.reduce_mean(tf.cast(tf.equal(one_hot_labels, tf.cast(predict, dtype=tf.float32)), "float"), name="accuracy")

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logist, labels=one_hot_labels))
    # cost = loss + tf.add_n(tf.get_collection('losses'))
    # predict = tf.argmax(logist, axis=1, name="predictions", output_type=tf.int32)
    # acc = tf.reduce_mean(tf.cast(tf.equal(y, tf.cast(predict, dtype=tf.int32)), "float"), name="accuracy")

    """
    * 错误方法注意:此时使用sigmoid应该保证logist得到的结果>0,所以更好的方法是调用API
    output = tf.nn.sigmoid(logist)
    cost = tf.reduce_mean(-(y * tf.log(output) + (1 - y) * tf.log(1 - output)))
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    display_step = 1
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        trained_weights = w.eval()
        # print(trained_weights)
        for epoch in range(epochs):
            epoch_loss = 0
            acc_average = 0
            for data_choose in range(0, 3):
                total_batch_all = 0
                input_path = data_path + input_list[data_choose]
                content_data_false_list = list(
                    sorted(glob(os.path.join(input_path, 'content_data_false' + '*.npy'))))
                # content_data_false_list = list(
                #     sorted(glob(os.path.join(input_path, 'negtive_data' + '*.npy'))))

                content_data_true_list = list(
                    sorted(glob(os.path.join(input_path, 'content_data_true' + '*.npy'))))

                print(len(content_data_true_list), len(content_data_false_list))

                for evidence_pair_path_id in range(len(content_data_false_list[:3])):

                    print(content_data_true_list[evidence_pair_path_id], content_data_false_list[evidence_pair_path_id])
                    content_data_true = np.load(content_data_true_list[evidence_pair_path_id], allow_pickle=True)
                    negtive_data = np.load(content_data_false_list[evidence_pair_path_id], allow_pickle=True)
                    print(content_data_true.shape, negtive_data.shape)

                    row_rand_array = np.arange(negtive_data.shape[0])
                    np.random.shuffle(row_rand_array)
                    negtive_data = negtive_data[row_rand_array[0:len(content_data_true)]]
                    # Content_data = np.vstack((np.vstack((content_data_left, negtive_data)), content_data_right))
                    Content_data = np.vstack((content_data_true, negtive_data))
                    print(content_data_true.shape, negtive_data.shape, Content_data.shape)
                    Content_data = shuffle(Content_data)
                    x_list = Content_data[:, :-1]

                    y_list = Content_data[:, -1]
                    # x_list, y_list = list(x_list), list(y_list)
                    total_batch = len(y_list) // batch_size
                    total_batch_all += total_batch
                    for i in range(total_batch):
                        # print("batch:", i)
                        start_id = i * batch_size
                        end_id = (i + 1) * batch_size
                        batch_x = x_list[start_id:end_id]
                        batch_y = y_list[start_id:end_id]
                        # batch_y = np.reshape(batch_y, [-1, 1])
                        _, loss, acc_now = sess.run([optimizer, cost, acc], feed_dict={x: batch_x, y: batch_y})

                        epoch_loss += loss
                        acc_average += acc_now
                    # trained_weights = w.eval()
                    # trained_weights = trained_weights.flatten()
                    # print(list(trained_weights))
                # print(list(sess.run([logist, one_hot_labels, y, predict], feed_dict={x: batch_x, y: batch_y})))

                epoch_loss /= total_batch_all
                acc_average /= total_batch_all
                if epoch % display_step == 0:
                    trained_weights = w.eval()
                    print(list(trained_weights))
                    trained_weights = trained_weights.flatten()
                    print(list(trained_weights))

                    print("epoch: %d,data_kind: %d, loss: %.4f, acc: %.4f" % (
                        epoch, data_choose, epoch_loss, acc_average))

        saver.save(sess, 'model/loigstic_model.ckpt')
        print("done..")


def predict_model(checkpoint_dir='model', net_choose='ligistic'):
    batch_size = 1
    input_list = ['to_cause_effect_with_label/icw', 'to_cause_effect_with_label/bok',
                  'to_cause_effect_with_label/gut']
    output_list = ['classifier_data/get_socre_used_fonction/icw', 'classifier_data/get_socre_used_fonction/bok',
                   'classifier_data/get_socre_used_fonction/gut']

    def generate_data(total_data):
        X_list = []

        for data in total_data:
            word_bags = [0] * (2 * len_str_id_word + 2)
            cause_clause_orig, effect_clause_orig, cause_list, effect_list = data

            for word in cause_list:
                # a = str_id_word[word]
                word_bags[str_id_word[word]] += 1

            for word in effect_list:
                word_bags[len_str_id_word + 1 + str_id_word[word]] += 1
                word_bags = word_bags[:input_dim]
                word_bags = preprocessing.minmax_scale(word_bags)
            # print(cause_list, effect_list, word_bags)
            X_list.append([cause_clause_orig, effect_clause_orig, cause_list, effect_list, word_bags])
        return X_list

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[None, input_dim])

    regularizer = tf.contrib.layers.l2_regularizer(l2_learning_rate)
    logist = logistic_model(x, regularizer, net_choose=net_choose)
    logist = tf.nn.softmax(logist)
    predict = tf.argmax(logist, axis=1, name="predictions")

    w = tf.get_collection('weight')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            epoch_loss = 0
            w = w[0]
            trained_weights = w.eval()
            trained_weights = trained_weights.flatten()
            print(list(trained_weights))
            for data_choose in range(0, 3):
                total_batch_all = 0
                input_path = data_path + input_list[data_choose]
                Content_file_list = list(
                    sorted(glob(os.path.join(input_path, evidence_list[evidence_choose] + '*.npy'))))
                print(Content_file_list)
                for evidence_pair_path in Content_file_list:
                    now_file_name = evidence_pair_path.split('/')[-1]
                    save_data = []
                    full_data_with_predic = Generate_Data_for_predict().generate_data_for_predict(evidence_pair_path)
                    x_list = generate_data(full_data_with_predic)

                    for data in x_list:
                        cause_clause_orig, effect_clause_orig, cause_list, effect_list, word_bags = data
                        pred_y, max_y = sess.run([logist, predict], feed_dict={x: [word_bags]})
                        print(pred_y, max_y)

                    #     resu_y_left = pred_y[0][0][0]
                    #     resu_y_right = max_y[0][0][0]
                    #     if resu_y_left > resu_y_right:
                    #         direction_use_fonction = 1
                    #     elif resu_y_left < resu_y_right:
                    #         direction_use_fonction = -1
                    #     else:
                    #         direction_use_fonction = 0
                    #     resu_y = round(max(resu_y_left, resu_y_right), 2)
                    #     #
                    #     if resu_y > 0.2:
                    #         # if resu_y < -0.8:
                    #         # if (resu_y > 0.1) & (resu_y < 0.2):
                    #         print(resu_y)
                    #         if direction_use_fonction == -1:
                    #             print('反向')
                    #             cause_clause_orig1, effect_clause_orig1 = effect_clause_orig, cause_clause_orig
                    #         elif direction_use_fonction == 0:
                    #             print('无法判断')
                    #             cause_clause_orig1, effect_clause_orig1 = cause_clause_orig, effect_clause_orig
                    #         else:
                    #             cause_clause_orig1, effect_clause_orig1 = cause_clause_orig, effect_clause_orig
                    #         cause_clause_orig1 = sorted(cause_clause_orig1, key=functools.cmp_to_key(cmp))
                    #         effect_clause_orig1 = sorted(effect_clause_orig1, key=functools.cmp_to_key(cmp))
                    #         print(' '.join([word[0] for word in cause_clause_orig1]))
                    #         print(' '.join([word[0] for word in effect_clause_orig1]))
                    #         print('----')
                    #
                    #     score_dict[resu_y] += 1
                    #     save_data.append(
                    #         [cause_clause_orig, effect_clause_orig, cause_list, effect_list, direction_use_pmi, score,
                    #          direction_use_fonction, resu_y])
                    # output_paht_name = data_path + output_list[data_choose] + '/' + now_file_name
                    # print(output_paht_name)
                    # np.save(output_paht_name, save_data)
                    # print(score_dict)
                    # print(len(score_dict))
        else:
            pass
        print("done..")


if __name__ == '__main__':
    data_path = '../../data/'

    # evidence_list = ['cause_label_to']
    evidence_list = ['cause_label_to', 'phrase_and']
    evidence_choose = 0
    word_total_list = defaultdict(int)

    input_list = ['classifier_data/fonction_str_id/icw', 'classifier_data/fonction_str_id/bok',
                  'classifier_data/fonction_str_id/gut']
    str_id_word = get_str_id_word(data_path)

    len_str_id_word = len(str_id_word)
    input_dim = 2 * len_str_id_word + 2

    epochs = 8
    num_labels = 1
    batch_size = 64
    learning_rate = 0.01
    print(str_id_word)
    print(input_dim)
    l2_learning_rate = 1e-4
    n_hidden = 50
    print('epochs:', epochs)
    print('l2_learning_rate:', l2_learning_rate)
    print('n_hidden:', n_hidden)
    score_dict = defaultdict(int)

    Generate_Data().main()
    # generate_merge_data()
    train_model()
    # predict_model()
    # train_model(net_choose='tanhNN')
    # predict_model(net_choose='tanhNN')
