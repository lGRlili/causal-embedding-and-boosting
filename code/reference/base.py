# -*- coding: utf-8 -*-
import sys
sys.path.extend(['../'])
import numpy as np
from collections import Counter
import tensorflow as tf
import math
import codecs
import random
import functools


class FilterData(object):
    def __init__(self):
        self.pos_filter = {'X', 'NUM', 'PRON', 'SYM', 'SPACE', 'DET', 'PUNCT'}
        self.pos_function = {'PART', 'ADP', 'CCONJ'}
        self.pos_content = {'VERB', 'ADJ', 'NOUN'}
        # 这个部分那不一定注释,其中包括了副词和表示情感的词语
        self.pos_other = {'INTJ', 'ADV', 'PROPN', }

        with open('stop_word.txt') as file:
            eng_stop_word = file.readlines()
            self.eng_stop_word = set([word.strip() for word in eng_stop_word])
        # print(self.eng_stop_word)
        pass

    def filter_pos_other(self, word):
        return word[1] not in self.pos_filter

    def filter_stop_word(self, word):
        return word[0] not in self.eng_stop_word

    def filter_isalpha(self, word):
        return word[0].encode('UTF-8').isalpha()

    def filter_pos_function(self, word):
        return word[1] not in self.pos_function

    def filter_pos_not_function_or_content(self, word):
        return word[1] not in self.pos_other

    @staticmethod
    def filter_now(cause_clause, effect_clause, filter_kind):
        cause_clause = filter(filter_kind, cause_clause)
        effect_clause = filter(filter_kind, effect_clause)
        cause_clause, effect_clause = list(cause_clause), list(effect_clause)
        return cause_clause, effect_clause

    @staticmethod
    def cmp(x, y):
        if x[3] > y[3]:
            return 1
        if x[3] < y[3]:
            return -1
        else:
            return 0


class BaseData(object):
    def __init__(self, sample_neg_randomly, num_samples=None):
        # 这个参数用于决定生成正样本还是进行max采样
        self.sample_neg_randomly = sample_neg_randomly
        self.vocab_left, self.vocab_rev_left, self.vocab_left_size = [], {}, 0
        self.vocab_right, self.vocab_rev_right, self.vocab_right_size = [], {}, 0
        self.c2e_test, self.e2c_test = [], []
        self.x_left, self.x_right, self.x_target = None, None, None
        self.max_length = 0
        self.test_left, self.test_right, self.test_pairs = None, None, None
        self.num_samples = num_samples
        self.labels = []

    @staticmethod
    def load_samples(data_file_path):
        """
        samples: 地震导致多人死亡----地震----多人 死亡
        """
        input_left, input_right = [], []
        count = 0
        with codecs.open(data_file_path, 'r', 'utf-8') as fin:
            while True:
                line = fin.readline()
                count += 1
                print(len(line), line)
                if not line:
                    break
                if len(line) > 1:
                    items = line.strip().split('----')
                    s1 = items[0].split(' ')
                    s2 = items[1].split(' ')
                    if '' in s1 or '' in s2:
                        continue
                    # 其中s1 和s2都是包含不止一个单词
                    input_left.append(s1)
                    input_right.append(s2)
        return input_left, input_right

    @staticmethod
    def load_samples_from_numpy(
            data_file_path, filter_pos_other_flag=1, filter_stop_word_flag=1, filter_isalpha_flag=1,
            filter_pos_not_function_or_content_flag=1, filter_pos_function_flag=1):
        """
        samples: 从原始获得的numpy语料中抽取出对应的数据,首先过滤数据,只保留内容词的部分
        """

        filter_data = FilterData()
        input_left, input_right = [], []
        for data_file in data_file_path:
            cause_effect_list = np.load(data_file, allow_pickle=True)
            for cause_effect_pair in cause_effect_list:
                cause_clause, effect_clause = cause_effect_pair
                cause_clause = sorted(cause_clause, key=functools.cmp_to_key(FilterData.cmp))
                effect_clause = sorted(effect_clause, key=functools.cmp_to_key(FilterData.cmp))

                # 过滤其他类型
                if filter_pos_other_flag:
                    cause_clause, effect_clause = filter_data.filter_now(
                        cause_clause, effect_clause, filter_data.filter_pos_other)
                # 过滤停用词
                if filter_stop_word_flag:
                    cause_clause, effect_clause = filter_data.filter_now(
                        cause_clause, effect_clause, filter_data.filter_stop_word)
                # 　过滤含有其他字符的词
                if filter_isalpha_flag:
                    cause_clause, effect_clause = filter_data.filter_now(
                        cause_clause, effect_clause, filter_data.filter_isalpha)
                # 过滤第三类成分词
                if filter_pos_not_function_or_content_flag:
                    cause_clause, effect_clause = filter_data.filter_now(
                        cause_clause, effect_clause, filter_data.filter_pos_not_function_or_content)
                # 过滤功能词
                if filter_pos_function_flag:
                    cause_clause, effect_clause = filter_data.filter_now(
                        cause_clause, effect_clause, filter_data.filter_pos_function)

                if len(cause_clause) > 0 and len(effect_clause) > 0:
                    if len(cause_clause) > 50 or len(effect_clause) > 50:
                        pass
                    else:
                        # print(' '.join([word[0] for word in sorted(cause_effect_pair[0], key=functools.cmp_to_key(FilterData.cmp))]))
                        # print(' '.join([word[0] for word in sorted(cause_effect_pair[1], key=functools.cmp_to_key(FilterData.cmp))]))
                        # print(' '.join([word[0] for word in cause_clause]))
                        # print(' '.join([word[0] for word in effect_clause]))
                        # print('-------')
                        input_left.append([word[0] for word in cause_clause])
                        input_right.append([word[0] for word in effect_clause])
                    # for word in cause_clause:
                    #     input_left.append(word[0])
                    # for word in effect_clause:
                    #     input_right.append(word[0])
            print(len(input_left), len(input_right))
        return input_left, input_right

    def load_dev(self, labeled_path):
        # cause_dev = ['侵扰', '事故', '爆炸', '台风', '冲突', '矛盾', '地震', '农药', '违章', '腐蚀',
        #              '感染', '病毒', '暴雨', '疲劳', '真菌', '贫血', '感冒', '战乱', '失调', '摩擦']
        # effect_dev = ['污染', '愤怒', '困境', '损失', '不适', '疾病', '失事', '悲剧', '危害', '感染',
        #               '故障', '死亡', '痛苦', '失败', '矛盾', '疲劳', '病害', '塌陷', '洪灾']
        cause_dev = ['claim', 'mercy', 'law', 'accident', 'explosion', 'earthquake', 'virus', 'storm', 'war']
        effect_dev = ['happy', 'heard', 'surprised', 'pollution', 'death', 'loss', 'failure', 'disease', 'illness', 'flood']
        """
        load test word to show similarity
        """
        for w in cause_dev:
            try:
                self.c2e_test.append(self.vocab_rev_left[w])
            except KeyError as e:
                print('{} is not existed in cause vocab!'.format(e))
        for w in effect_dev:
            try:
                self.e2c_test.append(self.vocab_rev_right[w])
            except KeyError as e:
                print('{} is not existed in effect vocab!'.format(e))
        # self.load_labeled_data(labeled_path)

    def load_labeled_data_from_numpy(self, data_path):
        lefts, rights, pairs = [], [], []
        with codecs.open(data_path, 'r', 'utf-8') as f:
            line = f.readline()
            while line:
                if line.strip() == '':
                    continue
                result = line.strip().split('##')
                pair = result[1].split(' ')
                left = result[0].split('----')[1].split(' ')
                right = result[0].split('----')[2].split(' ')
                try:
                    pair_left, pair_right = self.vocab_rev_left[pair[0]], self.vocab_rev_right[pair[1]]
                    temp_left, temp_right = [], []
                    for l in left:
                        try:
                            temp_left.append(self.vocab_rev_left[l])
                        except KeyError as e:
                            pass
                    for w in right:
                        try:
                            temp_right.append(self.vocab_rev_right[w])
                        except KeyError as e:
                            pass
                    if pair_left in temp_left and pair_right in temp_right:
                        lefts.append(temp_left)
                        rights.append(temp_right)
                        pairs.append([pair_left, pair_right])
                except KeyError:
                    c = 0
                line = f.readline()
        self.test_left, self.test_right, self.test_pairs = lefts, rights, pairs
        print('num of valid labelled data is {}.'.format(len(self.test_pairs)))

    def load_labeled_data(self, data_path):
        """
        注意:这些样本中,只有正例的句子
        :param data_path: 输入数据中包含信息: 原始语料, cause,effect cause中核心词,effect中的核心词
        :return:
        """
        # 58冬季放水不当引起的故障----冬季 放水 不当----故障##不当 故障
        lefts, rights, pairs = [], [], []
        with codecs.open(data_path, 'r', 'utf-8') as f:
            line = f.readline()
            while line:
                if line.strip() == '':
                    continue
                result = line.strip().split('##')
                pair = result[1].split(' ')
                left = result[0].split('----')[1].split(' ')
                right = result[0].split('----')[2].split(' ')
                try:
                    pair_left, pair_right = self.vocab_rev_left[pair[0]], self.vocab_rev_right[pair[1]]
                    temp_left, temp_right = [], []
                    for l in left:
                        try:
                            temp_left.append(self.vocab_rev_left[l])
                        except KeyError as e:
                            pass
                    for w in right:
                        try:
                            temp_right.append(self.vocab_rev_right[w])
                        except KeyError as e:
                            pass
                    if pair_left in temp_left and pair_right in temp_right:
                        lefts.append(temp_left)
                        rights.append(temp_right)
                        pairs.append([pair_left, pair_right])
                except KeyError:
                    c = 0
                line = f.readline()
        self.test_left, self.test_right, self.test_pairs = lefts, rights, pairs
        print('num of valid labelled data is {}.'.format(len(self.test_pairs)))

    def build_vocab(self, new_x_left, new_x_right, min_count):
        """
        1. build vocab, most frequent words while have low index in vocab
        2. the word <pad> will be insert into vocab at head position in nce node because of padding
        """
        # assert mode in ['pad', 'not_pad']
        data_left = [word for x in new_x_left for word in x]
        data_right = [word for x in new_x_right for word in x]
        # data_left = new_x_left
        # data_right = new_x_right

        number = 0
        c = Counter(data_left)
        for key in c:
            if c[key] < min_count:
                number += 1
        counter = c.most_common(len(c) - number)
        self.vocab_left = [counter[i][0] for i in range(len(counter))]
        del c, counter

        number = 0
        c = Counter(data_right)
        for key in c:
            if c[key] < min_count:
                number += 1
        counter = c.most_common(len(c) - number)
        self.vocab_right = [counter[i][0] for i in range(len(counter))]
        del c, counter
        # 保存所有的词
        self.vocab_left.insert(0, '<pad>')
        self.vocab_right.insert(0, '<pad>')
        # 保存所有的词对应id
        self.vocab_rev_left = {x: i for i, x in enumerate(self.vocab_left)}
        self.vocab_rev_right = {x: i for i, x in enumerate(self.vocab_right)}
        # 保存id_str和str_id

        # print(self.vocab_rev_left)
        # print(self.vocab_rev_right)

        self.vocab_left_size = len(self.vocab_left)
        self.vocab_right_size = len(self.vocab_right)
        del data_left, data_right

    def convert2one_hot(self, input_left, input_right):
        """
        1. convert to ont hot representation
        2. get max_length of left and right
        """
        one_hot_left, one_hot_right = list(), list()
        max_length = 0
        count1, count2 = 0, 0
        for i in range(len(input_left)):
            cause, effect = list(), list()
            left = input_left[i]
            right = input_right[i]
            for w_l in left:
                try:
                    cause.append(self.vocab_rev_left[w_l])
                except KeyError as e:
                    count1 += 1
            for w_r in right:
                try:
                    effect.append(self.vocab_rev_right[w_r])
                except KeyError as e:
                    count2 += 1
            if cause and effect:
                one_hot_left.append(cause)
                one_hot_right.append(effect)
                # 更新最大句子长度
                local_max_length = max(len(cause), len(effect))
                if max_length < local_max_length:
                    max_length = local_max_length
        return one_hot_left, one_hot_right, max_length

    def padding_data(self, one_hot_left, one_hot_right):
        # 将全部的单词加上padding
        padding_word = 0
        new_x_left = list()
        new_x_right = list()
        for i in range(len(one_hot_left)):
            x_left = one_hot_left[i]
            num_padding = self.max_length - len(x_left)
            new_x = x_left + [padding_word] * num_padding
            new_x_left.append(new_x)
            x_right = one_hot_right[i]
            num_padding = self.max_length - len(x_right)
            new_y = x_right + [padding_word] * num_padding
            new_x_right.append(new_y)
        return new_x_left, new_x_right

    def neg_sent(self, batch_size):
        """
        get negative samples from sentence
        """
        neg_left, neg_right = [], []
        L = len(self.x_left) - 1
        for i in range(self.num_samples * batch_size):
            k = random.randint(0, L)
            neg_left.append(self.x_left[k])
            j = random.randint(0, L)
            neg_right.append(self.x_right[j])
        return neg_left, neg_right

    @staticmethod
    def get_len(one_hot_left, one_hot_right):
        left_len, right_len = [], []
        for i in range(len(one_hot_right)):
            left_len.append(len(one_hot_left[i]))
            right_len.append(len(one_hot_right[i]))
        return left_len, right_len

    def load_data(self, data_path, min_count):
        if self.sample_neg_randomly:
            # 获取样本,构建词典
            samples_left, samples_right = self.load_samples(data_path['pos_path'])
            # samples_left, samples_right = self.load_samples_from_numpy(data_path['pos_path'])
            self.build_vocab(samples_left, samples_right, min_count)
            # 将单词进行one-hot编码
            onehot_left, onehot_right, max_len = self.convert2one_hot(samples_left, samples_right)
            self.x_left, self.x_right, self.max_length = onehot_left, onehot_right, max_len
        else:
            pos_left, pos_right = self.load_samples(data_path['pos_path'])
            neg_left, neg_right = self.load_samples(data_path['neg_path'])
            self.build_vocab(pos_left, pos_right, min_count)
            pos_onehot_left, pos_onehot_right, pos_max_len = self.convert2one_hot(pos_left, pos_right)
            pos_target = [1.0 for _ in range(len(pos_onehot_left))]
            neg_onehot_left, neg_onehot_right, neg_max_len = self.convert2one_hot(neg_left, neg_right)
            neg_target = [0.0 for _ in range(len(neg_onehot_left))]
            self.x_left = pos_onehot_left + neg_onehot_left
            self.x_right = pos_onehot_right + neg_onehot_right
            self.x_target = pos_target + neg_target
            self.max_length = max(pos_max_len, neg_max_len)
        self.load_dev(data_path['labeled_path'])


class BaseModel(object):

    def __init__(self, embedding_size, batch_size, num_epochs, num_samples, learning_rate, data_loader):
        self.dataLoader = data_loader
        self.sample_neg_randomly = data_loader.sample_neg_randomly
        self.max_len, self.num_samples, self.num_epochs = 0, num_samples, num_epochs
        self.embedding_size, self.batch_size = embedding_size, batch_size
        # used for showing similarity of cause->effect and effect->cause
        self.cause_word_id, self.effect_word_id = None, None
        self.cause_normed_embed, self.c2e_similar = None, None
        self.effect_normed_embed, self.e2c_similar = None, None

        self.input_left, self.input_right = None, None
        self.left_len, self.right_len, self.targets = None, None, None
        self.input_left_embed, self.input_right_embed = None, None
        # embedding of cause and effect vocabs
        self.cause_embed_dict, self.effect_embed_dict = None, None
        self.vocab_left_size, self.vocab_right_size = 0, 0

        self.train_op, self.loss, self.global_steps, self.init = None, None, None, None
        self.learning_rate, self.average_loss = learning_rate, 0.0
        self.graph, self.sess = tf.Graph(), None

    def load_data(self, data_path, min_count):
        self.dataLoader.load_data(data_path, min_count)
        self.max_len = self.dataLoader.max_length
        self.vocab_left_size = self.dataLoader.vocab_left_size
        self.vocab_right_size = self.dataLoader.vocab_right_size
        if '' in self.dataLoader.vocab_left or '' in self.dataLoader.vocab_right:
            print('error')
            exit(1)
        print('max length of phrase: {}!'.format(self.max_len))
        print('length of left vocab: {}'.format(self.dataLoader.vocab_left_size))
        print('length of right vocab: {}'.format(self.dataLoader.vocab_right_size))

    def construct_graph(self):
        pass

    def train_stage(self, cause_output_path, effect_output_path):
        pass

    @staticmethod
    def shuffle_pos_neg(pos_data, neg_data):
        data = np.array(pos_data + neg_data)
        shuffle_indices = np.random.permutation(np.arange(len(data)))
        return data[shuffle_indices]

    def show_loss(self, feed_dict):
        _, global_step, loss_val = self.sess.run([self.train_op, self.global_steps, self.loss],
                                                 feed_dict=feed_dict)
        self.average_loss += loss_val
        current_step = tf.train.global_step(self.sess, self.global_steps)
        # print(current_step)
        if current_step % 1000 == 0:
            self.average_loss /= 1000
            print('Average loss at step ', current_step, ': ', self.average_loss)
            self.average_loss = 0.0
        if current_step % 10000 == 0:
            self.show_similar()

    def init_embedding(self):
        self.cause_embed_dict = tf.Variable(tf.truncated_normal(
            [self.vocab_left_size, self.embedding_size],
            stddev=0.01 / math.sqrt(self.embedding_size), dtype=tf.float32))

        self.effect_embed_dict = tf.Variable(tf.truncated_normal(
            [self.vocab_right_size, self.embedding_size],
            stddev=0.01 / math.sqrt(self.embedding_size), dtype=tf.float32))

    def calculate_similar(self):
        """
        分别对原因的embedding和结果的embedding做归一化,然后再计算成绩
        获取test中的部分词汇的id ,然后分别计算这些词在cause空间和effect空间的相似性
        :return:
        """
        self.cause_word_id = tf.constant(self.dataLoader.c2e_test, dtype=tf.int32)
        cause_norm = tf.sqrt(tf.reduce_sum(tf.square(self.cause_embed_dict), 1, keep_dims=True))
        self.cause_normed_embed = self.cause_embed_dict / cause_norm
        c_test_embed = tf.nn.embedding_lookup(self.cause_normed_embed, self.cause_word_id)

        self.effect_word_id = tf.constant(self.dataLoader.e2c_test, dtype=tf.int32)
        effect_norm = tf.sqrt(tf.reduce_sum(tf.square(self.effect_embed_dict), 1, keep_dims=True))
        self.effect_normed_embed = self.effect_embed_dict / effect_norm
        e_test_embed = tf.nn.embedding_lookup(self.effect_normed_embed, self.effect_word_id)

        self.c2e_similar = tf.matmul(c_test_embed, tf.transpose(self.effect_normed_embed))
        self.e2c_similar = tf.matmul(e_test_embed, tf.transpose(self.cause_normed_embed))

    def show_similar(self):
        """
        获取原空间中的单词在对应的空间中排序最靠前的15个单词,然后打印观察相似性,是否能判断具有相似性
        :return:
        """
        sim = self.c2e_similar.eval()
        for i in range(len(self.dataLoader.c2e_test)):
            valid_word = self.dataLoader.vocab_left[self.dataLoader.c2e_test[i]]
            top_k = 15
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest effect words to %s:' % valid_word
            for k in range(top_k):
                close_word = self.dataLoader.vocab_right[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
        print('\n\n')
        sim = self.e2c_similar.eval()
        for i in range(len(self.dataLoader.e2c_test)):
            valid_word = self.dataLoader.vocab_right[self.dataLoader.e2c_test[i]]
            top_k = 15
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest cause words to %s:' % valid_word
            for k in range(top_k):
                close_word = self.dataLoader.vocab_left[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    def accuracy(self):
        """
        self.dataLoader.test_left, self.dataLoader.test_right 中保存的都是单词的id

        :return:
        """
        left_words, right_words = self.dataLoader.test_left, self.dataLoader.test_right
        labelled_pairs = self.dataLoader.test_pairs
        causeVec, effectVec = self.cause_embed_dict.eval(), self.effect_embed_dict.eval()

        def predict(left, right):
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
                    l_vec, r_vec = causeVec[l], effectVec[r]

                    d[' '.join([str(l), str(r)])] = l_vec.dot(r_vec.T)
            result = sorted(d.items(), key=lambda item: item[1], reverse=True)
            return result

        count, mrr = 0, []
        for i in range(len(left_words)):
            res = predict(left_words[i], right_words[i])
            s = ' '.join([str(labelled_pairs[i][0]), str(labelled_pairs[i][1])])
            if res[0][0] == s:
                count += 1
            for index, [k, _] in enumerate(res):
                if k == s:
                    mrr.append(1.0 / float(index + 1))

        return round(count / float(len(labelled_pairs)), 4), round(sum(mrr) / float(len(mrr)), 4)

    @staticmethod
    def generate_batches(data, batch_size, shuffle=True):
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

    def write_embedding(self, cause_output_path, effect_output_path, step):
        tail = '_' + step + '.txt'
        cause_path, effect_path = cause_output_path + tail, effect_output_path + tail
        with codecs.open(cause_path, 'w', 'utf-8') as fcause, codecs.open(effect_path, 'w', 'utf-8') as feffect:
            with self.sess.as_default():
                cause_embeddings, effect_embeddings = self.cause_embed_dict.eval(), self.effect_embed_dict.eval()
                if '<pad>' in self.dataLoader.vocab_left:
                    length = self.dataLoader.vocab_left_size - 1
                else:
                    length = self.dataLoader.vocab_left_size
                fcause.write(str(length) + ' ' + str(self.embedding_size) + '\n')
                if '<pad>' in self.dataLoader.vocab_right:
                    length = self.dataLoader.vocab_right_size - 1
                else:
                    length = self.dataLoader.vocab_right_size
                feffect.write(str(length) + ' ' + str(self.embedding_size) + '\n')
                for i in range(self.dataLoader.vocab_left_size):
                    s = self.dataLoader.vocab_left[i]
                    if s != '<pad>':
                        for j in range(self.embedding_size):
                            s += ' ' + str(cause_embeddings[i][j])
                        fcause.write(s + '\n')
                for i in range(self.dataLoader.vocab_right_size):
                    s = self.dataLoader.vocab_right[i]
                    if s != '<pad>':
                        for j in range(self.embedding_size):
                            s += ' ' + str(effect_embeddings[i][j])
                        feffect.write(s + '\n')
        print('word embedding are stored in {} and {} respectively!'.format(cause_path, effect_path))

    @staticmethod
    def mask_softmax(match_matrix, mask_matrix):
        """
        :param match_matrix: (batch, max_len, max_len)
        :param mask_matrix: (batch, max_len, max_len)
        :return:
        """
        match_matrix_masked = match_matrix * mask_matrix

        match_matrix_shifted_1 = mask_matrix * tf.exp(
            match_matrix_masked - tf.reduce_max(match_matrix_masked, axis=1, keep_dims=True))
        match_matrix_shifted_2 = mask_matrix * tf.exp(
            match_matrix_masked - tf.reduce_max(match_matrix_masked, axis=2, keep_dims=True))

        Z1 = tf.reduce_sum(match_matrix_shifted_1, axis=1, keep_dims=True)
        Z2 = tf.reduce_sum(match_matrix_shifted_2, axis=2, keep_dims=True)
        softmax_1 = match_matrix_shifted_1 / (Z1 + 1e-12)  # weight of left words
        softmax_2 = match_matrix_shifted_2 / (Z2 + 1e-12)  # weight of right words
        return softmax_1, softmax_2

    @staticmethod
    def make_attention(input_left_embed, input_right_embed):
        return tf.matmul(input_left_embed, tf.transpose(input_right_embed, perm=[0, 2, 1]))
