# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter
import tensorflow as tf
import math
import codecs
import random


class BaseData(object):
    def __init__(self, sample_neg_randomly, num_samples=None):
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
        with codecs.open(data_file_path, 'r', 'utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                items = line.strip().split('----')
                # s1 = items[1].split(' ')
                # s2 = items[2].split(' ')
                s1 = items[0].split(' ')
                s2 = items[1].split(' ')
                if '' in s1 or '' in s2:
                    continue
                input_left.append(s1)
                input_right.append(s2)
        return input_left, input_right

    def load_dev(self, labeled_path):
        # cause_dev = ['侵扰', '事故', '爆炸', '台风', '冲突', '矛盾', '地震', '农药', '违章', '腐蚀',
        #              '感染', '病毒', '暴雨', '疲劳', '真菌', '贫血', '感冒', '战乱', '失调', '摩擦']
        # effect_dev = ['污染', '愤怒', '困境', '损失', '不适', '疾病', '失事', '悲剧', '危害', '感染',
        #               '故障', '死亡', '痛苦', '失败', '矛盾', '疲劳', '病害', '塌陷', '洪灾']
        cause_dev = ['accident', 'explosion', 'earthquake', 'virus', 'storm', 'war']
        effect_dev = ['pollution', 'death', 'loss', 'failure', 'disease', 'illness', 'flood']
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

    def load_labeled_data(self, data_path):
        # 58冬季放水不当引起的故障----冬季 放水 不当----故障##不当 故障
        lefts, rights, pairs = [], [], []
        with codecs.open(data_path, 'r', 'utf-8') as f:
            line = f.readline()
            while line:
                if line.strip() == '':
                    continue
                result = line.strip().split('##')
                pair = result[1].split(' ')
                # left = result[0].split('----')[1].split(' ')
                # right = result[0].split('----')[2].split(' ')
                left = result[0].split('----')[0].split(' ')
                right = result[0].split('----')[1].split(' ')
                try:
                    pair_left, pair_right = self.vocab_rev_left[pair[0]], self.vocab_rev_right[pair[1]]
                    temp_left, temp_right = [], []
                    for l in left:
                        try:
                            temp_left.append(self.vocab_rev_left[l])
                        except KeyError as e:
                            c = 0
                    for w in right:
                        try:
                            temp_right.append(self.vocab_rev_right[w])
                        except KeyError as e:
                            c = 0
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

        self.vocab_left.insert(0, '<pad>')
        self.vocab_right.insert(0, '<pad>')

        self.vocab_rev_left = {x: i for i, x in enumerate(self.vocab_left)}
        self.vocab_rev_right = {x: i for i, x in enumerate(self.vocab_right)}

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
                local_max_length = max(len(cause), len(effect))
                if max_length < local_max_length:
                    max_length = local_max_length
        return one_hot_left, one_hot_right, max_length

    def padding_data(self, one_hot_left, one_hot_right):
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
        L = len(self.x_left)-1
        for i in range(self.num_samples*batch_size):
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
            samples_left, samples_right = self.load_samples(data_path['pos_path'])
            self.build_vocab(samples_left, samples_right, min_count)
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
            self.x_left = pos_onehot_left+neg_onehot_left
            self.x_right = pos_onehot_right+neg_onehot_right
            self.x_target = pos_target+neg_target
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
        data = np.array(pos_data+neg_data)
        shuffle_indices = np.random.permutation(np.arange(len(data)))
        return data[shuffle_indices]

    def show_loss(self, feed_dict):
        _, global_step, loss_val = self.sess.run([self.train_op, self.global_steps, self.loss],
                                                 feed_dict=feed_dict)
        self.average_loss += loss_val
        current_step = tf.train.global_step(self.sess, self.global_steps)
        if current_step % 1000 == 0:
            self.average_loss /= 1000
            print('Average loss at step ', current_step, ': ', self.average_loss)
            self.average_loss = 0.0
        if current_step % 100000 == 0:
            self.show_similar()

    def init_embedding(self):
        self.cause_embed_dict = tf.Variable(tf.truncated_normal(
            [self.vocab_left_size, self.embedding_size],
            stddev=0.01 / math.sqrt(self.embedding_size), dtype=tf.float32))

        self.effect_embed_dict = tf.Variable(tf.truncated_normal(
            [self.vocab_right_size, self.embedding_size],
            stddev=0.01 / math.sqrt(self.embedding_size), dtype=tf.float32))

    def calculate_similar(self):
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
        left_words, right_words = self.dataLoader.test_left, self.dataLoader.test_right
        labelled_pairs = self.dataLoader.test_pairs
        causeVec, effectVec = self.cause_embed_dict.eval(), self.effect_embed_dict.eval()

        def predict(left, right):
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
                    mrr.append(1.0/float(index+1))

        return round(count / float(len(labelled_pairs)), 4), round(sum(mrr)/float(len(mrr)), 4)

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
        tail = '_'+step+'.txt'
        cause_path, effect_path = cause_output_path+tail, effect_output_path+tail
        with codecs.open(cause_path, 'w', 'utf-8') as fcause, codecs.open(effect_path, 'w', 'utf-8') as feffect:
            with self.sess.as_default():
                cause_embeddings, effect_embeddings = self.cause_embed_dict.eval(), self.effect_embed_dict.eval()
                if '<pad>' in self.dataLoader.vocab_left:
                    length = self.dataLoader.vocab_left_size-1
                else:
                    length = self.dataLoader.vocab_left_size
                fcause.write(str(length) + ' ' + str(self.embedding_size) + '\n')
                if '<pad>' in self.dataLoader.vocab_right:
                    length = self.dataLoader.vocab_right_size-1
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
        # 做归一化
        softmax_1 = match_matrix_shifted_1 / (Z1 + 1e-12)  # weight of left words
        softmax_2 = match_matrix_shifted_2 / (Z2 + 1e-12)  # weight of right words
        return softmax_1, softmax_2

    @staticmethod
    def make_attention(input_left_embed, input_right_embed):
        return tf.matmul(input_left_embed, tf.transpose(input_right_embed, perm=[0, 2, 1]))
