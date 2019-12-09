# -*- coding: utf-8 -*-
import sys
import os
from time import time
import numpy as np
import tensorflow as tf
from reference.base import BaseModel
from reference.base import BaseData
# import os
# os.chdir("/home/xiezp/KE_Projects/CausalEmbedding/")
from sklearn.utils import shuffle
import random


class MaxData(BaseData):
    def __init__(self, sample_neg_randomly, num_samples=None):
        BaseData.__init__(self, sample_neg_randomly, num_samples)

    def neg_sent_fixed_one(self, batch_size, per_batch):
        """
        get negative samples from sentence
        """
        neg_left, neg_right = [], []
        L = len(self.x_left) - 1
        for assembl_list in per_batch:
            padded_left, padded_right, x_target, left_len, right_len = assembl_list
            # print(padded_left, padded_right)
            padded_left = padded_left[:left_len]
            padded_right = padded_right[:right_len]
            # print('---')
            # print(padded_left, padded_right)
            for i in range(self.num_samples):
                k = random.randint(0, L)
                neg_left.append(self.x_left[k])
                neg_right.append(padded_right)

                j = random.randint(0, L)
                neg_left.append(padded_left)
                neg_right.append(self.x_right[j])

        return neg_left, neg_right

    def sample_negative(self, batch_size, per_batch):
        # 进行负采样
        neg_left, neg_right = self.neg_sent_fixed_one(batch_size, per_batch)
        neg_labels = []
        L = len(neg_left)
        for i in range(L):
            neg_labels.append(0.0)
        left_len, right_len = self.get_len(neg_left, neg_right)
        padded_left, padded_right = self.padding_data(neg_left, neg_right)
        # del neg_left, neg_right
        return list(zip(padded_left, padded_right, neg_labels, left_len, right_len))

    def generate_pos_data(self):
        # 生成正样本
        pos_labels = []
        L = len(self.x_left)
        for i in range(L):
            pos_labels.append(1.0)
        padded_left, padded_right = self.padding_data(self.x_left, self.x_right)
        left_len, right_len = self.get_len(self.x_left, self.x_right)
        return list(zip(padded_left, padded_right, pos_labels, left_len, right_len))

    def generate_mixed_data(self):
        left_len, right_len = self.get_len(self.x_left, self.x_right)
        padded_left, padded_right = self.padding_data(self.x_left, self.x_right)
        return list(zip(padded_left, padded_right, self.x_target, left_len, right_len))


class MaxModel(BaseModel):
    def __init__(self, embedding_size, batch_size, num_epochs, num_samples, learning_rate, data_loader):
        BaseModel.__init__(self, embedding_size, batch_size, num_epochs, num_samples, learning_rate, data_loader)
        self.alpha, self.gamma = None, None

    def train_stage(self, cause_output_path, effect_output_path):

        with self.sess.as_default():
            if self.sample_neg_randomly:
                data = self.dataLoader.generate_pos_data()
            else:
                data = self.dataLoader.generate_mixed_data()

            for current_epoch in range(self.num_epochs):
                print('current epoch: {} started.'.format(current_epoch + 1))
                start_time = time()
                train_batches = self.generate_batches(data, self.batch_size)
                for per_batch in train_batches:
                    if self.sample_neg_randomly:
                        neg_data = self.dataLoader.sample_negative(self.batch_size, per_batch)
                        new_data = np.concatenate([per_batch, np.array(neg_data)], 0)
                        new_data = shuffle(new_data)
                        # mixed_data = self.shuffle_pos_neg(pos_batch, neg_data)
                        input_left, input_right, input_labels, left_len, right_len = zip(*new_data)
                    else:
                        input_left, input_right, input_labels, left_len, right_len = zip(*per_batch)
                    feed_dict = {
                        self.input_left: np.array(input_left),
                        self.input_right: np.array(input_right),
                        self.targets: input_labels,
                        self.left_len: left_len,
                        self.right_len: right_len,
                        self.alpha: 0.8,
                        self.gamma: 2.0
                    }
                    self.show_loss(feed_dict)
                    # print(self.sess.run([self._probs], feed_dict=feed_dict))
                    # print(self.sess.run([self.probs_reshape], feed_dict=feed_dict))
                # acc = self.accuracy()
                # print('accuracy at epoch:{} is {}.'.format(current_epoch+1, acc))
                if current_epoch % 1 == 0 and current_epoch != 0:
                    self.write_embedding(cause_output_path, effect_output_path, str(current_epoch + 1))
                end_time = time()
                print('epoch: {} uses {} mins.\n'.format(current_epoch + 1, float(end_time - start_time) / 60))

    def construct_graph_max(self):
        print('model: Max started!\n')
        with self.graph.as_default():
            # 用于对网络进行设定
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            self.sess = tf.Session(config=session_conf)

            self.input_left = tf.placeholder(tf.int32, [None, self.max_len])
            self.input_right = tf.placeholder(tf.int32, [None, self.max_len])
            self.left_len = tf.placeholder(tf.float32, [None, ])
            self.right_len = tf.placeholder(tf.float32, [None, ])
            self.targets = tf.placeholder(tf.float32, [None, ])
            self.alpha = tf.placeholder(tf.float32, name='alpha')
            self.gamma = tf.placeholder(tf.float32, name='gamma')
            # 初始化原因词embedding 和结果词embedding
            self.init_embedding()
            self.input_left_embed = tf.nn.embedding_lookup(self.cause_embed_dict, self.input_left)
            self.input_right_embed = tf.nn.embedding_lookup(self.effect_embed_dict, self.input_right)

            """
            focal loss:
                for positive samples, loss is -alpha*((1-p)**gamma)*log(p)

                for negative samples, loss is -(1-alpha)*(p**gamma)*log(1-p)

            """
            left_mask = tf.sequence_mask(self.left_len, self.max_len, dtype=tf.float32)
            right_mask = tf.sequence_mask(self.right_len, self.max_len, dtype=tf.float32)
            mask_matrix = tf.matmul(tf.expand_dims(left_mask, 2), tf.expand_dims(right_mask, 1))

            logits = self.make_attention(self.input_left_embed, self.input_right_embed)

            print(self.input_left_embed, self.input_right_embed)
            print(mask_matrix)
            print(logits)
            _probs = tf.sigmoid(logits)
            # 将其中没有单词的部分全部打掩码,然后,选择出,最大的单词对
            maxed_probs = tf.reduce_max(tf.reduce_max(_probs * mask_matrix, axis=1), axis=1)
            # 对结果做一个约束
            pos_probs = tf.clip_by_value(maxed_probs, 1e-5, 1.0 - 1e-5)
            pos_fl = -self.alpha * tf.pow(1 - pos_probs, self.gamma) * tf.log(pos_probs) * self.targets
            # 针对反向的情况,需要计算全部的存在组合的loss
            neg_probs = tf.clip_by_value(_probs, 1e-5, 1.0 - 1e-5)
            _3d_neg_fl = (self.alpha - 1) * tf.pow(neg_probs, self.gamma) * tf.log(1 - neg_probs)
            neg_fl = tf.reduce_sum(tf.reduce_sum(_3d_neg_fl * mask_matrix, axis=1), axis=1) * (1.0 - self.targets)

            self.loss = tf.reduce_sum([pos_fl, neg_fl])

            # self.loss = tf.reduce_sum(tf.maximum(0.0, 1.0-pos_logits+neg_logits))

            self.calculate_similar()
            self.global_steps = tf.Variable(0, trainable=False)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss, global_step=self.global_steps)
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)

    def construct_graph_top_k(self, top_k_para=3):
        print('model: Top k started!\n')
        with self.graph.as_default():
            # 用于对网络进行设定
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            self.sess = tf.Session(config=session_conf)

            self.input_left = tf.placeholder(tf.int32, [None, self.max_len])
            self.input_right = tf.placeholder(tf.int32, [None, self.max_len])
            self.left_len = tf.placeholder(tf.float32, [None, ])
            self.right_len = tf.placeholder(tf.float32, [None, ])
            self.targets = tf.placeholder(tf.float32, [None, ])
            self.alpha = tf.placeholder(tf.float32, name='alpha')
            self.gamma = tf.placeholder(tf.float32, name='gamma')
            # 初始化原因词embedding 和结果词embedding
            self.init_embedding()
            self.input_left_embed = tf.nn.embedding_lookup(self.cause_embed_dict, self.input_left)
            self.input_right_embed = tf.nn.embedding_lookup(self.effect_embed_dict, self.input_right)
            print(self.input_left_embed, self.input_right_embed)
            """
            focal loss:
                for positive samples, loss is -alpha*((1-p)**gamma)*log(p)

                for negative samples, loss is -(1-alpha)*(p**gamma)*log(1-p)

            """
            # 构建掩码矩阵,确保其他位置不设置掩码
            left_mask = tf.sequence_mask(self.left_len, self.max_len, dtype=tf.float32)
            right_mask = tf.sequence_mask(self.right_len, self.max_len, dtype=tf.float32)
            mask_matrix = tf.matmul(tf.expand_dims(left_mask, 2), tf.expand_dims(right_mask, 1))

            logits = self.make_attention(self.input_left_embed, self.input_right_embed)
            _probs = tf.sigmoid(logits)
            # 将其中没有单词的部分全部打掩码,然后,选择出,最大的单词对
            # maxed_probs = tf.reduce_max(tf.reduce_max(_probs*mask_matrix, axis=1), axis=1)
            probs_reshape = tf.reshape(_probs * mask_matrix, [-1, self.max_len * self.max_len])
            self._probs = tf.sigmoid(logits)
            self.probs_reshape = tf.reshape(_probs * mask_matrix, [-1, self.max_len * self.max_len])
            top_k = tf.nn.top_k(probs_reshape, k=top_k_para)[0]

            pos_probs = tf.clip_by_value(top_k, 1e-5, 1.0 - 1e-5)
            top_k_pos_fl = -self.alpha * tf.pow(1 - pos_probs, self.gamma) * tf.log(pos_probs)
            pos_fl = tf.reduce_sum(top_k_pos_fl, axis=1) * self.targets
            # 针对反向的情况,需要计算全部的存在组合的loss
            neg_probs = tf.clip_by_value(_probs, 1e-5, 1.0 - 1e-5)
            _3d_neg_fl = (self.alpha - 1) * tf.pow(neg_probs, self.gamma) * tf.log(1 - neg_probs)
            neg_fl = tf.reduce_sum(tf.reduce_sum(_3d_neg_fl * mask_matrix, axis=1), axis=1) * (1.0 - self.targets)

            self.loss = tf.reduce_sum([pos_fl, neg_fl])
            # print(_probs, probs_reshape, top_k)
            # print(_probs.shape, probs_reshape.shape, top_k.shape)
            # print(pos_probs, top_k_pos_fl, pos_fl)
            # print(pos_probs.shape, top_k_pos_fl.shape, pos_fl.shape)
            # print(neg_probs, _3d_neg_fl, neg_fl)
            # print(neg_probs.shape, _3d_neg_fl.shape, neg_fl.shape)

            # self.loss = tf.reduce_sum(tf.maximum(0.0, 1.0-pos_logits+neg_logits))

            self.calculate_similar()
            self.global_steps = tf.Variable(0, trainable=False)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss, global_step=self.global_steps)
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)

    def construct_graph_pair_wise_match(self):
        print('model: pair wise match started!\n')
        with self.graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            self.sess = tf.Session(config=session_conf)

            self.input_left = tf.placeholder(tf.int32, [None, self.max_len])
            self.input_right = tf.placeholder(tf.int32, [None, self.max_len])
            self.left_len = tf.placeholder(tf.int32, [None, ])
            self.right_len = tf.placeholder(tf.int32, [None, ])
            self.targets = tf.placeholder(tf.float32, [None, ])
            self.alpha = tf.placeholder(tf.float32, name='alpha')
            self.gamma = tf.placeholder(tf.float32, name='gamma')
            self.init_embedding()
            self.input_left_embed = tf.nn.embedding_lookup(self.cause_embed_dict, self.input_left)
            self.input_right_embed = tf.nn.embedding_lookup(self.effect_embed_dict, self.input_right)
            left_mask = tf.sequence_mask(self.left_len, self.max_len, dtype=tf.float32)
            right_mask = tf.sequence_mask(self.right_len, self.max_len, dtype=tf.float32)
            mask_matrix = tf.matmul(tf.expand_dims(left_mask, 2), tf.expand_dims(right_mask, 1))

            logits = self.make_attention(self.input_left_embed, self.input_right_embed)
            probs = tf.clip_by_value(tf.sigmoid(logits), 1e-5, 1 - 1e-5)
            target = tf.expand_dims(tf.expand_dims(self.targets, -1), -1)
            labels = tf.tile(target, [1, self.max_len, self.max_len])

            unmasked_loss = -labels*tf.log(probs) - (1 - labels) * tf.log(1 - probs)
            self.loss = tf.reduce_mean(unmasked_loss * mask_matrix)

            # pos_fl = -self.alpha*tf.pow(1 - probs, self.gamma) * tf.log(probs)
            # pos_fl = tf.reduce_sum(tf.reduce_sum(pos_fl*mask_matrix, axis=1), axis=1)*self.targets
            #
            # neg_fl = (self.alpha-1)*tf.pow(probs, self.gamma) * tf.log(1 - probs)
            # neg_fl = tf.reduce_sum(tf.reduce_sum(neg_fl*mask_matrix, axis=1), axis=1)*(1-self.targets)
            #
            # self.loss = tf.reduce_sum([pos_fl, neg_fl])

            self.calculate_similar()
            self.global_steps = tf.Variable(0, trainable=False)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss, global_step=self.global_steps)
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)

    def construct_graph_pair_wise_match_old(self):

        with self.graph.as_default():
            # 用于对网络进行设定
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            self.sess = tf.Session(config=session_conf)

            self.input_left = tf.placeholder(tf.int32, [None, self.max_len])
            self.input_right = tf.placeholder(tf.int32, [None, self.max_len])
            self.left_len = tf.placeholder(tf.float32, [None, ])
            self.right_len = tf.placeholder(tf.float32, [None, ])
            self.targets = tf.placeholder(tf.float32, [None, ])
            self.alpha = tf.placeholder(tf.float32, name='alpha')
            self.gamma = tf.placeholder(tf.float32, name='gamma')
            # 初始化原因词embedding 和结果词embedding
            self.init_embedding()
            self.input_left_embed = tf.nn.embedding_lookup(self.cause_embed_dict, self.input_left)
            self.input_right_embed = tf.nn.embedding_lookup(self.effect_embed_dict, self.input_right)
            print(self.input_left_embed, self.input_right_embed)
            """
            focal loss:
                for positive samples, loss is -alpha*((1-p)**gamma)*log(p)

                for negative samples, loss is -(1-alpha)*(p**gamma)*log(1-p)

            """
            # 构建掩码矩阵,确保其他位置不设置掩码
            left_mask = tf.sequence_mask(self.left_len, self.max_len, dtype=tf.float32)
            right_mask = tf.sequence_mask(self.right_len, self.max_len, dtype=tf.float32)
            mask_matrix = tf.matmul(tf.expand_dims(left_mask, 2), tf.expand_dims(right_mask, 1))

            logits = self.make_attention(self.input_left_embed, self.input_right_embed)
            _probs = tf.sigmoid(logits)
            # 将其中没有单词的部分全部打掩码,然后,选择出,最大的单词对
            # maxed_probs = tf.reduce_max(tf.reduce_max(_probs*mask_matrix, axis=1), axis=1)

            pos_probs = tf.clip_by_value(_probs, 1e-5, 1.0 - 1e-5)
            _pos_fl = -self.alpha * tf.pow(1 - pos_probs, self.gamma) * tf.log(pos_probs)
            pos_fl = tf.reduce_sum(tf.reduce_sum(_pos_fl * mask_matrix, axis=1), axis=1) * self.targets
            # 针对反向的情况,需要计算全部的存在组合的loss
            neg_probs = tf.clip_by_value(_probs, 1e-5, 1.0 - 1e-5)
            _3d_neg_fl = (self.alpha - 1) * tf.pow(neg_probs, self.gamma) * tf.log(1 - neg_probs)
            neg_fl = tf.reduce_sum(tf.reduce_sum(_3d_neg_fl * mask_matrix, axis=1), axis=1) * (1.0 - self.targets)

            self.loss = tf.reduce_sum([pos_fl, neg_fl])

            # self.loss = tf.reduce_sum(tf.maximum(0.0, 1.0-pos_logits+neg_logits))

            self.calculate_similar()
            self.global_steps = tf.Variable(0, trainable=False)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss, global_step=self.global_steps)
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)

    def construct_graph_attentive_match(self):
        print('model: attentive match started!\n')
        with self.graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            self.sess = tf.Session(config=session_conf)

            self.input_left = tf.placeholder(tf.int32, [None, self.max_len])
            self.input_right = tf.placeholder(tf.int32, [None, self.max_len])
            self.left_len = tf.placeholder(tf.float32, [None, ])
            self.right_len = tf.placeholder(tf.float32, [None, ])
            self.targets = tf.placeholder(tf.float32, [None, ])
            self.alpha = tf.placeholder(tf.float32, name='alpha')
            self.gamma = tf.placeholder(tf.float32, name='gamma')
            self.init_embedding()

            self.input_left_embed = tf.nn.embedding_lookup(self.cause_embed_dict, self.input_left)
            self.input_right_embed = tf.nn.embedding_lookup(self.effect_embed_dict, self.input_right)
            left_mask = tf.sequence_mask(self.left_len, self.max_len, dtype=tf.float32)
            right_mask = tf.sequence_mask(self.right_len, self.max_len, dtype=tf.float32)
            mask_matrix = tf.matmul(tf.expand_dims(left_mask, 2), tf.expand_dims(right_mask, 1))

            logits = self.make_attention(self.input_left_embed, self.input_right_embed)

            # 按行按列做softmax
            softmax_1, softmax_2 = self.mask_softmax(logits, mask_matrix)
            left_attentive = tf.matmul(tf.transpose(softmax_1, [0, 2, 1]), self.input_left_embed)  # (batch, r, dims)
            right_attentive = tf.matmul(softmax_2, self.input_right_embed)  # (batch, l, dims)
            right_interaction = tf.reduce_sum(left_attentive * self.input_right_embed, axis=2)  # (batch, r)
            left_interaction = tf.reduce_sum(right_attentive * self.input_left_embed, axis=2)  # (batch, l)

            right_probs = tf.clip_by_value(
                tf.reduce_max(tf.sigmoid(right_interaction) * right_mask, 1), 1e-5, 1.0 - 1e-5
            )  # (batch,)
            left_probs = tf.clip_by_value(
                tf.reduce_max(tf.sigmoid(left_interaction) * left_mask, 1), 1e-5, 1.0 - 1e-5
            )  # (batch,)

            left_pos_fl = tf.reduce_sum(
                -self.alpha * tf.pow(1 - left_probs, self.gamma) * tf.log(left_probs)*self.targets
            )
            right_pos_fl = tf.reduce_sum(
                -self.alpha * tf.pow(1 - right_probs, self.gamma) * tf.log(right_probs)*self.targets
            )

            _pro = tf.clip_by_value(tf.sigmoid(logits), 1e-5, 1.0 - 1e-5)
            _3d_focal = (self.alpha - 1) * tf.pow(_pro, self.gamma) * tf.log(1 - _pro) * mask_matrix
            neg_fl = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(_3d_focal, axis=1), axis=1)*(1.0-self.targets))
            # neg_fl = tf.reduce_sum(
            #     ((self.alpha - 1) * tf.pow(_pro, self.gamma) * tf.log(1 - _pro) * mask_matrix)*(1.0-self.targets)
            # )

            self.loss = tf.reduce_sum([left_pos_fl, right_pos_fl, neg_fl])
            # self.loss = tf.reduce_sum(tf.maximum(0.0, 1.0-pos_logits+neg_logits))

            # self.calculate_similar()
            self.global_steps = tf.Variable(0, trainable=False)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss, global_step=self.global_steps)
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)

"""
学习因果词向量:
第一步,获取正样本,构建词袋,
第二步:随机采样获得负样本
第三步:构建因果向量的模型,将因果向量做内积,学习目标是,正向结果的概率尽可能的大,非因果的概率尽可能的小


当前问题:学习得到的词向量,不同的词语,对应的因/果向量的排序都是相同的几个单词
"""
