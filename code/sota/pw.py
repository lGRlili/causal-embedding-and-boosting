# -*- coding: utf-8 -*-
import sys
import numpy as np
import tensorflow as tf
from sota.base import BaseModel
from sota.base import BaseData
from time import time
# import os
# os.chdir("/home/xiezp/KE_Projects/CausalEmbedding/")


class PWData(BaseData):
    def __init__(self, sample_neg_randomly, num_samples=None):
        BaseData.__init__(self, sample_neg_randomly, num_samples)

    def sample_negative(self, batch_size):
        neg_left, neg_right = self.neg_sent(batch_size)
        neg_labels = []
        L = len(neg_left)
        for i in range(L):
            neg_labels.append(0.0)
        left_len, right_len = self.get_len(neg_left, neg_right)
        padded_left, padded_right = self.padding_data(neg_left, neg_right)
        # del neg_left, neg_right
        return list(zip(padded_left, padded_right, neg_labels, left_len, right_len))

    def generate_pos_data(self):
        pos_labels = []
        L = len(self.x_left)
        for i in range(L):
            pos_labels.append(1.0)
        padded_left, padded_right = self.padding_data(self.x_left, self.x_right)
        left_len, right_len = self.get_len(self.x_left, self.x_right)
        return list(zip(padded_left, padded_right, pos_labels, left_len, right_len))
        # return list(zip(padded_left, padded_right, pos_mask))

    def generate_mixed_data(self):
        left_len, right_len = self.get_len(self.x_left, self.x_right)
        padded_left, padded_right = self.padding_data(self.x_left, self.x_right)
        return list(zip(padded_left, padded_right, self.x_target, left_len, right_len))


class PairWise(BaseModel):
    def __init__(self, embedding_size, batch_size, num_epochs, num_samples, learning_rate, data_loader):
        BaseModel.__init__(self, embedding_size, batch_size, num_epochs, num_samples, learning_rate, data_loader)

    def train_stage(self, cause_output_path, effect_output_path):
        print('model: Pairwise started!\n')
        with self.sess.as_default():
            if self.sample_neg_randomly:
                data = self.dataLoader.generate_pos_data()
            else:
                data = self.dataLoader.generate_mixed_data()
            for current_epoch in range(self.num_epochs):
                start_time = time()
                print('current epoch: {} started.'.format(current_epoch+1))
                train_batches = self.generate_batches(data, self.batch_size)
                for per_batch in train_batches:
                    if self.sample_neg_randomly:
                        neg_data = self.dataLoader.sample_negative(self.batch_size)
                        # mixed_data = self.shuffle_pos_neg(pos_batch, neg_data)
                        new_data = np.concatenate([per_batch, np.array(neg_data)], 0)
                        input_left, input_right, input_labels, left_len, right_len = zip(*new_data)
                    else:
                        input_left, input_right, input_labels, left_len, right_len = zip(*per_batch)
                    feed_dict = {
                        self.input_left: np.array(input_left),
                        self.input_right: np.array(input_right),
                        self.targets: input_labels,
                        self.left_len: left_len,
                        self.right_len: right_len
                    }
                    self.show_loss(feed_dict)
                acc = self.accuracy()
                print('accuracy at epoch:{} is {}.'.format(current_epoch+1, acc))
                if current_epoch % 10 == 0 and current_epoch != 0:
                    self.write_embedding(cause_output_path, effect_output_path, str(current_epoch+1))
                end_time = time()
                print('epoch: {} uses {} mins.\n'.format(current_epoch+1, float(end_time-start_time)/60))

    def construct_graph(self):
        with self.graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            self.sess = tf.Session(config=session_conf)
            self.init_embedding()
            self.input_left = tf.placeholder(tf.int32, [None, self.max_len])
            self.input_right = tf.placeholder(tf.int32, [None, self.max_len])
            self.left_len = tf.placeholder(tf.int32, [None, ])
            self.right_len = tf.placeholder(tf.int32, [None, ])
            self.targets = tf.placeholder(tf.float32, [None, ])

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

            # self.input_left = tf.placeholder(tf.int32, [None, self.max_len])
            # self.input_right = tf.placeholder(tf.int32, [None, self.max_len])
            # self.left_len = tf.placeholder(tf.int32, [None, ])
            # self.right_len = tf.placeholder(tf.int32, [None, ])
            # self.targets = tf.placeholder(tf.float32, [None, ])
            # self.alpha = tf.placeholder(tf.float32, name='alpha')
            # self.gamma = tf.placeholder(tf.float32, name='gamma')
            # self.input_left_embed = tf.tutorial.embedding_lookup(self.cause_embed_dict, self.input_left)
            # self.input_right_embed = tf.tutorial.embedding_lookup(self.effect_embed_dict, self.input_right)
            # left_mask = tf.sequence_mask(self.left_len, self.max_len, dtype=tf.float32)
            # right_mask = tf.sequence_mask(self.right_len, self.max_len, dtype=tf.float32)
            # mask_matrix = tf.matmul(tf.expand_dims(left_mask, 2), tf.expand_dims(right_mask, 1))
            # logits = self.make_attention(self.input_left_embed, self.input_right_embed)
            # probs = tf.clip_by_value(tf.sigmoid(logits), 1e-5, 1 - 1e-5)
            #
            # pos_fl = -self.alpha*tf.pow(1 - probs, self.gamma) * tf.log(probs)
            # pos_fl = tf.reduce_sum(tf.reduce_sum(pos_fl*mask_matrix, axis=1), axis=1)*self.targets
            #
            # neg_fl = (self.alpha-1)*tf.pow(probs, self.gamma) * tf.log(1 - probs)
            # neg_fl = tf.reduce_sum(tf.reduce_sum(neg_fl*mask_matrix, axis=1), axis=1)*(1-self.targets)
            #
            # self.loss = tf.reduce_sum([pos_fl, neg_fl])

            # self.calculate_similar()
            self.global_steps = tf.Variable(0, trainable=False)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss, global_step=self.global_steps)
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)
