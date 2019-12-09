import numpy as np
import tensorflow as tf

labels = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
logits = np.array([[11., 8., 7.], [10., 14., 3.], [1., 2., 4.]])

y_pred = tf.math.sigmoid(logits)
prob_error1 = -labels * tf.math.log(y_pred) - (1 - labels) * tf.math.log(1 - y_pred)

labels1 = np.array([[0., 1., 0.], [1., 1., 0.], [0., 0., 1.]])  # 不一定只属于一个类别
logits1 = np.array([[1., 8., 7.], [10., 14., 3.], [1., 2., 4.]])
y_pred1 = tf.math.sigmoid(logits1)
prob_error11 = -labels1 * tf.math.log(y_pred1) - (1 - labels1) * tf.math.log(1 - y_pred1)

with tf.Session() as sess:
    print(sess.run(y_pred))
    print("1:")
    print(sess.run(prob_error1))
    print("2:")
    print(sess.run(prob_error11))
    print("3:")
    print(sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)))
    print("4:")
    print(sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels1, logits=logits1)))