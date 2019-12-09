import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(3)
x1_label0 = np.random.normal(1, 1, (1000, 1))
x2_label0 = np.random.normal(1, 1, (1000, 1))
x1_label1 = np.random.normal(5, 1, (1000, 1))
x2_label1 = np.random.normal(4, 1, (1000, 1))
x1_label2 = np.random.normal(8, 1, (1000, 1))
x2_label2 = np.random.normal(0, 1, (1000, 1))
x1_label0 = x1_label0.reshape(100, 10)
x2_label0 = x2_label0.reshape(100, 10)
x1_label1 = x1_label1.reshape(100, 10)
x2_label1 = x2_label1.reshape(100, 10)
x1_label2 = x1_label2.reshape(100, 10)
x2_label2 = x2_label2.reshape(100, 10)

# 并入行
xs_label0 = np.hstack((x1_label0, x2_label0))
xs_label1 = np.hstack((x1_label1, x2_label1))
xs_label2 = np.hstack((x1_label2, x2_label2))
# 并入列
xs = np.vstack((xs_label0, xs_label1, xs_label2))
labels = np.matrix([[1., 0., 0.]] * len(x1_label0) + [[0., 1., 0.]] * len(x1_label1) + [[0.,
                    0., 1.]] * len(x1_label2))
arr = np.arange(xs.shape[0])
np.random.shuffle(arr)
xs = xs[arr, :]

labels = labels[arr, :]
test_x1_label0 = np.random.normal(1, 1, (100, 1))
test_x2_label0 = np.random.normal(1, 1, (100, 1))
test_x1_label1 = np.random.normal(5, 1, (100, 1))
test_x2_label1 = np.random.normal(4, 1, (100, 1))
test_x1_label2 = np.random.normal(8, 1, (100, 1))
test_x2_label2 = np.random.normal(0, 1, (100, 1))
test_x1_label0 = test_x1_label0.reshape(10, 10)
test_x2_label0 = test_x2_label0.reshape(10, 10)
test_x1_label1 = test_x1_label1.reshape(10, 10)
test_x2_label1 = test_x2_label1.reshape(10, 10)
test_x1_label2 = test_x1_label2.reshape(10, 10)
test_x2_label2 = test_x2_label2.reshape(10, 10)
test_xs_label0 = np.hstack((test_x1_label0, test_x2_label0))
test_xs_label1 = np.hstack((test_x1_label1, test_x2_label1))
test_xs_label2 = np.hstack((test_x1_label2, test_x2_label2))
test_xs = np.vstack((test_xs_label0, test_xs_label1, test_xs_label2))
test_labels = np.matrix([[1., 0., 0.]] * 10 + [[0., 1., 0.]] * 10 + [[0., 0., 1.]] * 10)
train_size, num_features = xs.shape

learning_rate = 0.01
training_epochs = 1000
num_labels = 3
batch_size = 100
X = tf.placeholder("float", shape=[None, num_features])
Y = tf.placeholder("float", shape=[None, num_labels])
# W = tf.Variable(tf.zeros([num_features, num_labels]))
# b = tf.Variable(tf.zeros([num_labels]))

W = tf.Variable(initial_value=tf.random_normal([num_features, num_labels]), name="weight1")  # 2 x 1
b = tf.Variable(initial_value=tf.random_normal([num_labels]), name="bias1")  # 1

logist = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logist, labels=Y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predict = tf.argmax(tf.nn.softmax(logist), axis=1, name="predictions")
tf.cast(predict, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), predict), "float"), name="accuracy")
# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.cast(predict, dtype=tf.float32)), "float"), name="accuracy")
# y_model = tf.nn.softmax(tf.matmul(X, W) + b)
# cost = -tf.reduce_sum(Y * tf.log(y_model))
# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for step in range(training_epochs * train_size // batch_size):
        offset = (step * batch_size) % train_size
        batch_xs = xs[offset:(offset + batch_size), :]
        batch_labels = labels[offset:(offset + batch_size)]
        err, _ = sess.run([cost, train_op], feed_dict={X:batch_xs, Y:batch_labels})
        print(step, err)
    W_val = sess.run(W)
    print('w', W_val)
    b_val = sess.run(b)
    print('b', b_val)
    print("accuracy", accuracy.eval(feed_dict={X:test_xs, Y:test_labels}))

# w [[-2.087371    0.47534525  1.9333048 ]
#  [-1.018466    0.96481436 -1.6764901 ]]
# b [ 9.649531  -2.6709933 -8.362488 ]

# w [[-7.28930533e-01 -5.71702719e-01 -8.33698153e-01]
#  [ 7.24015236e-01  6.65346026e-01  6.99907422e-01]
#  [-4.13682938e-01  1.93497073e-03  2.88499415e-01]
#  [-4.28662866e-01 -2.93151498e-01  1.46056920e-01]
#  [-8.26939464e-01 -5.44626653e-01  6.94389164e-01]
#  [ 1.47338676e+00  1.48113108e+00  1.23548508e+00]
#  [-4.94176030e-01 -1.15365095e-01  1.25624865e-01]
#  [-2.55301416e-01 -3.04328837e-02  8.50321446e-03]
#  [ 1.00161284e-02  4.41384196e-01  1.35330260e-01]
#  [-3.89877766e-01 -2.24063516e-01  7.92081356e-02]
#  [-4.46118154e-02  2.02607989e-01 -1.74157631e+00]
#  [ 1.64054191e+00  1.45049202e+00 -3.43394697e-01]
#  [-6.93511739e-02  6.47712648e-02 -1.51901498e-01]
#  [ 5.59840024e-01  7.08429635e-01 -2.30513453e+00]
#  [ 1.39872038e+00  1.81583452e+00 -3.47408682e-01]
#  [ 9.31227565e-01  1.08921003e+00  1.31975198e+00]
#  [ 1.10349096e-02  9.42238495e-02  3.41288269e-01]
#  [ 1.13120034e-01  5.69436610e-01 -1.59392595e+00]
#  [-9.75227952e-01 -6.55715585e-01 -8.37434351e-01]
#  [ 1.52496025e-01  2.59026557e-01 -7.99061954e-01]]
# b [ 4.5173645 -5.714694  -1.7029107]