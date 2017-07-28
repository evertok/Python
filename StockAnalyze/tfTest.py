'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

91%的概率

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
'''
'''
97%成功率

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# 第二层卷积
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())
for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x: batch[0],
                       y_: batch[1],
                       keep_prob: 1.0})
        print("step %d training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %g" % accuracy.eval(
    feed_dict={x: mnist.test.images,
               y_: mnist.test.labels,
               keep_prob: 1.0}))
'''               
'''
偏微分方程
import tensorflow as tf
import numpy as np
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display


def DisplayArray(a, fmt="jpeg", rng=[0, 1]):
    a = (a - rng[0]) / float(rng[1] - rng[0]) * 255
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def make_kernel(a):
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1, 1])
    return tf.constant(a, dtype=1)


def simple_conv(x, k):
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding="SAME")
    return y[0, :, :, 0]


def laplace(x):
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                             [1.0, -6, 1.0],
                             [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)

sess = tf.InteractiveSession()
N = 500
u_init = np.zeros([N, N], dtype="float32")
ut_init = np.zeros([N, N], dtype="float32")
for n in range(40):
    a, b = np.random.randint(0, N, 2)
    u_init[a, b] = np.random.uniform()
DisplayArray(u_init, rng=[-0.1, 0.1])
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())
U = tf.Variable(u_init)
Ut = tf.Variable(ut_init)
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

step = tf.group(U.assign(U_), Ut.assign(Ut_))

tf.global_variables_initializer().run()
for i in range(1000):
    step.run({eps: 0.03, damping: 0.04})
    if i % 50 == 0:
        clear_output()
        DisplayArray(U.eval(), rng=[-0.1, 0.1])
'''


'''
股票预测
'''
# -*- coding: utf-8 -*-
# @DATE    : 2017/2/14 17:50
# @Author  : 
# @File    : stock_predict.py

import os
import sys
import datetime

import tensorflow as tf
import pandas as pd
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt



class StockRNN(object):
    def __init__(self, seq_size=12, input_dims=1, hidden_layer_size=12, stock_id="002708", days=365, log_dir="stock_model/"):
        self.seq_size = seq_size
        self.input_dims = input_dims
        self.hidden_layer_size = hidden_layer_size
        self.stock_id = stock_id
        self.days = days
        self.data = self._read_stock_data()["close"].astype(float).values
        self.log_dir = log_dir

    def _read_stock_data(self):
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() + datetime.timedelta(-200)).strftime("%Y-%m-%d")
        # print(start_date, end_date)

        his_data = ts.get_hist_data(self.stock_id, start_date, end_date)
        stock_pd = pd.DataFrame(his_data)
        stock_pd["close"] = stock_pd["close"].astype(float)
        stock_pd = stock_pd[::-1]
        stock_pd.reset_index(inplace=True)
        return stock_pd[["date", "close"]]

    def _create_placeholders(self):
        with tf.name_scope(name="data"):
            self.X = tf.placeholder(tf.float32, [None, self.seq_size, self.input_dims], name="x_input")
            self.Y = tf.placeholder(tf.float32, [None, self.seq_size], name="y_input")

    def init_network(self, log_dir):
        print("Init RNN network")
        self.log_dir = log_dir
        self.sess = tf.Session()
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restore")

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(self.sess, self.coord)

    def _create_rnn(self):
        W = tf.Variable(tf.random_normal([self.hidden_layer_size, 1], name="W"))
        b = tf.Variable(tf.random_normal([1], name="b"))
        with tf.variable_scope("cell_d"):
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_layer_size)
        with tf.variable_scope("rnn_d"):
            outputs, states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)

        W_repeated = tf.tile(tf.expand_dims(W, 0), [tf.shape(self.X)[0], 1, 1])
        out = tf.matmul(outputs, W_repeated) + b
        out = tf.squeeze(out)
        return out

    def _data_prepare(self):
        self.train_x = []
        self.train_y = []
        # data
        data = np.log1p(self.data)
        for i in range(len(data) - self.seq_size - 1):
            self.train_x.append(np.expand_dims(data[i: i + self.seq_size], axis=1).tolist())
            self.train_y.append(data[i + 1: i + self.seq_size + 1].tolist())

    def train_pred_rnn(self):

        self._create_placeholders()

        y_hat = self._create_rnn()
        self._data_prepare()
        loss = tf.reduce_mean(tf.square(y_hat - self.Y))
        train_optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        feed_dict = {self.X: self.train_x, self.Y: self.train_y}

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(1, 20001):
                _, loss_ = sess.run([train_optim, loss], feed_dict=feed_dict)
                if step % 100 == 0:
                    print("{} {}".format(step, loss_))
            saver.save(sess, self.log_dir + "model.ckpt")

            # prediction
            prev_seq = self.train_x[-1]
            predict = []
            for i in range(20):
                next_seq = sess.run(y_hat, feed_dict={self.X: [prev_seq]})
                predict.append(next_seq[-1])
                prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
            predict = np.exp(predict) - 1
            print(predict)
            self.pred = predict

    def visualize(self):
        pred = self.pred
        plt.figure()
        plt.legend(prop={'family': 'SimHei', 'size': 15})
        plt.plot(list(range(len(self.data))), self.data, color='b')
        plt.plot(list(range(len(self.data), len(self.data) + len(pred))), pred, color='r')
        plt.title(u"{}股价预测".format(self.stock_id), fontproperties="SimHei")
        plt.xlabel(u"日期", fontproperties="SimHei")
        plt.ylabel(u"股价", fontproperties="SimHei")
        plt.savefig("stock.png")
        plt.show()


if __name__ == "__main__":
    stock = StockRNN(stock_id="002708")
    # print(stock.read_stock_data())
    log_dir = "stock_model"
    stock.train_pred_rnn()
    stock.visualize()
