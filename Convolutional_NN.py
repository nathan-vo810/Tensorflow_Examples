import numpy as np
import tensorflow as tf
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

#Extract data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
Input -> Conv1 -> Relu -> Max Pool 1 -> Conv2 -> Relu -> Max Pool 2 -> Fully Connected -> Relu -> FC -> Softmax

(Input) -> [?, 28, 28, 1] >> Apply 32 filter of [5x5]
(Convolutional layer 1) -> [?, 28, 28, 32]
(ReLU 1) -> [?, 28, 28, 32]
(Max pooling 1) -> [?, 14, 14, 32]
(Convolutional layer 2) -> [?, 14, 14, 64]
(ReLU 2) -> [?, 14, 14, 64]
(Max pooling 2) -> [?, 7, 7, 64]
[fully connected layer 3] -> [1x1024]
[ReLU 3] -> [1x1024]
[Drop out] -> [1x1024]
[fully connected layer 4] -> [1x10]

...

'''

def initialize_parameters():
    W_conv1 = tf.get_variable("W_conv1", shape = [5,5,1,32], initializer=tf.contrib.layers.xavier_initializer())
    b_conv1 = tf.Variable(tf.constant(0.1, shape = [32]))
    W_conv2 = tf.get_variable("W_conv2", shape = [5,5,32,64], initializer = tf.contrib.layers.xavier_initializer())
    b_conv2 = tf.Variable(tf.constant(0.1, shape = [64]))
    # W_conv1 = tf.Variable(tf.truncated_normal(shape = [5,5,1,32], stddev=0.01))
    # W_conv2 = tf.Variable(tf.truncated_normal(shape = [5,5,32,64], stddev=0.01))
    W_fc1 = tf.Variable(tf.truncated_normal(shape = [7*7*64,1024],stddev=0.01))
    b_fc1 = tf.Variable(tf.constant(0.1, shape = [1024]))
    W_fc2 = tf.Variable(tf.truncated_normal(shape = [1024,10], stddev=0.01))
    b_fc2 = tf.Variable(tf.constant(0.1, shape = [10]))

    parameters = {"W_conv1" : W_conv1,
                  "b_conv1" : b_conv1,
                  "W_conv2" : W_conv2,
                  "b_conv2" : b_conv2,
                  "W_fc1" : W_fc1,
                  "b_fc1" : b_fc1,
                  "W_fc2" : W_fc2,
                  "b_fc2" : b_fc2}
    return parameters;

def conv(X,W):
    return tf.nn.conv2d(input = X, filter = W, strides = [1,1,1,1], padding = "SAME")

def maxpool(X):
    return tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

height = 28
width =28
flat = height * width
class_output = 10

x = tf.placeholder(tf.float32, shape = [None, flat])
y = tf.placeholder(tf.float32, shape = [None, class_output])
x_image = tf.reshape(x, shape = [-1, 28, 28, 1])

parameters = initialize_parameters()
W_conv1 = parameters['W_conv1']
b_conv1 = parameters['b_conv1']
W_conv2 = parameters['W_conv2']
b_conv2 = parameters['b_conv2']
W_fc1 = parameters['W_fc1']
b_fc1 = parameters['b_fc1']
W_fc2 = parameters['W_fc2']
b_fc2 = parameters['b_fc2']

convolve_1 = conv(x_image,W_conv1) + b_conv1
relu_1 = tf.nn.relu(convolve_1)
max_1 = maxpool(relu_1)

convolve_2 = conv(max_1,W_conv2) + b_conv2
relu_2 = tf.nn.relu(convolve_2)
max_2 = maxpool(relu_2)

layer2_matrix = tf.reshape(max_2, [-1, 7*7*64])
fc1 = tf.matmul(layer2_matrix,W_fc1) + b_fc1
relu_fc1 = tf.nn.relu(fc1)

keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(relu_fc1, keep_prob)

fc = tf.matmul(layer_drop,W_fc2) + b_fc2
y_CNN = tf.nn.softmax(fc)

#loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_CNN), reduction_indices=[1]))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fc, labels = y))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_prediciton = tf.equal(tf.argmax(y_CNN,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediciton,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    for i in range(1100):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y:batch[1], keep_prob:1.0})
            print("Step %d, training accuracy %g"%(i,float(train_accuracy)))
        optimizer.run(feed_dict={x:batch[0], y:batch[1], keep_prob:0.5})







