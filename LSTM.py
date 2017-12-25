'''
Our simple RNN consists of
    - One input layer which converts a  28*28  dimensional input to an  128  dimensional hidden layer
    - One intermediate recurrent neural network (LSTM)
    - One output layer which converts an  128  dimensional output of the LSTM to  10 dimensional output
    indicating a class label
'''

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".",one_hot=True)

trainimgs = mnist.train.images
trainlabels = mnist.train.labels
testimgs = mnist.test.images
testlabels = mnist.test.labels

ntrain = trainimgs.shape[0]
ntest = testimgs.shape[0]
dim = trainimgs.shape[1]
nclasses = trainlabels.shape[1]

n_input = 28 # MNIST Data input (img shape: 28 x 28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST Total classes (0-9)

learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

x = tf.placeholder(tf.float32, shape = [None, n_steps, n_input], name = "x")
y = tf.placeholder(tf.float32, shape = [None, n_classes], name = "y")

weights = {'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))}
biases = {'out':tf.Variable(tf.random_normal([n_classes]))}

lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=x,dtype=tf.float32)

output = tf.reshape(tf.split(outputs, 28, axis=1, num=None, name='split')[-1],[-1,128])
pred = tf.matmul(output,weights['out']) + biases['out']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        sess.run(optimizer,feed_dict={x:batch_x, y:batch_y})
        if step % display_step == 0:
            accuracy = sess.run(acc, feed_dict={x:batch_x, y:batch_y})
            loss = sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss =  "+\
                  "{:.6f}".format(loss) + ", Training accuracy = "+\
                  "{:.5f}".format(accuracy))

        step += 1
    print("Optimizer finished!")

    test_len = 500
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len].reshape((-1,n_classes))
    print("Testing accuracy: ", sess.run(acc, feed_dict={x:test_data,y:test_label}))

sess.close()