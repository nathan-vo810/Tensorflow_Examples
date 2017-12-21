import numpy as np
import tensorflow as tf

#Initialize data
X = np.random.rand(1000)
Y = X * 3 + 2
Y = np.vectorize(lambda Y : Y + np.random.normal(loc = 0.0, scale = 0.1))(Y)

#Initialize tensor
a = tf.Variable(2.0)
b = tf.Variable(1.0)
y = a * X + b

loss = tf.reduce_mean(tf.square(Y-y))

optimizer = tf.train.AdamOptimizer(0.4)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #Train model
    train_data = []
    for step in range(100):
            evals = sess.run([train, loss, a, b])[1:]
            if step % 5 == 0:
                print(step, evals)
                train_data.append(evals)