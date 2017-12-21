import tensorflow as tf
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

iris = load_iris()
iris_X, iris_Y = iris.data[:-1,:], iris.target[:-1]
iris_Y = pd.get_dummies(iris_Y).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_Y, test_size=0.33, random_state=42)

numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]

X = tf.placeholder(tf.float32,[None,numFeatures], name = "X")
Y = tf.placeholder(tf.float32,[None,numLabels], name = "Y")

W = tf.get_variable(name = "W", shape = [numFeatures,numLabels], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable(name = "b", shape = [1,numLabels], initializer=tf.zeros_initializer())

Z = tf.add(tf.matmul(X,W),b)

logits = Z
labels = Y

num_epochs = 2000

learning_rate = 0.05

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels= labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


init = tf.global_variables_initializer()

epoch_costs = []
oldCost = 0
diff = 1

with tf.Session() as sess:
    sess.run(init)

    correct_predict = tf.equal(tf.argmax(Z, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))
    for epoch in range(num_epochs):
        if epoch > 1 and diff <.00001:
            print("Stop at epoch: ",epoch, " - Cost: ",oldCost, " - Accuracy: ",acc)
            break
        _, acc, epoch_cost = sess.run([optimizer,accuracy, cost],feed_dict={X:trainX, Y:trainY})
        diff = abs(epoch_cost - oldCost)
        oldCost = epoch_cost
        if epoch % 50 == 0:
            print(epoch,acc,epoch_cost)
        epoch_costs.append(epoch_cost)
