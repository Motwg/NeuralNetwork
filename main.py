from layers import Dense
from network import Network

'''
from tensorflow import keras
import numpy as np

fashion = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print([test_labels[0]])
net = Network([
    Dense(28*28, 100, activation='log'),
    Dense(100, 10, activation='log')
])

 for image, label in zip(train_images, train_labels):
    net.learn(image.flatten(), label)

print(net.run_once(train_images, [0 * 9, 1]))
print(test_labels[0])
# f_train_images = [image.flatten() for image in train_images]
# print(f_train_images)
'''
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
labels = iris.target
df = pd.DataFrame(iris.data, columns=['sl', 'sw', 'pl', 'pw'])
print(df.head())
print(labels)
print(df.iterrows())

net = Network([
    Dense(4, 3, activation='log', learning_rate=0.002),
    Dense(3, 3, activation='log', learning_rate=0.002),
    Dense(3, 3, activation='log', learning_rate=0.001),
    Dense(3, 3, activation='log', learning_rate=0.001)
])

train, tr_labs, err = [], [], []
for row, label in zip(df.iterrows(), labels):
    new_list = [row[1][i] for i in range(4)]
    new_label = [0] * 3
    new_label[label] = 1
    train.append(new_list)
    tr_labs.append(new_label)

for _ in range(20):
    print(_)
    for row, label in zip(train, tr_labs):
        print(row, label)
        print(net.learn(row, label))
        print(net.error)
        err.append(net.error)

for row, label in zip(train, tr_labs):
    print(label, ': ', net.run_once(row, label))

print(err)
