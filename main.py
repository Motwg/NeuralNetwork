from random import choices

import visualisation as vis
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

net = Network([
    # Dense(4, 8, activation='relu', learning_rate=0.02),
   #Dense(4, 5, activation='none', learning_rate=0.01),
    # Dense(8, 8, activation='none', learning_rate=0.01),
    Dense(4, 4, activation='relu', learning_rate=0.003),
    Dense(4, 3, activation='log', learning_rate=0.01)
])

train, tr_labs, err, acc = [], [], [], []
for row, label in zip(df.iterrows(), labels):
    new_list = [row[1][i] for i in range(4)]
    new_label = [0] * 3
    new_label[label] = 1
    train.append(new_list)
    tr_labs.append(new_label)

packet = list(map(list, zip(train, tr_labs)))
print(packet[0])

for _ in range(65):
    print(_)
    for row, label in choices(packet, k=150):
        net.learn(row, label)
        err.append(net.error)

for row, label in zip(train, tr_labs):
    output = net.run_once(row, label)
    if label.index(max(label)) == output.tolist().index(max(output.tolist())):
        acc.append(1)
        print(label, ': ', output, '  OK')
    else:
        acc.append(0)
        print(label, ': ', output, '  WRONG')

print(acc.count(1) / len(acc))
print(err)
vis.show(err)

