from core import visualisation as vis
from core.layers import Dense
from core.network import Network

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

'''
logging.basicConfig(level=logging.DEBUG, filename='../logs/iris.log', filemode='w',
                    format='%(levelname)-8s %(funcName)-20s %(message)s')
'''


iris = load_iris()
labels = iris.target
df = pd.DataFrame(iris.data, columns=['sl', 'sw', 'pl', 'pw'])

net = Network([
    Dense(4, 4, activation='log', learning_rate=0.4),
    Dense(4, 3, activation='log', learning_rate=0.4)
], use_normalise=False)

train, tr_labs, errors, acc = [], [], [], []
for row, label in zip(df.iterrows(), labels):
    new_list = [row[1][i] for i in range(4)]
    new_label = [0] * 3
    new_label[label] = 1
    train.append(new_list)
    tr_labs.append(new_label)

packet = list(map(list, zip(train, tr_labs)))

for _ in range(1000):
    # choices(packet, k=150)
    err = []
    for row, label in packet:
        net.learn(row, label)
        err.append(net.error)
    errors.append(sum(err) / len(err))
    print("Epoch: %d  MSE: %f" % (_, errors[-1]))

for row, label in zip(train, tr_labs):
    output = net.run_once(row, label)
    if label.index(max(label)) == output.tolist().index(max(output.tolist())):
        acc.append(1)
        print(label, ': ', output, '  OK')
    else:
        acc.append(0)
        print(label, ': ', output, '  WRONG')

print('{} % Correct!'.format(acc.count(1) / len(acc) * 100))
print(errors)
vis.show(errors)

