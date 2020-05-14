import visualisation as vis
from layers import Dense
from network import Network

float_formatter = "{:.4f}".format

# length, width
train_flowers = [[2, 1],
                 [1, 3],
                 [2, 1],
                 [1, 4],
                 [3, 1],
                 [2, 5],
                 [3, 2],
                 [1, 5]]
train_labels = [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]]

test_flowers = [[4, 1], [4, 2], [1, 4]]
test_labels = [[1, 0], [1, 0], [0, 1]]

err, acc = [], []

layer = Dense(2, 3, activation='relu', learning_rate=0.02)
layer1 = Dense(3, 2, activation='log', learning_rate=0.1)

net = Network([
    layer,
    layer1
], use_normalise=False)

for _ in range(500):
    print(_)
    for row, label in zip(train_flowers, train_labels):
        print('{}'.format(net.learn(row, label)))
        err.append(net.error)

for row, label in zip(train_flowers + test_flowers, train_labels + test_labels):
    output = net.run_once(row, label)
    if label.index(max(label)) == output.tolist().index(max(output.tolist())):
        acc.append(1)
    else:
        acc.append(0)
    print('{} : {}'.format(label, output))

print(err)
vis.show(err)
print(acc.count(1) / len(acc))