import numpy as np

from core.layers import Dense as MyDense
from core.network import Network as MyNetwork

float_formatter = "{:.4f}".format


def flatten_labels(labels):
    for i, label in enumerate(labels):
        labels[i] = label.index(1)
    return labels


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
my = True  # True or False
layers = 3  # 3 or 2

if my:
    if layers == 3:
        layer = MyDense(2, 3, activation='log', learning_rate=0.01)
        layer1 = MyDense(3, 2, activation='log', learning_rate=0.01)
        my_net = MyNetwork([
            layer,
            layer1
        ], use_normalise=False)
        layer.weights = np.array([[0.1, 0.2, 0.3],
                                  [0.4, 0.5, 0.6]])
        layer1.weights = np.array([[0.7, 0.8],
                                   [0.9, 0.1],
                                   [0.2, 0.3]])

    elif layers == 2:
        layer = MyDense(2, 2, activation='log', learning_rate=0.01)
        my_net = MyNetwork([
            layer
        ], use_normalise=False)
        layer.weights = np.array([[0.1, 0.2],
                                  [0.4, 0.5]])

    my_net.learn(train_flowers[0], train_labels[0])
    print(layer)
    if layers == 3:
        print(layer1)
else:
    from tensorflow import keras

    layer1 = keras.layers.Input(2)
    layer2 = keras.layers.Dense(3, activation='sigmoid')
    layer3 = keras.layers.Dense(2, activation='sigmoid')
    if layers == 3:
        model = keras.Sequential([layer1, layer2, layer3])
    elif layers == 2:
        model = keras.Sequential([layer1, layer3])
    train_flowers, train_labels = np.array(train_flowers), np.array(train_labels)
    print(train_flowers.shape)
    print(train_labels.shape)

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, nesterov=True),
                  # loss='mean_squared_error',
                  loss=keras.losses.MSE,
                  # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    if layers == 3:
        layer2.set_weights([
            np.array([[0.1, 0.2, 0.3],
                      [0.4, 0.5, 0.6]]),
            # bias
            np.array([.0, .0, .0])
        ])

        layer3.set_weights([
            np.array([[0.7, 0.8],
                      [0.9, 0.1],
                      [0.2, 0.3]]),
            np.array([.0, .0])
        ])
    elif layers == 2:
        layer3.set_weights([
            np.array([[0.1, 0.2],
                      [0.4, 0.5]]),
            np.array([.0, .0])
        ])

    model.fit(np.array([train_flowers[0]]), np.array([train_labels[0]]), epochs=1)
    print(model.get_config())
    if layers == 3:
        print(layer2.weights)
    print(layer3.weights)

    # test_loss, test_acc = model.evaluate(train_xy, correct_xy, verbose=2)
    # print(test_loss)
    # print(test_acc)
