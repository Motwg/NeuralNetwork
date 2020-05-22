import csv

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

from robot.utils import *

df = pd.read_excel('../pozyx/pozyxAPI_only_localization_measurement1.xlsx', sheet_name='measurement')
previous_samples = 10
train_xy, correct_xy = [], []

for index, row in df.iterrows():
    new_list_meas = [row['measurement y'], row['measurement x']]
    new_list_ref = [row['reference x'], row['reference y']]
    new_list_dif = [row['difx'], row['dify']]
    train_xy.append(new_list_meas)
    correct_xy.append(new_list_ref)
train_xy = reduce(normalise(train_xy))
correct_xy = normalise(correct_xy)
train_xy = [xy for xy in get_previous(train_xy, previous_samples * 2)]
print(train_xy[:3])
print(correct_xy[:3])

model = keras.Sequential([
    keras.layers.Input(previous_samples * 2),
    keras.layers.Dense(200, activation='relu',
                       kernel_initializer=keras.initializers.RandomUniform(minval=0.01, maxval=0.05)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(25, activation='relu'),
    keras.layers.Dense(2, activation='relu')
])
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001, nesterov=True),
              loss=keras.losses.MSE,
              metrics=['accuracy'])

model.fit(train_xy, correct_xy, epochs=100, batch_size=1)

test_loss, test_acc = model.evaluate(train_xy, correct_xy, verbose=2)
print('Overall     loss:  ', test_loss)
print('Overall accuracy:  ', test_acc)
predictions = model.predict(train_xy)
for i in range(5):
    print('Prediction: ', predictions[i], '  Correct: ', correct_xy[i])

plt.plot(*split(correct_xy), linestyle='-', color='red')
plt.plot(*split(predictions), linestyle=' ', color='blue', marker='o', markersize=1.2)
plt.show()


with open('output.csv', mode='w') as result_file:
    writer = csv.writer(result_file, delimiter=',')
    for line in predictions:
        writer.writerow(line)
