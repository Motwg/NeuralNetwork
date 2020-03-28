from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

df = pd.read_excel('pozyx/pozyxAPI_only_localization_measurement1.xlsx', sheet_name='measurement')
train_xy, correct_xy, err, acc = [], [], [], []
for index, row in df.iterrows():
    new_list_pm = []
    new_list_meas = [row['measurement x'], row['measurement y']]
    new_list_ref = [row['reference x'], row['reference y']]
    new_list_dif = [row['difx']*row['difx'] + row['dify']*row['dify']]
    new_list_dif1 = [row['difx'], row['dify']]
    if row['difx'] > 0:
        new_list_pm.append(2)
    elif row['difx'] < 0:
        new_list_pm.append(0)
    else:
        new_list_pm.append(0)
    train_xy.append(new_list_meas)
    correct_xy.append(new_list_pm)

'''
packet = list(map(list, zip(train_xy, correct_xy)))
print(packet[0])'''

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential([
    keras.layers.Dense(2),
keras.layers.Dense(10),
    keras.layers.Dense(3, activation='sigmoid')
])
train_xy = np.array(train_xy)
correct_xy = np.array(correct_xy)
print(train_xy.shape)
print(correct_xy.shape)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_xy, correct_xy, epochs=100)

test_loss, test_acc = model.evaluate(train_xy, correct_xy, verbose=2)
print(test_loss)
print(test_acc)