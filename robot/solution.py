from math import sqrt
from random import choices

import matplotlib.pyplot as plt
import pandas as pd

from core import visualisation as vis
from core.layers import Dense
from core.network import Network
from robot.utils import *

# logging.basicConfig(level=logging.DEBUG, filename='../logs/robot.log', filemode='w',
#                     format='%(levelname)-8s %(funcName)-20s %(message)s')

df = pd.read_excel('../pozyx/pozyxAPI_only_localization_measurement1.xlsx', sheet_name='measurement')
previous_samples = 4
packets = 1000
train_xy, correct_xy = [], []

for index, row in df.iterrows():
    if sqrt((row['measurement x'] - row['reference x'])**2 + (row['measurement y'] - row['reference y'])**2) < 240:
        new_list_meas = [row['measurement y'], row['measurement x']]
        new_list_ref = [row['reference x'], row['reference y']]
        new_list_dif = [row['difx'], row['dify']]
        train_xy.append(new_list_meas)
        correct_xy.append(new_list_ref)
train_xy = reduce(normalise(train_xy))
correct_xy = normalise(correct_xy)
train_xy = [xy for xy in get_previous(train_xy, previous_samples * 2)]

packet = list(map(list, zip(train_xy, correct_xy)))
print(packet[0])

net = Network([
    Dense(previous_samples * 2, 8, activation='relu', learning_rate=0.01),
    Dense(8, 8, activation='relu', learning_rate=0.01),
    Dense(8, 2, activation='relu', learning_rate=0.01)
])

epoch = 0
error = []
while len(error) == 0 or error[-1] > 0.00001 and epoch < 1:
    err = []
    for row, label in packet[:packets]: #  choices(packet, k=100):
        net.learn(row, label)
        err.append(net.error)
    epoch += 1
    error.append(sum(err) / len(err))
    print("Epoch: %d   MSE: %f" % (epoch, error[-1]))
    print(net)

for row, label in choices(packet[:packets], k=9):
    output = net.run_once(row, label)
    print(row, ': ', label, ': ', output)

vis.show(error)

results = []
for row, label in packet[:packets]:
    output = net.run_once(row, label)
    results.append(output)

# correct_p = Path(correct_xy[:packets], [Path.MOVETO if i == 0 else Path.LINETO for i in range(len(correct_xy[:packets]))])
# got_p = Path(results, [Path.MOVETO if i == 0 else Path.LINETO for i in range(len(results))])
# fig, ax = plt.subplots()
plt.grid(True)
print(correct_xy[:3])
print(results[:3])
plt.plot(*split(correct_xy[:packets]), linestyle='-', color='red')
# plt.plot(*split(train_xy[:packets]), linestyle='-', color='cyan')
plt.plot(*split(results), linestyle=' ', color='blue', marker='o', markersize=1.2)
# patch = patches.PathPatch(correct_p, facecolor='#FFFFFF', edgecolor='red')
# ax.add_patch(patch)
# patch = patches.PathPatch(got_p, facecolor='#FFFFFF', edgecolor='blue')
# ax.add_patch(patch)
# ax.set_xlim(-3000, 8000)
# ax.set_ylim(-1000, 4500)
plt.show()
