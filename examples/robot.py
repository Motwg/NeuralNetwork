import pandas as pd

import visualisation as vis
from layers import Dense
from network import Network

df = pd.read_excel('pozyx/pozyxAPI_only_localization_measurement1.xlsx', sheet_name='measurement')

train_xy, correct_xy, err, acc = [], [], [], []
for index, row in df.iterrows():
    new_list_meas = [row['measurement x'], row['measurement y']]
    new_list_ref = [row['reference x'], row['reference y']]
    new_list_dif = [row['difx'], row['dify']]
    train_xy.append(new_list_meas)
    correct_xy.append(new_list_dif)

net = Network([
    # Dense(4, 8, activation='relu', learning_rate=0.02),
    # Dense(4, 5, activation='none', learning_rate=0.01),
    # Dense(8, 8, activation='none', learning_rate=0.01),
    #Dense(2, 20, activation='log', learning_rate=0.03),
    #Dense(20, 4, activation='log', learning_rate=0.08),
    Dense(2, 2, activation='none', learning_rate=0.03)
], use_normalise=True)


packet = list(map(list, zip(train_xy, correct_xy)))
print(packet[0])

for _ in range(150):
    print(_)
    for row, label in packet[:9]:#choices(packet, k=100):
        net.learn(row, label)
        err.append(net.error)

for row, label in packet[:9]:
    output = net.run_once(row, label)
    print(row, ': ', label, ': ', output, '   ')
'''    
    if label.index(max(label)) == output.tolist().index(max(output.tolist())):
        acc.append(1)
        print(label, ': ', output, '  OK')
    else:
        acc.append(0)
        print(label, ': ', output, '  WRONG')
'''

#print(acc.count(1) / len(acc))
print(err)
vis.show(err)
