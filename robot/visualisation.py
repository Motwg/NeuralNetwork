from math import sqrt

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path

df = pd.read_excel('../pozyx/pozyxAPI_only_localization_measurement1.xlsx', sheet_name='measurement')

train_xy, correct_xy = [], []
for index, row in df.iterrows():
    if sqrt((row['measurement x'] - row['reference x']) ** 2 + (row['measurement y'] - row['reference y']) ** 2) < 250:
        new_list_meas = (row['measurement x'], row['measurement y'])
        new_list_ref = (row['reference x'], row['reference y'])
        new_list_dif = (row['difx'], row['dify'])
        train_xy.append(new_list_meas)
        correct_xy.append(new_list_ref)

correct_p = Path(correct_xy, [Path.MOVETO if i == 0 else Path.LINETO for i in range(len(correct_xy))])
got_p = Path(train_xy, [Path.MOVETO if i == 0 else Path.LINETO for i in range(len(train_xy))])
fig, ax = plt.subplots()
plt.grid(True)
patch = patches.PathPatch(correct_p, facecolor='#FFFFFF', edgecolor='red')
ax.add_patch(patch)
patch = patches.PathPatch(got_p, facecolor='#FFFFFF', edgecolor='blue')
ax.add_patch(patch)
ax.set_xlim(-3000, 8000)
ax.set_ylim(-1000, 4500)
plt.show()
