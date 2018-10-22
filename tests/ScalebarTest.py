#%% IMPORT MODULES

import matplotlib.pyplot as plt

import sys
sys.path.append('./src/')
import pltools
from cell_class import Cell

#%% LOAD TEST RECORDING

TEST_REC_PATH = './tests/example-data/17n28000.abf'
test_rec = Cell().read_ABF(TEST_REC_PATH)[0]
test_rec.plot()


#%%

plt.figure(figsize = (10, 5))

plt.subplot2grid((4, 1), (0, 0), rowspan = 3)
plt.plot(test_rec[0, :, :].mean(axis = 1), 'k-')

plt.xlim(1000, 8000)
plt.ylim(-200, 200)
pltools.add_scalebar(y_units = 'pA', x_units = 'ms', anchor = (0.98, 0.2))

plt.subplot2grid((4, 1), (3, 0))
plt.plot(test_rec[1, :, :].mean(axis = 1), 'k-')

plt.xlim(1000, 8000)

pltools.add_scalebar(y_units = 'mV', x_units = 'ms', anchor = (0.7, 0.15),
text_spacing = (0.007, 0), bar_spacing = 0.03, omit_x = True)

plt.show()
