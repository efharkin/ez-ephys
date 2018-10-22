#%% IMPORT MODULES

import matplotlib.pyplot as plt

import sys
sys.path.append('./src/')
from cell_class import Cell

#%% LOAD TEST RECORDING

TEST_REC_PATH = './tests/17n28000.abf'
test_rec = Cell().read_ABF(TEST_REC_PATH)[0]
test_rec.plot()


#%% TEST SUPPORT STRUCTURES

# Set timestep in ms
test_rec.set_dt(0.1)

# Test t_vec
plt.figure()
plt.plot(test_rec.t_vec, test_rec[1, :, 0], 'k-')
plt.xlabel('Time (ms)')
plt.show()

# Test t_mat, which should be the same shape as test_rec
plt.figure()
plt.plot(test_rec.t_mat[0, :, :], test_rec[0, :, :], 'k-', alpha = 0.5)
plt.xlabel('Time (ms)')
plt.show()
