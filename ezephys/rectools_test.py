# IMPORT MODULES

import os

import matplotlib.pyplot as plt

from rectools import Cell


# LOAD TEST RECORDING

# Load all ABF files in the example data directory
EX_DATA_DIR = 'test_data'
PATHS = []
for fname in os.listdir(EX_DATA_DIR):
    if fname[-4:].lower() == '.abf':
        PATHS.append(os.path.join(EX_DATA_DIR, fname))
test_rec_ls = Cell().read_ABF(PATHS)

# Assign one of the recordings to test_rec and plot it
test_rec = test_rec_ls[1]
test_rec.plot()


# TEST SUPPORT STRUCTURES

# Set timestep in ms
test_rec.set_dt(0.1)

# Test t_vec
plt.figure()
plt.title('t_vec test')
plt.plot(test_rec.t_vec, test_rec[0, :, 0], 'k-')
plt.xlabel('Time (ms)')
plt.show()

# Test t_mat, which should be the same shape as test_rec
plt.figure()
plt.title('t_mat test')
plt.plot(test_rec.t_mat[0, :, :], test_rec[0, :, :], 'k-', alpha=0.5)
plt.xlabel('Time (ms)')
plt.show()
