"""Unit tests for `rectools`."""

import unittest

import ezephys.rectools as rt


class TestRecording(unittest.TestCase):
    """Tests for `rectools.Recording`."""

    def test_type_preservation(self):
        """Test that `Recording` objects are returned from various ops on `Recording` objects."""
        test_rec = rt.Recording([[[0, 1, 2]]])
        self.assertTrue(
            isinstance(test_rec, rt.Recording),
            'Result of `Recording` constructor is not of `Recording` type.'
        )
        self.assertTrue(
            isinstance(test_rec[..., 1:], rt.Recording),
            'Result of slicing `Recording` is not of `Recording` type.'
        )
        self.assertTrue(
            isinstance(test_rec[..., 1], rt.Recording),
            'Result of retrieving single element of `Recording` is not of '
            '`Recording` type.'
        )

    def test_dt_preservation(self):
        """Test that `dt` is preserved after indexing `Recording`."""
        dt = 0.6767335
        test_rec = rt.Recording([[[0, 1, 2]]], dt=dt)
        self.assertEqual(
            test_rec.dt, dt,
            'Assigned `dt` not equal to attribute `dt`; test code probably '
            'broken.'
        )
        self.assertEqual(
            test_rec[..., 1:].dt, dt,
            '`Recording.dt` attribute altered by slicing.'
        )
        self.assertEqual(
            test_rec[..., 1].dt, dt,
            '`dt` attribute altered by retrieving single element of `Recording`.'
        )


if __name__ == '__main__':
    unittest.main()

"""
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
"""
