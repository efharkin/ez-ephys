"""Unit tests for `rectools`."""

import unittest

import numpy as np
import numpy.testing as npt

import ezephys.rectools as rt

class TestRecording(unittest.TestCase):
    """Tests for `rectools.Recording`."""

    def test_type_preservation(self):
        """Test that `Recording` objects are returned from various ops on `Recording` objects."""
        test_rec = rt.Recording([[[0, 1, 2]]])
        self.assertTrue(
            isinstance(test_rec, rt.Recording),
            'Result of `Recording` constructor is not of `Recording` type.',
        )
        self.assertTrue(
            isinstance(test_rec[..., 1:], rt.Recording),
            'Result of slicing `Recording` is not of `Recording` type.',
        )
        self.assertTrue(
            isinstance(test_rec[..., 1], rt.Recording),
            'Result of retrieving single element of `Recording` is not of '
            '`Recording` type.',
        )

    def test_dt_preservation(self):
        """Test that `dt` is preserved after indexing `Recording`."""
        dt = 0.6767335
        test_rec = rt.Recording([[[0, 1, 2]]], dt=dt)
        self.assertEqual(
            test_rec.dt,
            dt,
            'Assigned `dt` not equal to attribute `dt`; test code probably '
            'broken.',
        )
        self.assertEqual(
            test_rec[..., 1:].dt,
            dt,
            '`Recording.dt` attribute altered by slicing.',
        )
        self.assertEqual(
            test_rec[..., 1].dt,
            dt,
            '`dt` attribute altered by retrieving single element of `Recording`.',
        )

    def test_no_timesteps_property(self):
        """Test that no_timesteps reflects number of timesteps in recording."""
        expected_values = {
            'no_timesteps': 1000,
            'no_sweeps': 10,
            'no_channels': 4,
        }
        test_rec = rt.Recording(
            np.zeros(
                [
                    expected_values['no_channels'],
                    expected_values['no_timesteps'],
                    expected_values['no_sweeps'],
                ]
            ),
            dt=0.1,
        )
        self.assertEqual(
            test_rec.no_timesteps,
            expected_values['no_timesteps'],
            'Expected {} for `no_timesteps` property; got {} instead.'.format(
                expected_values['no_timesteps'], test_rec.no_timesteps
            ),
        )

    def test_no_sweeps_property(self):
        """Test that no_sweeps reflects number of sweeps in recording."""
        expected_values = {
            'no_timesteps': 1000,
            'no_sweeps': 10,
            'no_channels': 4,
        }
        test_rec = rt.Recording(
            np.zeros(
                [
                    expected_values['no_channels'],
                    expected_values['no_timesteps'],
                    expected_values['no_sweeps'],
                ]
            ),
            dt=0.1,
        )
        self.assertEqual(
            test_rec.no_sweeps,
            expected_values['no_sweeps'],
            'Expected {} for `no_sweeps` property; got {} instead.'.format(
                expected_values['no_sweeps'], test_rec.no_sweeps
            ),
        )

    def test_no_channels_property(self):
        """Test that no_channels reflects number of channels in recording."""
        expected_values = {
            'no_timesteps': 1000,
            'no_sweeps': 10,
            'no_channels': 4,
        }
        test_rec = rt.Recording(
            np.zeros(
                [
                    expected_values['no_channels'],
                    expected_values['no_timesteps'],
                    expected_values['no_sweeps'],
                ]
            ),
            dt=0.1,
        )
        self.assertEqual(
            test_rec.no_channels,
            expected_values['no_channels'],
            'Expected {} for `no_channels` property; got {} instead.'.format(
                expected_values['no_channels'], test_rec.no_channels
            ),
        )

    def test_duration_property(self):
        """Test that duration reflects number of duration in recording."""
        recording_dt = 0.1
        recording_shape = {
            'no_timesteps': 1000,
            'no_sweeps': 10,
            'no_channels': 4,
        }
        expected_duration = recording_shape['no_timesteps'] * recording_dt
        test_rec = rt.Recording(
            np.zeros(
                [
                    recording_shape['no_channels'],
                    recording_shape['no_timesteps'],
                    recording_shape['no_sweeps'],
                ]
            ),
            dt=recording_dt,
        )
        npt.assert_almost_equal(
            test_rec.duration,
            expected_duration,
            err_msg='Expected {} for `duration` property; got {} instead.'.format(
                expected_duration, test_rec.duration
            ),
        )

    def test_time_supp_length_matches_no_timesteps(self):
        """Ensure time_supp is a valid time support vector for input signal."""
        for no_timesteps in [5, 578, 993, 300072]:
            for dt in [0.1, 0.5, 3.0]:
                test_rec = rt.Recording(np.empty([6, no_timesteps, 1]), dt=dt)
                self.assertEqual(
                    len(test_rec.time_supp),
                    no_timesteps,
                    'Expected length of time_supp {} to match no_timesteps of '
                    'input {}.'.format(len(test_rec.time_supp), no_timesteps),
                )

    def test_time_supp_length_matches_no_timesteps_after_slicing(self):
        test_rec = rt.Recording(np.empty([3, 4387, 2]), dt=0.4)
        test_sliced_rec = test_rec[:, 300:557, :]
        self.assertEqual(
            len(test_sliced_rec.time_supp),
            test_sliced_rec.no_timesteps,
            'Expected length of `time_supp` {} to equal no_timesteps of '
            'Recording {} after slicing.'.format(
                len(test_sliced_rec.time_supp), test_sliced_rec.no_timesteps
            ),
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
loader = rt.ABFLoader()
test_rec_ls = loader.load(PATHS)

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
