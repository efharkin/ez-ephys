"""Unit tests for `stimtools`."""

import unittest

import numpy as np
import numpy.testing as npt

from ezephys import stimtools as st


class TestCompoundStimulus(unittest.TestCase):
    """Unit tests for `stimtools.CompoundStimulus` stimulus class."""

    def test_add(self):
        """Test `stimtools.CompoundStimulus.__add__` behaviour."""
        dt = 1
        initial_command = np.array([1, 2, 3, 2])
        compound = st.CompoundStimulus(initial_command, dt=dt)

        for const_offset in np.linspace(-10, 10, 7):
            tmp_compound = compound + const_offset
            npt.assert_array_almost_equal(
                tmp_compound.command,
                initial_command + const_offset,
                err_msg='Add constant to `CompoundStimulus` not '
                'equivalent to adding to command.'
            )

        variable_offset = np.array([3, 3, 2, 1])
        tmp_compound = compound + variable_offset
        npt.assert_array_almost_equal(
            tmp_compound.command,
            initial_command + variable_offset,
            err_msg='Add vector to `CompoundStimulus` not '
            'equivalent to adding to command.'
        )

        tmp_compound = compound + st.CompoundStimulus(variable_offset, dt=dt)
        npt.assert_array_almost_equal(
            tmp_compound.command,
            initial_command + variable_offset,
            err_msg='Adding `CompoundStimulus` objects not equivalent to '
            'adding commands.'
        )

    def test_subtract(self):
        """Test `stimtools.CompoundStimulus.__sub__` behaviour."""
        dt = 1
        initial_command = np.array([1, 2, 3, 2])
        compound = st.CompoundStimulus(initial_command, dt=dt)

        for const_offset in np.linspace(-10, 10, 7):
            tmp_compound = compound - const_offset
            npt.assert_array_almost_equal(
                tmp_compound.command,
                initial_command - const_offset,
                err_msg='Subtract constant from `CompoundStimulus` not '
                'equivalent to subtraction from command.'
            )

        variable_offset = np.array([3, 3, 2, 1])
        tmp_compound = compound - variable_offset
        npt.assert_array_almost_equal(
            tmp_compound.command,
            initial_command - variable_offset,
            err_msg='Subtract vector from `CompoundStimulus` not '
            'equivalent to subtraction from command.'
        )

        tmp_compound = compound - st.CompoundStimulus(variable_offset, dt=dt)
        npt.assert_array_almost_equal(
            tmp_compound.command,
            initial_command - variable_offset,
            err_msg='Adding `CompoundStimulus` objs not equivalent to adding '
            'commands.'
        )


class TestConcatenate(unittest.TestCase):
    """Unit tests for `stimtools.concatenate`."""

    def test_CompoundStimulus_only(self):
        """Test concatenation of CompoundStimulus objects."""
        dt = 0.7
        command1 = np.array([1, 2, 3, 2])
        command2 = np.array([1, 1, 2, 2, 3])

        with self.subTest('Test behaviour with good dt.'):
            compound1 = st.CompoundStimulus(command1, dt=dt)
            compound2 = st.CompoundStimulus(command2, dt=dt)

            new_compound = st.concatenate([compound1, compound2])

            npt.assert_array_almost_equal(
                new_compound.command,
                np.concatenate([command1, command2], axis=-1),
                err_msg='Concatenatation of vector with CompoundStimulus not equivalent to '
                'numpy concatenation.'
            )
            self.assertEqual(
                new_compound.no_timesteps,
                len(command1) + len(command2),
                'no_timesteps does not match length of commands inputs.'
            )
            self.assertEqual(
                new_compound.dt, dt, 'Incorrect dt in concatenated stimulus.'
            )

        with self.subTest('Test behaviour when bad dt is supplied.'):
            compound1 = st.CompoundStimulus(command1, dt=1)
            compound2 = st.CompoundStimulus(command2, dt=2)
            with self.assertRaises(
                ValueError,
                msg='Value error not raised for mismatched dt using auto dt.'
            ):
                st.concatenate([compound1, compound2])

            compound1 = st.CompoundStimulus(command1, dt=dt)
            compound2 = st.CompoundStimulus(command2, dt=dt)
            with self.assertRaises(
                ValueError,
                msg='Value error not raised for mismatched dt using specified dt.'
            ):
                st.concatenate([compound1, compound2], dt=dt*2)

    def test_vectors_only(self):
        """Test concatenation of numpy arrays using stimtools.concatenate."""
        dt = 0.7
        command1 = np.array([1, 2, 3, 2])
        command2 = np.array([1, 1, 2, 2, 3])

        new_compound = st.concatenate([command1, command2], dt=dt)

        npt.assert_array_almost_equal(
            new_compound.command,
            np.concatenate([command1, command2], axis=-1),
            err_msg='Concatenatation of vector with CompoundStimulus not equivalent to '
            'numpy concatenation.'
        )
        self.assertEqual(
            new_compound.no_timesteps,
            len(command1) + len(command2),
            'no_timesteps does not match length of commands inputs.'
        )
        self.assertEqual(
            new_compound.dt, dt, 'Incorrect dt in concatenated stimulus.'
        )

        with self.subTest('Test behaviour when bad dt is supplied.'):
            with self.assertRaises(
                ValueError,
                msg='Value error not raised when required dt is not supplied.'
            ):
                st.concatenate([command1, command2])


if __name__ == '__main__':
    unittest.main()
