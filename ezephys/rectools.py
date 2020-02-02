"""
Created on Tue Jan 30 12:27:39 2018

@author: Emerson

`Recording` class to hold recs as a np.array with built-in methods for plotting
and test-pulse fitting. Implements a factory function for loading recordings in
Axon binary format (.abf).

Compatible with python 2 and 3 as of Feb. 5, 2018.
"""

# IMPORT MODULES

from __future__ import division

from warnings import warn
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from neo.io import AxonIO


# DEFINE CELL CLASS

def read_ABF(self, fnames):
    """Import ABF files into a list of np.arrays.

    Inputs
    ------
        fnames  --  list of files to import, or str for a single file

    Returns
    -------
        List of np.arrays of recordings with dimensionality [channels,
        samples, sweeps]
    """
    # Convert str to iterable if only one fname is provided.
    if isinstance(fnames, str):
        fnames = [fnames]

    # Initialize list to hold output.
    output = []

    # Iterate over fnames.
    for fname in fnames:

        # Try reading the file.
        try:
            sweeps = AxonIO(fname).read()[0].segments
        except FileNotFoundError:
            warn('{} file {} not found. Skipping.'.format(
                self.name, fname))
            continue

        # Allocate np.array to hold recording.
        no_channels = len(sweeps[0].analogsignals)
        no_samples = len(sweeps[0].analogsignals[0])
        no_sweeps = len(sweeps)

        sweeps_arr = Recording(np.empty(
            (no_channels, no_samples, no_sweeps),
            dtype=np.float64))

        # Fill the array one sweep at a time.
        for sweep_ind in range(no_sweeps):

            for chan_ind in range(no_channels):

                signal = sweeps[sweep_ind].analogsignals[chan_ind]
                signal = np.squeeze(signal)

                assert len(signal) == sweeps_arr.shape[1], (
                    'Not all channels in {} are sampled at the same '
                    'rate.'.format(fname)
                )

                sweeps_arr[chan_ind, :, sweep_ind] = signal

        # Add recording to output list.
        output.append(sweeps_arr)

    return output


class Recording(np.ndarray):
    """Subclass of np.ndarray with additional methods for common ephys tasks.

    Recording objects are arrays with dimensionality [channel, time, sweep].

    Extra methods:
        plot
        fit_test_pulse
    """

    def __new__(cls, input_array, dt=0.1):
        """Instantiate new Recording given an array of data.

        Allows new Recording objects to be created using np.array-type syntax;
        i.e., by passing Recording a nested list or existing np.array.
        """
        if np.ndim(input_array) != 3:
            raise ValueError(
                'Expected `input_array` ndim == 3, got {} instead. '
                'Dimensionality must be `[channel, time, sweep]`.'.format(
                    np.ndim(input_array)
                )
            )

        # Convert input_array to a np.ndarray, and subsequently to a Recording.
        obj = np.asarray(input_array).view(cls)
        obj.dt = dt

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return None
        self.dt = getattr(obj, 'dt', None)

    def set_dt(self, dt):
        """Set Recording timestep width.

        Used for calculation of timestamps in t_vec/t_mat
        """
        self.dt = dt

    @property
    def t_vec(self):
        """Support vector of timestamps."""
        t_vec = np.arange(0, (self.shape[1] - 0.5) * self.dt, self.dt)

        assert len(
            t_vec) == self.shape[1], 'Unequal dimensions error in Recording.t_vec'

        return t_vec

    @property
    def t_mat(self):
        """Support array of timestamps with same shape as Recording."""
        t_mat = np.tile(
            self.t_vec[np.newaxis, :, np.newaxis], (self.shape[0], 1, self.shape[2]))

        assert t_mat.shape == self.shape, 'Unequal dimensions error in Recording.t_mat'

        return t_mat

    def plot(self, single_sweep=False, downsample=10):
        """Quick inspection of Recording.

        Note that x-axis values correspond to inds of the time axis of the array.
        """
        ### Check for correct input ###

        # Check single_sweep.
        if not isinstance(single_sweep, bool):
            raise TypeError('`single_sweep` must be bool.')

        # Check downsample.
        if downsample is None:
            downsample = 1

        elif not isinstance(downsample, int):
            raise TypeError('`downsample` must be int or None.')

        elif downsample < 1:
            raise ValueError('`downsample` must be an int > 0. or None.')

        ### Select data to plot ###
        if not single_sweep:
            plotting_data = self
        else:
            plotting_data = self[:, :, 0][:, :, np.newaxis]

        ### Make plot ###
        # Preserves indexes.
        x_vector = np.arange(0, self.shape[1], downsample)
        plt.figure(figsize=(10, 7))

        for i in range(self.shape[0]):

            # Force all subplots to share x-axis.
            if i == 0:
                ax0 = plt.subplot(self.shape[0], 1, 1)
            else:
                plt.subplot(self.shape[0], 1, i + 1, sharex=ax0)

            plt.title('Channel {}'.format(i))
            plt.plot(x_vector, plotting_data[i, ::downsample, :],
                     'k-',
                     linewidth=0.5)
            plt.xlabel('Time (timesteps)')

        plt.tight_layout()
        plt.show()

    def fit_test_pulse(self, baseline, steady_state, **kwargs):
        """Extract R_input and (optionally) R_a from test pulse.

        `baseline` and `steady_state` should be passed tuples of indexes over
        which to take measurements on each sweep.

        Set `verbose` to False to prevent printing results.

        tau: 3 tuple, optional
        --  Tuple of test pulse start and range over which to calculate tau in *indexes*.
        """
        ### Inputs ###

        # Set kwarg defaults.
        kwargs.setdefault('V_chan', 1)
        kwargs.setdefault('I_chan', 0)
        kwargs.setdefault('V_clamp', True)
        kwargs.setdefault('verbose', True)
        kwargs.setdefault('tau', None)

        # Check for correct inputs.
        if not isinstance(baseline, tuple):
            raise TypeError('Expected type tuple for `baseline`; got {} '
                            'instead.'.format(type(baseline)))
        elif any([not isinstance(entry, int) for entry in baseline]):
            raise TypeError('Expected tuple of ints for `baseline`.')
        elif len(baseline) != 2:
            raise TypeError('Expected tuple of len 2 specifying start and '
                            'stop positions for `baseline`.')
        elif any([entry > self.shape[1] for entry in baseline]):
            raise ValueError('`baseline` selection out of bounds for channel '
                             'of length {}.'.format(self.shape[1]))

        if not isinstance(steady_state, tuple):
            raise TypeError('Expected type tuple for `steady_state`; got {} '
                            'instead.'.format(type(steady_state)))
        elif any([not isinstance(entry, int) for entry in steady_state]):
            raise TypeError('Expected tuple of ints for `steady_state`.')
        elif len(steady_state) != 2:
            raise TypeError('Expected tuple of len 2 specifying start and '
                            'stop positions for `steady_state`.')
        elif any([entry > self.shape[1] for entry in steady_state]):
            raise ValueError('`steady_state` selection out of bounds for '
                             'channel of length {}.'.format(self.shape[1]))

        if steady_state[0] < baseline[1]:
            raise ValueError('Steady state measurement must be taken after '
                             ' end of baseline.')

        if not isinstance(kwargs['V_clamp'], bool):
            raise TypeError('Expected `V_clamp` to be type bool; got {} '
                            'instead.'.format(type(kwargs['V_clamp'])))

        if not isinstance(kwargs['verbose'], bool):
            raise TypeError('Expected `verbose` to be type bool; got {} '
                            'instead.'.format(type(kwargs['verbose'])))

        ### Main ###

        # Create dict to hold output.
        output = {}

        # Calculate R_input.
        V_baseline = self[kwargs['V_chan'], slice(*baseline), :].mean(axis=0)
        I_baseline = self[kwargs['I_chan'], slice(*baseline), :].mean(axis=0)
        V_test = self[kwargs['V_chan'], slice(*steady_state), :].mean(axis=0)
        I_test = self[kwargs['I_chan'], slice(*steady_state), :].mean(axis=0)

        delta_V_ss = V_test - V_baseline
        delta_I_ss = I_test - I_baseline

        R_input = 1000 * delta_V_ss / delta_I_ss
        output['R_input'] = R_input

        # Calculate R_a.
        if kwargs['V_clamp']:

            if delta_V_ss.mean() < 0:
                I_peak = self[kwargs['I_chan'],
                              slice(baseline[1], steady_state[0]),
                              :].min(axis=0)
            else:
                I_peak = self[kwargs['I_chan'],
                              slice(baseline[1], steady_state[0]),
                              :].max(axis=0)

            R_a = 1000 * delta_V_ss / (I_peak - I_baseline)
            output['R_a'] = R_a

        if kwargs['tau'] is not None:
            try:
                self.dt
            except NameError:
                raise RuntimeError('dt (timestep) must be set to fit tau')

            if not kwargs['V_clamp']:

                V_copy = deepcopy(self[kwargs['V_chan'], :, :])
                V_copy = V_copy.mean(axis=1)

                V_copy -= V_copy[slice(*steady_state)].mean()
                V0 = V_copy[slice(*baseline)].mean()

                pulse_start = kwargs['tau'][0]
                fitting_range = kwargs['tau'][1:3]

                t = (
                    np.arange(
                        fitting_range[0],
                        fitting_range[1]) - pulse_start) * self.dt
                x = np.log(V_copy[slice(*fitting_range)] / V0)

                mask = np.isnan(x)

                tau = - np.sum(x[~mask] * t[~mask]) / \
                    np.sum(x[~mask] * x[~mask])

                output['tau'] = tau

            else:
                raise NotImplementedError(
                    'Tau fitting for V-clamp is not implemented.')

        # Optionally, print results.
        if kwargs['verbose']:
            print('\n\n### Test-pulse results ###')
            print('R_in: {} +/- {} MOhm'.format(round(R_input.mean(), 1),
                                                round(R_input.std())))

            if kwargs['V_clamp']:
                print('R_a: {} +/- {} MOhm'.format(round(R_a.mean()),
                                                   round(R_a.std())))

        return output
