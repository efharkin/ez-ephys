"""
Created on Tue Jan 30 12:27:39 2018

@author: Emerson

`Recording` class to hold recs as a np.array with built-in methods for plotting
and test-pulse fitting. Implements a factory class `ABFLoader` for loading recordings in
Axon binary format (.abf).

Compatible with python 2 and 3 as of Feb. 5, 2018.
"""

# IMPORT MODULES

from __future__ import division

from warnings import warn
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from neo.io import AxonIO


# OBJECTS FOR LOADING RECORDINGS

class BaseRecordingLoader(object):
    """Abstract base class for RecordingLoaders.

    Children implement the following methods needed to load
    electrophysiological recordings of a given format:
    - _read_data_from_file(file_name)
    - _get_sampling_interval_in_ms(file_data)
    - _coerce_to_recording(file_data, sampling_interval)

    These private methods are called by _load_single_file(file_name).

    """
    def __init__(self):
        raise NotImplementedError

    def load(self, file_names):
        """Load recordings from files.

        Arguments
        ---------
        file_names : list or str
            list of files to import, or str for a single file

        Returns
        -------
        List of np.array-like recordings each with dimensionality
        [channels, samples, sweeps].

        """
        if isinstance(file_names, str):
            file_names = [file_names]

        recordings = []
        for file_name in file_names:
            recordings.append(self._load_single_file(file_name))
        return recordings

    def _load_single_file(self, file_name):
        file_data = self._read_data_from_file(file_name)
        sampling_intervals = self._get_sampling_intervals_in_ms(file_data)
        assert all(np.isclose(sampling_intervals, sampling_intervals[0]))
        recording = self._coerce_to_recording(file_data, sampling_intervals[0])
        return recording

    def _read_data_from_file(self, file_name):
        raise NotImplementedError

    def _get_sampling_intervals_in_ms(self, file_data):
        raise NotImplementedError

    def _coerce_to_recording(self, file_data, sampling_interval):
        raise NotImplementedError


class ABFLoader(BaseRecordingLoader):
    """Load recordings in Axon binary format (.abf).

    Recordings are loaded by passing a list of file names to the `load()`
    method.

    """
    def __init__(self):
        pass

    def _read_data_from_file(self, file_name):
        return AxonIO(file_name).read()[0].segments

    def _get_sampling_intervals_in_ms(self, file_data):
        sampling_intervals = []
        for sweep in file_data:
            for signal in sweep.analogsignals:
                sampling_intervals.append(1e3 / signal.sampling_rate.item())
        return sampling_intervals

    def _coerce_to_recording(self, file_data, sampling_interval):
        sweeps = file_data

        no_channels = len(sweeps[0].analogsignals)
        no_samples = len(sweeps[0].analogsignals[0])
        no_sweeps = len(sweeps)

        sweeps_arr = np.empty(
            (no_channels, no_samples, no_sweeps), dtype=np.float64
        )

        # Fill the array one sweep at a time.
        for sweep_ind in range(no_sweeps):

            for chan_ind in range(no_channels):

                signal = sweeps[sweep_ind].analogsignals[chan_ind]
                signal = np.squeeze(signal)

                assert len(signal) == sweeps_arr.shape[1], (
                    'Not all channels are sampled at the same rate.'
                )

                sweeps_arr[chan_ind, :, sweep_ind] = signal

        return Recording(sweeps_arr, dt=sampling_interval)


# PYTHON OBJECT FOR REPRESENTING RECORDINGS

class Recording(np.ndarray):
    """Thin wrapper of numpy.ndarray with add-ons for common ephys tasks.

    Recording objects are 3D arrays with dimensionality [channel, time, sweep].

    Extra attributes
    ----------------
    no_channels: int
        Number of channels in the recording (e.g., current, voltage, etc).
    no_sweeps, no_timesteps: int
        Number of sweeps, timesteps in the recording.
    duration: float
        Duration of one sweep in ms.
    dt: float
        Sampling interval in ms. 1000.0/sampling rate in Hz.
    time_supp: float 1D array
        Time support vector for one sweep of the recording in ms. Always starts
        at zero.

    Extra methods
    -------------
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

        # Initialize attributes.
        obj.dt = dt

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return None

        # Copy time step.
        self.dt = getattr(obj, 'dt', None)

    @property
    def no_channels(self):
        """Number of channels in instance."""
        assert self.ndim == 3
        return self.shape[0]

    @property
    def no_sweeps(self):
        """Number of sweeps in instance."""
        assert self.ndim == 3
        return self.shape[2]

    @property
    def no_timesteps(self):
        """Number of sweeps in instance."""
        assert self.ndim == 3
        return self.shape[1]

    @property
    def duration(self):
        """Duration of one sweep in ms."""
        return self.no_timesteps * self.dt

    @property
    def time_supp(self):
        """Time support vector for one sweep.

        Gives time from start of sweep in ms. Always starts at zero.

        """
        if getattr(self, '_time_supp', None) is None:
            self._init_time_supp()
        return self._time_supp

    def _init_time_supp(self):
        self._time_supp = np.arange(0, self.duration - 0.5 * self.dt, self.dt)

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
        plot_tau: bool, default False
        --  Optionally plot the tau fit.
        """
        ### Inputs ###

        # Set kwarg defaults.
        kwargs.setdefault('V_chan', 1)
        kwargs.setdefault('I_chan', 0)
        kwargs.setdefault('V_clamp', True)
        kwargs.setdefault('verbose', True)
        kwargs.setdefault('tau', None)
        kwargs.setdefault('plot_tau', False)

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

                pulse_start = kwargs['tau'][0]
                fitting_range = kwargs['tau'][-2:]

                p0 = [V_copy[slice(*baseline)].mean(), V_copy[slice(*steady_state)].mean(), 10]
                p, fitted_pts = self._exponential_optimizer_wrapper(V_copy[slice(*fitting_range)], p0, self.dt)

                output['tau'] = p[2]

                if kwargs['plot_tau']:
                    plt.figure()
                    plt.plot(
                        np.arange(0, (len(V_copy) - 0.5) * self.dt, self.dt), V_copy,
                        'k-', lw=0.5
                    )
                    plt.plot(
                        np.linspace(fitting_range[0] * self.dt, fitting_range[1] * self.dt, fitted_pts.shape[1]),
                        fitted_pts[0, :],
                        'b--'
                    )
                    plt.show()

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

    def _exponential_curve(self, p, t):
        """Three parameter exponential.

        I = (A + C) * exp (-t/tau) + C

        p = [A, C, tau]
        """

        A = p[0]
        C = p[1]
        tau = p[2]

        return (A + C) * np.exp(-t/tau) + C

    def _compute_residuals(self, p, func, Y, X):
        """Compute residuals of a fitted curve.

        Inputs:
            p       -- vector of function parameters
            func    -- a callable function
            Y       -- real values
            X       -- vector of points on which to compute fitted values

        Returns:
            Array of residuals.
        """

        if len(Y) != len(X):
            raise ValueError('Y and X must be of the same length.')

        Y_hat = func(p, X)

        return Y - Y_hat

    def _exponential_optimizer_wrapper(self, I, p0, dt=0.1):

        t = np.arange(0, len(I) * dt, dt)[:len(I)]

        p = optimize.least_squares(self._compute_residuals, p0, kwargs={
        'func': self._exponential_curve,
        'X': t,
        'Y': I
        })['x']

        no_pts = 500

        fitted_points = np.empty((2, no_pts))
        fitted_points[1, :] = np.linspace(t[0], t[-1], no_pts)
        fitted_points[0, :] = self._exponential_curve(p, fitted_points[1, :])

        return p, fitted_points


