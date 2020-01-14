"""Construct complex electrophysiological stimuli.

@author: Emerson

"""
# IMPORT MODULES

import csv
import warnings
from copy import deepcopy

import numpy as np
import numba as nb
import matplotlib.pyplot as plt


# SIMULUS BASE CLASS

class BaseStimulus(object):
    """Base class for stimulus objects."""

    # Methods that must be implemented by derived classes. (Pure virtual.)
    def __init__(self):
        """Initialize Stimulus (implemented by derived classes)."""
        raise NotImplementedError(
            'Initialization must be implemented by derived stimulus classes.'
        )

    # Methods that should not be changed by derived classes.
    def __add__(self, x):
        """Add x to Stimulus object."""
        # Adding is implemented for CompoundStimulus objects,
        # so coerce self to a CompoundStimulus and add.
        tmpstimulus = CompoundStimulus(self)
        return tmpstimulus + x

    def __sub__(self, x):
        """Subtract x from Stimulus object."""
        # Subtraction is implemented for CompoundStimulus objects,
        # so coerce self to a CompoundStimulus and subtract.
        tmpstimulus = CompoundStimulus(self)
        return tmpstimulus - x

    @property
    def no_sweeps(self):
        """Number of stimulus sweeps."""
        if not hasattr(self, 'command'):
            return 0
        elif self.command.ndim == 1:
            return 1
        else:
            assert self.command.ndim == 2
            return self.command.shape[0]

    @property
    def no_timesteps(self):
        """Number of stimulus timesteps."""
        if not hasattr(self, 'command'):
            return 0
        elif self.command.ndim == 1:
            return len(self.command)
        else:
            assert self.command.ndim == 2
            return self.command.shape[1]

    @property
    def duration(self):
        """Duration of stimulus in ms."""
        if not hasattr(self, 'dt'):
            raise AttributeError(
                '`duration` not defined for un-generated stimulus.'
            )
        return self.no_timesteps * self.dt

    def replicate(self, replicates):
        """Replicate stimulus for `replicates` sweeps.

        `replicates=2` results in a two sweep stimulus. `replicates=1` does
        nothing.
        """
        if replicates <= 1:
            raise ValueError('Replicates must be int >1.')
        self.command = np.tile(self.command, (replicates, 1))

    def copy(self):
        """Return deep copy of stimulus instance."""
        return deepcopy(self)

    def simulate_response(self, R, C, E, plot=True, verbose=True):
        """Simulate response of passive membrane to stimulus.

        Model the neuronal membrane as an RC circuit.

        Inputs
        ------
            R: float
            --  Membrane resistance in MOhm.
            C: float
            -- Membrane capacitance in pF.
            E: float
            --  Membrane resting potential in mV.
            plot: bool (default True)
            --  Plot the simulated voltage response.
            verbose: bool (default True)
            --  Print information about progress.

        Returns
        -------
            [sweeps, time] array with simulated voltage response.

        """
        # Check that command to export actually exists.
        if not (hasattr(self, 'command') and hasattr(self, 'time_supp')):
            raise AttributeError('`Stimulus<type>.command` must be initialized '
                                 'by calling `Stimulus<type>.generate()` '
                                 'before it can be integrated.')

        # Data type to use for simulations.
        # Needed for numba.jit accelerated current integrator.
        dtype = np.float64

        # Put command into 2D array if it isn't already.
        if self.command.ndim == 1:
            input_ = self.command.copy()[np.newaxis, :].astype(dtype)
        else:
            input_ = self.command.copy().astype(dtype)

        # Unit conversions.
        input_ *= 1e-3
        leak_conductance = 1 / R
        C *= 1e-3
        if verbose:
            print('tau = {}ms'.format(R * C))

        # Run current integrator.
        if verbose:
            print('Integrating input...')
        output_ = np.empty_like(input_)  # Buffer for integrated voltage.
        V = self._integrate(
            input_, output_,
            dtype(leak_conductance), dtype(C), dtype(E), dtype(self.dt)
        )
        #V *= 1e3
        if verbose:
            print('Done!')

        # Plot integrated voltage.
        if plot:
            if verbose:
                print('Plotting...')
            plt.figure()

            ax = plt.subplot(211)
            plt.plot(self.time_supp, V.T, 'k-')
            plt.ylabel('Voltage (mV)')
            plt.xlabel('Time (ms)')

            plt.subplot(212, sharex=ax)
            plt.plot(self.time_supp, self.command.T, 'k-')
            plt.ylabel('Command (pA)')
            plt.xlabel('Time (ms)')

            plt.show()

            if verbose:
                print('Done!')

        return V

    def plot(self, ax=None, **pltargs):
        """Show generated stimulus."""
        # Check that command to plot actually exists.
        if not (hasattr(self, 'command') and hasattr(self, 'time_supp')):
            raise AttributeError('`Stimulus<type>.command` must be initialized '
                                 'by calling `Stimulus<type>.generate()` '
                                 'before it can be plotted.')

        # Get default mpl axes.
        if ax is None:
            ax = plt.gca()

        # Make plot.
        ax.plot(self.time_supp, self.command.T, **pltargs)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Command')

        return ax

    def export(self, fname):
        """Export generated stimulus in Axon Text Format."""
        # Check that command to export actually exists.
        if not (hasattr(self, 'command') and hasattr(self, 'time_supp')):
            raise AttributeError('`Stimulus<type>.command` must be initialized '
                                 'by calling `Stimulus<type>.generate()` '
                                 'before it can be exported.')

        # Coerce command and time support to column vectors, if applicable.
        arrs_to_export = {}
        for attr in ['command', 'time_supp']:
            if getattr(self, attr).ndim == 1:
                arrs_to_export[attr] = getattr(self, attr)[:, np.newaxis]
            else:
                arrs_to_export[attr] = getattr(self, attr).T
        # Join into a single array to write to file.
        arr_to_export = np.concatenate(
            [arrs_to_export['time_supp'], arrs_to_export['command']],
            axis=1
        )
        del arrs_to_export

        with open(fname, 'w', newline='') as f:
            # Write header.
            f.write(
                'ATF1.0\n1\t{}\nType=1\nTime (ms)\t{}\n'.format(
                    arr_to_export.shape[1] - 1,
                    'Command (AU)\t' * (arr_to_export.shape[1] - 1)
                )
            )

            # Write input.
            stimwriter = csv.writer(f, delimiter='\t')
            stimwriter.writerows(arr_to_export)

            # Close file.
            f.close()

    # Accelerated methods.

    @staticmethod
    @nb.jit(nb.float64[:, :](nb.float64[:, :], nb.float64[:, :], nb.float64,
                             nb.float64, nb.float64, nb.float64), nopython=True)
    def _integrate(input_, output_, leak_conductance, C, E, dt):
        """Integrate `input_` with a passive membrane filter.

        `output_` is a buffer to hold output voltage. Accelerated using
        `numba.jit`.

        """
        for i in range(input_.shape[0]):
            output_[0, i] = E
            for t in range(1, input_.shape[1]):
                dV = (
                    -leak_conductance * (output_[i, t - 1] - E) + input_[i, t - 1]
                ) * dt / C
                output_[i, t] = output_[i, t - 1] + dV

        return output_


# ARBITRARY ARRAY STIMULUS

class ArrayStimulus(BaseStimulus):
    """Stimulus constructed from an arbitrary array."""

    def __init__(self, command, dt=0.1, label=None):
        """Initialize ArrayStimulus."""
        self.label = label

        if issubclass(type(command), BaseStimulus):
            self.command = command.command
        elif issubclass(type(command), StimulusKernel):
            self.command = command.kernel
        else:
            self.command = np.asarray(command)

        self.dt = dt
        self.time_supp = np.arange(
            0, self.duration - 0.5 * self.dt, self.dt
        )

    def __str__(self):
        """Return string representation of ArrayStimulus."""
        return "ArrayStimulus of size {:.1f}ms x {} sweeps".format(
            self.duration, self.no_sweeps
        )


class ConvolvedStimulus(BaseStimulus):
    """Stimulus defined by a kernel convolved with a basis."""

    # Methods that must be implemented by derived classes.
    def __init__(
        self, loc, kernel, basis=None, dt=0.1, kernel_max_len=None, label=None
    ):
        """Initialize CompoundStimulus."""
        self.label = label

        self.loc = loc
        self.kernel = kernel
        self.basis = basis

        if basis is not None:
            self.generate(basis, dt, kernel_max_len)

    # Methods that should not be reimplemented by derived classes.
    def generate(self, basis, dt, kernel_max_len=None):
        """Generate stimulus vector.."""
        self.basis = ArrayStimulus(basis, dt)
        self.time_supp = np.arange(0, self.basis.duration - 0.5 * dt, dt)

        # Compute length of kernel to generate.
        if kernel_max_len is None:
            kernel_duration = self.basis.duration
        else:
            kernel_duration = min(kernel_max_len * dt, self.basis.duration)

        # Generate kernel.
        self.kernel.generate(
            duration=kernel_duration, dt=dt, front_padded=False
        )

        # Run discrete convolution.
        output = np.zeros_like(self.basis.command, dtype=np.float64)
        output = self._sparse_convolve(
            self.kernel.kernel.astype(np.float64),
            np.atleast_2d(self.basis.command).astype(np.float64),
            np.atleast_2d(output)
        )
        output += self.loc  # Add offset.

        self.command = output

    @staticmethod
    @nb.jit(
        nb.float64[:, :](nb.float64[:], nb.float64[:, :], nb.float64[:, :]),
        nopython=True
    )
    def _sparse_convolve(kernel, basis, output):
        """Discrete convolution of kernel and basis vector.

        Arguments
        ---------
        kernel : list-like
            Non-sparse stimulus kernel.
        basis : array-like
            Sparse array with which kernel is convolved.
        output : array-like
            Pre-allocated array to hold output.

        Returns
        -------
        output : array-like
            Array containing the convolved stimulus. Same shape as basis.

        """
        for i in range(basis.shape[0]):
            for j in range(basis.shape[1]):
                if basis[i, j] == 0.:
                    continue
                else:
                    end_ind = min(len(kernel), output.shape[1] - j)
                    output[i, j:j + end_ind] += basis[i, j] * kernel[:end_ind]
        return output


class StimulusKernel(object):

    def __init__(self):
        raise NotImplementedError

    @property
    def no_timesteps(self):
        """Number of kernel timesteps."""
        if not hasattr(self, 'kernel'):
            return 0
        elif self.kernel.ndim == 1:
            return len(self.kernel)
        else:
            raise NotImplementedError(
                '`no_timesteps` not implemented for non-vector kernel.'
            )

    def generate(self):
        raise NotImplementedError

    def plot(self, ax=None, **pltargs):
        # Check that kernel to plot actually exists.
        if not (hasattr(self, 'kernel') and hasattr(self, 'time_supp')):
            raise AttributeError('`<type>Kernel.kernel` must be initialized '
                                 'by calling `<type>Kernel.generate()` '
                                 'before it can be plotted.')

        # Get default mpl axes.
        if ax is None:
            ax = plt.gca()

        # Make plot.
        ax.plot(self.time_supp, self.kernel, **pltargs)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')

        return ax


class ArrayKernel(StimulusKernel):
    """Stimulus kernel constructed from an arbitrary array."""

    def __init__(self, kernel, dt=0.1, label=None):
        """Initialize ArrayKernel."""
        self.label = label

        if issubclass(type(kernel), BaseStimulus):
            self.kernel = kernel.command
        elif issubclass(type(kernel), StimulusKernel):
            self.kernel = kernel.kernel
        else:
            self.kernel = np.asarray(kernel)

        self.dt = dt
        self.duration = self.no_timesteps * self.dt
        self.time_supp = np.arange(
            0, self.duration - 0.5 * self.dt, self.dt
        )

    def __str__(self):
        """Return string representation of ArrayStimulus."""
        return "ArrayStimulus of size {:.1f}ms x {} sweeps".format(
            self.duration, self.no_sweeps
        )

    def generate(self, duration, dt, front_padded=False):
        """Return the ArrayKernel. Arguments are for compatibility only."""
        return self.kernel


class BiexponentialSynapticKernel(StimulusKernel):
    """Synaptic kernel with exponential rise and decay."""

    def __init__(
        self, size, tau_rise, tau_decay, size_method='amplitude',
        duration=None, dt=0.1, front_padded=False, label=None
    ):
        """Initialize BiexponentialSynapticKernel."""
        self.label = label

        # Store kernel parameters.
        self.size = size
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.size_method = size_method

        # Generate kernel if optional time params are given.
        if all([x is not None for x in [duration, dt]]):
            self.generate(duration, dt, front_padded)

    def __repr__(self):
        """Return repr(self)."""
        reprstr = (
            'ez.stimtools.BiexponentialSynapticKernel('
            'size={size}, '
            'tau_rise={tau_rise}, '
            'tau_decay={tau_decay}, '
            'size_method={size_method}, '
            'label={label}'
            ')'.format(
                size=self.size, tau_rise=self.tau_rise,
                tau_decay=self.tau_decay, size_method=self.size_method,
                label=self.label
            )
        )
        return reprstr

    def generate(self, duration, dt, front_padded=False):
        """Generate BiexponentialSynapticKernel vector."""
        self.time_supp = np.arange(0, duration - 0.5 * dt, dt)

        waveform = (
            np.exp(-self.time_supp / self.tau_decay)
            - np.exp(-self.time_supp / self.tau_rise)
        )
        if self.size_method == 'amplitude':
            waveform /= np.max(waveform)
            waveform *= self.size
        elif self.size_method.upper() == 'AUC':
            waveform *= self.size / (self.tau_decay - self.tau_rise)
        else:
            raise ValueError(
                'Expected `size_method` to be `amplitude` or `AUC`, got {} '
                'instead.'.format(self.size_method)
            )

        if front_padded:
            # Pad with zeros to center kernel.
            waveform = waveform[:(len(self.time_supp) // 2)]
            waveform = np.concatenate(
                [np.zeros(len(waveform) + len(self.time_supp) % 2), waveform]
            )

        self.kernel = waveform
        self.dt = dt

        return waveform


class MonoexponentialSynapticKernel(StimulusKernel):
    """Synaptic kernel with exponential decay."""

    def __init__(
        self, size, tau_decay, size_method='amplitude',
        duration=None, dt=0.1, front_padded=False, label=None
    ):
        """Initialize BiexponentialSynapticKernel."""
        self.label = label

        # Store kernel parameters.
        self.size = size
        self.tau_decay = tau_decay
        self.size_method = size_method

        # Generate kernel if optional time params are given.
        if all([x is not None for x in [duration, dt]]):
            self.generate(duration, dt, front_padded)

    def __repr__(self):
        """Return repr(self)."""
        reprstr = (
            'ez.stimtools.MonoexponentialSynapticKernel('
            'size={size}, '
            'tau_decay={tau_decay}, '
            'size_method={size_method}, '
            'label={label}'
            ')'.format(
                size=self.size, tau_rise=self.tau_rise,
                size_method=self.size_method,
                label=self.label
            )
        )
        return reprstr

    def generate(self, duration, dt, front_padded=False):
        """Generate MonoexponentialSynapticKernel vector."""
        self.time_supp = np.arange(0, duration - 0.5 * dt, dt)

        waveform = (
            np.exp(-self.time_supp / self.tau_decay)
        )
        if self.size_method == 'amplitude':
            waveform /= np.max(waveform)
            waveform *= self.size
        elif self.size_method.upper() == 'AUC':
            waveform *= self.size / self.tau_decay
        else:
            raise ValueError(
                'Expected `size_method` to be `amplitude` or `AUC`, got {} '
                'instead.'.format(self.size_method)
            )

        if front_padded:
            # Pad with zeros to center kernel.
            waveform = waveform[:(len(self.time_supp) // 2)]
            waveform = np.concatenate(
                [np.zeros(len(waveform) + len(self.time_supp) % 2), waveform]
            )

        self.kernel = waveform
        self.dt = dt

        return waveform


# SIMPLE STIMULI

class SimpleStimulus(BaseStimulus):
    """Interface template for simple stimuli."""

    def __init__(self, loc, ampli, reqd_args, duration, dt, label=None):
        """Initialize SimpleStimulus.

        Must be implemented by derived classes.

        Suggested implementation
        ------------------------
        1. Store loc, ampli, and stimulus-specific positional arguments in
           attributes.
        2. If duration and dt are not None, call generate method.

        """
        raise NotImplementedError(
            'Initialization must be implemented by derived stimulus classes.'
        )

    def generate(self, duration, dt=0.1):
        """Generate stimulus vector.

        Required call signature
        -----------------------
        SimpleStimulus.generate(duration, dt=0.1)

        """
        raise NotImplementedError(
            '`generate` must be implemented by derived stimulus classes.'
        )


class OUStimulus(SimpleStimulus):
    """Ornstein-Uhlenbeck noise stimulus."""

    def __init__(
        self, mean, amplitude, tau, ampli_modulation, mod_period,
        seed=None, duration=None, dt=0.1,
        label=None
    ):
        """Initialize OUStimulus."""
        self.label = label

        # Store stimulus parameters.
        self.mean = mean
        self.amplitude = amplitude
        self.tau = tau
        self.ampli_modulation = ampli_modulation
        self.mod_period = mod_period
        self.seed = seed

        # Generate stimulus if optional time params are given.
        if all([x is not None for x in [duration, dt]]):
            self.generate(duration, dt)

    def __repr__(self):
        """Return repr(self)."""
        reprstr = (
            'ez.stimtools.OUStimulus('
            'mean={mean}, '
            'amplitude={amplitude}, '
            'tau={tau}, '
            'ampli_modulation={ampli_modulation}, '
            'mod_period={mod_period}, '
            'seed={seed}, '
            'label={label}'
            ')'.format(
                mean=self.mean,
                amplitude=self.amplitude,
                tau=self.tau,
                ampli_modulation=self.ampli_modulation,
                mod_period=self.mod_period,
                seed=self.seed,
                label=self.label
            )
        )
        return reprstr

    def generate(self, duration, dt):
        """Generate OUStimulus vector."""
        dtype = np.float64  # Datatype to use for realizing noise.
        self.time_supp = np.arange(0, duration - 0.5 * dt, dt)

        # Precompute sinusoidal amplitude modulation.
        ampli = self.amplitude * (
            1 + self.ampli_modulation * np.sin(
                (2 * np.pi / self.mod_period) * self.time_supp
            )
        )
        # Sample noise.
        np.random.seed(self.seed)
        rands = np.random.standard_normal(len(self.time_supp))

        # Leakily integrate random walk to get Ornstein-Uhlenbeck noise.
        output_ = np.empty_like(self.time_supp).astype(dtype)
        noise = self._integrate_walk(
            output_.astype(dtype), rands.astype(dtype), ampli.astype(dtype),
            dtype(self.mean), dtype(self.tau), dtype(dt)
        )

        # Assign attributes.
        self.command = noise
        self.dt = dt

    @staticmethod
    @nb.jit(
        nb.float64[:](
            nb.float64[:], nb.float64[:], nb.float64[:],
            nb.float64, nb.float64, nb.float64
        ),
        nopython=True
    )
    def _integrate_walk(output_, rands, amplitude, mean, tau, dt):
        """Leakily integrate random walk."""
        output_[0] = mean
        for t in range(1, len(output_)):
            adaptive_term = mean - output_[t - 1]
            random_term = (
                np.sqrt(2 * amplitude[t - 1]**2 * dt / tau)
                * rands[t - 1]
            )
            doutput_ = adaptive_term * dt / tau + random_term
            output_[t] = output_[t - 1] + doutput_

        return output_


class SinStimulus(SimpleStimulus):
    """Sinusoidal stimulus."""

    def __init__(
        self, mean, amplitude, frequency,
        duration=None, dt=0.1,
        label=None
    ):
        """Initialize SinStimulus."""
        self.label = label

        # Store stimulus parameters.
        self.mean = mean
        self.amplitude = amplitude
        self.frequency = frequency

        # Generate stimulus if optional time params are given.
        if all([x is not None for x in [duration, dt]]):
            self.generate(duration, dt)

    def __repr__(self):
        """Return repr(self)."""
        reprstr = (
            'ez.stimtools.SinStimulus('
            'mean={mean}, '
            'amplitude={amplitude}, '
            'frequency={frequency}, '
            'label={label}'
            ')'.format(
                mean=self.mean,
                amplitude=self.amplitude,
                frequency=self.frequency,
                label=self.label
            )
        )
        return reprstr

    def generate(self, duration, dt):
        """Generate SinStimulus vector."""
        self.time_supp = np.arange(0, duration - 0.5 * dt, dt)

        # Generate sine wave.
        wave = self.mean + self.amplitude * np.sin(
            2 * np.pi * self.frequency * 1e-3 * self.time_supp  # Convert to Hz
        )

        # Assign attributes.
        self.command = wave
        self.dt = dt


class ChirpStimulus(SimpleStimulus):
    """Sine wave stimulus of exponentially changing frequency."""

    def __init__(
        self, mean, amplitude, initial_frequency, final_frequency,
        duration=None, dt=0.1,
        label=None
    ):
        """Initialize ChirpStimulus."""
        # Input checks.
        if np.isclose(initial_frequency, final_frequency):
            warnings.warn(
                '`initial_frequency` and `final_frequency` should '
                'not be identical. Change one of these values or use '
                '`ez.stimtools.SinStimulus` instead.'
            )

        self.label = label

        # Store stimulus parameters.
        self.mean = mean
        self.amplitude = amplitude
        self.initial_frequency = initial_frequency
        self.final_frequency = final_frequency

        # Generate stimulus if optional time params are given.
        if all([x is not None for x in [duration, dt]]):
            self.generate(duration, dt)

    def __repr__(self):
        """Return repr(self)."""
        reprstr = (
            'ez.stimtools.ChirpStimulus('
            'mean={mean}, '
            'amplitude={amplitude}, '
            'initial_frequency={initial_frequency}, '
            'final_frequency={final_frequency}, '
            'label={label}'
            ')'.format(
                mean=self.mean,
                amplitude=self.amplitude,
                initial_frequency=self.initial_frequency,
                final_frequency=self.final_frequency,
                label=self.label
            )
        )
        return reprstr

    def generate(self, duration, dt):
        """Generate ChirpStimulus vector."""
        self.time_supp = np.arange(0, duration - 0.5 * dt, dt)

        # Generate sine wave.
        freq = np.logspace(
            np.log10(self.initial_frequency),
            np.log10(self.final_frequency),
            endpoint=True,
            num=len(self.time_supp),
            base=10,
        )
        wave = self.mean + self.amplitude * np.sin(
            2 * np.pi * freq * 1e-3 * self.time_supp  # Convert freq to Hz.
        )

        # Assign attributes.
        self.command = wave
        self.dt = dt


# STEP SIMULUS
class StepStimulus(BaseStimulus):
    """Stimulus consisting of square steps."""
    def __init__(self, durations, amplitudes, dt=0.1, label=None):
        if len(durations) != len(amplitudes):
            raise ValueError(
                'Expected lengths of durations and amplitudes to be equal; '
                'got {} and {} instead.'.format(
                    len(durations), len(amplitudes)
                )
            )

        self.dt = dt
        self.durations = durations
        self.amplitudes = amplitudes

        self.label = label

        self._generate()

    def __repr__(self):
        """Return repr(self)."""
        reprstr = (
            'ez.stimtools.StepStimulus('
            'durations={durations}, '
            'amplitudes={amplitudes}, '
            'dt={dt}, '
            'label={label}, '
            ')'.format(
                durations=self.durations,
                amplitudes=self.amplitudes,
                dt=self.dt,
                label=self.label,
            )
        )

    def _generate(self):
        command_tmp = []
        for dur, ampli in zip(self.durations, self.amplitudes):
            command_tmp.append(ampli * np.ones(int(dur / self.dt)))
        self.command = np.concatenate(command_tmp, axis=-1)
        self.time_supp = np.arange(0, self.duration - 0.5 * self.dt, self.dt)



# COMPOUND STIMULI

class CompoundStimulus(BaseStimulus):
    """Stimulus constructed from sub-stimuli."""

    def __init__(self, stimulus=None, dt=0.1, label=None):
        self.label = label
        self.dt = dt

        if stimulus is None:
            self.recipe = ''
            pass
        elif issubclass(type(stimulus), BaseStimulus):
            if isinstance(stimulus, CompoundStimulus):
                self.recipe = stimulus.recipe
            else:
                self.recipe = repr(stimulus)

            # If stimulus is a single Stimulus object, just copy attributes.
            self.command = stimulus.command
            self.time_supp = stimulus.time_supp
            if stimulus.dt != dt:
                warnings.warn(
                    'stimulus.dt = {} not equal to argument dt = {}; using '
                    '{} from stimulus.'.format(stimulus.dt, dt)
                )
            self.dt = stimulus.dt
        else:
            self.recipe = 'array_like'

            # Assume stimulus is array_like.
            self.command = np.asarray(stimulus)
            self.time_supp = np.arange(
                0,
                self.duration - 0.5 * self.dt,
                self.dt
            )
            assert len(self.time_supp) == self.no_timesteps

    def __str__(self):
        """Return recipe."""
        return self.recipe

    def __add__(self, x):
        """Add CompoundStimulus and x."""
        # Method to add two Stimulus objects together.
        if issubclass(type(x), BaseStimulus):
            # Check that command arrays have compatible shapes.
            if not (hasattr(self, 'command') and hasattr(x, 'command')):
                raise AttributeError('`command` must be initialized by '
                                     'calling `generate` before Stimulus '
                                     'objects can be added.')
            if self.no_timesteps != x.no_timesteps:
                raise ValueError('`command` arrays must have same number of '
                                 'time steps.')

            # Check that time attributes match.
            for attr_ in ['duration', 'dt']:
                if not np.isclose(getattr(self, attr_), getattr(x, attr_)):
                    raise ValueError(
                        'Stimulus {} must be equal.'.format(attr_)
                    )

            # Get recipe for x.
            if isinstance(x, CompoundStimulus):
                x_recipe = x.recipe
            else:
                x_recipe = repr(x)

            # Add command arrays.
            newcommand = self.command + x.command

        # Method to add Stimulus to array_like object.
        else:
            x_recipe = 'array_like'
            newcommand = self.command + np.asarray(x)

        newstimulus = CompoundStimulus()
        newstimulus.recipe = '\n+ '.join([self.recipe, x_recipe])
        newstimulus.command = newcommand
        newstimulus.time_supp = self.time_supp
        newstimulus.dt = self.dt

        return newstimulus

    def __sub__(self, x):
        """Subtract x from CompoundStimulus."""
        # Method to add two Stimulus objects together.
        if issubclass(type(x), BaseStimulus):
            # Check that command arrays have compatible shapes.
            if not (hasattr(self, 'command') and hasattr(x, 'command')):
                raise AttributeError('`command` must be initialized by '
                                     'calling `generate` before Stimulus '
                                     'objects can be subtracted.')
            if self.no_timesteps != x.no_timesteps:
                raise ValueError('`command` arrays must have same number of '
                                 'time steps.')

            # Check that time attributes match.
            for attr_ in ['duration', 'dt']:
                if not np.isclose(getattr(self, attr_), getattr(x, attr_)):
                    raise ValueError(
                        'Stimulus {} must be equal.'.format(attr_)
                    )

            # Get recipe for x.
            if isinstance(x, CompoundStimulus):
                x_recipe = x.recipe
            else:
                x_recipe = repr(x)

            # Add command arrays.
            newcommand = self.command - x.command

        # Method to add Stimulus to array_like object.
        else:
            x_recipe = 'array_like'
            newcommand = self.command - np.asarray(x)

        newstimulus = CompoundStimulus()
        newstimulus.recipe = '\n- '.join([self.recipe, x_recipe])
        newstimulus.command = newcommand
        newstimulus.time_supp = self.time_supp
        newstimulus.dt = self.dt

        return newstimulus


def concatenate(stimuli, dt='auto'):
    """Join Stimulus objects together end to end."""
    # Infer dt.
    dts = [
        stimulus.dt for stimulus in stimuli
        if issubclass(type(stimulus), BaseStimulus)
    ]
    if len(dts) == 0:
        if dt == 'auto':
            raise ValueError(
                'stimuli must contain at least one Stimulus object or '
                'dt must be specified.'
            )
        else:
            pass
    elif dt == 'auto' and not all([dt_ == dts[0] for dt_ in dts]):
        raise ValueError('stimulus.dt is not equal for all stimuli.')
    elif dt != 'auto' and not all([dt_ == dt for dt_ in dts]):
        raise ValueError(
            'stimulus.dt is not equal to argument dt for all stimuli.'
        )
    else:
        # Successful auto dt.
        dt = dts[0]
        del dts

    # Concatenate stimuli together.
    ingredients = []  # List to hold recipe of each stimulus in stimuli.
    if isinstance(stimuli[0], CompoundStimulus):  # Initialize result.
        result = stimuli[0].copy()
    else:
        result = CompoundStimulus(deepcopy(stimuli[0]), dt=dt)
    # Concatenate each additional stimulus onto result.
    for stimulus in stimuli[1:]:
        # Convert stimulus to CompoundStimulus if necessary.
        if not isinstance(stimulus, CompoundStimulus):
            stimulus = CompoundStimulus(stimulus, dt=dt)

        # Concatenate command arrays.
        if result.command.ndim == 1:
            result.command = np.concatenate(
                [result.command, stimulus.command]
            )
        else:
            assert result.command.ndim == 2
            result.command = np.concatenate(
                [result.command,
                 np.broadcast_to(stimulus.command, result.command.shape)]
            )

        # Store recipe ingredient.
        ingredients.append(stimulus.recipe)

    # Update duration and support vector.
    result.time_supp = np.arange(
        0, result.duration - result.dt * 0.5, result.dt
    )

    # Store result recipe.
    result.recipe = '\n'.join(
        ['concatenate(', '\t' + ',\n\t'.join(ingredients), ')']
    )

    return result
