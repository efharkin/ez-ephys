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
    """Abstract base class for stimulus objects.

    Attributes
    ----------
    no_sweeps: int
        Number of sweeps in stimulus.
    no_timesteps: int
        Number of timesteps in one stimulus sweep.
    duration: float
        Duration of one stimulus sweep in ms.
    dt: float
        Timestep of stimulus in ms.
    label: str
        Descriptive label.

    Methods
    -------
    generate
        Generate a discrete representation of the stimulus. Useful for
        visualization and export.
    replicate(replicates)
        Add sweeps to the stimulus by replicating a single sweep.
    copy
    simulate_response(R, C, E, plot, verbose)
        Simulate the voltage response of a neuron to the stimulus.
    plot
        Create a matplotlib plot to visualize the stimulus.
    export
        Export the stimulus for use in ephys experiments.

    """

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

        Useful to see how a neuron recorded in current clamp would be expected
        to respond to the stimulus.

        Models the neuronal membrane as an RC circuit.

        Inputs
        ------
        R: float
            Membrane resistance in MOhm.
        C: float
            Membrane capacitance in pF.
        E: float
            Membrane resting potential in mV.
        plot: bool (default True)
            Plot the simulated voltage response.
        verbose: bool (default True)
            Print information about progress.

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
        """Export generated stimulus in Axon text format.

        Compatible with Axon Instruments software.

        Arguments
        ---------
        fname: str
            Name of file to which stimulus will be written.

        """
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
    """Stimulus constructed from an arbitrary array.

    See documentation of BaseStimulus for more information about Stimulus
    objects.

    """

    def __init__(self, command, dt=0.1, label=None):
        """Initialize ArrayStimulus.

        Arguments
        ---------
        command: 1D or 2D array
            Stimulus waveform.
        dt: float, default 0.1
            Timestep in ms.
        label: str
            Descriprive label for stimulus instance.

        """
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
    """Stimulus defined by a kernel convolved with a basis.

    See documentation of BaseStimulus for more information about Stimulus
    objects.

    """

    # Methods that must be implemented by derived classes.
    def __init__(
        self, loc, kernel, basis=None, dt=0.1, kernel_max_len=None, label=None
    ):
        """Initialize ConvolvedStimulus.

        Arguments
        ---------
        loc: float
            Constant offset to apply to generated stimulus. Equivalent to
            adding a constant to the generated stimulus.
        kernel: StimulusKernel
            Kernel to convolve with basis. Example: for a train of synaptic-
            like currents, you could use a BiexponentialSynapticKernel to
            specify the shape of a unitary synaptic current.
        basis: 1D array, Stimulus, or StimulusKernel
            Basis with which to convolve kernel. Stimulus command will have the
            same shape as `basis`. Example: for a 100 timestep train of
            synaptic-like currents, this would be a vector of length 100 that
            is 1.0 when a synaptic-like current begins and 0.0 otherwise.
            (Note: for generating synaptic trains, consider using
            PoissonProcess for a simpler interface.)
        dt: float, default 0.1
            Timestep of stimulus in ms.
        kernel_max_len: int or None
            Length at which to truncate the discrete StimulusKernel before
            convolution. Use only if you are using a basis with many timesteps
            and the convolution takes too long to compute.
        label: str
            Descriptive label for stimulus instance.

        """
        self.label = label
        self.dt = dt

        self.loc = loc
        self.kernel = kernel
        self.basis = basis

        if basis is not None:
            self.generate(basis, dt, kernel_max_len)

    def __repr__(self):
        """Return repr(self)."""
        reprstr = (
            'ez.stimtools.ConvolvedStimulus('
            'loc={loc}, '
            'kernel={kernel}, '
            'basis={basis}, '
            'dt={dt}, '
            'label={label}'
            ')'.format(
                loc=self.loc,
                kernel=repr(self.kernel),
                basis=repr(self.basis),
                dt=self.dt,
                label=self.label
            )
        )
        return reprstr

    # Methods that should not be reimplemented by derived classes.
    def generate(self, basis, dt, kernel_max_len=None):
        """Generate stimulus vector by convolving basis with attached kernel.

        Uses an efficient algorithm for sparse convolution. For convolution
        with a sparse signal with a constant offset (e.g., a synaptic train
        with a baseline offset), it is more efficient to specify the offset
        using the `loc` parameter at ConvolvedStimulus initialization than to
        add the offset to `basis`.

        Arguments
        ---------
        basis: 1D array, Stimulus, or StimulusKernel
            Basis with which to convolve kernel. Stimulus command will have the
            same shape as `basis`. Example: for a 100 timestep train of
            synaptic-like currents, this would be a vector of length 100 that
            is 1.0 when a synaptic-like current begins and 0.0 otherwise.
            (Note: for generating synaptic trains, consider using
            PoissonProcess for a simpler interface.)
        dt: float, default 0.1
            Timestep of the stimulus in ms.
        kernel_max_len: int or None
            Length at which to truncate the discrete StimulusKernel before
            convolution. Use only if you are using a basis with many timesteps
            and the convolution takes too long to compute.

        Result
        ------
        Initializes `command` attribute with result of discrete convolution
        of `kernel` attribute with `basis`.

        Side-effects
        ------------
        Initializes `basis` and `time_supp` attributes. Calls `generate` on
        attached `kernel`.

        """
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


# KERNELS

class StimulusKernel(object):
    """Abstract base class of StimulusKernels for use by ConvolvedStimulus.

    Similar to Stimulus objects, but uses a `kernel` rather than `command`
    attribute to store the vector representation of the object.

    """

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
    """Stimulus kernel constructed from an arbitrary array.

    Analogous to ArrayStimulus.

    See StimulusKernel documentation for for more information about kernels.

    """

    def __init__(self, kernel, dt=0.1, label=None):
        """Initialize ArrayKernel.

        Arguments
        ---------
        kernel: 1D array
            Array to use for kernel.
        dt: float, default 0.1
            Timestep of kernel in ms.
        label: str
            Descriptive label.

        """
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
        """Return string representation of ArrayKernel."""
        return "ArrayKernel of duration {:.1f}ms".format(self.duration)

    def generate(self, duration, dt, front_padded=False):
        """Return the ArrayKernel.

        Arguments are only for compatibility with other StimulusKernels and are
        ignored.

        """
        return self.kernel


class BiexponentialSynapticKernel(StimulusKernel):
    """Synaptic kernel with exponential rise and decay.

    See StimulusKernel documentation for more information about kernels.

    """

    def __init__(
        self, size, tau_rise, tau_decay, size_method='amplitude',
        duration=None, dt=0.1, front_padded=False, label=None
    ):
        """Initialize BiexponentialSynapticKernel.

        Arguments
        ---------
        size: float
            Size of waveform. Usually amplitude, but see `size_method`.
        tau_rise, tau_decay: float
            Time constants of mono-exponential rise and decay of the synaptic-
            like current in ms.
        size_method: str, `amplitude` or `AUC`
            Interpret `size` argument as amplitude or area under the curve
            (AUC).
        duration: float
            Duration of stimulus kernel in ms.
        dt: float
            Timestep of kernel in ms.
        front_padded: bool, default False
            Zero-pad the front of the kernel so the onset of the synaptic
            waveform is centered.
        label: str
            Descriptive label.

        """
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
        """Generate BiexponentialSynapticKernel vector.

        Arguments
        ---------
        duration: float
            Duration of stimulus kernel in ms.
        dt: float
            Timestep of kernel in ms.
        front_padded: bool, default False
            Zero-pad the front of the kernel so the onset of the synaptic
            waveform is centered.

        Result
        ------
        Initializes `kernel` attribute with a biexponential synaptic current-
        like waveform.

        Side-effects
        ------------
        Initializes `time_supp` and `dt` attributes.

        """
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
    """Synaptic kernel with exponential decay.

    See StimulusKernel documentation for more information about kernels.

    """

    def __init__(
        self, size, tau_decay, size_method='amplitude',
        duration=None, dt=0.1, front_padded=False, label=None
    ):
        """Initialize BiexponentialSynapticKernel.

        Arguments
        ---------
        size: float
            Size of waveform. Usually amplitude, but see `size_method`.
        tau_decay: float
            Time constants of mono-exponential decay of the synaptic-like
            current in ms.
        size_method: str, `amplitude` or `AUC`
            Interpret `size` argument as amplitude or area under the curve
            (AUC).
        duration: float
            Duration of stimulus kernel in ms.
        dt: float
            Timestep of kernel in ms.
        front_padded: bool, default False
            Zero-pad the front of the kernel so the onset of the synaptic
            waveform is centered.
        label: str
            Descriptive label.

        """
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
                size=self.size, tau_decay=self.tau_decay,
                size_method=self.size_method,
                label=self.label
            )
        )
        return reprstr

    def generate(self, duration, dt, front_padded=False):
        """Generate MonoexponentialSynapticKernel vector.

        Arguments
        ---------
        duration: float
            Duration of stimulus kernel in ms.
        dt: float
            Timestep of kernel in ms.
        front_padded: bool, default False
            Zero-pad the front of the kernel so the onset of the synaptic
            waveform is centered.

        Result
        ------
        Initializes `kernel` attribute with an exponential synaptic current-
        like waveform.

        Side-effects
        ------------
        Initializes `time_supp` and `dt` attributes.

        """
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

# POISSON PROCESS
class PoissonProcess(object):
    #TODO: Inherit from some kind of process object?
    def __init__(self, rate, dt=0.1, label=None):
        """Initialize PoissonProcess.

        Arguments
        ---------
        rate: float 1D array or Stimulus
            Rate of Poisson process in Hz.
        dt: float, default 0.1
            Timestep in ms.
        label: str
            Descriptive label.

        """
        self.dt = dt
        self.label = label
        self.rate = self._coerce_to_stimulus(rate)

    def _coerce_to_stimulus(self, x):
        if issubclass(type(x), BaseStimulus):
            if not np.isclose(x.dt, self.dt):
                raise ValueError(
                    'Expected instance dt={} and argument dt={} '
                    'to be equal.'.format(self.dt, x.dt)
                )
            else:
                return x
        else:
            return ArrayStimulus(x, self.dt, 'Rate of Poisson process.')

    def sample(self, no_samples='auto'):
        """Sample from inhomogenous Poisson process.

        Arguments
        ---------
        no_samples : int
            Number of samples to draw from Poisson process.

        Returns
        -------
        Array with same shape as instance rate containing samples from point
        process.

        """
        rate_in_mHz = 1e-3 * self.rate.command  # Must convert rate in Hz (s^-1) to mHz (ms^-1)
        event_probability = 1.0 - np.exp(-rate_in_mHz * self.dt)
        if no_samples == 'auto':
            samples = (
                np.random.uniform(0.0, 1.0, size=event_probability.shape)
                < event_probability
            )
        elif self.no_sweeps == 1:
            samples = (
                np.random.uniform(0.0, 1.0, size=(no_samples, len(event_probability)))
                < event_probability
            )
        else:
            raise ValueError(
                'Argument `no_samples={}` must be `auto` because '
                'instance `no_sweeps={}` is not one.'.format(
                    no_samples,
                    self.no_sweeps
                )
            )
        return samples.astype(np.int8)

    @property
    def no_sweeps(self):
        return self.rate.no_sweeps

    @property
    def no_timesteps(self):
        return self.rate.no_timesteps

    @property
    def duration(self):
        return self.rate.duration

# SIMPLE STIMULI

class SimpleStimulus(BaseStimulus):
    """Abstract base class for simple stimuli."""

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
    """Ornstein-Uhlenbeck noise stimulus.

    Ornstein-Uhlenbeck (OU) noise is the limiting case of synaptic noise for
    an infinitely large population of synapses with instantaneous rise and the
    same monoexponential decay.

    """

    def __init__(
        self, mean, amplitude, tau, ampli_modulation, mod_period,
        seed=None, duration=None, dt=0.1,
        label=None
    ):
        """Initialize OUStimulus.

        Arguments
        ---------
        mean: float
            Mean of the generative process. Close to the mean of the generated
            signal for very long signals.
        amplitude: float
            Scale of fluctuations.
        tau: float
            Time constant of the OU process.
        ampli_modulation: float >= 0.0
            Fractional amplitude modulation. Set to zero for no amplitude
            modulation.
        mod_period: float > 0.0
            Period of amplitude modulation in ms. Can be zero if
            `ampli_modulation` is also zero.
        seed: int
            Seed value for pseudorandom number generator.
        duration: float
            Duration of OU noise to realize in ms.
        dt: float, default 0.1
            Timestep of signal in ms.
        label: str
            Descriptive label.

        """
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
        """Generate OUStimulus vector.

        Arguments
        ---------
        duration: float
            Duration of signal to realize in ms.
        dt: float
            Timestep of signal to realize in ms.

        Result
        ------
        Initialize `command` attribute with OU noise.

        Side-effects
        ------------
        Initialize `time_supp` and `dt` attributes. Re-seed pseudorandom number
        generator.

        """
        dtype = np.float64  # Datatype to use for realizing noise.
        self.time_supp = np.arange(0, duration - 0.5 * dt, dt)

        # Precompute sinusoidal amplitude modulation.
        if self.ampli_modulation != 0.:
            ampli = self.amplitude * (
                1 + self.ampli_modulation * np.sin(
                    (2 * np.pi / self.mod_period) * self.time_supp
                )
            )
        else:
            ampli = self.amplitude * np.ones_like(self.time_supp)

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
        """Initialize SinStimulus.

        Arguments
        ---------
        mean: float
            Mean of oscillations.
        amplitude: float
            Amplitude of oscillations.
        frequency: float
            Frequency of oscillations in Hz.
        duration: float
            Duration of stimulus in ms.
        dt: float, default 0.1
            Timestep of stimulus in ms.
        label: str
            Descriptive label.

        """
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
        """Generate SinStimulus vector.

        Arguments
        ---------
        duration: float
            Duration of stimulus in ms.
        dt: float
            Timestep of stimulus in ms.

        Result
        ------
        Initializes `command` attribute with sinusoidal oscillations.

        Side-effects
        ------------
        Initializes `time_supp` and `dt` attributes.

        """
        self.time_supp = np.arange(0, duration - 0.5 * dt, dt)

        # Generate sine wave.
        wave = self.mean + self.amplitude * np.sin(
            2 * np.pi * self.frequency * 1e-3 * self.time_supp  # Convert to Hz
        )

        # Assign attributes.
        self.command = wave
        self.dt = dt


class CosStimulus(SimpleStimulus):
    """Cosinusoidal stimulus."""

    def __init__(
        self, mean, amplitude, frequency,
        duration=None, dt=0.1,
        label=None
    ):
        """Initialize CosStimulus.

        Arguments
        ---------
        mean: float
            Mean of oscillations.
        amplitude: float
            Amplitude of oscillations.
        frequency: float
            Frequency of oscillations in Hz.
        duration: float
            Duration of stimulus in ms.
        dt: float, default 0.1
            Timestep of stimulus in ms.
        label: str
            Descriptive label.

        """
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
            'ez.stimtools.CosStimulus('
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
        """Generate CosStimulus vector.

        Arguments
        ---------
        duration: float
            Duration of stimulus in ms.
        dt: float
            Timestep of stimulus in ms.

        Result
        ------
        Initializes `command` attribute with cosinusoidal oscillations.

        Side-effects
        ------------
        Initializes `time_supp` and `dt` attributes.

        """
        self.time_supp = np.arange(0, duration - 0.5 * dt, dt)

        # Generate cosine wave.
        wave = self.mean + self.amplitude * np.cos(
            2 * np.pi * self.frequency * 1e-3 * self.time_supp  # Convert to Hz
        )

        # Assign attributes.
        self.command = wave
        self.dt = dt


class SquareWaveStimulus(SimpleStimulus):
    """Square wave stimulus."""

    def __init__(
        self, mean, amplitude, frequency,
        duration=None, dt=0.1,
        label=None
    ):
        """Initialize SquareWaveStimulus.

        Arguments
        ---------
        mean: float
            Mean of oscillations.
        amplitude: float
            Amplitude of oscillations.
        frequency: float
            Frequency of oscillations in Hz.
        duration: float
            Duration of stimulus in ms.
        dt: float, default 0.1
            Timestep of stimulus in ms.
        label: str
            Descriptive label.

        """
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
            'ez.stimtools.SquareWaveStimulus('
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
        """Generate SquareWaveStimulus vector.

        Arguments
        ---------
        duration: float
            Duration of stimulus in ms.
        dt: float
            Timestep of stimulus in ms.

        Result
        ------
        Initializes `command` attribute with cosinusoidal oscillations.

        Side-effects
        ------------
        Initializes `time_supp` and `dt` attributes.

        """
        self.time_supp = np.arange(0, duration - 0.5 * dt, dt)

        period = 1e3 / self.frequency
        num_full_alternations = int(len(self.time_supp) // (period / dt))
        trailing_duration = duration % period

        # Implementation note: square wave will be constructed with
        # StepStimulus. The duration of each step will be one half-wave.
        durations = [period / 2. for i in range(2 * num_full_alternations)]
        if trailing_duration < period / 2.:
            durations.append(trailing_duration)
        else:
            durations.append(period / 2.)
            durations.append(trailing_duration % (period / 2.))

        amplitudes = [self.amplitude, -self.amplitude] * num_full_alternations
        amplitudes.append(self.amplitude)  # For trailing_duration.
        if trailing_duration >= period / 2.:
            amplitudes.append(-self.amplitude)

        wave = StepStimulus(durations, amplitudes, dt=dt).command + self.mean

        assert len(wave) == len(self.time_supp), "Length of wave {} does not match time_supp {}".format(len(wave), len(self.time_supp))

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
        """Initialize ChirpStimulus.

        Arguments
        ---------
        mean: float
            Mean of oscillations.
        amplitude: float
            Amplitude of oscillations.
        initial_frequency, final_frequency: float > 0.0
            Initial and final frequencies of oscillations in Hz.
        duration: float
            Duration of oscillations in ms.
        dt: float
            Timestep of stimulus in ms.
        label: str
            Descriptive label.

        """
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
        """Generate ChirpStimulus vector.

        Arguments
        ---------
        duration: float
            Duration of stimulus in ms.
        dt: float
            Timestep of stimulus in ms.

        Result
        ------
        Initializes `command` attribute with chirp.

        Side-effects
        ------------
        Initializes `time_supp` and `dt` attributes.

        """
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
    """Stimulus consisting of square steps.

    See documentation of BaseStimulus for more information about stimulus
    objects.

    """
    def __init__(self, durations, amplitudes, dt=0.1, label=None):
        """Initialize StepStimulus.

        Arguments
        ---------
        durations: float 1D array
            Duration of each step. Must be of same length as `amplitudes`.
        amplitudes: float 1D array
            Amplitude of each step. Must be of same length as `durations`.
        dt: float, default 0.1
            Timestep of signal.
        label:
            Descriptive label.

        """
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
        return reprstr

    def _generate(self):
        command_tmp = []
        for dur, ampli in zip(self.durations, self.amplitudes):
            command_tmp.append(ampli * np.ones(int(dur / self.dt)))
        self.command = np.concatenate(command_tmp, axis=-1)
        self.time_supp = np.arange(0, self.duration - 0.5 * self.dt, self.dt)



# COMPOUND STIMULI

class CompoundStimulus(BaseStimulus):
    """Stimulus constructed from other stimuli.

    Generally the result of adding or concatenating other Stimulus objects.
    Not intended for direct use.

    See documentation of BaseStimulus for more information about Stimulus
    objects.

    Attributes
    ----------
    recipe: str
        Recipe for generating CompoundStimulus from its parts. Not necessarily
        callable.
    command, time_supp, no_sweeps, no_timesteps, duration, dt, label
        Same as BaseStimulus.

    """

    def __init__(self, stimulus=None, dt=0.1, label=None):
        """Initialize CompoundStimulus.

        Arguments
        ---------
        stimulus: Stimulus or array
            Stimulus to coerce to a CompoundStimulus. Copies command, recipe,
            time_supp, and dt if possible.
        dt: float, default 0.1
            Timestep of stimulus. Can be overridden by `stimulus`.
        label: str
            Descriptive label.

        """
        self.label = label
        self.dt = dt

        if stimulus is None:
            self.recipe = ''
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
                    'dt from stimulus.'.format(stimulus.dt, dt)
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
    """Join Stimulus objects and/or arrays together end to end.

    Arguments
    ---------
    stimuli: list of Stimulus objects and/or 1D or 2D array-like
        Stimuli to concatenate.
    dt: float or `auto`
        Timestep of stimulus to be returned. Use `auto` to try to infer
        timestep from arguments.

    Returns
    -------
    CompoundStimulus containing stimuli joined end to end.

    """
    # Infer dt.
    #TODO: extract into private method.
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
    if isinstance(stimuli[0], CompoundStimulus):  # Initialize result.
        result = stimuli[0].copy()
    else:
        result = CompoundStimulus(deepcopy(stimuli[0]), dt=dt)
    ingredients = [result.recipe]  # To hold recipies of all stimuli.

    # Concatenate each additional stimulus onto result
    # and build recipe from ingredients.
    for stimulus in stimuli[1:]:
        if not isinstance(stimulus, CompoundStimulus):
            stimulus = CompoundStimulus(stimulus, dt=dt)
        result.command = _concatenate_along_columns(
            result.command,
            stimulus.command
        )
        ingredients.append(stimulus.recipe)

    # Update support vector.
    result.time_supp = np.arange(
        0, result.duration - result.dt * 0.5, result.dt
    )

    # Store result recipe.
    concatenate_arg_str = ',\n'.join(ingredients)
    concatenate_arg_str_indented = '\t' + concatenate_arg_str.replace(
        '\n',
        '\n\t'
    )
    result.recipe = '\n'.join(
        ['concatenate(', concatenate_arg_str_indented, ')']
    )

    return result


def _concatenate_along_columns(a, b):
    """Concatenate along columns, broadcasting if necessary.

    Treat a and b as row vectors or matrices and concatenate along column axis.
    Raise a ValueError if either a or b has more than two dimensions.

    """
    # Check whether a and b are both 1D. If so, output should still be 1D.
    if (np.ndim(a) == 1) and (np.ndim(b) == 1):
        output_as_1D = True
    else:
        output_as_1D = False

    # Concatenation is easier if both a and b have same number of dimensions,
    # so coerce to matrices first.
    a = np.atleast_2d(np.copy(a))
    b = np.atleast_2d(np.copy(b))

    # This function is not valid if a or b is a multidimensional (>2D) array,
    # so raise a ValueError if wrong input is supplied.
    if (a.ndim > 2) or (b.ndim > 2):
        raise ValueError(
            'Expected arguments to be of dimensionality at most 2, got {} '
            'and {} instead.'.format(np.ndim(a), np.ndim(b))
        )

    # Do concatenation.
    if a.shape[0] == b.shape[0]:
        # Shapes match: no broadcasting required.
        result = np.concatenate([a, b], axis=1)
    elif a.shape[0] == 1:
        # Broadcast a.
        result = np.concatenate([np.tile(a, (b.shape[0], 1)), b], axis=1)
    elif b.shape[0] == 1:
        # Broadcast b.
        result = np.concatenate([a, np.tile(b, (a.shape[0], 1))], axis=1)
    else:
        # Shapes do not match and cannot broadcast since neither a nor b is a
        # row vector.
        raise ValueError(
            'Matrices cannot be concatenated with incompatible number of '
            'rows: {} and {}.'.format(a.shape[0], b.shape[0])
        )

    # Coerce output to vector if both a and b were vectors.
    if output_as_1D:
        result = result.flatten()

    return result

