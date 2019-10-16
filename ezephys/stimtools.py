"""Construct complex electrophysiological stimuli.

@author: Emerson

"""

__all__ = [
    'BaseStimulus', 'SynapticStimulus', 'OUStimulus', 'SinStimulus',
    'ChirpStimulus', 'CompoundStimulus', 'concatenate'
]

# IMPORT MODULES

import csv
import warnings
from copy import deepcopy

import numpy as np
import numba as nb
import matplotlib.pyplot as plt


# DEFINE SIMULUS PARENT CLASS

class BaseStimulus(object):
    """Base class for stimulus objects."""

    def __init__(self):
        """Initialize BaseStimulus (does nothing)."""
        pass

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
        leak_conductance = 1/R
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
                    -leak_conductance * (output_[i, t-1] - E) + input_[i, t-1]
                ) * dt / C
                output_[i, t] = output_[i, t-1] + dV

        return output_


# DEFINE STIMULUS SUBCLASSES

class SynapticStimulus(BaseStimulus):
    """Biexponential synaptic stimulus."""

    def __init__(
        self, amplitude, tau_rise, tau_decay,
        start_time=None, duration=None, dt=0.1,
        label=None
    ):
        """Initialize SynapticStimulus."""
        self.label = label

        # Store stimulus parameters.
        self.amplitude = amplitude
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay

        # Generate stimulus if optional time params are given.
        if all([x is not None for x in [start_time, duration, dt]]):
            self.generate(start_time, duration, dt)

    def __repr__(self):
        """Return repr(self)."""
        reprstr = (
            'ez.stimtools.SynapticStimulus('
            'amplitude={amplitude}, '
            'tau_rise={tau_rise}, '
            'tau_decay={tau_decay}, '
            'label={label}'
            ')'.format(
                amplitude=self.amplitude, tau_rise=self.tau_rise,
                tau_decay=self.tau_decay, label=self.label
            )
        )
        return reprstr

    def generate(self, start_time, duration, dt):
        """Generate SynapticStimulus vector."""
        self.time_supp = np.arange(0, duration - 0.5 * dt, dt)

        # Generate waveform based on time constants then normalize amplitude.
        waveform = (
            np.exp(-self.time_supp / self.tau_decay)
            - np.exp(-self.time_supp / self.tau_rise)
        )
        waveform /= np.max(waveform)
        waveform *= self.amplitude

        # Pad with zeros for convolution.
        waveform = waveform[:(len(self.time_supp) // 2)]
        waveform = np.concatenate(
            [np.zeros(len(waveform) + len(self.time_supp) % 2), waveform]
        )
        assert len(waveform) == len(self.time_supp)

        # Convolve waveform with indicator vector for onset times.
        indicator = np.zeros_like(self.time_supp)
        indicator[int(start_time/dt)] = 1
        convolved = np.convolve(waveform, indicator, mode='same')

        # Assign attributes.
        self.command = convolved
        self.start_time = start_time
        self.dt = dt


class OUStimulus(BaseStimulus):
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
            adaptive_term = mean - output_[t-1]
            random_term = (
                np.sqrt(2 * amplitude[t-1]**2 * dt / tau)
                * rands[t-1]
            )
            doutput_ = adaptive_term * dt / tau + random_term
            output_[t] = output_[t-1] + doutput_

        return output_


class SinStimulus(BaseStimulus):
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


class ChirpStimulus(BaseStimulus):
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
