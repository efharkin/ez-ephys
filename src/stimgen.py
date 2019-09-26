"""
STIMULUS GENERATOR

Created on Tue Sep  5 10:42:19 2017

@author: Emerson


Class with built-in methods for generating commonly used stimuli and writing
them to ATF files for use with AxonInstruments hardware.

Example usage:

    # Initialize the class and simulate a synaptic current.
    s = Stim('Slow EPSC')
    s.generate_PS(duration = 200, ampli = 10, tau_rise = 1.5, tau_decay = 15)

    # Display some information about the generated waveform.
    print(s)
    s.plot()

    # Create a set of synaptic-like currents of increasing amplitude.
    s.set_replicates(5)
    s.command *= np.arange(1, 6)
    s.plot()

    # Write the stimulus to an ATF file.
    s.write_ATF()
"""




#%% IMPORT PREREQUISITE MODULES

import numpy as np
import types
import numba as nb
import matplotlib.pyplot as plt




#%% DEFINE MAIN STIM CLASS

class Stim(object):

    """
    Class with built-in methods for generating commonly used stimuli and writing them to ATF files for use with AxonInstruments hardware.

    Attributes:

        label           -- string descriptor of the class instance.
        stim_type       -- string descriptor of the type of stimulus.
        dt              -- size of the time step in ms.
        command         -- 2D array containing stimuli; time across rows, sweeps across cols.
        time            -- time support vector.
        stim_params     -- object containing attributes for each stim parameter for the current stim_type.


    Methods:

        generate_PS         -- generate a synaptic current/potential-like waveform, with total amplitude defined.
        generate_PS_bycharge-- generates a synaptic current/potential-like waveform, with total charge defined.
        generate_OU         -- generate Ornstein-Uhlenbeck noise.
        set_replicates      -- set the number of replicates of the stimulus.
        plot                -- plot the stimulus.
        write_ATF           -- write the stimulus to an ATF file.


    Example usage:

        # Initialize the class and simulate a synaptic current.
        s = Stim('Slow EPSC')
        s.generate_PS(duration = 200, ampli = 10, tau_rise = 1.5, tau_decay = 15)

        # Display some information about the generated waveform.
        print(s)
        s.plot()

        # Create a set of synaptic-like currents of increasing amplitude.
        s.set_replicates(5)
        s.command *= np.arange(1, 6)
        s.plot()

        # Write the stimulus to an ATF file.
        s.write_ATF()
    """


    ### MAGIC METHODS

    # Initialize class instance.
    def __init__(self, label, dt=0.1):

        """Initialize self."""

        self.label      = label
        self.stim_type  = 'Empty'

        self.dt         = dt        # Sampling interval in ms.

        self.command    = None      # Attribute to hold the command (only current is currently supported).
        self.time       = None      # Attribute to hold a time support vector.
        self.stim_params = None     # Attribute to hold stim parameters for given stim_type


    # Method for unambiguous representation of Stim instance.
    def __repr__(self):

        """Return repr(self)."""

        if self.time is not None:
            time_range = '[{}, {}]'.format(self.time[0], self.time[-1])
            command_str   = np.array2string(self.command)
        else:
            time_range  = str(self.time)
            command_str   = str(self.command)

        output_ls = [
            'Stim object\n\nLabel: ', self.label, '\nStim type: ',
            self.stim_type, '\nTime range (ms): ', time_range,
            '\nTime step (ms):', str(self.dt), '\nStim Parameters',
            vars(self.stim_params), '\nCommand:\n', command_str
            ]

        return ''.join(output_ls)

    # Pretty print self.command and some important details.
    # (Called by print().)
    def __str__(self):

        """
        Return str(self).
        """

        # Include more details about the object if it isn't empty.
        if self.command is not None:

            header = '{} Stim object with {} sweeps of {}s each.\n\n'.format(
                self.stim_type,
                self.command.shape[1],
                (self.time[-1] + self.dt) * self.dt / 1000
                )

            content = np.array2string(self.command)

            footer_ls = ['Stim parameters are: ']

            for key, value in vars(self.stim_params).items():
                keyval_str = '\n\t{}: {}'.format(key, value)
                footer_ls.append(keyval_str)

            footer_ls.append('\n\n')
            footer = ''.join(footer_ls)

        else:

            header = '{} Stim object.'.format(self.stim_type)
            content = ''
            footer = ''

        output_ls = [str(self.label), '\n\n', header, footer, content]

        return ''.join(output_ls)



    ### MAIN METHODS

    # Generate a synaptic current-like waveform with defined amplitude
    def generate_PS(self, duration, ampli, tau_rise, tau_decay):

        """
        Generate a post-synaptic potential/current-like waveform.

        Note that the rise and decay time constants are only good approximations of fitted rise/decay taus (which are more experimentally relevant) if the provided values are separated by at least approx. half an order of magnitude.

        Inputs:
            duration          -- length of the simulated waveform in ms ^ -1.
            ampli             -- peak height of the waveform.
            tau_rise          -- time constant of the rising phase of the waveform in ms ^ -1.
            tau_decay         -- time constant of the falling phase of the waveform in ms ^ -1.
        """

        # Initialize time support vector.
        offset = 500
        self.time = np.arange(0, duration, self.dt)

        # Generate waveform based on time constants then normalize amplitude.
        waveform = np.exp(-self.time/tau_decay) - np.exp(-self.time/tau_rise)
        waveform /= np.max(waveform)
        waveform *= ampli

        # Convert waveform into a column vector.
        waveform = np.concatenate(
            (np.zeros((int(offset / self.dt))), waveform), axis = 0
            )
        waveform = waveform[np.newaxis].T

        # Compute total charge transfer using the equation AUC = ampli * (tau_decay - tau_rise). (Derived from integrating PS equation from 0 to inf)
        charge = ampli * (tau_decay - tau_rise)

        # Assign output.
        self.time = np.arange(0, duration + offset, self.dt)
        self.command    = waveform
        self.stim_type  = "Post-synaptic current-like"
        self.stim_params = types.SimpleNamespace(
            tau_rise = tau_rise, tau_decay = tau_decay,
            ampli = ampli, charge = charge
            )


    # Generate a synaptic current-like waveform with defined area under curve (total charge transfer)
    def generate_PS_bycharge(self, duration, charge, tau_rise, tau_decay):

        """
        Generate a post-synaptic potential/current-like waveform.

        Note that the rise and decay time constants are only good approximations of fitted rise/decay taus (which are more experimentally relevant) if the provided values are separated by at least approx. half an order of magnitude.

        Inputs:
            duration          -- length of the simulated waveform in ms ^ -1.
            charge            -- total charge transfer in units of pA*ms
            tau_rise          -- time constant of the rising phase of the waveform in ms ^ -1.
            tau_decay         -- time constant of the falling phase of the waveform in ms ^ -1.
        """

        # Initialize time support vector.
        offset = 500
        self.time = np.arange(0, duration, self.dt)

        # Generate waveform based on time constants
        waveform = np.exp(-self.time/tau_decay) - np.exp(-self.time/tau_rise)

        # Calculate ratio between desired and current charge and use to normalize waveform
        curr_charge = tau_decay - tau_rise
        scalefactor_waveform = charge / curr_charge
        waveform *= scalefactor_waveform

        # Convert waveform into a column vector.
        waveform = np.concatenate(
            (np.zeros((int(offset / self.dt))), waveform), axis = 0
            )
        waveform = waveform[np.newaxis].T

        # Compute amplitude of PS based on charge sign
        if charge > 0:
            ampli = np.max(waveform)
        else:
            ampli = np.min(waveform)

        # Assign output.
        self.time = np.arange(0, duration + offset, self.dt)
        self.command    = waveform
        self.stim_type  = "Post-synaptic current-like"
        self.stim_params = types.SimpleNamespace(
            tau_rise = tau_rise, tau_decay = tau_decay,
            ampli = ampli, charge = charge
            )


    # Realize OU noise and assign to self.command. (Wrapper for _gen_OU_internal.)
    def generate_OU(self, duration, I0, tau, sigma0, dsigma, sin_per):

        """
        Realize Ornstein-Uhlenbeck noise.

        Parameters are provided to allow the noise SD to vary sinusoidally over time.

        sigma[t] = sigma0 * ( 1 + dsigma * sin(2pi * sin_freq)[t] )

        Inputs:
            duration        -- duration of noise to realize in ms.
            I0              -- mean value of the noise.
            tau             -- noise time constant in ms ^ -1.
            sigma0          -- mean SD of the noise.
            dsigma          -- fractional permutation of noise SD.
            sin_per         -- period of the sinusoidal SD permutation in ms.
        """


        # Initialize support vectors.
        self.time       = np.arange(0, duration, self.dt)
        self.command    = np.zeros(self.time.shape)
        S               = sigma0 * (1 + dsigma * np.sin((2 * np.pi / sin_per) * self.time))
        rands           = np.random.standard_normal( len(self.time) )

        # Perform type conversions for vectors.
        self.time.dtype         = np.float64
        self.command.dtype      = np.float64
        S.dtype                 = np.float64
        rands.dtype             = np.float64

        # Perform type conversions for constants.
        self.dt                 = np.float64(self.dt)
        I0                      = np.float64(I0)
        tau                     = np.float64(tau)

        # Realize noise using nb.jit-accelerated function.
        noise = self._gen_OU_internal(
            self.time, rands, self.dt, I0,
            tau, S
            )

        # Convert noise to a column vector.
        noise = noise[np.newaxis].T

        # Assign output.
        self.command    = noise
        self.stim_type  = 'Ornstein-Uhlenbeck noise'
        self.stim_params = types.SimpleNamespace(
            I0 = I0, tau = tau, sigma0 = sigma0,
            dsigma = dsigma, sin_per = sin_per
            )


    # Generate sinusoidal input
    def generate_sin(self, duration, I0, ampli, period):

        """
        Generate a sine wave with time-dependent amplitude and/or period.

        Inputs:
            duration        -- duration of the wave in ms.
            I0              -- offset of the wave.
            ampli           -- amplitude of the wave.
            period          -- period of the wave in ms.

        Amplitude and/or period can be time-varied by passing one-dimensional vectors of length duration/dt instead of constants.
        """

        # Initialize time support vector.
        self.time = np.arange(0, duration, self.dt)

        # Convert ampli to a vector if need be;
        # otherwise check that it's the right shape.
        try:
            tmp = iter(ampli); del tmp # Verify that ampli is iterable.
            assert len(ampli) == len(self.time)

        except TypeError:
            ampli = np.array([ampli] * len(self.time))

        except AssertionError:
            raise ValueError('len of ampli must correspond to duration.')

        # Do the same with period.
        try:
            tmp = iter(period); del tmp # Verify that period is iterable.
            assert len(period) == len(self.time)

        except TypeError:
            period = np.array([period] * len(self.time))

        except AssertionError:
            raise ValueError('len of period must correspond to duration.')

        # Calculate the sine wave over time.
        sinewave = I0 + ampli * np.sin((2 * np.pi / period) * self.time)

        # Convert sine wave to column vector.
        sinewave = sinewave[np.newaxis].T

        # Assign output.
        self.command    = sinewave
        self.stim_type  = 'Sine wave'
        self.stim_params = types.SimpleNamespace(
            I0 = I0, ampli = ampli, period = period
            )

    @staticmethod
    @nb.jit(nb.float64[:, :](nb.float64[:, :], nb.float64, nb.float64, nb.float64, nb.float64))
    def _internal_V_integrator(input_, R, C, E, dt):

        V = np.empty_like(input_)

        for i in range(input_.shape[1]):

            V[0, i] = E

            for t in range(1, input_.shape[0]):

                dV = ((-(V[t-1, i] - E)/R + input_[t, i])) * dt/C
                V[t, i] = V[t-1, i] + dV

        return V


    # Simulate response of RC circuit.
    def simulate_RC(self, R, C, E, plot = True, verbose = True):

        """
        Simulate response of RC circuit to command.

        Inputs:

        R: float
        --  Resistance of RC circuit in MOhm

        C: float
        --  Capacitance of RC circuit in pF

        E: float
        --  Equilibrium potential/reversal poential/resting potential of the cell in mV

        plot: bool (default True)
        --  Plot the integrated stimulation

        verbose: bool (default True)
        --  Print some helpful output. Set to False to run quietly.
        """

        input_ = self.command.copy() * 1e-12 # Convert pA to A
        dt_ = self.dt * 1e-3
        R *= 1e6 # Convert R from MOhm to Ohm
        C *= 1e-12 # Convert C to F from pF
        E *= 1e-3 # Convert E from mV to V
        if verbose: print('tau = {}ms'.format(R * C * 1e3))


        if verbose: print('Integrating voltage...')
        V = self._internal_V_integrator(input_, R, C, E, dt_)
        V *= 1e3
        if verbose: print('Done integrating voltage!')

        if plot:
            if verbose: print('Plotting...')
            plt.figure()

            t_vec = np.arange(0, int(input_.shape[0] * self.dt), self.dt)

            ax = plt.subplot(211)
            plt.plot(t_vec, V, 'k-')
            plt.ylabel('Voltage (mV)')
            plt.xlabel('Time (ms)')

            plt.subplot(212, sharex = ax)
            plt.plot(t_vec, input_ * 1e12, 'k-')
            plt.ylabel('Command (pA)')
            plt.xlabel('Time (ms)')

            plt.show()

            if verbose: print('Done!')

        return V


    # Set number of replicates of the command array.
    def set_replicates(self, reps):

        """
        Set number of replicates of the existing command array.
        """

        # Check that command has been initialized.
        try:
            assert self.command is not None
        except AssertionError:
            raise RuntimeError('No command array to replicate!')

        # Create replicates by tiling.
        self.command = np.tile(self.command, (1, reps))
        self.stim_params.array_replicates = reps


    # Plot command, time, and additional data.
    def plot(self, **data):

        """
        Plot command (and any additional data) over time.

        Produces a plot of self.command over self.time as its primary output.

        Additional data of interest may be plotted as supplementary plots by passing them to the function as named arguments each containing a numerical vector of the same length as self.command.
        """

        d_keys  = data.keys()
        l_dk    = len(d_keys)

        plt.figure(figsize = (9, 3 + 3 * l_dk))
        plt.suptitle(str(self.label))

        # Plot generated noise over time.
        plt.subplot(1 + l_dk, 1, 1)
        plt.title('Generated stimulus')
        plt.xlabel('Time (ms)')
        plt.ylabel('Command')

        plt.plot(self.time, self.command, '-k', linewidth = 0.5)

        # Add plots from data passed as named arguments.
        i = 2
        for key in d_keys:

            plt.subplot(1 + l_dk, 1, i)
            plt.title( key )
            plt.xlabel('Time (ms)')

            plt.plot(self.time, data[ key ], '-k', linewidth = 0.5)

            i += 1


        # Final formatting and show plot.
        plt.tight_layout(rect = (0, 0, 1, 0.95))
        plt.show()


    # Write command and time to an ATF file.
    def write_ATF(self, fname = None):

        """
        Write command and time to an ATF file in the current working directory.
        """

        # Check whether there is any data to write.
        try:
            assert self.command is not None
            assert self.time is not None
        except AssertionError:
            raise RuntimeError('Command and time must both exist!')

        if fname is None:
            fname = self.label + '.ATF'
        elif fname[-4:].upper() != '.ATF':
            fname = fname + '.ATF'

        header_ls = [
            'ATF1.0\n1\t{}\nType=1\nTime (ms)\t'.format(self.command.shape[1]),
            *['Command (AU)\t' for sweep in range(self.command.shape[1])],
            '\n'
            ]
        header = ''.join(header_ls)

        # Convert numeric arrays to strings.
        str_command     = self.command.astype(np.unicode_)
        str_time        = self.time.astype(np.unicode_)

        # Initialize list to hold arrays.
        content_ls = []

        # Tab-delimit data one row (i.e., time step) at a time.
        for t in range(len(str_time)):

            tmp = str_time[t] + '\t' + '\t'.join(str_command[t, :])
            content_ls.append(tmp)

        # Turn the content list into one long string.
        content = '\n'.join(content_ls)

        # Write the header and content strings to the file.
        with open(fname, 'w') as f:

            f.write(header)
            f.write(content)
            f.close()


    ### HIDDEN METHODS

    # Fast internal method to realize OU noise. (Called by generate_OU.)
    @staticmethod
    @nb.jit(
        nb.float64[:](
            nb.float64[:], nb.float64[:], nb.float64,
            nb.float64, nb.float64, nb.float64[:]
            ),
        nopython = True
        )
    def _gen_OU_internal(T, rands, dt, I0, tau, sigma):

        I       = np.zeros(T.shape, dtype = np.float64)
        I[0]    = I0

        for t in range(1, len(T)):

            adaptive_term = (I0 - I[t - 1])
            random_term = np.sqrt(2 * sigma[t]**2 * dt / tau) * rands[t]

            dV = adaptive_term * dt / tau + random_term

            I[t] = I[t - 1] + dV

        return I
