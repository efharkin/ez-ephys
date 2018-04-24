"""
Created on Tue Jan 30 12:27:39 2018

@author: Emerson

`Cell` class to hold ABF recordings associated with a single neuron, along with 
support `Recording` class to hold recs as a np.array with built-in methods for
plotting and test-pulse extraction.

Compatible with python 2 and 3 as of Feb. 5, 2018.
"""

#%% IMPORT MODULES

# Built-ins.
from __future__ import division
from warnings import warn

# Third-party.
import numpy as np
import matplotlib.pyplot as plt
from neo.io import AxonIO
import numba as nb


#%% DEFINE CELL CLASS

"""
Defines a class to load and store multiple ABF recordings associated with a
single neuron.
"""


class Cell(object):
    
    # Initialize cell instance.
    def __init__(self, name = None, **recordings):
        
        """Initialize cell.
        """
        
        # Assign predetermined arguments.
        self.name = name
        self.rec_names = tuple(recordings.keys())
        self._rec_dict = recordings
        
        
        # Read in ABF recordings named in kwargs and place in eponymous attrs.
        for key in recordings.keys():
            
            # Initialize list to hold recordings.
            self.__setattr__(key, self.read_ABF(recordings[key]))
        
                
                
    # Define repr magic method.            
    def __repr__(self):
        
        """Unambiguous representation of cell instance.
        """
        
        if self.name is not None:
            cellname = 'Cell {}'.format(self.name)
        else:
            cellname = 'Unnamed cell'
            
        recnames = '\n'.join(self.rec_names)
        
        representation = (
                cellname 
                + ' with recordings:\n\n' 
                + recnames
                + '\n\nLocated at {}.'.format(hex(id(self)))
                )
                
        return representation
    
    
    # Method to read ABF files into a list of np.arrays.
    # Arrays have dimensionality [channels, samples, sweeps].
    def read_ABF(self, fnames):
        
        """Import ABF files into a list of np.arrays.
        
        \rInputs:
        \r\tfnames  --  list of files to import, or str for a single file
        
        \rReturns:
        \r\tList of np.arrays of recordings with dimensionality [channels, samples, sweeps]
        """
        
        # Convert str to iterable if only one fname is provided.
        if type(fnames) is str:
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
                    dtype = np.float64))
            
            # Fill the array one sweep at a time.
            for sweep_ind in range(no_sweeps):
                
                for chan_ind in range(no_channels):
                    
                    signal = sweeps[sweep_ind].analogsignals[chan_ind]
                    signal = np.squeeze(signal)
                    
                    assert len(signal) == sweeps_arr.shape[1], ('Not all '
                    'channels in {} are sampled at the same '
                    'rate.'.format(fname))
                    
                    sweeps_arr[chan_ind, :, sweep_ind] = signal
            
            # Add recording to output list.
            output.append(sweeps_arr)
        
        return output
    

class Recording(np.ndarray):
    
    """Subclass of np.ndarray with additional methods for common ephys tasks.
    
    Extra methods:
        plot
        fit_test_pulse
    """
    
    def __new__(cls, input_array):
        
        """Instantiate new Recording given an array of data.
        
        Allows new Recording objects to be created using np.array-type syntax;
        i.e., by passing Recording a nested list or existing np.array.
        """
        
        # Convert input_array to a np.ndarray, and subsequently to a Recording.
        obj = np.asarray(input_array).view(cls)
        
        # Check that newly-created recording has correct ndim.
        if obj.ndim != 3:
            raise ValueError('Recording dimensionality must be '
                             '[channel, time, sweep].')
        
        return obj
    
    
    def plot(self, single_sweep = False, downsample = 10):
        
        """Plotting function for quick inspection of Recording.
        
        Note that x-axis values correspond to inds of the time axis of the array.
        """
        
        ### Check for correct input ###
        
        # Check single_sweep.
        if type(single_sweep) is not bool:
            raise TypeError('`single_sweep` must be bool.')
            
        # Check downsample.
        if downsample is None:
            downsample = 1
            
        elif type(downsample) is not int:
            raise TypeError('`downsample` must be int or None.')
        
        elif downsample < 1:
            raise ValueError('`downsample` must be an int > 0. or None.')
            
        
        ### Select data to plot ###
        if not single_sweep:
            plotting_data = self
        else:
            plotting_data = self[:, :, 0][:, :, np.newaxis]
            
        
        ### Make plot ###
        x_vector = np.arange(0, self.shape[1], downsample) # Preserves indexes.
        plt.figure(figsize = (10, 7))
        
        for i in range(self.shape[0]):
            
            # Force all subplots to share x-axis.
            if i == 0:
                ax0 = plt.subplot(self.shape[0], 1, 1)
            else:
                plt.subplot(self.shape[0], 1, i + 1, sharex = ax0)
            
            plt.title('Channel {}'.format(i))
            plt.plot(x_vector, plotting_data[i, ::downsample, :],
                     'k-',
                     linewidth = 0.5)
            plt.xlabel('Time (timesteps)')
            
        plt.tight_layout()
        plt.show()
        
    
    def fit_test_pulse(self, baseline, steady_state, **kwargs):
        
        """Extract R_input and (optionally) R_a from test pulse.
        
        `baseline` and `steady_state` should be passed tuples of indexes over
        which to take measurements on each sweep.
        
        Set `verbose` to False to prevent printing results.
        """
        
        ### Inputs ###
        
        # Set kwarg defaults.
        kwargs.setdefault('V_chan', 1)
        kwargs.setdefault('I_chan', 0)
        kwargs.setdefault('V_clamp', True)
        kwargs.setdefault('verbose', True)
        
        # Check for correct inputs.
        if type(baseline) is not tuple:
            raise TypeError('Expected type tuple for `baseline`; got {} '
                            'instead.'.format(type(baseline)))
        elif any([type(entry) != int for entry in baseline]):
            raise TypeError('Expected tuple of ints for `baseline`.')
        elif len(baseline) != 2:
            raise TypeError('Expected tuple of len 2 specifying start and '
                             'stop positions for `baseline`.')
        elif any([entry > self.shape[1] for entry in baseline]):
            raise ValueError('`baseline` selection out of bounds for channel '
                             'of length {}.'.format(self.shape[1]))
            
        if type(steady_state) is not tuple:
            raise TypeError('Expected type tuple for `steady_state`; got {} '
                            'instead.'.format(type(steady_state)))
        elif any([type(entry) != int for entry in steady_state]):
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
            
        if type(kwargs['V_clamp']) is not bool:
            raise TypeError('Expected `V_clamp` to be type bool; got {} '
                            'instead.'.format(type(kwargs['V_clamp'])))
            
        if type(kwargs['verbose']) is not bool:
            raise TypeError('Expected `verbose` to be type bool; got {} '
                            'instead.'.format(type(kwargs['verbose'])))
            
        
        ### Main ###
        
        # Create dict to hold output.
        output = {}
        
        # Calculate R_input.
        V_baseline = self[kwargs['V_chan'], slice(*baseline), :].mean(axis = 0)
        I_baseline = self[kwargs['I_chan'], slice(*baseline), :].mean(axis = 0)
        V_test = self[kwargs['V_chan'], slice(*steady_state), :].mean(axis = 0)
        I_test = self[kwargs['I_chan'], slice(*steady_state), :].mean(axis = 0)
        
        delta_V_ss = V_test - V_baseline
        delta_I_ss = I_test - I_baseline
        
        R_input = 1000 * delta_V_ss / delta_I_ss
        output['R_input'] = R_input
        
        # Calculate R_a.
        if kwargs['V_clamp']:
            
            if delta_V_ss.mean() < 0:
                I_peak = self[kwargs['I_chan'], 
                              slice(baseline[1], steady_state[0]),
                              :].min(axis = 0)
            else:
                I_peak = self[kwargs['I_chan'],
                              slice(baseline[1], steady_state[0]),
                              :].max(axis = 0)
            
            R_a = 1000 * delta_V_ss / (I_peak - I_baseline)
            output['R_a'] = R_a
            
        
        # Optionally, print results.
        if kwargs['verbose']:
            print('\n\n### Test-pulse results ###')
            print('R_in: {} +/- {} MOhm'.format(round(R_input.mean(), 1), 
                  round(R_input.std())))
            
            if kwargs['V_clamp']:
                print('R_a: {} +/- {} MOhm'.format(round(R_a.mean()), 
                      round(R_a.std())))
                
        
        return output