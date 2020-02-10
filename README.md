[![Build Status](https://travis-ci.com/efharkin/ez-ephys.svg?branch=master)](https://travis-ci.com/efharkin/ez-ephys)

# ez-ephys

Easy IO, inspection, and manipulation of electrophysiological data in Python.

## Philosophy

Keep things simple so you can spend your time running experiments instead of
learning a complicated API.

## Highlights

- `rectools.Recording`: thin `np.ndarray` wrapper with a `plot` method for
  visualizing recordings
- `pltools.add_scalebar`: add a set of automatically-sized scalebars to a
  `matplotlib` plot
- `stimtools`: take your experiments beyond current steps with a simple system
  for generating complicated electrophysiological stimuli ([examples](https://github.com/efharkin/ez-ephys/blob/master/examples/stimtools_demo.ipynb))
