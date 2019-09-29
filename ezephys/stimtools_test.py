# IMPORT MODULES

import os
import tempfile

from ezephys import stimtools


# INITIALIZE A SET OF STIMULATIONS

PS_stim = stimtools.Stim('synaptic waveform')
PS_stim.generate_PS(100, 10, 1, 20)

OU_stim = stimtools.Stim('OU noise')
OU_stim.generate_OU(1000, 0, 1, 10, 0.5, 100)

sin_stim = stimtools.Stim('sinewave')
sin_stim.generate_sin(1000, 0, 10, 100)


# TEST HELP

help(PS_stim.generate_PS)
help(OU_stim.generate_OU)
help(sin_stim.generate_sin)


# PLOT STIMULATIONS

PS_stim.plot()
OU_stim.plot()
sin_stim.plot()


# TRY SIMULATING RC DYNAMICS

"""
Handy if you're trying to figure out how much to scale a stim for a particular
cell on-the-fly.
"""

# Define RC circuit parameters
R = 100  # Resistance in MOhm
C = 70  # Capacitance in pF
E = -70  # Equilibrium potential in mV

PS_stim.simulate_RC(R, C, E)
OU_stim.simulate_RC(R, C, E)
sin_stim.simulate_RC(R, C, E)


# WRITE THE STIM TO AN ATF

with tempfile.TemporaryDirectory() as tmpdir:

    PS_stim.write_ATF(os.path.join(tmpdir, 'testSynATF'))
    OU_stim.write_ATF(os.path.join(tmpdir, 'testOUATF.atf'))
    sin_stim.write_ATF(os.path.join(tmpdir, 'testSin.ATF'))
