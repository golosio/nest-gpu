# -*- coding: utf-8 -*-
#
# hpc_benchmark.py
#
# This file is part of NEST GPU.
#
# Copyright (C) 2021 The NEST Initiative
#
# NEST GPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST GPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.


"""
Random balanced network HPC benchmark
-------------------------------------
Warning: the NEST GPU implementation still presents differencies with respect
to NEST in the average firing rate of neurons.

This script produces a balanced random network of `scale*11250` neurons
connected with static connections. The number of incoming connections 
per neuron is fixed and independent of network size (indegree=11250).

Furthermore, the scale can be also increased through running the script
using several MPI processes using the command mpirun -np nproc python hpc_benchmark.py
This way, a network of `scale*11250` neurons is built in every MPI process,
with indegrees equally distributed across the processes.

This is the standard network investigated in [1]_, [2]_, [3]_.
A note on connectivity
~~~~~~~~~~~~~~~~~~~~~~
Each neuron receives :math:`K_{in,{\\tau} E}` excitatory connections randomly
drawn from population E and :math:`K_{in,\\tau I}` inhibitory connections from
population I. Autapses are prohibited while multapses are allowed. Each neuron
receives additional input from an external stimulation device. All delays are
constant, all weights but excitatory onto excitatory are constant.

A note on scaling
~~~~~~~~~~~~~~~~~
This benchmark was originally developed for very large-scale simulations on
supercomputers with more than 1 million neurons in the network and
11.250 incoming synapses per neuron. For such large networks, synaptic input
to a single neuron will be little correlated across inputs and network
activity will remain stable over long periods of time.
The original network size corresponds to a scale parameter of 100 or more.
In order to make it possible to test this benchmark script on desktop
computers, the scale parameter is set to 1 below, while the number of
11.250 incoming synapses per neuron is retained.
Over time, network dynamics will therefore become unstable and all neurons
in the network will fire in synchrony, leading to extremely slow simulation
speeds.
Therefore, the presimulation time is reduced to 50 ms below and the
simulation time to 250 ms, while we usually use 100 ms presimulation and
1000 ms simulation time.
For meaningful use of this benchmark, you should use a scale > 10 and check
that the firing rate reported at the end of the benchmark is below 10 spikes
per second.
References
~~~~~~~~~~
.. [1] Morrison A, Aertsen A, Diesmann M (2007). Spike-timing-dependent
       plasticity in balanced random networks. Neural Comput 19(6):1437-67
.. [2] Helias et al (2012). Supercomputers ready for use as discovery machines
       for neuroscience. Front. Neuroinform. 6:26
.. [3] Kunkel et al (2014). Spiking network simulation code for petascale
       computers. Front. Neuroinform. 8:78
.. [4] Senk et al (2021). Connectivity Concepts in Neuronal Network Modeling.
       arXiv. 2110.02883
"""

import os
import sys
import json
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

from time import perf_counter_ns

print("ok0")
import nestgpu as ngpu
print("ok1")

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str, default=".")
parser.add_argument("--seed", type=int, default=12345)
args = parser.parse_args()

M_INFO = 10
M_ERROR = 30

ngpu.ConnectMpiInit()
ngpu.SetKernelStatus({
        "rnd_seed": args.seed
})



mpi_id = ngpu.HostId()
mpi_np = 640 #ngpu.HostNum()
print("OK MPI ID ", mpi_id)


#hg = ngpu.CreateHostGroup([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
hg = ngpu.CreateHostGroup(list(range(mpi_np)))

