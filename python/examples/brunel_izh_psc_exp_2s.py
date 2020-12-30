import sys
import ctypes
import neurongpu as ngpu
from random import randrange

if len(sys.argv) != 2:
    print ("Usage: python %s n_neurons" % sys.argv[0])
    quit()
    
order = int(sys.argv[1])//5

ngpu.SetTimeResolution(1.0)

print("Building ...")

ngpu.SetRandomSeed(1234) # seed for GPU random numbers

sim_time = 10.0 # simulation time in seconds

n_receptors = 2

NE = 4 * order       # number of excitatory neurons
NI = 1 * order       # number of inhibitory neurons
n_neurons = NE + NI  # number of neurons in total

CE = 80     # number of excitatory synapses per neuron
CI = CE//4  # number of inhibitory synapses per neuron

fact=0.42
Wex = 0.5*fact
Win = -3.5*fact

tau_plus = 20.0
tau_minus = 20.0
lambd = 1.0e-5
alpha = 1.0
mu_plus = 1.0
mu_minus = 1.0
Wmax = 10.0

syn_group = ngpu.CreateSynGroup \
            ("stdp", {"tau_plus":tau_plus, "tau_minus":tau_minus, \
                      "lambda":lambd, "alpha":alpha, "mu_plus":mu_plus, \
                      "mu_minus":mu_minus,  "Wmax":Wmax})

# poisson generator parameters
poiss_rate = 20000.0 # poisson signal rate in Hz
poiss_weight = 0.37*fact
poiss_delay = 1.0 # poisson signal delay in ms

# create poisson generator
pg = ngpu.Create("poisson_generator")
ngpu.SetStatus(pg, "rate", poiss_rate)

# Create n_neurons neurons with n_receptor receptor ports
neuron = ngpu.Create("izhikevich_psc_exp_2s", n_neurons, n_receptors)
ngpu.ActivateSpikeCount(neuron)
exc_neuron = neuron[0:NE]      # excitatory neurons
inh_neuron = neuron[NE:n_neurons]   # inhibitory neurons
  
# receptor parameters

delay = 2.0

# Excitatory connections
# connect excitatory neurons to port 0 of all neurons
# normally distributed delays, weight Wex and CE connections per neuron
exc_conn_dict={"rule": "fixed_indegree", "indegree": CE}
exc_syn_dict={"weight": Wex, "delay": delay, "receptor":0,
              "synapse_group":syn_group}
ngpu.Connect(exc_neuron, neuron, exc_conn_dict, exc_syn_dict)


# Inhibitory connections
# connect inhibitory neurons to port 1 of all neurons
# normally distributed delays, weight Win and CI connections per neuron
inh_conn_dict={"rule": "fixed_indegree", "indegree": CI}
inh_syn_dict={"weight": Win, "delay":delay, "receptor":1}

ngpu.Connect(inh_neuron, neuron, inh_conn_dict, inh_syn_dict)


#connect poisson generator to port 0 of all neurons
pg_conn_dict={"rule": "all_to_all"}
pg_syn_dict={"weight": poiss_weight, "delay": poiss_delay, "receptor":0}

ngpu.Connect(pg, neuron, pg_conn_dict, pg_syn_dict)


filename = "test_brunel_net.dat"
i_neuron_arr = [neuron[37], neuron[randrange(n_neurons)], neuron[n_neurons-1]]
i_receptor_arr = [0, 0, 0]
# any set of neuron indexes
# create multimeter record of V_m
var_name_arr = ["V_m", "V_m", "V_m"]
record = ngpu.CreateRecord(filename, var_name_arr, i_neuron_arr,
                                i_receptor_arr)

ngpu.Simulate(sim_time*1000.0)

spike_count = ngpu.GetStatus(neuron, "spike_count")
spike_count = [s[0] for s in spike_count] 
print("Average firing rate: ", sum(spike_count)/len(spike_count)/sim_time)

data_list = ngpu.GetRecordData(record)
t=[row[0] for row in data_list]
V1=[row[1] for row in data_list]
V2=[row[2] for row in data_list]
V3=[row[3] for row in data_list]

import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(t, V1)

plt.figure(2)
plt.plot(t, V2)

plt.figure(3)
plt.plot(t, V3)

plt.draw()
plt.pause(0.5)
ngpu.waitenter("<Hit Enter To Close>")
plt.close()
