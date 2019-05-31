"""
Version uploaded on ModelDB October 2017.
Author:
Basabdatta Sen Bhattacharya, APT group, School of Computer Science,
University of Manchester, 2017.

If you are using the code,
please cite the original work on the model - details are:

B. Sen-Bhattacharya, T. Serrano-Gotarredona, L. Balassa, A. Bhattacharya,
A.B. Stokes, A. Rowley, I. Sugiarto, S.B. Furber,
"A spiking neural network model of the Lateral Geniculate Nucleus on the
SpiNNaker machine", Frontiers in Neuroscience, vol. 11 (454), 2017.

Free online access:
http://journal.frontiersin.org/article/10.3389/fnins.2017.00454/abstract

"""
import spynnaker8 as p
import time

start_time = time.time()


''' Initialising Time and Frequency parameters'''

# total duration of simulation
TotalDuration = 6000.0

# time-step of model equation solver
time_resol = 0.1
TimeInt = 1.0 / time_resol

# this is in ms.
Duration_Inp = 5900.0

# 50 ms at both start and end are disregarded to avoid transients
Start_Inp = 50
End_Inp = int(Start_Inp + Duration_Inp)

# setting the input frequency of the spike train input
Rate_Inp = 50
Inp_isi = int(1000 / Rate_Inp)


''' Initialising Model connectivity parameters'''
intra_pop_delay = 4.0
intra_nucleus_delay = 6.0
inter_nucleus_delay = 8.0
inter_pop_delay = 10.0

tcr_weights = 5.0
in_weights = 4.0
input_delay = inter_pop_delay

# input_delay is the delay of the spike source hitting the neuronal populations
# inter_pop_delay is the delay of spike communication between the different
# populations of the model

p_in2tcr = 0.232

# WHICH IS 1/4th of 0.309 THIS IS KEPT AT A REDUCED VALUE UNDER NORMAL
# SIMULATIONS - HOWEVER, FOR REDUCED EFFECT OF IN, THIS IS INCREASED TO 0.232
# WHICH IS 3/4TH OF 30.9
p_trn2tcr = 0.077

w_tcr2trn = 3.0

w_trn2tcr = 2.0
w_trn2trn = 2.0

# SET TO 1 WHEN TESTING FOR REDUCED EFFECT OF THE IN ALONG WITH REDUCING
# P_IN2TCR
w_in2tcr = 8.0
w_in2in = 2.0


''' Initialising Izhikevich spiking neuron model parameters.
We have used the current-based model here.'''

# Tonic mode parameters
tcr_a_tonic = 0.02
tcr_b_tonic = 0.2
tcr_c_tonic = -65.0
tcr_d_tonic = 6.0
tcr_v_init_tonic = -65.0

in_a_tonic = 0.1
in_b_tonic = 0.2
in_c_tonic = -65.0
in_d_tonic = 6.0
in_v_init_tonic = -70.0

trn_a_tonic = 0.02
trn_b_tonic = 0.2
trn_c_tonic = -65.0
trn_d_tonic = 6.0
trn_v_init_tonic = -75.0

tcr_a = tcr_a_tonic
tcr_b = tcr_b_tonic
tcr_c = tcr_c_tonic
tcr_d = tcr_d_tonic
tcr_v_init = tcr_v_init_tonic


in_a = in_a_tonic
in_b = in_b_tonic
in_c = in_c_tonic
in_d = in_d_tonic
in_v_init = in_v_init_tonic


trn_a = trn_a_tonic
trn_b = trn_b_tonic
trn_c = trn_c_tonic
trn_d = trn_d_tonic
trn_v_init = trn_v_init_tonic


tcr_u_init = tcr_b * tcr_v_init
in_u_init = in_b * in_v_init
trn_u_init = trn_b * trn_v_init

# a constant DC bias current; this is used here for testing the RS and FS
# characteristics of IZK neurons
current_Pulse = 0.0

# excitatory input time constant
tau_ex = 1.7

# inhibitory input time constant
tau_inh = 2.5


'''Starting the SpiNNaker Simulator'''
p.setup(timestep=0.1, min_delay=1.0, max_delay=14.0)
## set number of neurons per core to 50, for the spike source to avoid clogging
p.set_number_of_neurons_per_core(p.SpikeSourceArray, 50)

'''Defining each cell type as dictionary'''

# THALAMOCORTICAL RELAY CELLS (TCR)
TCR_cell_params = {'a': tcr_a_tonic, 'b': tcr_b, 'c': tcr_c, 'd': tcr_d,
                   'v_init': tcr_v_init, 'u_init': tcr_u_init,
                   'tau_syn_E': tau_ex, 'tau_syn_I': tau_inh,
                   'i_offset': current_Pulse
                   }

# THALAMIC INTERNEURONS (IN)
IN_cell_params = {'a': in_a, 'b': in_b, 'c': in_c, 'd': in_d,
                  'v_init': in_v_init, 'u_init': in_u_init,
                  'tau_syn_E': tau_ex, 'tau_syn_I': tau_inh,
                  'i_offset': current_Pulse
                  }

# THALAMIC RETICULAR NUCLEUS (TRN)
TRN_cell_params = {'a': trn_a, 'b': trn_b, 'c': trn_c, 'd': trn_d,
                   'v_init': trn_v_init, 'u_init': trn_u_init,
                   'tau_syn_E': tau_ex, 'tau_syn_I': tau_inh,
                   'i_offset': current_Pulse
                   }


'''Creating populations of each cell type'''
scale_fact = 10
NumCellsTCR = 8*scale_fact
NumCellsIN = 2*scale_fact
NumCellsTRN = 4*scale_fact
TCR_pop = p.Population(
    NumCellsTCR, p.Izhikevich(**TCR_cell_params), label='TCR_pop')
IN_pop = p.Population(
    NumCellsIN, p.Izhikevich(**IN_cell_params), label='IN_pop')
TRN_pop = p.Population(
    NumCellsTRN, p.Izhikevich(**TRN_cell_params), label='TRN_pop')


''' Periodic spike train input defined'''

spike_source = p.Population(
    NumCellsTCR, p.SpikeSourceArray(
        spike_times=[i for i in range(Start_Inp, End_Inp, Inp_isi)]),
    label='spike_source')


'''Source to TCR population projections'''
Proj0 = p.Projection(
    spike_source, TCR_pop, p.FixedProbabilityConnector(p_connect=0.07),
    p.StaticSynapse(weight=tcr_weights, delay=input_delay),
    receptor_type='excitatory')


'''Source2IN'''
Proj1 = p.Projection(
    spike_source, IN_pop, p.FixedProbabilityConnector(p_connect=0.47),
    p.StaticSynapse(weight=in_weights, delay=input_delay),
    receptor_type='excitatory')


'''TCR2TRN'''
Proj2 = p.Projection(
    TCR_pop, TRN_pop, p.FixedProbabilityConnector(p_connect=0.35),
    p.StaticSynapse(weight=w_tcr2trn, delay=inter_nucleus_delay),
    receptor_type='excitatory')


'''TRN2TCR'''
Proj3 = p.Projection(
    TRN_pop, TCR_pop, p.FixedProbabilityConnector(p_connect=p_trn2tcr),
    p.StaticSynapse(weight=w_trn2tcr, delay=inter_nucleus_delay),
    receptor_type='inhibitory')


'''TRN2TRN'''
Proj4 = p.Projection(
    TRN_pop, TRN_pop, p.FixedProbabilityConnector(p_connect=0.2),
    p.StaticSynapse(weight=w_trn2trn, delay=intra_pop_delay),
    receptor_type='inhibitory')


'''IN2TCR'''
Proj5 = p.Projection(
    IN_pop, TCR_pop, p.FixedProbabilityConnector(p_connect=p_in2tcr),
    p.StaticSynapse(weight=w_in2tcr, delay=intra_nucleus_delay),
    receptor_type='inhibitory')


'''IN2IN'''
Proj6 = p.Projection(
    IN_pop, IN_pop, p.FixedProbabilityConnector(p_connect=0.236),
    p.StaticSynapse(weight=w_in2in, delay=intra_pop_delay),
    receptor_type='inhibitory')


''' Recording simulation data'''

# recording the spikes and voltage
spike_source.record("spikes")
TCR_pop.record(("spikes", "v"))
IN_pop.record(("spikes", "v"))
TRN_pop.record(("spikes", "v"))


''' Run the simulation for the total duration set'''
p.run(TotalDuration)


''' On simulation completion, extract the data off the spinnaker machine
memory'''

# extracting the spike time data
spikesourcepattern = spike_source.get_data("spikes")
TCR_spikes = TCR_pop.get_data("spikes")
IN_spikes = IN_pop.get_data("spikes")
TRN_spikes = TRN_pop.get_data("spikes")

# extracting the membrane potential data (in millivolts)
TCR_membrane_volt = TCR_pop.get_data("v")
IN_membrane_volt = IN_pop.get_data("v")
TRN_membrane_volt = TRN_pop.get_data("v")

''' Now release the SpiNNaker machine'''
p.end()

''' The user can now either save the data for further use, or plot this
using standard python tools'''

print "--- {} SECONDS ELAPSED ---".format(time.time() - start_time)
print "validating input isi used in the model: {}".format(Inp_isi)
