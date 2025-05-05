# %%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from SaVer_Toolbox import signedDistanceFunction, verify
import time

# Seed the random number generator for reproducibility
# np.random.seed(22)

# Set the seed for reproducibility
# torch.manual_seed(128)

# %%
betaDKW = 0.001
epsilonDKW = 0.001
Delta = 1-0.999
verifDKW = verify.usingDKW(betaDKW,epsilonDKW,Delta)
betaScenario = 0.001
verifScenario = verify.usingScenario(betaScenario,Delta)


class ReLUFeedforwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ReLUFeedforwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc1.bias, mean=0, std=0.01)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc1.bias, mean=0, std=0.01)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    


# Fix the output dimension and num of samples. Run verif task.

input_dim_list = np.array([10,50,100,500,1000])
hidden_size = 10
output_size = 2
input_sampling_time_duration = np.empty(input_dim_list.shape[0])
input_verif_time_duration = np.empty((2,input_dim_list.shape[0]))
input_set_modif_duration = np.empty((2,input_dim_list.shape[0]))

for i in range(0,input_dim_list.shape[0]):


    # Generate samples from a standard Cauchy distribution
    samplesDKW = np.random.standard_cauchy((verifDKW.samplesRequired(), input_dim_list[i]))
    samplesScenario = np.random.standard_cauchy((verifScenario.samplesRequired(), input_dim_list[i]))
    # Create an instance of the ReLU feedforward network
    network = ReLUFeedforwardNetwork(input_dim_list[i], hidden_size, output_size)
    torch.nn.init.xavier_uniform_(network.fc1.weight)
    torch.nn.init.xavier_uniform_(network.fc2.weight)

    # Convert samples to a PyTorch tensor
    samples_tensor_DKW = torch.tensor(samplesDKW, dtype=torch.float32)
    samples_tensor_Scenario = torch.tensor(samplesScenario, dtype=torch.float32)

    start_time = time.perf_counter()
    # Evaluate the network using the samples
    output_tensor_DKW = network(samples_tensor_DKW)
    output_tensor_Scenario = network(samples_tensor_Scenario)
    sampling_time = time.perf_counter()
    input_sampling_time_duration[i] = sampling_time - start_time

    # Convert the output to a numpy array
    samplesDKW = output_tensor_DKW.detach().cpu().numpy()
    samplesScenario = output_tensor_Scenario.detach().cpu().numpy()

    # %%
    # Center of the norm-ball
    center = np.array(0.0)
    zero_radius_fixed = np.array(20000)
    normSDF = signedDistanceFunction.norm(center,zero_radius_fixed,norm=2)

    # Add SDF: 
    verifDKW.specification(normSDF)
    verifScenario.specification(normSDF)

    # %%
    # Add samples to the verifier
    verifDKW.addSamples(samplesDKW)
    verifScenario.addSamples(samplesScenario)

    start_time = time.perf_counter()
    # %%
    # Check if the samples satisfy the specification: 
    verifDKW.probability()
    verif_time_DKW = time.perf_counter()
    verifScenario.probability()
    verif_time_Scenario = time.perf_counter()
    input_verif_time_duration[0,i] = verif_time_DKW - start_time
    input_verif_time_duration[1,i] = verif_time_Scenario - verif_time_DKW

    start_time = time.perf_counter()
    # %%
    # Modify the set:
    setReductionDKW = verifDKW.modifySet()
    setMod_time_DKW = time.perf_counter()
    setReductionScenario = verifScenario.modifySet()
    setMod_time_Scenario = time.perf_counter()
    input_set_modif_duration[0,i] = setMod_time_DKW - start_time
    input_set_modif_duration[1,i] = setMod_time_Scenario - setMod_time_DKW

    # %%
    # Check if the samples satisfy the specification: 
    verifDKW.probability()
    verifScenario.probability()

print(input_sampling_time_duration)
print(input_verif_time_duration)
print(input_set_modif_duration)

# np.savetxt('inputScale.out', (sampling_time_duration,verif_time_duration,set_modif_duration))   # x,y,z equal sized 1D arrays





# Fix the input dimension and num of samples. Run verif task. 

input_size = 4
hidden_size = 10
output_dim_list = np.array([10,50,100,500,1000])
output_sampling_time_duration = np.empty(output_dim_list.shape[0])
output_verif_time_duration = np.empty((2,output_dim_list.shape[0]))
output_set_modif_duration = np.empty((2,output_dim_list.shape[0]))

for i in range(0,output_dim_list.shape[0]):


    # Generate samples from a standard Cauchy distribution  
    samplesDKW = np.random.standard_cauchy((verifDKW.samplesRequired(), input_size))
    samplesScenario = np.random.standard_cauchy((verifScenario.samplesRequired(), input_size))
    # Create an instance of the ReLU feedforward network
    network = ReLUFeedforwardNetwork(input_size, hidden_size, output_dim_list[i])
    torch.nn.init.xavier_uniform_(network.fc1.weight)
    torch.nn.init.xavier_uniform_(network.fc2.weight)

    # Convert samples to a PyTorch tensor
    samples_tensor_DKW = torch.tensor(samplesDKW, dtype=torch.float32)
    samples_tensor_Scenario = torch.tensor(samplesScenario, dtype=torch.float32)

    start_time = time.perf_counter()
    # Evaluate the network using the samples
    output_tensor_DKW = network(samples_tensor_DKW)
    output_tensor_Scenario = network(samples_tensor_Scenario)
    sampling_time = time.perf_counter()
    output_sampling_time_duration[i] = sampling_time - start_time

    # Convert the output to a numpy array
    samplesDKW = output_tensor_DKW.detach().cpu().numpy()
    samplesScenario = output_tensor_Scenario.detach().cpu().numpy()

    # %%
    # Center of the norm-ball
    center = np.array(0.0)
    zero_radius_fixed = np.array(20000)
    normSDF = signedDistanceFunction.norm(center,zero_radius_fixed,norm=2)

    # Add SDF: 
    verifDKW.specification(normSDF)
    verifScenario.specification(normSDF)

    # %%
    # Add samples to the verifier
    verifDKW.addSamples(samplesDKW)
    verifScenario.addSamples(samplesScenario)

    start_time = time.perf_counter()
    # %%
    # Check if the samples satisfy the specification: 
    verifDKW.probability()
    verif_time_DKW = time.perf_counter()
    verifScenario.probability()
    verif_time_Scenario = time.perf_counter()
    output_verif_time_duration[0,i] = verif_time_DKW - start_time
    output_verif_time_duration[1,i] = verif_time_Scenario - verif_time_DKW

    start_time = time.perf_counter()
    # %%
    # Modify the set:
    setReductionDKW = verifDKW.modifySet()
    setMod_time_DKW = time.perf_counter()
    setReductionScenario = verifScenario.modifySet()
    setMod_time_Scenario = time.perf_counter()
    output_set_modif_duration[0,i] = setMod_time_DKW - start_time
    output_set_modif_duration[1,i] = setMod_time_Scenario - setMod_time_DKW

    # %%
    # Check if the samples satisfy the specification: 
    verifDKW.probability()
    verifScenario.probability()

print(output_sampling_time_duration)
print(output_verif_time_duration)
print(output_set_modif_duration)

# np.savetxt('outputScale.out', (sampling_time_duration,verif_time_duration,set_modif_duration))   # x,y,z equal sized 1D arrays

# Plotting the results
fig, axs = plt.subplots(1, 2, figsize=(5, 1.5))

# Input scaling plots
axs[0].semilogy(input_dim_list, input_sampling_time_duration, marker='.', label='Cumulative Sampling Time')
axs[0].set_xlabel('Input Dimension')
axs[0].set_ylabel('Computation Time (s)')

axs[0].semilogy(input_dim_list, input_verif_time_duration[0], marker='o', markerfacecolor='none',markersize=10, color='orange', label='Verification - DKW')
axs[0].semilogy(input_dim_list, input_verif_time_duration[1], marker='o', markerfacecolor='none',markersize=10, color='green', label='Verification - Scenario')

axs[0].semilogy(input_dim_list, input_set_modif_duration[0], marker='.', color='orange', label='Modify Set - DKW')
axs[0].semilogy(input_dim_list, input_set_modif_duration[1], marker='.', color='green', label='Modify Set - Scenario')
# axs[0].legend()

# Output scaling plots
axs[1].semilogy(output_dim_list, output_sampling_time_duration, marker='.')
axs[1].set_xlabel('Output Dimension')

axs[1].semilogy(output_dim_list, output_verif_time_duration[0], marker='o',markerfacecolor='none', markersize=10, color='orange', label='Verification - DKW')
axs[1].semilogy(output_dim_list, output_verif_time_duration[1], marker='o',markerfacecolor='none',markersize=10, color='green', label='Verification - Scenario')

axs[1].semilogy(output_dim_list, output_set_modif_duration[0], marker='.', color='orange', label='Modify Set - DKW')
axs[1].semilogy(output_dim_list, output_set_modif_duration[1], marker='.', color='green', label='Modify Set - Scenario')
# axs[1].legend()
# axs[1].set_yticklabels([])
for ax in axs:
    ax.set_xlabel(ax.get_xlabel(), fontsize=9, fontname='serif')
    ax.set_ylabel(ax.get_ylabel(), fontsize=9, fontname='serif')
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.xaxis.label.set_fontsize(6)
    ax.yaxis.label.set_fontsize(6)
    ax.xaxis.label.set_fontname('serif')
    ax.yaxis.label.set_fontname('serif')
    ax.xaxis.label.set_text(ax.xaxis.label.get_text())
    ax.yaxis.label.set_text(ax.yaxis.label.get_text())
    ax.set_ylim(0, 2)
axs[0].grid(True)
axs[1].grid(True)
plt.tight_layout()
print('Saving plot...')
plt.savefig('./examples/scalabilityFfNN/Figure6.png', bbox_inches='tight',format='png', dpi=600)

 