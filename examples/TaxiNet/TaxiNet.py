# %%
import requests, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from nnSampleVerification import plotter, signedDistanceFunction, verify

# %% [markdown]
# 

# %%
betaDKW = 0.001
epsilonDKW = 0.1
Delta = 1-0.9
verifDKW = verify.usingDKW(betaDKW,epsilonDKW,Delta)
betaScenario = 0.001
verifScenario = verify.usingScenario(betaScenario,Delta)

f = h5py.File('./examples/TaxiNet/TaxiNetTraj.h5', 'r')
tot_full_runs = 408
y_pos = np.zeros((tot_full_runs,1))

for i in np.arange(0,tot_full_runs):

    run_num = 'run_' + str(i+1)
    group = f.get(run_num)
    y_pos[i] = group.get('cte')[-1]

# %%
# Center of the norm-ball
center = np.array(0.0)
zero_radius_fixed = np.array(1.0)
normSDF = signedDistanceFunction.norm(center,zero_radius_fixed,norm=1)

# Add SDF: 
verifDKW.specification(normSDF)
verifScenario.specification(normSDF)

# %%
# Generate samples
samples = y_pos
# Add samples to the verifier
verifDKW.samples(samples[0:verifDKW.samplesRequired()])    
verifScenario.samples(samples[0:verifScenario.samplesRequired()])

# %%
# Check if the samples satisfy the specification: 
verifDKW.probability()
verifScenario.probability()

# %%
# Modify the zero level set:
setEnlargementDKW = verifDKW.modifySet()
setEnlargementScenario = verifScenario.modifySet()

# %%
# Check if the samples satisfy the specification: 
verifDKW.probability()
verifScenario.probability()

# %%
plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
plt.figure(figsize=(3.3, 3.3))
plt.rcParams.update({'figure.dpi': 600})

# Add horizontal lines at 10 and -10
plt.axhline(y=0, color='k', linestyle='--', label='Centerline of Runway')

# Plot the samples
plt.plot(422*np.ones(samples.shape[0]),samples, 'o', label='Samples of Final Position', markersize=10, color='blue')

plt.plot([422, 422], [-1-setEnlargementScenario, 1+setEnlargementScenario], '-', color='green', linewidth=6, label='Modified Scenario Specification')
plt.plot([420.5, 423.5], [1+setEnlargementScenario, 1+setEnlargementScenario], '-', color='green', linewidth=4)
plt.plot([420.5, 423.5], [-1-setEnlargementScenario, -1-setEnlargementScenario], '-', color='green',linewidth=4)
plt.plot([422, 422], [-1-setEnlargementDKW, 1+setEnlargementDKW], '-', color='red', linewidth=4, label='Modified DKW Specification')
plt.plot([421, 423], [1+setEnlargementDKW, 1+setEnlargementDKW], '-', color='red', linewidth=3)
plt.plot([421, 423], [-1-setEnlargementDKW, -1-setEnlargementDKW], '-', color='red',linewidth=3)
# Add box and whisker like lines at y = ±1.5 and x = 422
plt.plot([422, 422], [-1, 1], '-', color='orange', linewidth=2)
plt.plot([421.5, 422.5], [1, 1], '-', color='orange', linewidth=2)
plt.plot([421.5, 422.5], [-1, -1], '-', color='orange',linewidth=2, label='Original Specification')




# Add labels and legend:
plt.xlabel('Position of Aircraft Down the Runway (m)')
plt.ylabel('Lateral Aircraft Position (m)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Show the plot
plt.show()


