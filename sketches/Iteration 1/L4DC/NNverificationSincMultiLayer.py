import sys
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
from scipy.signal import hilbert
import seaborn as sns

# # # # # # # # # # # # # # # # # # # # # 

def initializeUniform(gridpoint, a, b):
    return (jnp.exp(1j * gridpoint * b) - jnp.exp(1j * gridpoint * a)) / (1j * gridpoint * (b - a))

def initializeGaussian(gridpoint, mu, sigma):
    return jnp.exp(1j * mu * gridpoint - 0.5 * gridpoint ** 2 * sigma ** 2)

def affineLayer_node(grid, CFxj, gridpoint, Wij):
    return jnp.interp(Wij * gridpoint, grid, CFxj)

def affineLayer(grid, CFx, bi, Wi):
    return  jnp.exp(1j * grid * bi) * jnp.prod(vfunc_AL_node_jit(grid, CFx, grid, Wi), axis=1)

def hilbertTransform(f, grid, x, hilb_grid):
    eval_pt = (x - hilb_grid * h) / h
    return jnp.sum(jnp.interp(hilb_grid * h, grid, f) * jnp.sinc(eval_pt / 2) * jnp.sin(jnp.pi * eval_pt / 2))

def maxLayer(grid, CF, gridpoint, hilb_grid):
    CFz_eval = 0.5 * (1 + jnp.interp(gridpoint, grid, CF)) + 0.5 * 1j * (hilbertTransform(CF, grid, gridpoint, hilb_grid) - hilbertTransform(CF, grid, 0, hilb_grid))
    return CFz_eval

def oneLayerProp(CFx, grid, W, b):

    # Propagate through affine layer
    CFy = vfunc_AL_jit(grid, CFx, b, W)

    # Propagate through max layer
    CFz = maxLayer(CFy)
    
    # return output
    return CFz

def cdf(x, axs, *args, **kwargs):
    x, y = sorted(x), np.arange(len(x)) / len(x)
    return axs.plot(x, y, *args, **kwargs) 

def plotSampleCDF(data, axs, num):
    for j in range(0, num):
        cdf(data[j, :], axs, color=palette[j])

def plotSamplePDF(data, axs, num):
    for j in range(0, num):
        axs.hist(data[j, :], bins='auto', density=True, alpha=0.5, color=palette[j])

def plotSampleData(data, axs, num):
    plotSampleCDF(data, axs, num)
    plotSamplePDF(data, axs, num)
    axs.set_ylim([0, 1.25])

def plotEmpiricalCDF(CDF, CDFgrid, axs, num):
    for j in range(0, num):
        axs.scatter(CDFgrid, CDF[j, :], color=palette[j], marker="o")

def plotAllCDF(data, CDF, CDFgrid, axs, num):
    plotSampleCDF(data, axs, num)
    plotEmpiricalCDF(CDF, CDFgrid, axs, num)

def plotECFandCDF(grid, CDFgrid, ECF, CDF, axs, num):
    ECF_real = jnp.real(ECF)
    ECF_imag = jnp.imag(ECF)
    axs[0].plot(grid, ECF_real[0:num, :].T, 'k')
    axs[1].plot(grid, ECF_imag[0:num, :].T, 'k')
    axs[2].plot(CDFgrid, CDF[0:num, :].T, '.k')

def propagateSamplesfull(input_data, out_dim, W, b, num):
    output = np.zeros(shape=(out_dim, num))
    for k in range(0, num):
        input_k = input_data[:, k]
        IL_k = W.dot(input_k) + b
        for j in range(0, out_dim):
            output_kj = max(0, IL_k[j])
            output[j, k] = output_kj
    return output

def propagateSamplesAL(input_data, out_dim, W, b, num):
    output = np.zeros(shape=(out_dim, num))
    for k in range(0, num):
        input_k = input_data[:, k]
        output_k = W.dot(input_k) + b
        output[:, k] = output_k
    return output

def propagateSamplesML(input_data, out_dim, num):
    output = np.zeros(shape=(out_dim, num))
    for k in range(0, num):
        input_k = input_data[:, k]
        for j in range(0, out_dim):
            output_kj = max(0, input_k[j])
            output[j, k] = output_kj
    return output

def computeCDF(f, grid, x):
    numerator = jnp.exp(-1j * x * (hilb_grid - 0.5) * h) * jnp.interp((hilb_grid - 0.5) * h, grid, f)
    denominator = jnp.pi * (hilb_grid - 0.5)
    return jnp.sum(0.5 * 1j * numerator / denominator) + 0.5

def computeAllCDF(f, grid, CDFgrid):
    neurons = f.shape[0]
    CDFgridLength = CDFgrid.shape[0]
    CDFout = jnp.zeros(shape=(neurons, CDFgridLength), dtype=np.complex64)
    for i in range(0, neurons):
        fi = f[i, :]
        for j in range(0, CDFgridLength):
            gridpoint = CDFgrid[j]
            CDFgridpoint = computeCDF(fi, grid, gridpoint)
            CDFout = CDFout.at[i, j].set(CDFgridpoint)
    return CDFout

def computeECF(gridpoint, data):

    # For each gridpoint tk, we compute the sum (MoG) to get the ECF
    # Input: (scalar, vector)
    # Vectorization #1: gridpoint (L points)
    # Vectorization #2: dataset (# of neurons)

    return (1 / numSamples) * jnp.sum(jnp.exp(1j * gridpoint * data))


# # # # # # # # # # # # # # # # # # # # # 

# Disable warnings from complex numbers
warnings.filterwarnings("ignore")

# Define color palette for plotting
palette = sns.color_palette("husl", 10)

# # # vectorize function maps # # #
vfunc_IU      = vmap(initializeUniform, in_axes = (0, None, None))
vfunc_IG      = vmap(initializeGaussian, in_axes = (0, None, None))
vfunc_AL_node = vmap(vmap(affineLayer_node, in_axes = (None, 0, None, 0)), in_axes = (None, None, 0, None))
vfunc_AL      = vmap(affineLayer, in_axes = (None, None, 0, 0))
vfunc_ML      = vmap(vmap(maxLayer, in_axes = (None, None, 0, None)), in_axes = (None, 0, None, None))
vfunc_ECF     = vmap(vmap(computeECF, in_axes = (0, None)), in_axes = (None, 1))
vfunc_CDF     = vmap(vmap(computeCDF, in_axes = (None, None, 0)), in_axes = (0, None, None))

# # # jit it up # # #
vfunc_IU_jit      = jit(vfunc_IU)
vfunc_IG_jit      = jit(vfunc_IG)
vfunc_AL_node_jit = jit(vfunc_AL_node)
vfunc_AL_jit      = jit(vfunc_AL)
vfunc_ML_jit      = jit(vfunc_ML)
vfunc_ECF_jit     = jit(vfunc_ECF)
vfunc_CDF_jit     = jit(vfunc_CDF)

# define cutoff and resolution
d, L = 100, 10001

# define HT resolution
h, M = 0.5, 5000

# create grid along each axis
grid = jnp.linspace(-d, d, L)

# create grid for H
hilb_grid = jnp.linspace(-M, M, 2 * M + 1)

# generate input data
dmin, dmax, numSamples = -1., 0.1, 1000000
mu1, mu2, sigma1, sigma2 = 1, 1, np.sqrt(1), np.sqrt(2)
input_data1 = np.random.normal(mu1, sigma1, numSamples)
input_data2 = np.random.normal(mu2, sigma2, numSamples)
input_data = np.stack([input_data1, input_data2])

# compute cutoffs
min_input_data  = jnp.min(input_data)
max_input_data  = jnp.max(input_data)
CDFgrid_input   = jnp.linspace(min_input_data, max_input_data, 100)

# Compute CF and CDF of initial distribution
CFin1 = vfunc_IG_jit(grid, mu1, sigma1)
CFin2 = vfunc_IG_jit(grid, mu2, sigma2)
CFin = jnp.array([CFin1, CFin2])
CDFin = vfunc_CDF_jit(CFin, grid, CDFgrid_input)

# define layer architecture
n_in, n_hidden, n_out = 2, 10, 2

numTrials = 101
trials = np.arange(numTrials)
for trial in trials:
    
    # generate random weights and biases
    W0 = np.random.uniform(-1., 1., size=(n_hidden, n_in))
    W1 = np.random.uniform(-1., 1., size=(n_hidden, n_hidden))
    W2 = np.random.uniform(-1., 1., size=(n_hidden, n_hidden))
    W3 = np.random.uniform(-1., 1., size=(n_hidden, n_hidden))
    W4 = np.random.uniform(-1., 1., size=(n_out, n_hidden))
    b0 = np.random.uniform(-1., 1., n_hidden)
    b1 = np.random.uniform(-1., 1., n_hidden)
    b2 = np.random.uniform(-1., 1., n_hidden)
    b3 = np.random.uniform(-1., 1., n_hidden)
    b4 = np.random.uniform(-1., 1., n_out)

    # propagate MC samples
    hidden_data1_AL = propagateSamplesAL(input_data,      n_hidden, W0, b0, numSamples)
    hidden_data1_ML = propagateSamplesML(hidden_data1_AL, n_hidden,         numSamples)
    hidden_data2_AL = propagateSamplesAL(hidden_data1_ML, n_hidden, W1, b1, numSamples)
    hidden_data2_ML = propagateSamplesML(hidden_data2_AL, n_hidden,         numSamples)
    hidden_data3_AL = propagateSamplesAL(hidden_data2_ML, n_hidden, W2, b2, numSamples)
    hidden_data3_ML = propagateSamplesML(hidden_data3_AL, n_hidden,         numSamples)
    hidden_data4_AL = propagateSamplesAL(hidden_data3_ML, n_hidden, W3, b3, numSamples)
    hidden_data4_ML = propagateSamplesML(hidden_data4_AL, n_hidden,         numSamples)
    output_data     = propagateSamplesAL(hidden_data4_ML, n_out,    W4, b4, numSamples)

    # compute cutoffs
    min_hidden_data1_AL = jnp.min(hidden_data1_AL)
    min_hidden_data1_ML = jnp.min(hidden_data1_ML)
    min_hidden_data2_AL = jnp.min(hidden_data2_AL)
    min_hidden_data2_ML = jnp.min(hidden_data2_ML)
    min_hidden_data3_AL = jnp.min(hidden_data3_AL)
    min_hidden_data3_ML = jnp.min(hidden_data3_ML)
    min_hidden_data4_AL = jnp.min(hidden_data4_AL)
    min_hidden_data4_ML = jnp.min(hidden_data4_ML)
    min_output_data     = jnp.min(output_data)
    max_hidden_data1_AL = jnp.max(hidden_data1_AL)
    max_hidden_data1_ML = jnp.max(hidden_data1_ML)
    max_hidden_data2_AL = jnp.max(hidden_data2_AL)
    max_hidden_data2_ML = jnp.max(hidden_data2_ML)
    max_hidden_data3_AL = jnp.max(hidden_data3_AL)
    max_hidden_data3_ML = jnp.max(hidden_data3_ML)
    max_hidden_data4_AL = jnp.max(hidden_data4_AL)
    max_hidden_data4_ML = jnp.max(hidden_data4_ML)
    max_output_data     = jnp.max(output_data)
    CDFgrid_hidden1_AL  = jnp.linspace(min_hidden_data1_AL, max_hidden_data1_AL, 100)
    CDFgrid_hidden1_ML  = jnp.linspace(min_hidden_data1_ML, max_hidden_data1_ML, 100)
    CDFgrid_hidden2_AL  = jnp.linspace(min_hidden_data2_AL, max_hidden_data2_AL, 100)
    CDFgrid_hidden2_ML  = jnp.linspace(min_hidden_data2_ML, max_hidden_data2_ML, 100)
    CDFgrid_hidden3_AL  = jnp.linspace(min_hidden_data3_AL, max_hidden_data3_AL, 100)
    CDFgrid_hidden3_ML  = jnp.linspace(min_hidden_data3_ML, max_hidden_data3_ML, 100)
    CDFgrid_hidden4_AL  = jnp.linspace(min_hidden_data4_AL, max_hidden_data4_AL, 100)
    CDFgrid_hidden4_ML  = jnp.linspace(min_hidden_data4_ML, max_hidden_data4_ML, 100)
    CDFgrid_output      = jnp.linspace(min_output_data,     max_output_data,     100)

    # run verification
    tic = time.perf_counter()

    # Propagate through ReLU network
    CFhidden1_AL = vfunc_AL_jit(grid, CFin, b0, W0)
    CFhidden1_ML = vfunc_ML_jit(grid, CFhidden1_AL, grid, hilb_grid)
    CFhidden2_AL = vfunc_AL_jit(grid, CFhidden1_ML, b1, W1)
    CFhidden2_ML = vfunc_ML_jit(grid, CFhidden2_AL, grid, hilb_grid)
    CFhidden3_AL = vfunc_AL_jit(grid, CFhidden2_ML, b2, W2)
    CFhidden3_ML = vfunc_ML_jit(grid, CFhidden3_AL, grid, hilb_grid)
    CFhidden4_AL = vfunc_AL_jit(grid, CFhidden3_ML, b3, W3)
    CFhidden4_ML = vfunc_ML_jit(grid, CFhidden4_AL, grid, hilb_grid)
    CFout        = vfunc_AL_jit(grid, CFhidden4_ML, b4, W4)

    # Compute CDF from CF
    CDFhidden1_AL = vfunc_CDF_jit(CFhidden1_AL, grid, CDFgrid_hidden1_AL)
    CDFhidden1_ML = vfunc_CDF_jit(CFhidden1_ML, grid, CDFgrid_hidden1_ML)
    CDFhidden2_AL = vfunc_CDF_jit(CFhidden2_AL, grid, CDFgrid_hidden2_AL)
    CDFhidden2_ML = vfunc_CDF_jit(CFhidden2_ML, grid, CDFgrid_hidden2_ML)
    CDFhidden3_AL = vfunc_CDF_jit(CFhidden3_AL, grid, CDFgrid_hidden3_AL)
    CDFhidden3_ML = vfunc_CDF_jit(CFhidden3_ML, grid, CDFgrid_hidden3_ML)
    CDFhidden4_AL = vfunc_CDF_jit(CFhidden4_AL, grid, CDFgrid_hidden4_AL)
    CDFhidden4_ML = vfunc_CDF_jit(CFhidden4_ML, grid, CDFgrid_hidden4_ML)
    CDFout        = vfunc_CDF_jit(CFout, grid, CDFgrid_output)
    
    # report time taken
    toc = time.perf_counter()

    # plot CDFs
    fig1, axs1 = plt.subplots()
    plotAllCDF(input_data, CDFin, CDFgrid_input, axs1, 2)
    fig2, axs2 = plt.subplots(2)
    plotAllCDF(hidden_data1_AL, CDFhidden1_AL, CDFgrid_hidden1_AL, axs2[0], 5)
    plotAllCDF(hidden_data1_ML, CDFhidden1_ML, CDFgrid_hidden1_ML, axs2[1], 5)
    fig3, axs3 = plt.subplots(2)
    plotAllCDF(hidden_data2_AL, CDFhidden2_AL, CDFgrid_hidden2_AL, axs3[0], 5)
    plotAllCDF(hidden_data2_ML, CDFhidden2_ML, CDFgrid_hidden2_ML, axs3[1], 5)
    fig4, axs4 = plt.subplots(2)
    plotAllCDF(hidden_data3_AL, CDFhidden3_AL, CDFgrid_hidden3_AL, axs4[0], 5)
    plotAllCDF(hidden_data3_ML, CDFhidden3_ML, CDFgrid_hidden3_ML, axs4[1], 5)
    fig5, axs5 = plt.subplots(2)
    plotAllCDF(hidden_data4_AL, CDFhidden4_AL, CDFgrid_hidden4_AL, axs5[0], 5)
    plotAllCDF(hidden_data4_ML, CDFhidden4_ML, CDFgrid_hidden4_ML, axs5[1], 5)
    fig6, axs6 = plt.subplots()
    plotAllCDF(output_data, CDFout, CDFgrid_output, axs6, 2)
    plt.show()

    bkpt = 1


# # # # # # # # # # # # #