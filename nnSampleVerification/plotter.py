import matplotlib.pyplot as plt
import numpy as np

def plot2DNormLevelSet(center,zero_radius,samples):

    # Generate samples within the norm-ball
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + zero_radius * np.cos(theta)
    y = center[1] + zero_radius * np.sin(theta)

    # Plot the norm-ball
    plt.plot(x, y, color='blue', label='Norm-Ball')

    # Plot the samples
    plt.scatter(samples[:, 0], samples[:, 1], color='red', label='Samples')

    # Set plot labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    # Set the aspect ratio of the plot to 'equal'
    plt.axis('equal')

    # Limit the x-axis and y-axis
    plt.xlim(center[0] - zero_radius - 1, center[0] + 2*zero_radius + 1)
    plt.ylim(center[1] - zero_radius - 1, center[1] + 2*zero_radius + 1)

    # Show the plot
    plt.show()

def histogramPlot(empiricalCDF,eval,zero_radius,prob):

    eCDF = np.zeros(eval.shape[0])

    # Generate the histogram of the signed distance function
    for i in range(0,eval.shape[0]):
        eCDF[i] = empiricalCDF(eval[i])

    # Plot the histogram
    plt.plot(eval, eCDF, color='blue', label='Empirical CDF')
    plt.axvline(x=zero_radius,ymin=0, ymax=1, color='red', label='Probabilistic Level Set')
    plt.axhline(y=prob, color='green', label='Desired Probability')

    # Set plot labels
    plt.xlabel('Signed Distance Function')
    plt.ylabel('CDF Value')
    plt.legend()

    # Show the plot
    plt.show()