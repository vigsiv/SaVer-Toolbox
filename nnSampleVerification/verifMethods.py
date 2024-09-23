import numpy as np
from scipy import optimize


class dkw:

    def __init__(self,epsilon,alpha):

        self.epsilon = epsilon
        self.alpha = alpha

        self.num_samples = np.ceil(-np.log(self.alpha / 2) / (2 * self.epsilon**2)).astype(int)

    def addSpecification(self,sdf):
        self.sdf = sdf

    def addSamples(self,samples):
        self.samples = samples

    def findZeroOne(self):
        self.zeroOne = np.zeros(2)
        self.tol = 1e-12
        # Find the zero crossing
        self.zeroOne[0] = optimize.root_scalar(self.empiricalCDFGen, x0=1.2, x1=1.2+1, xtol=self.tol).root
        complementZero = lambda x: 1-self.empiricalCDFGen(x)
        self.zeroOne[1] = optimize.root_scalar(complementZero, x0=1.2, x1=1.2-1, xtol=self.tol).root

    def empiricalCDFGen(self,eval):
        # Calculate the empirical CDF of the signed distance function
        # This is a simple wrapper around the numpy function

        return np.sum(self.sdf.eval(self.samples,eval) <= 0)/self.samples.shape[0]
    
    def findLevelSet(self,prob):
        # Find the level set of the CDF that corresponds to the zero radius
        # This is a simple wrapper around the optimization function

        levelCDF = lambda x: self.empiricalCDFGen(x) - prob

        return optimize.bisect(levelCDF, self.zeroOne[0], self.zeroOne[1],xtol=self.tol)
    

