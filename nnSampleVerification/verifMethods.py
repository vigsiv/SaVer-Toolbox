import numpy as np
from scipy import optimize


class dkw:

    def __init__(self,epsilon,beta):

        self.epsilon = epsilon
        self.beta = beta

        self.num_samples = np.ceil(-np.log(self.beta / 2) / (2 * self.epsilon**2)).astype(int)

    def addSpecification(self,sdf):
        self.sdf = sdf

    def addSamples(self,samples):
        self.samples = samples

    def findZeroOne(self,initPoint):

        evalCheck = self.empiricalCDF(initPoint)

        if evalCheck == 1.0 or evalCheck == 0.0:
            raise ValueError("Choose a different initial point")

        self.zeroOne = np.zeros(2)
        self.tol = 1e-12
        # Find the zero crossing
        self.zeroOne[0] = optimize.root_scalar(self.empiricalCDF, x0=initPoint, x1=initPoint+0.1, xtol=self.tol).root
        complementZero = lambda x: 1-self.empiricalCDF(x)
        self.zeroOne[1] = optimize.root_scalar(complementZero, x0=initPoint, x1=initPoint-0.1, xtol=self.tol).root

    def empiricalCDF(self,eval):
        # Calculate the empirical CDF of the signed distance function
        # This is a simple wrapper around the numpy function

        return np.sum(self.sdf.eval(self.samples,eval) <= 0)/self.samples.shape[0]
    
    def findLevelSet(self,prob):
        # Find the level set of the CDF that corresponds to the zero radius
        # This is a simple wrapper around the optimization function

        levelCDF = lambda x: self.empiricalCDF(x) - prob

        return optimize.bisect(levelCDF, self.zeroOne[0], self.zeroOne[1],xtol=self.tol)
    

class scenario: 

    def __init__(self,Delta,beta):

        self.num_samples = np.ceil(1/Delta*(np.e/(np.e-1))*(np.log(1/beta)+1)).astype(int)

    def addSpecification(self,sdf):
        self.sdf = sdf

    def addSamples(self,samples):
        self.samples = samples

    def findLevelSet(self):

        levelSetEval = self.sdf.eval(self.samples,0)
        return np.max(levelSetEval)
    

