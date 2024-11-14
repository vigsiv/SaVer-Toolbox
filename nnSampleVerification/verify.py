import numpy as np
from scipy import optimize


class usingDKW:

    def __init__(self,beta,epsilon,Delta=0.1):
        self.epsilon = epsilon
        self.beta = beta
        self.Delta = Delta
        self.setScaling = 0.0
        self.numSamples = np.ceil(-np.log(self.beta / 2) / (2 * self.epsilon**2)).astype(int)

        print("-----------------------------------------------------------------")
        print(f"Error between true and empirical CDF (i.e. \sup_x(|cdfTrue(x) - cdfEmpirical(x)|) < \epsilon): epsilon = {self.epsilon}")
        print(f"Confidence level (i.e. P(\sup_x(|cdfTrue(x) - cdfEmpirical(x)|) < \epsilon) >= 1-\\beta): 1-beta = {1 - self.beta}")
        _ = self.samplesRequired()
        print("Please add the specification by calling \"usingDKW.addSpecification()\" function.")
        print("-----------------------------------------------------------------")

    def empiricalCDF(self,eval):
        # Calculate the empirical CDF of the signed distance function
        # This is a simple wrapper around the numpy function

        return np.sum(self.signedDistanceFunction.eval(self.samples,eval+self.setScaling) <= 0)/self.samples.shape[0]
    
    def SpecificationSatisfied(self):
        # Check if the specification is satisfied
        # This is a simple wrapper around the empirical CDF calculation

        probabilityEval = self.empiricalCDF(0)
        satisfied = probabilityEval + self.epsilon >= 1 - self.Delta
        print("-----------------------------------------------------------------")
        if satisfied:
            print(f"Specification satisfied with probability: {probabilityEval * 100}% +/- {self.epsilon* 100}% which is greater than {1 - self.Delta}")
        else:
            print(f"Specification NOT satisfied as probability is {probabilityEval * 100}% +/- {self.epsilon* 100}% which is less than {1 - self.Delta}")
            print("Run \"usingDKW.modifySetScaling()\" to find the scaling factor that satisfies the specification at satisfaction probability. Rerun \"usingDKW.SpecificationSatisfied()\" to check if the specification is satisfied.")
        print("-----------------------------------------------------------------")

    
    def modifySetScaling(self,rootFindingTolerance=1e-6,rootFindingMaxIter=100):
        # Find the level set of the CDF that corresponds to the zero radius
        # This is a simple wrapper around the optimization function

        levelCDF = lambda x: self.empiricalCDF(x) - 1 + self.Delta

        bound = np.median(self.signedDistanceFunction.eval(self.samples, 0))

        lb = bound - 10**np.floor(np.log10(abs(bound)))
        ub = bound + 10**np.floor(np.log10(abs(bound)))

        attempts = 0
        while attempts < rootFindingMaxIter:
            try: 
                setScaling = optimize.bisect(levelCDF, lb, ub, xtol=rootFindingTolerance)
                print("-----------------------------------------------------------------")
                print(f"DKW-based scaling factor is modified to {setScaling} from the previous value: {self.setScaling}")
                self.setScaling = setScaling
                break
            except:
                attempts += 1
                lb = bound - 100*attempts*10**np.floor(np.log10(abs(bound)))
                ub = bound + 100*attempts*10**np.floor(np.log10(abs(bound)))
                print(f"Root finding attempt {attempts} failed. Trying again with lower bound of {lb} and upper bound of {ub}.")

        if attempts == rootFindingMaxIter:
            print("Root finding failed for finding set scaling that meets satisfaction probability not found for given parameters.")

    def addSpecification(self,signedDistanceFunction):
        self.signedDistanceFunction = signedDistanceFunction

    def addSamples(self,samples):
        if not hasattr(self, 'signedDistanceFunction'):
            raise ValueError(f"Specification (sdf) not added. Please add the specification by calling '{self.__class__.__name__}.addSpecification()' function.")
        self.samples = samples

    def samplesRequired(self):
        print(f"Number of samples needed from simulator/sampler: {self.numSamples}")
        return self.numSamples
    


class usingScenario: 

    def __init__(self,beta,Delta=0.1):
        self.setScaling = 0.0
        self.beta = beta
        self.Delta = Delta
        self.numSamples = np.ceil(1/Delta*(np.e/(np.e-1))*(np.log(1/beta)+1)).astype(int)

        print("-----------------------------------------------------------------")
        print(f"Confidence level (i.e. P(P(g_C(f(x)) <= 0) >= 1-\Delta) >= 1-\\beta): 1-beta = {1 - self.beta}, 1-Delta = {1 - self.Delta}")
        _ = self.samplesRequired()
        print("Please add the specification by calling \"usingScenario.addSpecification()\" function.")
        print("-----------------------------------------------------------------")

    def SpecificationSatisfied(self):
        # Check if the specification is satisfied
        # This is a simple wrapper around scenario evaluation: 

        setScaling = self.solveScenario()
        satisfied = setScaling <= 0
        print("-----------------------------------------------------------------")
        if satisfied and setScaling <= 0:
            print(f"Specification satisfied as scaling is {setScaling} which is less than 0")
        elif satisfied and setScaling == 0:
            print(f"Specification satisfied as scaling is {setScaling} which is equal to 0")
        else:
            print(f"Specification NOT satisfied as scaling is {setScaling} which is greater than 0")
            print("Run \"usingScenario.modifySetScaling()\" to find the scaling factor that satisfies the specification at satisfaction probability. Rerun \"usingScenario.SpecificationSatisfied()\" to check if the specification is satisfied.")
        print("-----------------------------------------------------------------")

    def modifySetScaling(self):
        # Find the level set of the CDF that corresponds to the zero radius
        # This is a simple wrapper around the optimization function
        setScaling = self.solveScenario()
        print("-----------------------------------------------------------------")
        print(f"Scenario-based scaling factor is modified to {setScaling} from the previous value: {self.setScaling}")
        self.setScaling = setScaling
        print("-----------------------------------------------------------------")

    def addSpecification(self,signedDistanceFunction):
        self.signedDistanceFunction = signedDistanceFunction

    def addSamples(self,samples):
        if not hasattr(self, 'signedDistanceFunction'):
            raise ValueError(f"Specification (sdf) not added. Please add the specification by calling '{self.__class__.__name__}.addSpecification()' function.")
        self.samples = samples

    def samplesRequired(self):
        print(f"Number of samples needed from simulator/sampler: {self.numSamples}")
        return self.numSamples

    def solveScenario(self):
        if not hasattr(self, 'samples'):
            raise ValueError("Samples not added. Please add the samples by calling 'usingScenario.addSamples()' function.")
        levelSetEval = self.signedDistanceFunction.eval(self.samples,0) - self.setScaling
        return np.max(levelSetEval)


