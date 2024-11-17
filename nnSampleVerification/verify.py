import numpy as np
from scipy import optimize


class usingDKW:

    def __init__(self,beta,epsilon,Delta=0.1):
        self.epsilon = epsilon
        self.beta = beta
        self.Delta = Delta
        self.theta = 0.0
        self.numSamples = np.ceil(-np.log(self.beta / 2) / (2 * self.epsilon**2)).astype(int)

        print("-----------------------------------------------------------------")
        print(f"Error between true and empirical CDF (i.e. \sup_x(|cdfTrue(x) - cdfEmpirical(x)|) < \epsilon): epsilon = {self.epsilon}")
        print(f"Confidence level (i.e. P(\sup_x(|cdfTrue(x) - cdfEmpirical(x)|) < \epsilon) >= 1-\\beta): 1-beta = {1 - self.beta}")
        _ = self.samplesRequired()
        print("Please add the specification by calling \"usingDKW.addSpecification()\" function.")
        print("Then, add samples by calling \"usingDKW.samples()\" function.")
        print("-----------------------------------------------------------------")

    def empiricalCDF(self,eval):
        # Calculate the empirical CDF of the signed distance function
        # This is a simple wrapper around the numpy function

        return np.sum(self.signedDistanceFunction.eval(self.samples,eval+self.theta) <= 0)/self.samples.shape[0]
    
    def probability(self):
        # Check if the specification is satisfied
        # This is a simple wrapper around the empirical CDF calculation

        probabilityEval = self.empiricalCDF(0)
        satisfied = probabilityEval + self.epsilon >= 1 - self.Delta
        print("-----------------------------------------------------------------")
        if satisfied:
            print(f"Specification satisfied via DKW approach with probability: {probabilityEval * 100}% +/- {self.epsilon* 100}% which is near {(1 - self.Delta)*100}% with +/- {self.epsilon* 100}%")
        else:
            print(f"Specification NOT satisfied via DKW approach as probability is {probabilityEval * 100}% +/- {self.epsilon* 100}% which is less than {(1 - self.Delta) *100}%")
            print("Run \"usingDKW.modifySet()\" to find the set modification that satisfies the specification at satisfaction probability.")
        print("-----------------------------------------------------------------")

    
    def modifySet(self,rootFindingTolerance=1e-12,rootFindingMaxIter=100, verbose=False):
        # Find the level set of the CDF that corresponds to the zero radius
        # This is a simple wrapper around the optimization function

        levelCDF = lambda x: self.empiricalCDF(x) - 1 + self.Delta

        bound = np.median(self.signedDistanceFunction.eval(self.samples, 0))

        lb = bound - 10**np.floor(np.log10(abs(bound)))
        ub = bound + 10**np.floor(np.log10(abs(bound)))

        attempts = 0
        while attempts < rootFindingMaxIter:
            try: 
                theta = optimize.bisect(levelCDF, lb, ub, xtol=rootFindingTolerance)
                print("-----------------------------------------------------------------")
                print(f"DKW-based theta is modified to {theta} from the previous value: {self.theta}")
                print("Please rerun \"usingDKW.probability()\" to check if the specification is satisfied.")
                self.theta = theta
                print("-----------------------------------------------------------------")
                break
            except:
                attempts += 1
                lb = bound - 100*attempts*10**np.floor(np.log10(abs(bound)))
                ub = bound + 100*attempts*10**np.floor(np.log10(abs(bound)))
                if verbose:
                    print(f"Root finding attempt {attempts} failed. Trying again with lower bound of {lb} and upper bound of {ub}.")

        if attempts == rootFindingMaxIter:
            print("Root finding failed for finding set modification that meets satisfaction probability not found for given parameters.")

        return theta

    def specification(self,signedDistanceFunction):
        self.signedDistanceFunction = signedDistanceFunction
        if hasattr(self,'signedDistanceFunction'):
            print("-----------------------------------------------------------------")
            print("Specification updated in DKW approach. Please add samples by calling 'usingDKW.samples()' function.")
            print("----------------------------------------------------------------")
        else: 
            print("-----------------------------------------------------------------")
            print("Specification added in DKW approach. Please add samples by calling 'usingDKW.samples()' function.")
            print("----------------------------------------------------------------")

    def samples(self,samples):
        if not hasattr(self, 'signedDistanceFunction'):
            raise ValueError(f"Specification (SDF) not added. Please add the specification by calling '{self.__class__.__name__}.addSpecification()' function.")
        self.samples = samples

    def samplesRequired(self):
        print(f"Number of samples needed from simulator/sampler for DKW: {self.numSamples}")
        return self.numSamples
    


class usingScenario: 

    def __init__(self,beta,Delta=0.1):
        self.theta = 0.0
        self.beta = beta
        self.Delta = Delta
        self.numSamples = np.ceil(1/Delta*(np.e/(np.e-1))*(np.log(1/beta)+1)).astype(int)

        print("-----------------------------------------------------------------")
        print(f"Confidence level (i.e. P(P(g_C(f(x)) <= 0) >= 1-\Delta) >= 1-\\beta): 1-beta = {1 - self.beta}, 1-Delta = {1 - self.Delta}")
        _ = self.samplesRequired()
        print("Please add the specification by calling \"usingScenario.addSpecification()\" function.")
        print("Then, add samples by calling \"usingScenario.samples()\" function.")
        print("-----------------------------------------------------------------")

    def probability(self):
        # Check if the specification is satisfied
        # This is a simple wrapper around scenario evaluation: 

        theta = self.solveScenario()
        satisfied = theta <= 0
        print("-----------------------------------------------------------------")
        if satisfied and theta < 0:
            print(f"Specification satisfied via scenario approach as set signed distance function is reduced to {theta} which is less than 0")
        elif satisfied and theta == 0:
            print(f"Specification satisfied via scenario approach as set is {theta} which is equal to 0")
        else:
            print(f"Specification NOT satisfied via scenario approach as set signed distance function is expanded to {theta} which is greater than 0")
            print("Run \"usingScenario.modifySet()\" to find the set modification that satisfies the specification at satisfaction probability.")
        print("-----------------------------------------------------------------")

    def modifySet(self):
        # Find the level set of the CDF that corresponds to the zero radius
        # This is a simple wrapper around the optimization function
        theta = self.solveScenario()
        print("-----------------------------------------------------------------")
        print(f"Scenario-based theta is modified to {theta} from the previous value: {self.theta}")
        print("Please rerun \"usingScenario.probability()\" to check if the specification is satisfied.")
        self.theta = theta
        print("-----------------------------------------------------------------")
        return theta

    def specification(self,signedDistanceFunction):
        self.signedDistanceFunction = signedDistanceFunction
        if hasattr(self,'signedDistanceFunction'):
            print("-----------------------------------------------------------------")
            print("Specification updated in scenario approach. Please add samples by calling 'usingScenario.samples()' function.")
            print("----------------------------------------------------------------")
        else: 
            print("-----------------------------------------------------------------")
            print("Specification added in scenario approach. Please add samples by calling 'usingScenario.samples()' function.")
            print("----------------------------------------------------------------")

    def samples(self,samples):
        if not hasattr(self, 'signedDistanceFunction'):
            raise ValueError(f"Specification (SDF) not added. Please add the specification by calling '{self.__class__.__name__}.addSpecification()' function.")
        self.samples = samples

    def samplesRequired(self):
        print(f"Number of samples needed from simulator/sampler for Scenario: {self.numSamples}")
        return self.numSamples

    def solveScenario(self):
        if not hasattr(self, 'samples'):
            raise ValueError("Samples not added to scenario approach. Please add the samples by calling 'usingScenario.samples()' function.")
        levelSetEval = self.signedDistanceFunction.eval(self.samples,0) - self.theta
        return np.max(levelSetEval)


