# SAVER: A toolbox for SAmpling-based, probabilistic VERification of neural networks

This package provides a collection of sampling-based approaches which allow one to verify neural networks just using samples. 

## Installation: 

Please follow the installation guide [here](REP.pdf) for system requirements, installation, and running examples in the paper.

## Elements in Toolbox:

### Verification Methods:
- **verify.DKW**: Employs the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality to provide probabilistic guarantees on the verification of neural networks via cumulative distribution functions (CDFs).
    - To initialize the DKW approach, we need to provide the following arguments: 
        1. beta (float): The confidence level parameter.
        2. epsilon (float): The error tolerance between the true and empirical CDF.
        3. Delta (float, optional): A parameter with a default value of 0.1.
- **verify.Scenario**: Employs the Scenario approach to provide probabilistic guarantees on the verification of neural networks via convex optimization. 
    - To initialize the Scenario approach, we need to provide the following arguments: 
        1. beta (float): The confidence level parameter.
        3. Delta (float, optional): A parameter with a default value of 0.1.

### Set Representations via Signed Distance Functions:
- **signedDistanceFunction.norm**: Implements the norm-ball as a signed distance functions.
    - To initialize the norm-ball, we need to provide the following arguments: 
        1. center (np.ndarray): The center of the norm ball.
        2. fixed_zero_radius (float): The fixed radius where the SDF is zero.
        3. norm (int, optional): The norm to be used (default is 2, which corresponds to the 2 norm).
- **signedDistanceFunction.polytope**: Implements the polytope as a signed distance functions.
    - To initialize the polytope, we need to provide the following arguments: 
        1. W (numpy.ndarray): The A matrix.
        2. B (numpy.ndarray): The b vector.


## Functions in Toolbox Used for Verification: 

Once we intialize a verification method, e.g. "verificationMethod," we can use the following functions: 
- **verificationMethod.samplesRequired()**: Provides the number of samples required by the verification method.
- **verificationMethod.specification()**: Add the specification to the verification method.

- **verificationMethod.probability()**: Computes the probability of the verification method.
- **verificationMethod.modifySet()**: Modifies the set specification to satisfy the probability of satisfaction, 1-Delta.

## Using the Toolbox for Your Use Case (Code Reuse):

As suggested in the HSCC 2025 repeatability evaluation page [here](https://hscc.acm.org/2025/repeatability-evaluation/), one can adapt this toolbox to their verification task by using the following template:

```python
###################
## example.py
###################
import numpy as np
from SaVer_Toolbox import signedDistanceFunction, verify

# Intialize the sampling methods with user defined error, violation, and confidence: 
betaDKW = 0.001
epsilonDKW = 0.001
Delta = 1-0.999
verifDKW = verify.usingDKW(betaDKW,epsilonDKW,Delta)
betaScenario = 0.001
verifScenario = verify.usingScenario(betaScenario,Delta)

# Call your sampler or simulator with provided `samplesRequired()' function: 

samplesDKW = your_sampler(verifDKW.samplesRequired())
samplesScenario = your_sampler(verifScenario.samplesRequired())

# Add samples to the verifier: 

verifDKW.samples(samplesDKW)
verifScenario.samples(samplesScenario)

# Check if the samples satisfy the specification: 

verifDKW.probability()
verifScenario.probability()

# Modify the set specification:
setReductionDKW = verifDKW.modifySet()
setReductionScenario = verifScenario.modifySet()

# Check again the samples satisfy the specification now it is modified: 
verifDKW.probability()
verifScenario.probability()
```

You can run this directly in the directory via the following steps after loading/building the Docker image following the installation guide [here](REP.pdf):

1. Open terminal or command line interface. Move to the folder that contains your Python file:
    ```bash
    cd folder_where_example_python_file_is
    ```
2. Run the following command to run in the Docker image:

    macOS or Linux:
    ```bash
    docker run --rm -it -v ./:/current_run saver-toolbox sh -c "cd /current_run && python3 ./example.py"
    ```
    Windows:
    ```bash
    docker run --rm -it -v .\:/current_run saver-toolbox sh -c "cd /current_run && python3 ./example.py"
    ```
