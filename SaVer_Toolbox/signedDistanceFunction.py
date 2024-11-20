import numpy as np        
import cvxpy as cp

class norm:
    def __init__(self,center, fixed_zero_radius,norm=2):
        """
        Initializes the Signed Distance Function (SDF) for a norm ball.
        Parameters:
        center (np.ndarray): The center of the norm ball.
        fixed_zero_radius (float): The fixed radius where the SDF is zero.
        norm (int, optional): The norm to be used (default is 2, which corresponds to the 2 norm).
        Attributes:
        center (np.ndarray): The center of the norm ball.
        norm (int): The norm to be used.
        fixed_zero_radius (float): The fixed radius where the SDF is zero.
        """
        
        self.center = center
        self.norm = norm
        self.fixed_zero_radius = fixed_zero_radius

    def reviseCenter(self, center):
        """
        Update the center attribute of the norm.
        Parameters:
        center (np.ndarray): The new center coordinates to be assigned to the object.
        """

        self.center = center

    def eval(self, point, zero_radius):
        """
        Evaluate the signed distance function for a norm ball.
        Parameters:
        point (np.ndarray): The point(s) at which to evaluate the signed distance function. 
                            Should be an array of shape (n_points, n_dimensions).
        zero_radius (float): The radius to subtract from the computed distance.
        Returns:
        np.ndarray: The evaluated signed distance for each point.
        """
        
        eval = np.zeros(point.shape[0])

        eval = np.linalg.vector_norm(point-self.center,axis=1,ord=self.norm) - self.fixed_zero_radius - zero_radius

        return eval
    

class polytope: 
    def __init__(self,W,B):
        """
        Initializes the polytope signed distance function.
        Parameters:
            W (numpy.ndarray): The A matrix.
            B (numpy.ndarray): The b vector.
        """

        self.W = W
        self.B = B

    def reviseCenter(self,W,B):
        """
        Update the weight matrix and bias vector of the polytope.
        Parameters:
            W (numpy.ndarray): The A matrix.
            B (numpy.ndarray): The b vector.
        """
        self.W = W
        self.B = B

    def eval(self, point, zero_radius):
        # Replace this with your actual signed distance function implementation
        eval = np.array([signed_distance_function(l,self.W,self.B) for l in point]) - zero_radius 
        return eval
    
def generate_A_b(W,B):
    """
    Generate the matrix A and vector b from the list of halfspaces defined by
    w_i^T x + b_i >= 0 for each halfspace.
    
    Parameters:
        W (numpy.ndarray): Rows of w_i^T.
        B (numpy.ndarray): array of b_i.
    Returns: 
    A (np.ndarray): A matrix for the polytope defined by halfspaces.
    b (np.ndarray): b vector for the polytope defined by halfspaces.
    """
    # Initialize lists to store A and b
    A = []
    b = []
    
    # Iterate over the list of halfspaces
    for w, b_i in zip(W,B):
        # Convert w_i^T x + b_i >= 0 into standard form -w_i^T x <= b_i
        A.append(-np.array(w))
        b.append(b_i)
    
    # Convert lists to numpy arrays
    A = np.array(A)
    b = np.array(b)
    
    return A, b


def signed_distance_function(s, w_list, b_list):
    """
    Computes the signed distance of point s to the intersection of half-spaces
    defined by w_i and b_i for each half-space, as per the given function.

    Parameters:
    s (np.array): The point whose signed distance we want to compute.
    w_list (list of np.array): List of normal vectors defining the half-spaces.
    b_list (list of float): List of offsets defining the half-spaces.

    Returns:
    float: The signed distance to the intersection of the half-spaces.
    """

    # Compute the signed distance for each half-space
    distances = [(np.dot(w, s) + b) / np.linalg.norm(w) for w, b in zip(w_list, b_list)]
    
    # Check if the point is inside the intersection (all distances are non-negative)
    if all(d >= 0 for d in distances):
        # Inside: Return the minimum distance:
        return -1*min(distances)
    else:
        # Outside: Return the minimum distance via cvxpy:
        A,b =  generate_A_b(w_list,b_list)
        x = cp.Variable(A.shape[1])
        x0 = np.array(s)
        objective = cp.Minimize(cp.sum_squares(x - x0))
        constraints = [A @ x <= b]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        x_star = x.value
        distance = np.linalg.norm(x_star - x0)
        return distance