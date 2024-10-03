import numpy as np
import cvxpy as cp


def define_classification_polytope_w_b(logits, class_index):
    """
    Defines the polytope for a given class index in a neural network classification
    based on the output logits and returns the list of [w_i, b_i] for each inequality.

    Parameters:
    logits (list or np.array): The output logits from the neural network.
    class_index (int): The index of the class whose polytope we want to define.

    Returns:
    list: A list of tuples [(w_i, b_i)] defining the polytope inequalities.
    """
    W = []
    B = []
    n_classes = len(logits)
    
    # For the given class_index, create inequalities z_class_index > z_j for all j != class_index
    for j in range(n_classes):
        if j != class_index:
            # Define w_i for the inequality z[class_index] - z[j] > 0
            w = np.zeros(n_classes)
            w[class_index] = 1  # coefficient for z[class_index]
            w[j] = -1  # coefficient for z[j]
            b = 0  # No bias term in these inequalities

            W.append(w)
            B.append(b)

    return W,B

def generate_A_b(W,B):
    """
    Generate the matrix A and vector b from the list of halfspaces defined by
    w_i^T x + b_i >= 0 for each halfspace.
    
    :param halfspaces: A list of tuples, where each tuple is (w, b), 
                       with w as the normal vector and b as the scalar term.
    :return: A (matrix) and b (vector) for the polytope defined by halfspaces.
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
        # Inside: Return the minimum distance
        return -1*min(distances)
    else:
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


