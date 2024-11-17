import numpy as np


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


