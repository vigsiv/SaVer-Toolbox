import numpy as np


class norm:
    def __init__(self,center,norm=2):
        
        self.center = center
        self.norm = norm

    def reviseCenter(self,center):
        self.center = center

    def eval(self, point, zero_radius):
        # Calculate the signed distance of a point from the set
        # Replace this with your actual signed distance function implementation

        eval = np.zeros(point.shape[0])

        eval = np.linalg.vector_norm(point-self.center,axis=1,ord=self.norm) - zero_radius

        return eval