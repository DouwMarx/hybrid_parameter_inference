import numpy as np
from abc import ABC, abstractmethod


class H(ABC):
    """
    Abstract base class that defines how a H class should look like.
    The model h is the data-driven measurement model that links the measured reality with the physics based model (g) output.
    """

    def __init__(self, phi):
        """
        Initializes class H for a set of model parameters phi
        :param phi: numpy array of model parameters for model h
        """
        self.phi = phi
        self.n_parameters = len(self.phi)

    @abstractmethod
    def get_z(self, y):
        """
        Translates the model generated response to the real world measured response
        :param y: The response from a G object
        :return:z Observed variable
        """
        pass


class BasicConceptValidationH(H):
    """
    Simplest toy problem to aid implementation
    """

    def get_z(self, y):
        z = self.phi[0] * y[0]
        return np.array([z])


class MultiDimH(H):
    """
    Simple problem with multi dimensional measurement vector
    """

    def get_z(self, y):
        z = self.phi[0]*y
        return z

class Constant(H):
    """
    Constant multiplication factor as transfer function
    """

    def get_z(self, y):
        z = self.phi[0]*y
        return z

class InitCond2DOFv1H(H):
    """
    Measurement function for 2DOF system is a simple multiplication with a constant
    """

    def get_z(self, y):
        z = self.phi[0]*y
        return z

class NonLinQuadratic(H):
    """
    Measurement function for system is modification by quadratic equation
    """

    def get_z(self, y):
        z = self.phi[0]*y**2 + self.phi[1]*y + self.phi[2]
        return z
