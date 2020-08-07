import numpy as np
from abc import ABC, abstractmethod, abstractproperty


class G(ABC):
    """
    Abstract base class that defines what a G class should look like.
    The model g is the physics-based measurement model that links the damage state with the expected response
    """

    def __init__(self, theta, x):
        """
        Initializes class G for a set of model parameters phi
        :param phi: numpy array of model parameters for model g
        """
        self.theta = theta
        self.n_parameters = len(self.theta)
        self.x = x

    @abstractmethod
    def get_y(self, c):
        """
        Calculates the physics-based model response
        :param c: The operating condition
        :return:y Response of the physics based model
        """
        pass


class BasicConceptValidationG(G):
    """
    Simplest toy problem to aid implementation
    """

    def __init__(self, theta, x):
        super().__init__(theta, x)
        self.constants = None

    def get_y(self, c):
        y = c[0] * self.x[0] + self.theta[0]
        return np.array([y])


class MultiDimG(G):
    """
    Simple problem with multi dimensional measurement vector
    """
    def __init__(self,theta,x):
        super().__init__(theta,x)
        self.constants = {"t_range":np.linspace(0,1,100)}

    def get_y(self, c):
        y = self.x[0]*np.sin(2*np.pi*c[0] * self.constants["t_range"] + self.theta[0])
        return y

class InitCond2DOF(G):
    """
    Two degree of freedom spring mass system from initial conditions
    """
    def __init__(self,theta,x):
        super().__init__(theta,x)
        self.constants = {"t_range": np.linspace(0,1,100)}

    def get_y(self, c):
        y = self.x[0]*np.sin(2*np.pi*c[0] * self.constants["t_range"] + self.theta[0])
        return y
