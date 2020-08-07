import numpy as np
from abc import ABC, abstractmethod, abstractproperty
import src.features.state_space as sp


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

class InitCond2DOFv1G(G):
    """
    Two degree of freedom spring mass system from initial conditions
    Two masses connected with springs and dampers series, ground-m1-m2 with
    =|/\/\/[m1]/\/\[m2]
    Subject to step force @ m1 at t=0

    theta = [m1]
    x = [k1]

    all other parameters assumed to be known

    The response z is the accelerations at mas 2
    """
    def __init__(self,theta,x):
        super().__init__(theta,x)
        self.constants = {"t_range": np.linspace(0,1,100)}

    def get_y(self, c):
        M = np.array([[self.theta[0], 0],
                      [0, 1]])
        C = lambda t: np.array([[10, 0], [-10, 10]])
        K = lambda t: np.array([[self.x[0], 0], [-self.x[0], 100]])
        f = lambda t: np.array([[c[0]], [0]])

        X0 = np.array([0, 0])
        Xd0 = np.array([0, 0])

        lmm2dof = sp.LMM_sys(M, C, K, f)  #Define a lumped mass model

        init_cond = np.hstack((X0, Xd0))  # Set initial condition to zero
        t_range = self.constants["t_range"]

        de = sp.FirsOrderDESys(lmm2dof.E_Q, init_cond, t_range)
        y = de.get_Xdotdot("RK45")[:, 1]  # Measure accelerations at mass 2

        return y


class InitCond2DOFv2G(G):
    """
    Two degree of freedom spring mass system from initial conditions
    Two masses connected with springs and dampers series, ground-m1-m2 with
    =|/\/\/[m1]/\/\[m2]
    Subject to step force c @ m1 at t=0

    theta = [m1,m2,c1,c2,k2]
    x = [k1]

    all model parameters unknown. initial conditions known

    The response z is the accelerations at mas 2
    """
    def __init__(self,theta,x):
        super().__init__(theta,x)
        self.constants = {"t_range": np.linspace(0,1.5,100, dtype="float32")}


    def get_y(self, c):
        M = np.array([[self.theta[0], 0],
                      [0, self.theta[1]]])
        C = lambda t: np.array([[self.theta[2], 0], [-self.theta[2], self.theta[3]]])
        K = lambda t: np.array([[self.x[0], 0], [-self.x[0], self.theta[4]]])
        f = lambda t: np.array([[c[0]], [0]])

        X0 = np.array([0, 0])
        Xd0 = np.array([0, 0])

        lmm2dof = sp.LMM_sys(M, C, K, f)  #Define a lumped mass model

        init_cond = np.hstack((X0, Xd0))  # Set initial condition to zero
        t_range = self.constants["t_range"]

        de = sp.FirsOrderDESys(lmm2dof.E_Q, init_cond, t_range)
        y = de.get_Xdotdot("RK45")[:, 1]  # Measure accelerations at mass 2

        return y


class InitCond1DOFv1G(G):
    """
    1DOF model to model 2DOF reality
    Subject to step force c @ m1 at t=0

    theta = [m1,c1]
    x = [k1]

    all model parameters unknown. initial conditions known

    The response z is the accelerations at mas 2
    """

    def __init__(self, theta, x):
        super().__init__(theta, x)
        self.constants = {"t_range": np.linspace(0, 1.5, 100, dtype="float32")}

    def get_y(self, c):
        M = np.array([[self.theta[0]]])
        C = lambda t: np.array([[self.theta[1]]])
        K = lambda t: np.array([[self.x[0]]])
        f = lambda t: np.array([[c[0]]])

        X0 = np.array([0])
        Xd0 = np.array([0])

        lmm2dof = sp.LMM_sys(M, C, K, f)  # Define a lumped mass model

        init_cond = np.hstack((X0, Xd0))  # Set initial condition to zero
        t_range = self.constants["t_range"]

        de = sp.FirsOrderDESys(lmm2dof.E_Q, init_cond, t_range)
        y = de.get_Xdotdot("RK45")[:, 0]  # Measure accelerations at mass 1

        return y
