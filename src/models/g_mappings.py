import numpy as np
from abc import ABC, abstractmethod, abstractproperty
import src.features.state_space as sp
import matplotlib.pyplot as plt


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

    def __init__(self, theta, x):
        super().__init__(theta, x)
        self.constants = {"t_range": np.linspace(0, 1, 100)}

    def get_y(self, c):
        y = self.x[0] * np.sin(2 * np.pi * c[0] * self.constants["t_range"] + self.theta[0])
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

    def __init__(self, theta, x):
        super().__init__(theta, x)
        self.constants = {"t_range": np.linspace(0, 1, 100)}

    def get_y(self, c):
        M = np.array([[self.theta[0], 0],
                      [0, 1]])
        # C = lambda t: np.array([[10, 0], [-10, 10]])
        # K = lambda t: np.array([[self.x[0], 0], [-self.x[0], 100]])
        C = lambda t: np.array([[10 + 10, -10], [-10, 10]])
        K = lambda t: np.array([[self.x[0] + 100, -100], [-100, 100]])

        f = lambda t: np.array([[c[0]], [0]])

        X0 = np.array([0, 0])
        Xd0 = np.array([0, 0])

        lmm2dof = sp.LMM_sys(M, C, K, f)  # Define a lumped mass model

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

    def __init__(self, theta, x):
        super().__init__(theta, x)
        self.constants = {"t_range": np.linspace(0, 1.5, 100, dtype="float32")}

    def get_y(self, c):
        M = np.array([[self.theta[0], 0],
                      [0, self.theta[1]]])
        C = lambda t: np.array([[self.theta[2] + self.theta[3], -self.theta[3]], [-self.theta[3], self.theta[3]]])
        K = lambda t: np.array([[self.x[0] + self.theta[4], self.theta[4]], [-self.theta[4], self.theta[4]]])
        f = lambda t: np.array([[c[0]], [0]])

        X0 = np.array([0, 0])
        Xd0 = np.array([0, 0])

        lmm2dof = sp.LMM_sys(M, C, K, f)  # Define a lumped mass model

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


class InitCondnDOFv2G(G):
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

    def __init__(self, n_dof, x):
        self.n_dof = n_dof  # Number of degrees of freedom of LMM system
        theta = self.make_parameters()
        super().__init__(theta, x)
        self.constants = {"t_range": np.linspace(0, 1.5, 100, dtype="float32")}

    def make_parameters(self):
        m_multiplier = 1
        c_multiplier = 10
        k_multiplier = 100

        # theta_dim = self.n_dof * (1 + 1 + 1) - 1  # DOF*(n_mass + n_damp + n_stiff) - 1damaged_stiff (x)
        # theta = [m...,c...,k...]
        theta_m = (np.random.rand(self.n_dof) + 1) * m_multiplier
        theta_c = (np.random.rand(self.n_dof) + 1) * c_multiplier
        theta_k = (np.random.rand(self.n_dof - 1) + 1) * k_multiplier

        return np.hstack((theta_m, theta_c, theta_k))

    def get_y(self, c):
        Cmat = np.diag(self.theta[1 * self.n_dof:2 * self.n_dof]) + \
               np.diag(np.hstack((self.theta[1 * self.n_dof + 1:2 * self.n_dof], np.zeros(1)))) - \
               np.diag(self.theta[1 * self.n_dof + 1:2 * self.n_dof], -1) - \
               np.diag(self.theta[1 * self.n_dof + 1:2 * self.n_dof], 1)

        Kmat = np.diag(np.hstack((self.x, self.theta[2 * self.n_dof:]))) + \
               np.diag(np.hstack((self.theta[2 * self.n_dof:], np.zeros(1)))) - \
               np.diag(self.theta[2 * self.n_dof:], -1) - \
               np.diag(self.theta[2 * self.n_dof:], 1)

        M = np.diag(self.theta[0:self.n_dof])
        C = lambda t: Cmat
        K = lambda t: Kmat

        fcol = np.zeros((self.n_dof, 1))
        fcol[0, 0] = c[0]  # Apply the step force to the first
        f = lambda t: fcol

        # X0 = np.array([0, 0])
        # Xd0 = np.array([0, 0])

        lmm2dof = sp.LMM_sys(M, C, K, f)  # Define a lumped mass model

        # init_cond = np.hstack((X0, Xd0))  # Set initial condition to zero
        init_cond = np.zeros(self.n_dof * 2)  # Set initial condition to zero
        t_range = self.constants["t_range"]

        de = sp.FirsOrderDESys(lmm2dof.E_Q, init_cond, t_range)
        y = de.get_Xdotdot("RK45")[:, 1]  # Measure accelerations at mass 2

        return y


class Chen2011(G):
    """
    Three degree of freedom system from Chen 2011
    Two masses connected with springs and dampers series, ground-m1-m2 with

    The response z is the accelerations at mas 1
    """

    def __init__(self, theta, x):
        super().__init__(theta, x)
        self.constants = {"t_range": np.linspace(0, 0.5, 10000, dtype="float32")}

    def get_y(self, c):
        m1 = 3.5
        I1 = 0.0082
        r1 = (2 * 55 / 2) / 1000
        k1 = 5e8
        c1 = 1.2e5

        m2 = 4.5
        I2 = 0.011
        r2 = (2 * 75 / 2) / 1000
        k2 = 5e8
        c2 = 1.2e5

        mg = I1 * I2 / (I1 * r1 ** 2 + I2 * r1 ** 2)
        cg = 4e5

        F3 = (I2 * r1 * m1 + I1 * r2 * m2) / (I1 * r2 ** 2 + I2 * r2 ** 2)

        M = np.array([[m1, 0, 0],
                      [0, m2, 0],
                      [-mg, mg, mg]])

        kg = lambda t: np.cos(2*np.pi*458.3*t)*1.3e9

        C = lambda t: np.array([[c1, 0, -cg], [0, c2, cg], [0, 0, cg]])
        K = lambda t: np.array([[k1, 0, -kg(t)], [0, k2, kg(t)], [0, 0, kg(t)]])
        f = lambda t: np.array([[0], [0], [F3]])

        X0t = np.linalg.solve(K(0),f(0))  # initial conditions that would lead to zero initial accelerations
        X0 = X0t[:,0]
        Xd0 = np.array([0, 0, 0.00001])

        lmm2dof = sp.LMM_sys(M, C, K, f)  # Define a lumped mass model

        init_cond = np.hstack((X0, Xd0))  # Set initial condition to zero
        t_range = self.constants["t_range"]

        de = sp.FirsOrderDESys(lmm2dof.E_Q, init_cond, t_range)
        y = de.get_Xdotdot("RK45")[:, 1]  # Measure accelerations at mass 2

        return y


# obj = InitCondnDOFv2G(4,np.array([999]))
# obj.get_y(np.array([100]))

obj = Chen2011(np.array([22]), np.array([33]))
y = obj.get_y(12)
plt.figure()
plt.plot(y)
