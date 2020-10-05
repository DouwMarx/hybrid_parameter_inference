import matplotlib.pyplot as plt
import scipy.integrate as inter
import numpy as np


class FirsOrderDESys(object):
    def __init__(self, EQ_func, initial_condition, time_range):
        self.EQ_func = EQ_func  # A and B matrices as function of time
        self.time_range = time_range
        self.initial_condition = initial_condition
        self.dim = len(initial_condition)

    def X_dot(self, t, X):
        E, Q = self.EQ_func(t)
        X_dot = np.dot(E, X) + Q

        return X_dot

    def solve(self, method):
        """
        Solve the system of differential equations
        :param method: method of solution, "RK45" "BDF"
        :return:
        """
        sol = inter.solve_ivp(self.X_dot,
                              [self.time_range[0], self.time_range[-1]],
                              self.initial_condition,
                              method=method,
                              dense_output=True,
                              t_eval=self.time_range,
                              vectorized=True)#,
                              #rtol=1e-9,
                              #atol=1e-12)
        return sol.y.T

    def get_Xdotdot(self, method):
        """
        Calculates accelerations from computed displacements and velocities
        Parameters
        ----------
        sol

        Returns
        -------

        """
        sol = self.solve(method)
        XXd = np.zeros((len(self.time_range), self.dim))
        #for i, timestep, in enumerate(self.time_range):
        for i, timestep, in enumerate(self.time_range):
            E, Q = self.EQ_func(timestep)
            acc = np.dot(E, np.array([sol[i]]).T) + Q

            XXd[i] = acc.reshape(-1)

        return XXd[:, int(self.dim / 2):]

    def plot_solution(self, solution, state_time_der):
        nstate = int(np.shape(solution)[0] / 3)
        if state_time_der == "Displacement":
            start = 0

        if state_time_der == "Velocity":
            start = nstate * 1

        if state_time_der == "Acceleration":
            start = nstate * 2

        plt.figure("Rotational DOF, carrier, sun, planet" + state_time_der)
        plt.plot(self.time_range, solution[:, start + 0], label="u_c")
        plt.plot(self.time_range, solution[:, start + 2], label="u_r")
        plt.plot(self.time_range, solution[:, start + 8], label="u_s")
        plt.plot(self.time_range, solution[:, start + 11], label="u_1")
        plt.legend()

        plt.figure("x-translation, carrier, sun, planet" + state_time_der)
        plt.plot(self.time_range, solution[:, start + 0], label="x_c")
        plt.plot(self.time_range, solution[:, start + 3], label="x_r")
        plt.plot(self.time_range, solution[:, start + 6], label="x_s")
        plt.plot(self.time_range, solution[:, start + 9], label="x_p1")
        plt.legend()

        plt.figure("Planet displacement")
        plt.plot(self.time_range, solution[:, start + 9], label="zeta_1")
        plt.plot(self.time_range, solution[:, start + 10], label="nu_1")
        plt.legend()

        # plt.plot(timerange, sol_nm[:, -3], label="Newmark")
        # plt.ylim(np.min(sol_rk[:, -3]),np.max(sol_rk[:, -3]))


class LMM_sys(object):
    """
    Lumped mass model object for second order equation of Newtons second law
    Time variable stiffness, All other parameters are constant
    """

    def __init__(self, M, C, K, f):
        """

        Parameters
        ----------
        M: Numpy array 2nx2n
        C  Numpy array 2nx2n
        K  Time dependent function K(t) = Numpy array 2nx2N
        f  Time dependent function f(t) = Numpy array 2nx1
        X0 Numpy array 2nx1
        Xd0 Numpy array 2nx1
        time_range Numpy array
        """

        self.M = M
        self.M_inv = np.linalg.inv(M)  # pre-compute as it is used a lot
        self.C = C
        self.K = K
        self.f = f

        self.dof = np.shape(self.M)[0]  # number of degrees of freedom

        E = np.zeros((self.dof * 2, self.dof * 2))  # *2 as we are working with second order system
        E[0:self.dof, self.dof:] = np.eye(self.dof)
        self.E = E  #Create an E matrix beforehand to speed things up

        Q = np.zeros((2 * self.dof, 1))
        self.Q = Q  # Create Q matrix beforehand for speedup
        return

    def E_Q(self, t):
        """
        Converts the second order differential equation to first order (E matrix and Q vector)

        Parameters
        ----------
        t  : Float
             Time

        Returns
        -------
        E  : 2x(9+3xN) x 2x(9+3xN) Numpy array

        Based on Runge-kutta notes

        """

        c_over_m = np.dot(self.M_inv, self.C(t))
        k_over_m = np.dot(self.M_inv, self.K(t))
        f_over_m = np.dot(self.M_inv, self.f(t))

        # E = np.zeros((self.dof * 2, self.dof * 2))  # *2 as we are working with second order system
        # E[self.dof:, 0:self.dof] = -k_over_m
        # E[self.dof:, self.dof:] = -c_over_m
        # E[0:self.dof, self.dof:] = np.eye(self.dof)

        self.E[self.dof:, 0:self.dof] = -k_over_m
        self.E[self.dof:, self.dof:] = -c_over_m

        # E = np.block([[np.zeros((self.dof,self.dof)),np.eye(self.dof)],
        #              [-k_over_m,-c_over_m]])


        # Q = np.zeros((2 * self.dof, 1))
        # Q[self.dof:] = f_over_m

        self.Q[self.dof:] = f_over_m

        #return E, Q
        return self.E, self.Q


def test_2dof_system():
    M = np.array([[1, 0], [0, 1]])
    C = lambda t: np.array([[10, 0], [-10, 10]])
    K = lambda t: np.array([[100, 0], [-100, 100]])
    f = lambda t: np.array([[100], [0]])

    X0 = np.array([0, 0])
    Xd0 = np.array([0, 0])

    lmm2dof = LMM_sys(M, C, K, f)
    init_cond = np.hstack((X0, Xd0))
    t_range = np.linspace(0, 1, 1000)

    de = FirsOrderDESys(lmm2dof.E_Q, init_cond, t_range)
    s = de.solve("RK45")

    plt.figure()
    plt.plot(s)
    plt.show()

    xdd = de.get_Xdotdot("RK45")

    plt.figure()
    plt.plot(xdd)
    plt.show()


def test_1dof_system():
    M = np.array([[1]])
    C = lambda t: np.array([[10]])
    K = lambda t: np.array([[200]])
    f = lambda t: np.array([[0]])

    X0 = np.array([-0.5])
    Xd0 = np.array([0])

    lmm2dof = LMM_sys(M, C, K, f, X0, Xd0)
    init_cond = np.hstack((X0, Xd0))
    t_range = np.linspace(0, 1, 1000)

    de = FirsOrderDESys(lmm2dof.E_Q, init_cond, t_range)
    s = de.solve("RK45")

    plt.figure()
    plt.plot(s)
    plt.show()

    xdd = de.get_Xdotdot("RK45")

    plt.figure()
    plt.plot(xdd)
    plt.show()

#test_2dof_system()
