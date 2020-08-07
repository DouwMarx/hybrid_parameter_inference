import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from abc import ABC, abstractmethod


class System(object):
    """
    A total system consisting of physics-based model g and data driven model h
    """

    def __init__(self, g_obj, h_obj):
        self.g = g_obj
        self.h = h_obj

    def simulate(self, c, plot=False, noise=None):
        """
        Simulates a system for a given opperating condition
        :param c: array, operating conditions
        :return z: response
        """
        if isinstance(c, (tuple, list)):
            z = [self.h.get_z(self.g.get_y(ci)) for ci in c]

        else:
            y = self.g.get_y(c)
            z = self.h.get_z(y)

        if noise:  # Add noise to the measurements if noise is not None
            z = np.random.normal(z, noise["sd"])

        if plot:
            self.plot_simulate(z)

        return z

    def plot_simulate(self, z):
        plt.figure()
        for condition,zi in enumerate(z):
            plt.plot(zi, ".",label = "c" + str(condition+1))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def get_parameter_summary(self, print_addition = ""):
        print("Parameter summary: " + print_addition)
        print("=================")
        print("Damage state, x: ", self.g.x)
        print("Parameters for g, theta: ", self.g.theta)
        print("Parameters for h, phi: ", self.h.phi)


class ParameterInference(ABC):
    """
    Used to find the optimal model parameters for a given set of measurements
    """

    def __init__(self, sys_obj, measurements):
        """
        :param sys_obj:
        :param measurements: a dictionary with condition and corresponding measurement {"c":(c1,c2...cn),"z":(z1,z2...zn)}
        """
        self.sys = sys_obj
        self.measurements = measurements

    def build_candidate_sys(self, parameters):
        pass

    def cost(self, parameters):
        self.build_candidate_sys(parameters)
        z = self.sys.simulate(self.measurements["c"])  # Simulate model for all operating conditions
        cost_for_each_measurement = [np.log(np.linalg.norm(z[i] - self.measurements["z"][i])) for i in range(len(z))]
        return np.sum(cost_for_each_measurement)

    def run_optimisation(self, startpoint,bounds):
        #sol = opt.minimize(self.cost, startpoint)#, bounds=bounds)
        #sol = opt.shgo(self.cost, bounds=bounds)
        # sol = opt.dual_annealing(self.cost, bounds)
        sol = opt.differential_evolution(self.cost, bounds)
        if sol["success"] ==False:
            print("Optimisation considered unsuccessful due to: ", sol["message"])
        return sol


class Callibration(ParameterInference):
    """
    Used to find the optimal model parameters given healthy data
    """

    def build_candidate_sys(self, parameters):
        """
        Makes a candiate solution to the optimisation problem
        :param parameters: array of parameters [theta,phi]
        :return:
        """
        # Update the system with the latest candidate solution
        self.sys.g.theta = parameters[0:self.sys.g.n_parameters]
        self.sys.h.phi = parameters[self.sys.g.n_parameters:]


class DamageInference(ParameterInference):
    """
    Used to find the damage x given certain operating conditions and assuming model parameters
    are independent of damage x.
    """

    def build_candidate_sys(self, x):
        """
        Makes a candiate solution to the optimisation problem
        :param parameters: array of parameters [theta,phi]
        :return:
        """
        # Update the system with the latest candidate solution
        self.sys.g.x = x

