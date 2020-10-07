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
        subplots = False
        t = self.g.constants["t_range"][self.g.samples_before_fault:]
        if subplots:
            fig,axs = plt.subplots(len(z))
            for condition, zi in enumerate(z):
                #axs[condition].plot(zi, ".", label="c" + str(condition + 1))
                axs[condition].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if subplots == False:
            plt.figure()
            for condition, zi in enumerate(z):
                #plt.plot(zi, ".", label="c" + str(condition + 1))
                plt.scatter(t, zi, label="c" + str(condition + 1), s=1)
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.xlabel("time [s]")
                plt.ylabel("Measured acceleration")
        return

    def plot_model_vs_measured(self, measurements, plot_title_addition = ""):
        c = measurements["c"]
        z = measurements["z"]
        t = self.g.constants["t_range"]

        t = t[self.g.samples_before_fault:]  # Used for Chen2011 where a section of reponse is used.
        plt.figure()
        #plt.title("Model fit " + plot_title_addition)
        z_mod = self.simulate(c, False, noise=None)
        # TODO2: plot only one measurment
        for condition, zi in enumerate(z):
            plt.plot(t, zi, ".", label= str(c[condition][0]) + "Nm measurements" )
            plt.plot(t, z_mod[condition], label=str(c[condition][0]) + "Nm model fit")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel("Time [s]")
        plt.ylabel("Measured acceleration ($m/s^2$)")
        return

    def get_parameter_summary(self, print_addition=""):
        print("")
        print("Parameter summary: " + print_addition)
        print("=================")
        print("Damage state, x: ", self.g.x)
        print("Parameters for g, theta: ", self.g.theta)
        print("Parameters for h, phi: ", self.h.phi)
        print("")


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
        self.optimisation_complete = False

    def build_candidate_sys(self, parameters):
        pass

    def cost(self, parameters):
        self.build_candidate_sys(parameters)
        z = self.sys.simulate(self.measurements["c"])  # Simulate model for all operating conditions
        cost_for_each_measurement = [np.log(np.linalg.norm(z[i] - self.measurements["z"][i])) for i in range(len(z))]
        return np.sum(cost_for_each_measurement)

    def run_optimisation(self, cost_function, bounds, startpoint = np.array([1.1, 6.5e-3, 0.9e8, 0.9e7, 0.4])):
        # sol = opt.minimize(self.cost, startpoint)#, bounds=bounds)

        sol = opt.minimize(self.cost,startpoint, bounds=bounds)#, bounds=bounds)
        # sol = opt.shgo(self.cost, bounds=bounds)
        # sol = opt.dual_annealing(self.cost, bounds)
        # sol = opt.differential_evolution(cost_function,
        #                                  bounds,
        #                                  disp=True)
        self.optimisation_complete = True
        if sol["success"] == False:
            print("Optimisation considered unsuccessful due to: ", sol["message"])
        return sol

    def cost_g(self, theta):
        """
        Used to optimize the system model g separately from h
        Expect to have h as the residual if there is no model inadequacy
        """
        self.sys.g.theta = theta  # create a candidate solution keeping parameters for h constant

        z = self.sys.simulate(self.measurements["c"])  # Simulate model for all operating conditions
        cost_for_each_measurement = [np.log(np.linalg.norm(z[i] - self.measurements["z"][i])) for i in range(len(z))]
        return np.sum(cost_for_each_measurement)

    def cost_h(self, phi):
        """
        Used to optimize the measurement model h separately from g
        Expect to have h compensate for the inadequacy of g
        """
        self.sys.h.phi = phi  # create a candidate solution keeping parameters for g constant

        z = self.sys.simulate(self.measurements["c"])  # Simulate model for all operating conditions
        cost_for_each_measurement = [np.log(np.linalg.norm(z[i] - self.measurements["z"][i])) for i in range(len(z))]
        return np.sum(cost_for_each_measurement)


class Calibration(ParameterInference):
    """
    Used to find the optimal model parameters given healthy data
    """

    def build_candidate_sys(self, parameters):
        """
        Makes a candidate solution to the optimisation problem
        :param parameters: array of parameters [theta,phi]
        :return:
        """
        # Update the system with the latest candidate solution
        self.sys.g.theta = parameters[0:self.sys.g.n_parameters]
        self.sys.h.phi = parameters[self.sys.g.n_parameters:]

    def run_optimisation_separately(self, theta_bounds, phi_bounds,n_iter,plot_fit=False,verbose=False):

        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages('multipage.pdf')
        for i in range(n_iter): # Repeat the separate optimisations n_iter times
            # Optimize model g
            self.run_optimisation(self.cost_g, theta_bounds)

            if verbose:
                self.sys.get_parameter_summary()
            if plot_fit:
                self.sys.plot_model_vs_measured(self.measurements, plot_title_addition="g optimized, iteration " + str(i+1))
                pp.savefig()

            # Optimize model h
            self.run_optimisation(self.cost_h, phi_bounds)
            if verbose:
                self.sys.get_parameter_summary()
            if plot_fit:
                self.sys.plot_model_vs_measured(self.measurements, plot_title_addition="h optimized, iteration " + str(i + 1))
                pp.savefig()

        pp.close()

        return


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
