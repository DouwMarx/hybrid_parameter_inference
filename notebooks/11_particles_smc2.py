import numpy as np
import matplotlib.pyplot as plt
from particles import state_space_models as ssm
from particles import distributions as dists
import particles
import seaborn as sb
from particles import smc_samplers as ssp
import dill

delta_N = 10000

class ParisLaw(ssm.StateSpaceModel):
    """
    The state space mode that defines the Paris law
    """
    #default_parameters = {'c': 8.5e-11, 'm': 2.7, "a0": 1}

    def PX0(self):  # Distribution of X_0
        return dists.Normal(loc=self.a0, scale=0.01)

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
        return dists.Normal(loc=self.a(self.c, self.m, xp), scale=1e-2)  #TODO: Should this be normally distributed?

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return dists.Normal(loc=x, scale=1e-9) # If this sdev too small, warning: could not compute Colesky,
        # use diag in stead

    def a(self, cp, mp, ap):
        return (delta_N * (2 / (2 - mp)) * cp * self.delta_K(ap) ** mp + ap ** ((2 - mp) / 2)) ** (2 / (2 - mp))

    def delta_K(self, ap):
        return 50 + 50 * ap ** 2 # TODO: This is not true sifs. Could use analytical but it is calculated by FEM

rerun_simulation = False # If the simulation in not rerun, an older simulation is simply loaded
if rerun_simulation:
    # Generate data for the true model parameters
    true_c = 9.12e-11
    true_m = 1.4354
    true_a0 = 0.1
    true_model = ParisLaw(c=true_c, m=true_m, a0=true_a0)  # actual model, theta = [mu, rho, sigma]

    # Simulate from the model 10 data points
    true_states, data = true_model.simulate(10)
    data = np.array(data).flatten()

    # Define a prior
    prior_dict = {'c': dists.Normal(loc = true_c, scale=1e-11),
                   'm': dists.Normal(loc = true_m, scale=0.1),
                   'a0':dists.Normal(loc = true_a0, scale=0.01)}

    my_prior = dists.StructDist(prior_dict)

    # Run the smc2 inference
    fk_smc2 = ssp.SMC2(ssm_cls=ParisLaw, data=data, prior=my_prior, init_Nx=500)#, ar_to_increase_Nx=0.1)
    alg_smc2 = particles.SMC(fk=fk_smc2, N=5000, store_history=True, verbose=True)
    alg_smc2.run()

    alg_smc2.true_model = true_model  # Save the true model so it can be used in post processing
    alg_smc2.true_states = true_states  # Save the true states so RUL's can be calculated

    # Save the result
    with open('run_20200915.pkl', 'wb') as f:
        dill.dump(alg_smc2, f)


# Load the previously computed data
with open('run_20200915.pkl', 'rb') as f:
    alg_smc2 = dill.load(f)


class ProcessSmc2Obj(object):
    def __init__(self, smc2obj):
        self.smc2obj = smc2obj
        self.data = self.smc2obj.fk.data
        self.n_data_points = len(self.data)
        self.time_steps = np.arange(self.n_data_points)
        self.Xhist = self.smc2obj.hist.X
        self.true_states = self.smc2obj.true_states
        self.true_model = self.smc2obj.true_model


        self.af = 0.5  # This is the critical crack length

        self.pf_particles, self.pf_weights = self.extract_state_particles_and_weights()

    def plot_data(self):
        plt.figure()
        #plt.plot(range(len(self.data)), self.data)
        n_cycles = self.time_steps*delta_N
        plt.scatter(n_cycles, self.data,c="k",marker="x")
        #plt.scatter(range(len(self.data)), self.data,c="k",marker="x")
        plt.xlabel("Number of cycles")
        plt.ylabel("Crack length [mm]")
        self.save_plot_to_directory("measured_data")
        return

    def plot_theta_dists(self):
        plt.figure()
        plt.xlabel("Paris law m")
        plt.ylabel("Probability density")
        for i in [7,8,9,10]:#range(self.n_data_points):
            print("yo")
            #plt.hist(self.Xhist[i].theta["m"], label=str(i), bins=20, density=True)  # Check if this should perhaps
            sb.distplot(self.Xhist[i].theta["m"],hist_kws={"density":True},label = str(i), kde=False)#
            # include weights?
        plt.legend()
        #self.save_plot_to_directory("theta_update")
        return

    def extract_state_particles_and_weights(self):
        """
        Extract all the particles for the particle filter runs at each timestep and combine them
        :return:
        """
        state_particles = []
        state_weights = []
        for i in self.time_steps:
            all_particles_for_timestep = np.array([pf_run.X for pf_run in alg_smc2.hist.X[i].pfs.l]).flatten()
            all_weights_for_timestep = np.array([pf_run.W for pf_run in alg_smc2.hist.X[i].pfs.l]).flatten()
            state_particles.append(all_particles_for_timestep)
            state_weights.append(all_weights_for_timestep)

        return np.array(state_particles), np.array(state_weights)

    def plot_state_dists(self):
        plt.figure()
        for i in self.time_steps:
            # plt.hist(self.pf_particles[i], bins=20,weights=self.pf_weights[i], density=False, label=str(i))
            sb.distplot(self.pf_particles[i],hist_kws={"density":True},label = str(i), kde=False) # "
                                                                               #"TODO: I am not
            # using the weights.
            # Should I be
            # using them
            # kde takes long therefore set to False for now
        plt.xlabel("Crack length")
        plt.ylabel("Posterior probability density")
        plt.legend()
        self.save_plot_to_directory("posterior_states")
        return

    def plot_state_estimates_with_time(self):
        state_aves = np.array([np.mean(self.pf_particles[i]) for i in self.time_steps])
        state_sds= np.array([np.std(self.pf_particles[i]) for i in self.time_steps])
        n_cycles = self.time_steps*delta_N
        plt.figure()
        plt.scatter(n_cycles,self.data, marker="x", label="Measurements")
        plt.plot(n_cycles, state_aves, label="Average of state estimate")
        factor = 1
        top = state_aves + state_sds*factor
        bot = state_aves - state_sds*factor

        plt.fill_between(n_cycles, bot, top, alpha= 0.3, label="1 standard deviation")
        plt.legend()
        plt.xlabel("Number of cycles")
        plt.ylabel("state estimate")
        self.save_plot_to_directory("mean_and_sd_states_with_time")

        return

    def save_plot_to_directory(self, plot_name):
        path = "C:\\Users\\douwm\\repos\\Hybrid_Approach_To_Planetary_Gearbox_Prognostics\\reports\\masters_report" \
               "\\5_model_callibration\\Images"
        plt.savefig(path + "\\" + plot_name + ".pdf")

    def get_rul_pred_samples(self,t):
        m = self.Xhist[t].theta["m"]
        c = self.Xhist[t].theta["c"]
        a0 = self.Xhist[t].theta["a0"]

        n_theta_particles = len(m)

        # Randomly select particles (same number as there are theta particles) from the posterior
        a_t = np.random.choice(self.pf_particles[t], n_theta_particles)
        return m, c, a_t

    def estimate_rul(self,t):
        pl = ParisLaw() # Initialize paris law object so SIF's can be computed
        m,c,a_t = self.get_rul_pred_samples(t)
        # print("m",np.average(m))
        # print("c",np.average(c))
        # print("a_t", np.average(a_t))

        delta_K = pl.delta_K(a_t)
        #print("dK", np.average(delta_K))

        return self.cycles_to_failure(m,c,a_t,delta_K)

    def cycles_to_failure(self, m, c, a, delta_k):
        frac_exp = (2-m)/2
        return 1/(c*delta_k**m) * (self.af*frac_exp - a**frac_exp) * frac_exp  # TODO: should deltaK not change with a?

    def plot_RUL_dists(self):
        """ Used to show how the RUL posteriors update as more measurements become available"""
        plt.figure()
        rul_aves = []
        rul_sds = []
        for t in self.time_steps:
            # Estimate the RULs
            rul_dist = proc_obj.estimate_rul(t)
            rul_aves.append(np.average(rul_dist))
            rul_sds.append(np.std(rul_dist))

            # The the histograms of the RULs
            plt.hist(rul_dist,label = "Measurement " + str(t))#, density=True)

        plt.xlabel("Cycles to failure")
        plt.ylabel("RUL probability density")
        plt.legend()
        self.save_plot_to_directory("rul_dists")

        # Plot the mean Rul vs actual
        rul_aves = np.array(rul_aves)
        rul_sds = np.array(rul_sds)
        plt.figure()
        n_cycles = self.time_steps*delta_N
        plt.plot(n_cycles, rul_aves, label = "Average of predicted RUL")
        top = rul_aves + rul_sds
        bot = rul_aves - rul_sds
        plt.fill_between(n_cycles, bot, top, alpha=0.3, label = "1 standard \n deviation \n from mean")

        # Plot the true RUL line
        pl = ParisLaw()
        delta_K = pl.delta_K(np.array(self.true_states))
        # print("dK", np.average(delta_K))
        m = self.true_model.m
        c = self.true_model.c
        a_t = np.array(self.true_states)
        true_cycles_to_fail = self.cycles_to_failure(m, c, a_t, delta_K)

       # TODO : use truemodel c m and a0 for RUl
        #plt.plot(n_cycles, n_cycles[-1]-n_cycles)
        #plt.plot(n_cycles, true_cycles_to_fail[-1] - true_cycles_to_fail)
        plt.scatter(n_cycles, true_cycles_to_fail, label="True RUL", marker="x", c="k")
        #plt.plot(n_cycles, true_cycles_to_fail[0]-n_cycles, label="True RUL")

        plt.xlabel("Number of cycles applied")
        plt.ylabel("Remaining number of cycles")
        plt.legend()
        self.save_plot_to_directory("rul_vs_true")
        return


proc_obj = ProcessSmc2Obj(alg_smc2)
#proc_obj.plot_data()
#proc_obj.plot_state_dists()
#proc_obj.plot_theta_dists()
#proc_obj.plot_state_estimates_with_time()
# proc_obj.plot_RUL_dists()

# cyc = proc_obj.estimate_rul(3)
# plt.figure()
# plt.hist(cyc)
