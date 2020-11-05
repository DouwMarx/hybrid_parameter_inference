import numpy as np
import matplotlib.pyplot as plt
from particles import state_space_models as ssm
from particles import distributions as dists
import particles
import seaborn as sb
import src.features.paris_law_analytical as pla
from particles import smc_samplers as ssp
import dill
from datetime import datetime

#delta_N = 1e7
delta_N =5.4e7
af = 5


def get_measurements(true_c,true_m,true_a0,plot = False):
    obj = pla.Cracked_Gear_Tooth(true_a0,true_c,true_m,"dum")

    cyc_to_fail = obj.cycles_to_failure(true_a0, af)

    sol = obj.crack_length_for_cycles(delta_N,cyc_to_fail)


    N_range = sol["t"]
    a_range = sol["y"][0]

    if plot:
        plt.figure()
        plt.scatter(N_range,a_range)
        plt.xlabel("Number of cycles")
        plt.ylabel("Crack length [mm]")

    return a_range

class ParisLaw(ssm.StateSpaceModel):
    """
    The state space mode that defines the Paris law
    """
    #default_parameters = {'c': 8.5e-11, 'm': 2.7, "a0": 1}

    def PX0(self):  # Distribution of X_0
        return dists.Dirac(loc=self.a0)

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
        # Division by 4 done to scale distributions
        #return dists.Normal(loc=self.a(self.c, self.m, xp), scale=1e-2)
        return dists.Dirac(loc=self.a(self.c, self.m, xp))
        # normally distributed?

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return dists.Normal(loc=x, scale=self.v) # If this sdev too small, warning: could not compute Colesky,
        #return dists.Dirac(loc=x) # If this sdev too small, warning: could not compute Colesky,
        # use diag in stead

    def a(self, cp, mp, ap):
        #return (delta_N * (2 / (2 - mp)) * cp * self.delta_K(ap) ** mp + ap ** ((2 - mp) / 2)) ** (2 / (2 - mp))
        delta_a =  delta_N*cp*(self.delta_K(ap))**mp
        return ap + delta_N*cp*(self.delta_K(ap+delta_a/2))**mp

    def delta_K(self, ap):
        #return 2.52*ap**4 - 7.175*ap**3 + 7.499*ap**2 + 16.751*ap + 25.828
        #return 5.446*ap**6 - 56.16*ap**5 + 216.6*ap**4 - 372.6*ap**3 + 268.3*ap**2 - 23.46*ap**1 + 35.2
        return pla.Stess_Intensity(ap)

rerun_simulation =  True# If the simulation in not rerun, an older simulation is simply loaded
if rerun_simulation:
    # Generate data for the true model parameters
    true_c = 9.12e-11
    true_m = 1.4354
    true_a0 = 0.1
    true_v = 0.05  # Standard deviation of measurement model
    true_model = ParisLaw(c=true_c, m=true_m, a0=true_a0,v=true_v)  # actual model, theta = [mu, rho, sigma]

    # Simulate from the model 10 data points
    # true_states, data = true_model.simulate(4)
    # data = np.array(data).flatten()
    true_states = get_measurements(true_c,true_m,true_a0)
    print("number of measured states ", len(true_states))
    print("true EOL ", true_states[-1])
    data = np.random.normal(loc = true_states, scale = true_v)
    data = data#[0:6]#[0:]

    # Define a prior
    # This prior has the true parameters as mean
    # prior_dict = {'c': dists.Normal(loc=9.12e-11,scale=1e-11),
    #               'm': dists.Normal(loc = 1.4353, scale = 1e-1),
    #               'a0':dists.Gamma(1/2, 2), # Chi squared with k=1
    #               "v":dists.Gamma(1/2,2)}  # This is chi squared with k=1

    # This prior is not exactly the same as the true parameter
    prior_dict = {'c': dists.Normal(loc=8e-11,scale=2e-11),
                  'm': dists.Normal(loc = 1.5, scale = 1e-1),
                  'a0':dists.Gamma(1/2, 2), # Chi squared with k=1
                  "v":dists.Gamma(1/2,2)}  # This is chi squared with k=1

    # # This prior is significantly different from the true parameters
    # prior_dict = {'c': dists.Normal(loc=8e-11,scale=1e-11),
    #               'm': dists.Normal(loc = 1.5, scale = 5e-1),
    #               'a0':dists.Gamma(1/2, 2), # Chi squared with k=1
    #               "v":dists.Gamma(1/2,2)}  # This is chi squared with k=1

    my_prior = dists.StructDist(prior_dict)

    # Run the smc2 inference
    # fk_smc2 = ssp.SMC2(ssm_cls=ParisLaw, data=data, prior=my_prior, init_Nx=1000)#, ar_to_increase_Nx=0.1)
    fk_smc2 = ssp.SMC2(ssm_cls=ParisLaw, data=data, prior=my_prior, init_Nx=5000)#, ar_to_increase_Nx=0.1) #Pf
    # particles
    alg_smc2 = particles.SMC(fk=fk_smc2, N=5000, store_history=True, verbose=True)  # Theta_particles
    alg_smc2.run()

    alg_smc2.true_model = true_model  # Save the true model so it can be used in post processing
    alg_smc2.true_states = true_states  # Save the true states so RUL's can be calculated

    # Save the result
    with open('run_20201105transf.pkl', 'wb') as f:
        dill.dump(alg_smc2, f)


# Load the previously computed data
with open('run_20201105transf.pkl', 'rb') as f:
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
        self.af = 0.4  # This is the critical crack length

        self.pf_particles, self.pf_weights = self.extract_state_particles_and_weights()
        self.kde_status = True

    def plot_data(self):
        plt.figure()
        #plt.plot(range(len(self.data)), self.data)
        n_cycles = self.time_steps*delta_N
        plt.scatter(n_cycles, self.data,c="k",marker="x")
        #plt.scatter(range(len(self.data)), self.data,c="k",marker="x")
        plt.xlabel("Number of cycles")
        plt.ylabel("Crack length [mm]")
        plot_sim = False
        if plot_sim:
            true_c = 9.12e-11
            true_m = 1.4354
            true_a0 = 0.1
            true_v = 0.05  # Standard deviation of measurement model
            true_model = ParisLaw(c=true_c, m=true_m, a0=true_a0, v=true_v)  # actual model, theta = [mu, rho, sigma]
            true_states, data = true_model.simulate(len(n_cycles))
            data = np.array(data).flatten()
            plt.scatter(n_cycles,data)

        self.save_plot_to_directory("measured_data")
        return

    def plot_theta_dists(self):
        for parameter in ["m","c","a0","v"]:
            plt.figure()
            plt.xlabel(parameter)
            plt.ylabel("Probability density")
            for i in range(self.n_data_points):
                #plt.hist(self.Xhist[i].theta["m"], label=str(i), bins=20, density=True)  # Check if this should perhaps
                sb.distplot(self.Xhist[i].theta[parameter],hist_kws={"density":True},label = str(i+1),
                             kde=self.kde_status)#
                # include weights?
            plt.legend()
            if parameter=="a0":
                plt.xlim(0,self.true_model.a0*5)
                plt.axvline(x=self.true_model.a0, color="gray", linestyle="--")
            if parameter=="v":
                plt.xlim(0,self.true_model.v*5)
                plt.axvline(x=self.true_model.v, color="gray", linestyle="--")
                plt.xlabel(r"$\nu$")
            if parameter=="c":
                plt.axvline(x=self.true_model.c, color="gray", linestyle="--")
            if parameter=="m":
                plt.axvline(x=self.true_model.m, color="gray", linestyle="--")
            self.save_plot_to_directory("theta_update_" + parameter)
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
            ax = sb.distplot(self.pf_particles[i],hist_kws={"density":True},label = str(i+1), kde=self.kde_status) # "
            # weights are not included?

            plt.axvline(x=self.true_states[i],color="grey",linestyle="--")
            # using the weights.
            # Should I be
            # using them
            # kde takes long therefore set to False for now
        plt.xlabel("Crack length, a [mm]")
        plt.ylabel("Posterior probability density")
        plt.xlim(0,self.true_states[-1]*1.1)
        plt.legend()
        self.save_plot_to_directory("posterior_states")
        return

    def plot_state_estimates_with_time(self):
        state_aves = np.array([np.mean(self.pf_particles[i]) for i in self.time_steps])
        state_sds= np.array([np.std(self.pf_particles[i]) for i in self.time_steps])
        n_cycles = self.time_steps*delta_N
        plt.figure()
        plt.scatter(n_cycles,self.data, marker="x", label="Measurements")
        plt.scatter(n_cycles,self.true_states, marker=".", label="True crack length")

        plt.plot(n_cycles, state_aves, label="Average crack length estimate")
        factor = 1
        top = state_aves + state_sds*factor
        bot = state_aves - state_sds*factor

        plt.fill_between(n_cycles, bot, top, alpha= 0.3, label="1 standard deviation")
        plt.legend()
        plt.xlabel("Number of cycles")
        plt.ylabel("crack length (mm)")
        self.save_plot_to_directory("mean_and_sd_states_with_time")

        return

    def save_plot_to_directory(self, plot_name):
        path = "C:\\Users\\douwm\\repos\\Hybrid_Approach_To_Planetary_Gearbox_Prognostics\\reports\\masters_report" \
               "\\5_model_callibration\\Images"
        plt.savefig(path + "\\" + plot_name + datetime.today().strftime('%Y%m%d')+ "trans.pdf")

    def get_rul_pred_samples(self,t):
        m = self.Xhist[t].theta["m"]
        c = self.Xhist[t].theta["c"]
        a0 = self.Xhist[t].theta["a0"]

        n_theta_particles = len(m)

        # Randomly select particles (same number as there are theta particles) from the posterior
        a_t = np.random.choice(self.pf_particles[t], n_theta_particles)
        return m, c, a_t

    def estimate_rul(self,t):
        # pl = ParisLaw() # Initialize paris law object so SIF's can be computed
        m,c,a_t = self.get_rul_pred_samples(t)
        # print("m",np.average(m))
        # print("c",np.average(c))
        # print("a_t", np.average(a_t))

        #delta_K = pl.delta_K(a_t)
        #print("dK", np.average(delta_K))
        #return self.cycles_to_failure(m,c,a_t,delta_K)
        return self.cycles_to_failure(m,c,a_t)

    def dadN(self,a,c,m):
        return c*pla.Stess_Intensity(a)**m

    def dNda(self, a,c,m):
        return 1/self.dadN(a,c,m)

    def cycles_to_failure(self, m, c, a):
        increms = int(1e3)  # This gives a good trade off between accuracy and speed
        af = self.true_states[-1] # Actual final crack length
        a_range = np.logspace(np.log10(a), np.log10(af), increms)#  Using a non-linear time scale helps with accuracy
        N = 1
        #
        Nlist = [N]
        for i in range(len(a_range) - 1):
            delta_a = a_range[i + 1] - a_range[i]  # Calculates the length of the integration increment
            a = a_range[i]
            N = N + self.dNda(a,c,m) * delta_a
            # Nlist.append(N)
        #return a_range, np.array(Nlist)
        return N

    def plot_RUL_dists(self):
        """ Used to show how the RUL posteriors update as more measurements become available"""
        plt.figure()
        rul_aves = []
        rul_sds = []
        for t in self.time_steps:
            # Estimate the RULs
            # rul_dist = proc_obj.estimate_rul(t)
            rul_dist = self.estimate_rul(t)
            rul_dist =  rul_dist[~np.isnan(rul_dist)] # Remove NaNs
            rul_aves.append(np.average(rul_dist))
            rul_sds.append(np.std(rul_dist))

            # The the histograms of the RULs
            # plt.hist(rul_dist, label = str(t))#, density=True)
            sb.distplot(rul_dist, hist_kws={"density": True}, label=str(t+1), kde=self.kde_status)  #

        plt.xlabel("Cycles to failure")
        plt.xlim(0,delta_N*len(self.true_states)*1.1)
        plt.ylabel("RUL probability density")
        plt.legend()
        self.save_plot_to_directory("rul_dists")

       #  # Plot the mean and variance Rul vs actual
        rul_aves = np.array(rul_aves)
        rul_sds = np.array(rul_sds)
        plt.figure()
        n_cycles = self.time_steps*delta_N
        plt.plot(n_cycles, rul_aves, label = "mean RUL")
        top = rul_aves + rul_sds
        bot = rul_aves - rul_sds
        plt.fill_between(n_cycles, bot, top, alpha=0.3, label = "1 standard \n deviation \n from mean")

        #Plot the actual RUL
        plt.plot(n_cycles,-n_cycles+ n_cycles[-1],label= "true RUL")
        plt.legend()
        plt.xlabel("Number of cycles applied")
        plt.ylabel("Remaining number of cycles")
        self.save_plot_to_directory("rul_mean_and_variance")
        return

proc_obj = ProcessSmc2Obj(alg_smc2)
proc_obj.plot_data()
proc_obj.plot_state_dists()
proc_obj.plot_theta_dists()
proc_obj.plot_state_estimates_with_time()
proc_obj.plot_RUL_dists()

# cyc = proc_obj.estimate_rul(3)
# plt.figure()
# plt.hist(cyc)

#################
