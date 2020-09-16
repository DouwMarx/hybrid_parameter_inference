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
    #default_parameters = {'c': 8.5e-11, 'm': 2.7, "a0": 1}

    def PX0(self):  # Distribution of X_0
        return dists.Normal(loc=self.a0, scale=0.001)

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
        return dists.Normal(loc=self.a(self.c, self.m, xp), scale=1e-5)  #TODO: Should this be normally distributed?

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return dists.Normal(loc=x, scale=1e-5)

    def a(self, cp, mp, ap):
        return (delta_N * (2 / (2 - mp)) * cp * self.delta_K(ap) ** mp + ap ** ((2 - mp) / 2)) ** (2 / (2 - mp))

    def delta_K(self, ap):
        return 50 + 50 * ap ** 2

rerun_simulation = True
if rerun_simulation:
    # Generate data
    true_c = 8.5e-11
    true_m = 2.7
    true_a0 = 0.1
    my_model = ParisLaw(c=true_c, m=true_m, a0=true_a0)  # actual model, theta = [mu, rho, sigma]
    true_states, data = my_model.simulate(10)  # we simulate from the model 100 data points
    data = np.array(data).flatten()
    np.save("data", data)

    plt.figure(1)
    plt.scatter(range(len(data)),data)

    # Define a prior
    prior_dict = {'c': dists.Normal(loc = true_c, scale=1e-11),
                   'm': dists.Normal(loc = true_m, scale=0.1),
                   'a0':dists.Normal(loc = true_a0, scale=0.01)}
    my_prior = dists.StructDist(prior_dict)

    # Run the smc2 inference
    fk_smc2 = ssp.SMC2(ssm_cls=ParisLaw, data=data, prior=my_prior, init_Nx=500, ar_to_increase_Nx=0.1)
    alg_smc2 = particles.SMC(fk=fk_smc2, N=500, store_history=True, verbose=True)
    alg_smc2.run()

    # Save the result
    with open('run_20200915.pkl', 'wb') as f:
        dill.dump(alg_smc2, f)


# Load the previously computed data
with open('run_20200915.pkl', 'rb') as f:
    alg_smc2 = dill.load(f)


class ProcesSmc2Obj(object):
    def __init__(self, smc2obj):
        self.smc2obj = smc2obj
        self.data = self.smc2obj.fk.data
        self.n_data_points = len(self.data)
        self.time_steps = range(self.n_data_points)
        self.Xhist = self.smc2obj.hist.X

        self.pf_particles, self.pf_weights = self.extract_state_particles_and_weights()

    def plot_data(self):
        plt.figure()
        plt.plot(range(len(self.data)), self.data)
        plt.xlabel("Measurement interval")
        plt.ylabel("Crack_length [mm]")
        return

    def plot_theta_dists(self):
        plt.figure()
        for i in range(self.n_data_points):
            plt.hist(self.Xhist[i].theta["m"], label=str(i), bins=20, density=True)  # Check if this should perhaps
            # include weights?
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
            sb.distplot(self.pf_particles[i],hist_kws={"density":True}, kde=False)# TODO: I am not using the weights.
            # Should I be
            # using them
            # kde takes long therefore set to False for now
        return


    def plot_state_estimates_with_time(self):
        state_aves = np.array([np.mean(self.pf_particles[i]) for i in self.time_steps])
        state_sds= np.array([np.std(self.pf_particles[i]) for i in self.time_steps])

        plt.figure()
        plt.scatter(self.time_steps,self.data, marker="x", label="Measurements")
        plt.plot(self.time_steps, state_aves, label="Average of state estimate")
        factor = 100
        top = state_aves + state_sds*factor
        bot = state_aves - state_sds*factor

        plt.fill_between(self.time_steps, bot, top, alpha= 0.3, label="1 standard deviation")
        plt.legend()
        self.save_plot_to_directory("testname")

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
        print("m",np.average(m))
        print("c",np.average(c))
        print("a_t", np.average(a_t))

        delta_K = pl.delta_K(a_t)
        print("dK", np.average(delta_K))

        return self.cycles_to_failure(m,c,a_t,delta_K)

    def cycles_to_failure(self,m,c,a,delta_k):
        af = 0.5
        frac_exp = (2-m)/2
        #return c*delta_k**m * (af**frac_exp - a**frac_exp) * frac_exp
        return 1/(c*delta_k**m) * (af**frac_exp - a**frac_exp) * frac_exp
        #return c*delta_k**m
        # return (af**frac_exp - a**frac_exp)*frac_exp


proc_obj = ProcesSmc2Obj(alg_smc2)
proc_obj.plot_data()
# proc_obj.plot_state_dists()
# proc_obj.plot_theta_dists()
#proc_obj.plot_state_estimates_with_time()
# next = proc_obj.get_rul_pred_samples()
cyc = proc_obj.estimate_rul(3)
plt.figure()
plt.hist(cyc)
# # Plot the results
# plt.figure(2)
# for i in range(len(data)):
#     plt.hist(alg_smc2.hist.X[i].theta["m"], label=str(i), bins=20, density=True)  # Check if this should perhaps
#     # include weights?

# plt.figure(3)
# for i in range(len(data)):
#     #plt.hist(alg_smc2.hist.X[i].pfs.l[1].X, weights=alg_smc2.hist.X[i].pfs.l[1].W, bins =20, density=True,
#     #plt.hist(alg_smc2.hist.X[i].pfs.l[1].X, bins=20, density=True)
#     # all_particles_for_timestep = np.array([alg_smc2.hist.X[i].pfs.l[j].X for j in range(len(alg_smc2.hist.X[
#     all_particles_for_timestep = np.array([x.X for x in alg_smc2.hist.X[i].pfs.l]).flatten()
#     print(np.mean(all_particles_for_timestep))
#     #plt.hist(alg_smc2.hist.X[i].pfs.l[1].X, weights=alg_smc2.hist.X[i].pfs.l[1].W, bins =20, density=True,
#     plt.hist(all_particles_for_timestep, bins=20, density=True, label=str(i))

# label=str(i))
# Plot the first of N=50 particle filters for the entire history of the simulation

# plt.figure(2)
# plt.legend()
# plt.figure(3)
# plt.legend()
