import warnings; warnings.simplefilter('ignore')  # hide warnings

# standard libraries
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb

# modules from particles
import particles  # core module
from particles import distributions as dists  # where probability distributions are defined
from particles import state_space_models as ssm  # where state-space models are defined

delta_N = 0.1
# Define a type of system and simulate a few of its responses as a function of time
#X = [C,m,a]

class ParisLaw(ssm.StateSpaceModel):
    def PX0(self):  # Distribution of X_0
        return dists.MvNormal(loc=self.mu, cov=self.sigma,scale=1)

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
        loc = np.array([xp[0,0],xp[0,1],self.a(xp)]).T
        #return loc
        return dists.MvNormal(loc=loc, cov=self.sigma, scale=1e-9)

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return dists.Normal(loc=x[0,2], scale=0.01)

    def a(self,xp):
        Cp = xp[0,0]
        mp = xp[0,1]
        ap = xp[0,2]
        return (delta_N*(2/(2-mp))*Cp*self.delta_K(ap)**mp + ap**((2-mp)/2))**(2/(2-mp))

    def delta_K(self,ap):
        return 10 + 5*ap**2

mu0 = [8.5e-11,2.7,10]
sigma0 = np.diag([7e-13**2,0.005**2,1])
my_model = ParisLaw(mu=mu0, sigma=sigma0)  # actual model, theta = [mu, rho, sigma]
true_states, data = my_model.simulate(10)  # we simulate from the model 100 data points

plt.style.use('ggplot')
plt.figure()
plt.scatter(range(len(data)),data)

data = np.array(data).flatten()
fk_model = ssm.Bootstrap(ssm=my_model, data=data)  # we use the Bootstrap filter
pf = particles.SMC(fk=fk_model, N=1000, resampling='stratified', moments=False, store_history=True)  # the algorithm
pf.run()  # actual computation

# plot
#plt.figure()
#plt.plot([yt**2 for yt in data], label='data-squared')
#plt.plot([m['mean'] for m in pf.summaries.moments], label='filtered volatility')
#plt.legend()

# prior_dict = {'mu':dists.Normal(),
#               'sigma': dists.Gamma(a=1., b=1.),
#               'rho':dists.Beta(9., 1.)}
# my_prior = dists.StructDist(prior_dict)
#
# from particles import mcmc  # where the MCMC algorithms (PMMH, Particle Gibbs, etc) live
# pmmh = mcmc.PMMH(ssm_cls=StochVol, prior=my_prior, data=data, Nx=50, niter = 1000)
# pmmh.run()  # Warning: takes a few seconds
#
# burnin = 100  # discard the 100 first iterations
# for i, param in enumerate(prior_dict.keys()):
#     plt.subplot(2, 2, i+1)
#     sb.distplot(pmmh.chain.theta[param][burnin:], 40)
#     plt.title(param)

plt.figure()
#for timestepX in pf.hist.X:
plt.hist(pf.hist.X[2][:,0]) # This should be mu