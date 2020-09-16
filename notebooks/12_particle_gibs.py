from particles import state_space_models as ssm
import warnings; warnings.simplefilter('ignore')  # hide warnings

from matplotlib import pyplot as plt
import numpy as np
from particles import mcmc

from particles import distributions as dists

class StochVol(ssm.StateSpaceModel):
    default_parameters = {'mu': -1., 'rho': 0.95, 'sigma': 0.2}

    def PX0(self):  # Distribution of X_0
        return dists.Normal(loc=self.mu, scale=self.sigma / np.sqrt(1. - self.rho ** 2))

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
        return dists.Normal(loc=self.mu + self.rho * (xp - self.mu), scale=self.sigma)

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return dists.Normal(loc=0., scale=np.exp(x))


class PGStochVol(mcmc.ParticleGibbs):
    store_x = True
    def update_theta(self, theta, x):
        new_theta = theta.copy()
        sigma, rho = 0.2, 0.95  # fixed values
        xlag = np.array(x[1:] + [0.,])
        dx = (x - rho * xlag) / (1. - rho)
        s = sigma / (1. - rho)**2
        new_theta['mu'] = self.prior.laws['mu'].posterior(dx, sigma=s).rvs()
        return new_theta

prior_dict = {'mu': dists.Normal(scale=2.),
              'rho': dists.Uniform(a=-1., b=1.),
              'sigma':dists.Gamma()}
my_prior = dists.StructDist(prior_dict)

# real data
raw_data = np.loadtxt('data.txt', skiprows=2, usecols=(3,), comments='(C)')
full_data = np.diff(np.log(raw_data))
data = full_data[:50]

pg = PGStochVol(ssm_cls=StochVol, data=data, prior=my_prior, Nx=200, niter=1000)
pg.run()  # may take several seconds...

# plt.plot(pg.chain.theta['mu'])
# plt.xlabel('iter')
# plt.ylabel('mu')
#
# plt.figure()
# plt.hist(pg.chain.theta['mu'][20:], 50)
# plt.xlabel('mu')