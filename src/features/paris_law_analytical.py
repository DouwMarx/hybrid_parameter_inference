import numpy as np
import scipy.optimize as opt
import scipy.integrate as inter
import matplotlib.pyplot as plt


def Stess_Intensity(ap):
    # delta_K = 2.52 * ap ** 4 - 7.175 * ap ** 3 + 7.499 * ap ** 2 + 16.751 * ap + 25.828
    delta_K = 5.446 * ap ** 6 - 56.16 * ap ** 5 + 216.6 * ap ** 4 - 372.6 * ap ** 3 + 268.3 * ap ** 2 - 23.46 * ap +35.2
    return delta_K


class Cracked_Gear_Tooth(object):
    def __init__(self,ai, C, M,K1c):

        self.ai = ai # Initial crack length

        self.K1c = K1c # Plane stain fracture toughness
        self.C = C # Paris law constant 1
        self.M = M # Paris law constant 1

        #self.a_c = self.Critical_Crack_Length()  # Calculate the critical crack length




    def Critical_Crack_Length(self):
        obj = lambda a: Stess_Intensity(a) - self.K1c
        sol = opt.fsolve(obj, 1)
        return sol[0]

    def dadN(self, t,a):
        return self.C*(Stess_Intensity(a))**self.M

    def dNda(self, a):
        return 1/self.dadN("dummy", a)

    def cycles_to_failure(self,ai,af):
        """Returns the Number of cycles to failure for starting crack length"""
        # I make use of Euler integration because it is simple. Aslo scipy.integrate.odeint give problems.

        increms = int(1e3)  # This gives a good trade off between accuracy and speed
        a_range = np.logspace(np.log10(ai), np.log10(af), increms)#  Using a non-linear time scale helps with accuracy

        N = 1
        #
        Nlist = [N]
        for i in range(len(a_range) - 1):
            delta_a = a_range[i + 1] - a_range[i]  # Calculates the length of the integration increment
            a = a_range[i]
            N = N + self.dNda(a) * delta_a
            # Nlist.append(N)
        #return a_range, np.array(Nlist)
        return N

    def crack_length_for_cycles(self, delta_N,Nf):
        N_range = np.arange(0,Nf+delta_N,delta_N)
        initial_condition = np.array([self.ai])

        sol = inter.solve_ivp(self.dadN,
                              [N_range[0], N_range[-1]],
                              initial_condition,
                              dense_output=True,
                              t_eval=N_range)
                              # rtol=1e-6,
                              # atol=1e-9)
        # self.sol = sol
        return sol

    def akp1(self, cp, mp, ap,delta_N):
        #return (delta_N * (2 / (2 - mp)) * cp * self.Stess_Intensity(ap) ** mp + ap ** ((2 - mp) / 2)) ** (2 / (2 -
# mp))
        return ap + delta_N*cp*Stess_Intensity(ap)**mp

def show_integrate_and_discrete_similar():
    true_c = 9.12e-1
    true_m = 1.4354
    true_a0 = 0.1
    true_v = 0.05  # Standard deviation of measurement model
    af =4
    delta_N = 1e6

    obj = Cracked_Gear_Tooth(true_a0, true_c, true_m,"dum")

    cyc_to_fail = obj.cycles_to_failure(true_a0, af)

    sol = obj.crack_length_for_cycles(delta_N,cyc_to_fail)

    N_range = sol["t"]
    a_range = sol["y"][0]

    a = true_a0
    a_discrete_list = [a]
    for i in range(len(N_range)-1):
        a = obj.akp1(true_c,true_m,a,delta_N)
        a_discrete_list.append(a)


    plt.figure()
    plt.plot(N_range,a_range,label = "integration")
    plt.plot(N_range,a_discrete_list, label = "transition model")
    plt.legend()
    return



