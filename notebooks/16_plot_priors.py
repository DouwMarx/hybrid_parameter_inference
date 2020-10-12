import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as s

def save_plot_to_directory(plot_name):
    path = "C:\\Users\\douwm\\repos\\Hybrid_Approach_To_Planetary_Gearbox_Prognostics\\reports\\masters_report" \
           "\\5_model_callibration\\Images"
    plt.savefig(path + "\\" + plot_name + ".pdf")

def plot_chi_square():
    plt.figure()
    x = np.linspace(0,5,1000)
    chi = s.chi2.pdf(x,df=1)
    plt.plot(x,chi,"k")
    plt.fill_between(x,chi)
    plt.grid()
    plt.ylabel("Probability density")
    plt.xlabel(r"$a_0$")
    save_plot_to_directory("chi2_prior")
    return

plot_chi_square()
