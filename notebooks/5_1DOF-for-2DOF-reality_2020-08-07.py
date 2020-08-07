import matplotlib.pyplot as plt
import src.models.h_mappings as h_maps
import src.models.g_mappings as g_maps
import src.models.system_modeller as sys_model

import numpy as np

# Define the true model parameters
# theta = [m1, m2, c1, c2, k2]
# x = [k1]
theta_real = np.array([1, 1, 10, 10, 100], dtype="float32")  # Mass of m1
phi_real = np.array([2])  # Measurement model transfer function is multiplication by constant

# Define different states of health (Stiffness of k1)
x_real_d0 = np.array([100])  # Healthy condition
x_real_d1 = np.array([50])  # Damaged condition

# Define the real physics based model g for healthy and damaged states
g_real_d1 = g_maps.InitCond2DOFv2G(theta_real, x_real_d1)  # notice v2
g_real_d0 = g_maps.InitCond2DOFv2G(theta_real, x_real_d0)

# Define the real measurement model (Independent of health state)
h_real = h_maps.InitCond2DOFv1H(phi_real)  # still v1 (constant) measurement func

# Define the system for healthy and damaged states
sys_real_d0 = sys_model.System(g_real_d0, h_real)
sys_real_d0.get_parameter_summary(print_addition="True system parameters")
sys_real_d1 = sys_model.System(g_real_d1, h_real)

# Define the operating conditions measured at
c = [np.array([i]) for i in np.linspace(0.1, 1, 10, dtype="float32") * 150]  # c in this case if force

# Get the measured data under healthy conditions
noise = {"sd": 5}
z_real_d0 = sys_real_d0.simulate(c, noise=noise, plot=True)
plt.title("Measured data at healthy condition")
plt.ylabel("Acceleration of mass 1")
plt.savefig("..\\reports\\healthy_measured_1DOF_for_2DOF_all_param2.png")
plt.show()

# # Define a model that might be appropriate for the physics we expect
theta_mod = np.array([2, 5], dtype="float32")
phi_mod = np.array([1], dtype="float32")
x_mod = x_real_d0  # Assume we know the initial damage condition exactly

h_mod = h_maps.InitCond2DOFv1H(phi_mod)
g_mod = g_maps.InitCond1DOFv1G(theta_mod, x_mod)
sys_mod = sys_model.System(g_mod, h_mod)

# Solve for the most likely model parameters given the healthy measurements (fit sys_mod to data)
measurements_d0 = {"c": c,
                   "z": z_real_d0}
cal_obj = sys_model.Callibration(sys_mod, measurements_d0)
start_point = np.hstack((theta_mod, phi_mod))
bounds =((0.1,5),
         (0,5))

sol = cal_obj.run_optimisation(start_point,bounds)

print("")
sys_mod.get_parameter_summary(print_addition="learnt parameters for model from healthy data")

# # Get measured data for unhealthy condition (same operating conditions and noise a healthy)
z_real_d1 = sys_real_d1.simulate(c, noise=noise, plot=True)
plt.title("Measured data at damaged condition")
plt.ylabel("Acceleration of mass 1")
plt.savefig("..\\reports\\damaged_measured_1DOF_for_2DOF_all_param2.png")
plt.show()


# Infer the degree of damage x from the damaged data
measurements_d1 = {"c":c,
                "z":z_real_d1}
inf_obj = sys_model.DamageInference(sys_mod, measurements_d1)  # sys_mod is updated with most likely
                                                               # parameters since we ran the callibration above
start_point = np.ones(1)  # Use the healthy condition as starting condition
bounds = ((0,100),)
x_pred = inf_obj.run_optimisation(start_point, bounds)

print("")
print("Actual damaged health state: ", sys_real_d1.g.x)
print("Inferred damaged health state: ", x_pred["x"])
