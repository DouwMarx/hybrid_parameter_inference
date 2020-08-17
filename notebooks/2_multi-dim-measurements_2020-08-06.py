import src.models.h_mappings as h_maps
import src.models.g_mappings as g_maps
import src.models.system_modeller as sys_model


import numpy as np
# Define the true model parameters
phi_real = np.array([1])
theta_real = np.array([1])

# Define different states of health
x_real_d0 = np.array([1])  # Healthy condition
x_real_d1 = np.array([2])  # Damaged condition

# Define the measurement model (Independent of health state)
h_real = h_maps.MultiDimH(phi_real)

# Define the real physics based model g for healthy and damaged states
g_real_d1 = g_maps.MultiDimG(theta_real,x_real_d1)
g_real_d0 = g_maps.MultiDimG(theta_real,x_real_d0)

# Define the system for healthy and damaged states
sys_real_d0 = sys_model.System(g_real_d0,h_real)
sys_real_d1 = sys_model.System(g_real_d1,h_real)

# Simulate and plot real system without noise at different c
#c = [np.array([i]) for i in np.linspace(1,5,3)]
#z = sys_real_d0.simulate(c, plot=True, noise=None)

# Define a model that might be appropriate for the physics we expect
phi_mod = np.array([1.5])
theta_mod = np.array([2])
x_mod = np.array([1])

h_mod = h_maps.MultiDimH(phi_mod)
g_mod = g_maps.MultiDimG(theta_mod,x_mod)
sys_mod = sys_model.System(g_mod,h_mod)
#
# Define the operating conditions under which the data is gathered
c = [np.array([i]) for i in np.linspace(1,5,3)]

# Get the measured data under healthy conditions
noise = {"sd":0.1}
z_real_d0 = sys_real_d0.simulate(c, noise = noise, plot=False)

# Solve for the most likely model parameters given the healthy measurements (fit sys_mod to data)
measurements_d0= {"c":c,
                "z":z_real_d0}
cal_obj = sys_model.Calibration(sys_mod, measurements_d0)
start_point = np.ones(2)*2
sol = cal_obj.run_optimisation(start_point)
# sys_mod.get_parameter_summary()

# Get measured data for unhealthy condition (same operating conditions and noise a healthy)
z_real_d1 = sys_real_d1.simulate(c, noise=noise)

# Infer the damage from the damaged data
measurements_d1 = {"c":c,
                "z":z_real_d1}
cal_obj = sys_model.DamageInference(sys_mod, measurements_d1)  # sys_mod would is updated with most likely
                                                               # parameters since we ran the callibration
start_point = np.ones(1) # Use the healthy condition as starting condition
x_pred = cal_obj.run_optimisation(start_point)
print(x_pred)

print("Actual health state: ", sys_real_d1.g.x)
print("Predicted health state: ", x_pred["x"])
