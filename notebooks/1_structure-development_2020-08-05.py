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
h_real = h_maps.BasicConceptValidationH(phi_real)

# Define the real physics based model g for healthy and damaged states
g_real_d1 = g_maps.BasicConceptValidationG(theta_real,x_real_d1)
g_real_d0 = g_maps.BasicConceptValidationG(theta_real,x_real_d0)

# Define the system for healthy and damaged states
sys_real_d0 = sys_model.System(g_real_d0,h_real)
sys_real_d1 = sys_model.System(g_real_d1,h_real)

# Define a model that might be appropriate for the physics we expect
phi_mod = np.array([1.5])
theta_mod = np.array([2])
x_mod = np.array([1])

h_mod = h_maps.BasicConceptValidationH(phi_mod)
g_mod = g_maps.BasicConceptValidationG(theta_mod,x_mod)
sys_mod = sys_model.System(g_mod,h_mod)

# Define the operating conditions under which the data is gathered
c = [np.array([i]) for i in range(10)]

# Get the measured data under healthy conditions
noise = {"sd":0.1}
z_real_d0 = sys_real_d0.simulate(c, noise = noise, plot=True)

# Solve for the most likely model parameters given the healthy measurements (fit sys_mod to data)
measurements_d0= {"c":c,
                "z":z_real_d0}
cal_obj = sys_model.Calibration(sys_mod, measurements_d0)
start_point = np.ones(2)*2
cal_obj.run_optimisation(start_point)

# Get measured data for unhealthy condition (same operating conditions and noise a healthy)
z_real_d1 = sys_real_d1.simulate(c, noise=noise)

# Infer the damage from the damaged data
measurements_d1 = {"c":c,
                "z":z_real_d1}
cal_obj = sys_model.DamageInference(sys_mod, measurements_d1) # sys_mod would now be updated with most likely parameters
start_point = np.ones(1)
x_pred = cal_obj.run_optimisation(start_point)
print(x_pred)
