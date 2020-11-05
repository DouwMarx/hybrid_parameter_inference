import matplotlib.pyplot as plt
import src.models.h_mappings as h_maps
import src.models.g_mappings as g_maps
import src.models.system_modeller as sys_model
import pickle
import time
import numpy as np

t_start = time.time()
# Define the true model parameters
theta_real = np.array([1.116, 6.405,1,1]) # theta = [m_ring, I_ring, k1 ,k2]
# I is adjusted for 1e-3,
# K1 adjusted by 1e8
# K2 adjusted by 1e7
phi_real = np.array([np.sin(np.deg2rad(20))])  # Measurement model transfer function is multiplication by constant.
# Fraction of pressure angle

# Define  healthy state of health
x_real_d0 = np.array([0])  # 0 N/m mean stiffness reduction (healthy)

# Define the real measurement model (Independent of health state)
h_real = h_maps.Constant(phi_real)

# Define the real physics based model g for healthy and damaged states
g_real_d0 = g_maps.Chen2011(theta_real, x_real_d0)

# Define the system
sys_real_d0 = sys_model.System(g_real_d0, h_real)

# Define the operating conditions where measurements were made
c = [np.array([i]) for i in np.linspace(10, 1, 4) * 1]  # c is applied moment 10Nm-100Nm
#c = [np.array([i]) for i in [5]]  # c is applied moment 10Nm-100Nm

# Get the Measured data under healthy conditions
noise = {"sd": 0.1}  # Small noise

z_real_d0 = sys_real_d0.simulate(c, noise=noise, plot=True)  # INFO: Toggle if plot
#plt.title("Measured data at healthy condition")
#plt.ylabel("Acceleration of ring gear")
#plt.show()

with open("real_d0_20201102.pkl", "wb") as fname:
    pickle.dump(sys_real_d0, fname)

# Define a model that might be appropriate for the physics we expect
# theta_real = np.array([1.116, 6.405e-3, 1e8, 1e7])  # theta = [m_ring, I_ring, k1 ,k2, c_prop]
# phi_real = np.array([1])  # Measurement model transfer function is multiplication by constant
theta_mod = np.array([1,5,1.1,1.2])
phi_mod = np.array([1])
x_mod = x_real_d0  # Assume we know the initial damage condition exactly

g_mod = g_maps.Chen2011(theta_mod, x_mod)
h_mod = h_maps.Constant(phi_mod)
sys_mod = sys_model.System(g_mod, h_mod)

with open("sys_mod_20201102.pkl", "wb") as fname:
    pickle.dump(sys_mod, fname)

# Solve for the most likely model parameters given the healthy measurements (fit sys_mod to data)
measurements_d0= {"c": c,
                "z": z_real_d0}

with open("healthy_measurements_20201102.pkl", "wb") as fname:
    pickle.dump(measurements_d0, fname)

cal_obj = sys_model.Calibration(sys_mod, measurements_d0)

# INFO: Below for running optimisation separate
# # Solve Separately
# theta_bounds =((0.1,5),)
# phi_bounds = ((0.1,5),
#              (0.1,5),
#              (0.1,5))
# n_iter = 3
# cal_obj.run_optimisation_separately(theta_bounds, phi_bounds,n_iter,plot_fit=True, verbose=True)

# # Solve all parameters at once
bounds =((1, 1.2),  # theta_real = np.array([1.116, 6.405])
         (6.1, 6.5),
         (0.9, 1.1),
         (0.9, 1.1),
         (0.2, 0.4),) #phi_real = np.array([0.342])

sol = cal_obj.run_optimisation(bounds)
#
print("")
sys_mod.get_parameter_summary(print_addition="learnt parameters for model from healthy data")

with open("sys_calibrated_20201104.pkl", "wb") as fname:
    pickle.dump(cal_obj, fname)

with open("sys_calibrated_20201104.pkl", "rb") as fname:
    cal_obj = pickle.load(fname)

# Damage inference
# =======================================================
for damage in np.linspace(0,1,4)[1:]*0.5:  # 3 damage states
    x_real_d1 = np.array([damage])  # 0.3e8 N/m mean stiffness reduction (damaged)

    # define real physics-based model at damage state
    g_real_d1 = g_maps.Chen2011(theta_real, x_real_d1)

    # define system
    sys_real_d1 = sys_model.System(g_real_d1, h_real)  # Notice that transfer function is independent of damage

    # # Get measured data for unhealthy condition (same operating conditions and noise a healthy)
    z_real_d1 = sys_real_d1.simulate(c, noise=noise, plot=False)
#
    # Infer the degree of damage x from the damaged data
    measurements_d1 = {"c":c,
                    "z":z_real_d1}

    sys_mod = cal_obj.sys # Use the calibrated healthy object

    cal_obj_for_damaged = sys_model.DamageInference(sys_mod, measurements_d1)  # sys_mod is updated with most likely
                                                                   # parameters since we ran the callibration above

    bounds =((0.1, 0.5),)
    #x_pred = cal_obj_for_damaged.run_optimisation(cal_obj_for_damaged.cost, bounds, startpoint=np.array([0.15e8]))
    x_pred = cal_obj_for_damaged.run_optimisation(bounds, startpoint=np.array([0.15]))
    print("Actual health state: ", sys_real_d1.g.x)
    print("Predicted health state: ", x_pred["x"])
    print("")

    with open("mod_damage_inferred_20201102" + str(np.round(damage,2)) + ".pkl", "wb") as fname:
        pickle.dump(cal_obj_for_damaged, fname)
print((time.time() - t_start)/60," min runtime")

# #Compare the measured damaged state with model fit
# try:
#     sys_mod.plot_model_vs_measured(measurements_d1)
# except:
#     pass
