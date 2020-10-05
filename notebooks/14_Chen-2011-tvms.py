import matplotlib.pyplot as plt
import src.models.h_mappings as h_maps
import src.models.g_mappings as g_maps
import src.models.system_modeller as sys_model
import numpy as np

#1 get tvms with a fault
#     if t_fault_start<t<t_fault_end
#     use a completely different profile
#     Just make sure your synchronize the periods

obj = g_maps.Chen2011(np.array([22]), np.array([33]))
opperating_condition = 12


t = obj.constants["t_range"]
frequency = 10 #Hz
k = np.array([obj.tvms(t_inc,500) for t_inc in t])
plt.figure()
plt.plot(t,k)

# y = obj.get_y(opperating_condition)
# Plot the time varying stiffness


# Plot the lumped mass response. First transients and then the steady state response
# t = obj.constants["t_range"]
#
# plt.figure()
# plt.plot(t, y)
# plt.xlabel("time [s]")
# plt.ylabel(r"Measured ring gear acceleration [$m/s^2$]")

