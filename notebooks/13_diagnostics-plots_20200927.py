import matplotlib.pyplot as plt
import src.models.h_mappings as h_maps
import src.models.g_mappings as g_maps
import src.models.system_modeller as sys_model
import numpy as np
import pickle

def save_plot_to_directory(plot_name):
    path = "C:\\Users\\douwm\\repos\\Hybrid_Approach_To_Planetary_Gearbox_Prognostics\\reports\\masters_report" \
           "\\5_model_callibration\\Images"
    plt.savefig(path + "\\" + plot_name + ".pdf")
    return

theta = np.array([1.116, 6.405e-3, 1e8, 1e7]) # Using theta as global variable to ensure

# #stiffness_reduction = np.array([0.3e8])
# stiffness_reduction = np.array([0])
# obj = g_maps.Chen2011(theta, stiffness_reduction)
# t = obj.constants["t_range"]
# # Plot the time varying stiffness
# tvms = np.array([obj.tvms(ti,stiffness_reduction) for ti in t])

def plot_healthy_and_damged_TVMS():
    plt.figure()
    plt.xlabel("Time [s]")
    plt.ylabel("Meshing stiffness [N/m]")

    for stiff_red in np.linspace(0,1,3):
        stiffness_reduction = np.array([stiff_red])*1e8*0.5
        obj = g_maps.Chen2011(theta, stiffness_reduction)
        t = obj.constants["t_range"]
        # Plot the time varying stiffness
        tvms = np.array([obj.tvms(ti, stiffness_reduction) for ti in t])

        plt.plot(t,tvms, label = str(stiffness_reduction) + " N/m reduction")

    plt.legend(loc= "lower right")
    #save_plot_to_directory("TVMS")
    return



def plot_healthy_and_damaged_at_operating_condition():
    fig,axs = plt.subplots(2,2)
    response_list = []
    for oc in enumerate([10,50]):
        for health in enumerate([0, 1*1e8*0.5]):
            # oc = (oc[0],10)
            # health = (health[0],0)
            stiffness_reduction = np.array([health[1]])
            obj = g_maps.Chen2011(theta, stiffness_reduction)
            t = obj.constants["t_range"][obj.samples_before_fault:]
            #axs[oc[0],health[0]].text(0,0,str(stiffness_reduction[0])
            y = obj.get_y(np.array([oc[1]]))
            response_list.append(y)

    max_oc_0 = np.max([np.max(response_list[0]), np.max(response_list[1])])
    min_oc_0 = np.min([np.min(response_list[0]), np.min(response_list[1])])
    max_oc_1 = np.max([np.max(response_list[2]), np.max(response_list[3])])
    min_oc_1 = np.min([np.min(response_list[2]), np.min(response_list[3])])

    count = 0
    for oc in enumerate(["10 Nm","50 Nm"]):
        for health in enumerate(["Healthy","Damaged"]):
            axs[oc[0], health[0]].plot(t,response_list[count],"k")
            axs[oc[0], health[0]].set_xlabel("time [s]")
            axs[oc[0], health[0]].set_ylabel(r"ring acceleration [m/$s^2$]")
            axs[oc[0], health[0]].set_title(health[1] + " " + oc[1])
            if oc[0]>0:
                axs[oc[0], health[0]].set_ylim(min_oc_1,max_oc_1)
                axs[oc[0], health[0]].set_ylabel(r"acceleration [m/$s^2$]")
            else:
                axs[oc[0], health[0]].set_ylim(min_oc_0,max_oc_0)
                axs[oc[0], health[0]].set_ylabel(r"acceleration [m/$s^2$]")

            count+=1
            # Plot the time varying stiffness
    #     tvms = np.array([obj.tvms(ti, stiffness_reduction) for ti in t])
    #
    #     plt.plot(t,tvms, label = str(stiff_red) + " N/m reduction")

    # plt.legend(loc= "lower right")
    #plt.subplot_tool()
    plt.subplots_adjust(wspace=0.4, hspace=0.53)
    save_plot_to_directory("chen_2011_response")
    return

def plot_measured_data():
    fname = "real_d0.pkl"
    with open(fname,"rb") as f:
        sys_real_d0 = pickle.load(f)

    #c = [np.array([i]) for i in np.linspace(0,1,4)*10]  # c is applied moment 10Nm-100Nm
    c = [np.array([i]) for i in np.linspace(10, 1, 3) * 1]  # c is applied moment 10Nm-100Nm

    # Get the measured data under healthy conditions
    noise = {"sd": 1}  # Small noise

    z_real_d0 = sys_real_d0.simulate(c, noise=None, plot=True)  # INFO: Toggle if plot
    save_plot_to_directory("sys_real_d0_measurements")
    return

def plot_healthy_fit():
    fname = "sys_calibrated.pkl"
    with open(fname,"rb") as f:
        sys_calibrated = pickle.load(f)

    fname = "healthy_measurements.pkl"
    with open(fname,"rb") as f:
        healthy_measurements = pickle.load(f)

    sys_calibrated.sys.plot_model_vs_measured(healthy_measurements)
    save_plot_to_directory("fit_to_healthy_data")
    return

#plot_healthy_and_damged_TVMS()
#plot_healthy_and_damaged_at_operating_condition()
# plot_measured_data()
plot_healthy_fit()

# applied_torque = np.array([10])
# stiffness_reduction = np.array([1e8*0.5])
# obj = g_maps.Chen2011(theta, stiffness_reduction)
# y = obj.get_y(applied_torque)
# # Plot the lumped mass response. First transients and then the steady state response
# plt.figure()
# #plt.plot(t, y)
# plt.plot(y)
# plt.xlabel("time [s]")
# plt.ylabel(r"Measured ring gear acceleration [$m/s^2$]")

# save_plot_to_directory("chen_2011_response")  # Save the full figure
# plt.xlim(0.4, 0.41)
# save_plot_to_directory("chen_2011_response_zoom")  # Save the full figure

