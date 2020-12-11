import matplotlib.pyplot as plt
import src.models.h_mappings as h_maps
import src.models.g_mappings as g_maps
import src.models.system_modeller as sys_model
import numpy as np
import pickle
from datetime import  datetime

def save_plot_to_directory(plot_name):
    path = "C:\\Users\\douwm\\repos\\Hybrid_Approach_To_Planetary_Gearbox_Prognostics\\reports\\masters_report" \
           "\\5_model_callibration\\Images"
    plt.savefig(path + "\\" + plot_name + datetime.today().strftime('%Y%m%d') + ".pdf", bbox_inches = "tight")
    return

theta = np.array([1.116, 6.405e-3, 1e8, 1e7]) # Using theta as global

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

    for stiff_red in np.linspace(0,1,4):
        stiffness_reduction = np.array([stiff_red])*0.5
        obj = g_maps.Chen2011(theta, stiffness_reduction,cycles_before_fault=2)
        t = np.linspace(0,obj.constants["t_range"][-1]*2,1000)
        # Plot the time varying stiffness
        k_at_first_transition_end = obj.smooth_square(obj.first_transition_end_time) - obj.x*1e8
        k_at_second_transition_start = obj.smooth_square(obj.second_transition_start_time) - obj.x*1e8
        first_transition_gradient = (obj.k_at_first_transition_start - k_at_first_transition_end) /obj.transition_time
        second_transition_gradient = (k_at_second_transition_start - obj.k_at_second_transition_end) / obj.transition_time

        tvms = np.array([obj.tvms(ti, first_transition_gradient,second_transition_gradient,k_at_second_transition_start) for ti in t])

        plt.plot(t,tvms, label = str(np.round(stiffness_reduction[0],3)) + "e8 N/m reduction")

    plt.legend(loc= "lower right")
    save_plot_to_directory("TVMS_20201119")
    return



def plot_healthy_and_damaged_at_operating_condition():
    fig,axs = plt.subplots(2,2)
    response_list = []
    for oc in enumerate([5,10]):
        for health in enumerate([0, 0.5
                                 ]):
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
    for oc in enumerate(["5 Nm","10 Nm"]):
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
    fname = "sys_calibrated_20201119.pkl"
    with open(fname,"rb") as f:
        sys_cal_d0 = pickle.load(f)

    z = sys_cal_d0.measurements["z"]
    c = sys_cal_d0.measurements["c"]
    print(c)

    t = sys_cal_d0.sys.g.constants["t_range"][sys_cal_d0.sys.g.samples_before_fault:]

    plt.figure()
    for condition, zi in enumerate(z):
        # plt.plot(zi, ".", label="c" + str(condition + 1))
        plt.scatter(t, zi, label=  str(c[condition][0]) + "Nm", s=1)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel("time [s]")
        plt.ylabel(r"Measured acceleration $[m/s^2]$")
    # fname = "real_d0.pkl"
    # with open(fname,"rb") as f:
    #     sys_real_d0 = pickle.load(f)

    # #c = [np.array([i]) for i in np.linspace(0,1,4)*10]  # c is applied moment 10Nm-100Nm
    # c = [np.array([i]) for i in np.linspace(10, 1, 3) * 1]  # c is applied moment 10Nm-100Nm
    #
    # # Get the measured data under healthy conditions
    # noise = {"sd": 1}  # Small noise
    #
    # z_real_d0 = sys_real_d0.simulate(c, noise=None, plot=True)  # INFO: Toggle if plot
    save_plot_to_directory("sys_real_d0_measurements ")
    return

def plot_healthy_fit():
    fname = "sys_calibrated_20201119.pkl"
    with open(fname,"rb") as f:
        sys_calibrated = pickle.load(f)

    # fname = "healthy_measurements.pkl"
    # with open(fname,"rb") as f:
    #     healthy_measurements = pickle.load(f)
    # z = sys_calibrated.measurements["z"]
    # c = sys_calibrated.measurements["c"]
    meas = sys_calibrated.measurements
    sys_calibrated.sys.plot_model_vs_measured(meas,plot_only_one=1)
    save_plot_to_directory("fit_to_healthy_data")
    return

def plot_damaged_fit():
    fname = "mod_damage_inferred_202011190.33.pkl"
    with open(fname,"rb") as f:
        mod_damaged = pickle.load(f)

    # fname = "healthy_measurements.pkl"
    # with open(fname,"rb") as f:
    #     healthy_measurements = pickle.load(f)
    # z = mod_damaged.measurements["z"]
    # c = mod_damaged.measurements["c"]
    meas = mod_damaged.measurements
    mod_damaged.sys.plot_model_vs_measured(meas, plot_only_one=1)
    save_plot_to_directory("fit_to_damaged_data")
    return


    # fname = "healthy_measurements.pkl"
    # with open(fname,"rb") as f:
    #     healthy_measurements = pickle.load(f)
    # z = mod_damaged.measurements["z"]
    # c = mod_damaged.measurements["c"]
    meas = mod_damaged.measurements
    mod_damaged.sys.plot_model_vs_measured(meas, plot_only_one=1)



# plot_healthy_and_damged_TVMS()
# plot_healthy_and_damaged_at_operating_condition()
plot_measured_data()
plot_healthy_fit()
plot_damaged_fit()

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

