'''
Description: This file contains functions that are used by the DVFS mechanism and PEs to get performance, power, and thermal values.
'''
import random
import sys

import numpy as np
from math import exp


# Thermal model (Odroid XU3 board)
A_model = [[0.9928,    0.000566,   0.004281,  0.0003725, 1.34**-5 ],
           [0.006084,  0.9909,     0,         0.001016,  8.863**-5],
           [0,         0.0008608,  0.993,     0,         0.0008842],
           [0.006844,  -0.0005119, 0,         0.9904,    0.0003392],
           [0.0007488, 0.003932,   8.654**-5, 0.002473,  0.9905   ]]

# Prev_powers =[Prev_Power_CPU(2) (little); Prev_Power_CPU(1) (big); 0.18; Prev_Power_GPU];
# B_model = [[0.02399,   0.0471,     0.07423,   6.898**-7],
#            [0,         0.01265,    0,         0.001971 ],
#            [0.02819,   0.113,      0.6708,    2.108**-6],
#            [0.007198,  0.01646,    0,         0.01682  ],
#            [0.03902,   0.01476,    0.01404,   0.03811  ]]
# Big core model
B_model_big = [[0.0471  ],
               [0.01265 ],
               [0.113   ],
               [0.01646 ],
               [0.01476 ]]

B_model_little = [[0.02399 ],
                  [0       ],
                  [0.02819 ],
                  [0.007198],
                  [0.03902 ]]

def compute_DVFS_performance_slowdown(resource, current_frequency):
    '''
    returns the slowdown from running a given task with lower frequency
    '''
    # Calculate the slowdown based on the current frequency
    if current_frequency == 0 or len(resource.OPP) == 0:
        return 0
    else:
        max_freq = get_max_freq(resource.OPP)
        slowdown_ratio = float(max_freq) / float(current_frequency)
        return slowdown_ratio - 1
# end compute_DVFS_performance_slowdown(resource, current_frequency)

def compute_Cdyn_and_alpha(resource, pwr_consumption_max_freq):
    '''
    based on the maximum frequency, voltage, and measured power dissipation, compute the capacitance C times the switching activity Alpha for the given task
    Pdyn = Cdyn * alpha * f * V^2
    '''
    max_freq = get_frequency_in_Hz(get_max_freq(resource.OPP))
    max_volt = get_voltage_in_V(get_max_voltage(resource.OPP))
    if len(resource.OPP) > 0:
        Cdyn_alpha = pwr_consumption_max_freq / (max_freq * max_volt**2)
    else:
        Cdyn_alpha = 0
    return Cdyn_alpha
# end compute_Cdyn_and_alpha(resource, pwr_consumption_max_freq)

def compute_static_power_dissipation(args, PE, DSE_voltage = 0):
    '''
    compute the static power dissipation of the PE
    '''
    if PE.type == "ACC":
        static_power = 0
    else:
        current_temperature = PE.current_temperature_vector[0][0]
        if DSE_voltage == 0:
            current_voltage = PE.current_voltage
        else:
            current_voltage = DSE_voltage
        temp_K = 273 + current_temperature # Convert the temperature to Kelvin
        voltage_V = get_voltage_in_V(current_voltage)
        static_power = voltage_V * args.xu3_c1 * temp_K * temp_K * exp(-args.xu3_c2/temp_K) + args.xu3_igate*voltage_V
    return static_power
# end compute_static_power_dissipation()

def compute_dynamic_power_dissipation(current_frequency, current_voltage, Cdyn_alpha):
    '''
    compute the dynamic power dissipation for the current task based on the current state of the PE
    '''
    current_frequency_Hz = get_frequency_in_Hz(current_frequency)
    current_voltage_V = get_voltage_in_V(current_voltage)
    dynamic_power =  Cdyn_alpha * current_frequency_Hz * (current_voltage_V**2)
    return dynamic_power
# end compute_dynamic_power_dissipation(current_frequency, current_voltage, Cdyn_alpha)

def get_execution_time_max_frequency(args, task, resource):
    '''
    returns the execution time of the current task.
    '''
    task_ind = resource.supported_functionalities.index(task.name)                  # Retrieve the index of the task
    execution_time = resource.performance[task_ind]                                 # Retrieve the mean execution time of a task
    if(resource.performance[task_ind]): 
        # Randomize the execution time based on a gaussian distribution
        randomized_execution_time = max(round(
                random.gauss(execution_time,args.resource_std * execution_time)), 1)
        if args.verbose:
            print('Randomized execution time is %s, the original was %s' 
                  %(randomized_execution_time, execution_time))                                                       										# finding execution time using randomization values by mean value (expected execution time) 
        return randomized_execution_time      
    else:                                                                                        										# if the expected execution time is 0, ie. if it is dummy task, then no randomization
        # If a task has a 0 us of execution (dummy ending task), it should stay the same
        return execution_time
        
# end def get_execution_time_max_frequency(task,resource)


def get_power_consumption_max_frequency(task, resource):
    '''
    returns the power consumption of the current task.
    '''
    task_ind = resource.supported_functionalities.index(task.name)                  # Retrieve the index of the task
    return resource.power_consumption[task_ind]
# end def get_execution_time_max_frequency(task,resource)

def predict_temperature(args, PE):
    '''
    predicts the temperature based on the current status of the PE
    '''
    if PE.type == "BIG":
        print(PE.current_power)
        predicted_temperature = np.matmul(A_model, np.array(PE.current_temperature_vector) - args.xu3_t_ambient) + \
                                (np.array(B_model_big) * PE.current_power) + args.xu3_t_ambient
    elif PE.type == "LITTLE":
        predicted_temperature = np.matmul(A_model, np.array(PE.current_temperature_vector) - args.xu3_t_ambient) + \
                                (np.array(B_model_little) * PE.current_power) + args.xu3_t_ambient
    else: # Return ambient temperature for other types of PEs (e.g., accelerators)
        predicted_temperature = [[args.xu3_t_ambient],                  # Indicate the current PE temperature for each hotspot
                                [args.xu3_t_ambient],
                                [args.xu3_t_ambient],
                                [args.xu3_t_ambient],
                                [args.xu3_t_ambient]]
    return predicted_temperature
# end predict_temperature(current_temperature, leakage_power)

def get_voltage_constant_mode(OPP_list, constantFrequency):
    '''
    returns the voltage of the OPP that satisfies the defined constant frequency
    '''
    if constantFrequency < get_min_freq(OPP_list):
        print("[E] The frequency set in the constant DVFS mode is lower than the minimum frequency that the PE supports, please check the resource file")
        sys.exit()
    for OPP_i, OPP in enumerate(OPP_list):
        if constantFrequency == OPP[0]:
            return OPP[1]
    if constantFrequency > get_max_freq(OPP_list):
        print("[E] The frequency set in the constant DVFS mode is higher than the maximum frequency that the PE supports, please check the resource file")
        sys.exit()
    print("[E] Target frequency was not found in the OPP list:", constantFrequency)
    sys.exit()
# end get_voltage_constant_mode(OPP_list, constantFrequency)

def get_max_freq(OPP_list):
    if len(OPP_list) > 0:
        opp_tuple_max = OPP_list[len(OPP_list) - 1]
        return  opp_tuple_max[0]
    else:
        return 0

def get_min_freq(OPP_list):
    if len(OPP_list) > 0:
        opp_tuple_min = OPP_list[0]
        return opp_tuple_min[0]
    else:
        return 0

def get_max_voltage(OPP_list):
    if len(OPP_list) > 0:
        opp_tuple_max = OPP_list[len(OPP_list) - 1]
        return  opp_tuple_max[1]
    else:
        return 0

def get_min_voltage(OPP_list):
    if len(OPP_list) > 0:
        opp_tuple_min = OPP_list[0]
        return opp_tuple_min[1]
    else:
        return 0

def get_frequency_in_Hz(frequency_MHz):
    return frequency_MHz * 1e6

def get_voltage_in_V(voltage_mV):
    return voltage_mV * 1e-3
    
