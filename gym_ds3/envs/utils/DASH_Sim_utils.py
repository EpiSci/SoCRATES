'''
Description: This file contains functions that are used by DASH_Sim.
'''
import ast
import csv
import fnmatch
import os

import pandas as pd

fixed_header_checkpoint = ['Timestamp', 'PE_ID_MODULE', 'Idle_PE', 'Total_Energy', 'Available_Time', 'Utilization', 'Current_Frequency', 'Current_Voltage',
                           'Temperature_0', 'Temperature_1', 'Temperature_2', 'Temperature_3', 'Temperature_4', 'Current_Power', 'Current_Leakage', 'Energy_Sampling_Period']
fixed_header_checkpoint_task_queues = ['Queue']

def update_PE_utilization_and_info(args, env_storage, PE, current_timestamp):
    lower_bound = current_timestamp-args.sampling_rate             # Find the lower bound for the time window_list under consideration
    
    completed_info = []
    running_info = []
    for task in env_storage.TaskQueues.completed.list:
        if task.PE_ID == PE.ID:
            if ((task.start_time < lower_bound) and (task.finish_time < lower_bound)):
                continue
            elif ((task.start_time < lower_bound) and (task.finish_time >= lower_bound)):
                completed_info.append(lower_bound)
                completed_info.append(task.finish_time)
            else:
                completed_info.append(task.start_time)
                completed_info.append(task.finish_time)
        
    for task in env_storage.TaskQueues.running.list:
        if task.PE_ID == PE.ID:
            if (task.start_time < lower_bound):
                running_info.append(lower_bound)
            else:
                running_info.append(task.start_time)
                task_start_time = task.start_time
            running_info.append(current_timestamp)
            
    merged_list = completed_info + running_info
    # get the utilization for the PE
    sum_of_active_times  = sum([merged_list[i*2+1] - merged_list[i*2] for i in range(int(len(merged_list)/2))])
    PE.utilization = sum_of_active_times/args.sampling_rate
    
    full_list = [PE.ID, PE.utilization, current_timestamp]
    info_list = [0 if i > (len(merged_list)-1) else merged_list[i] for i in range(12)]
    full_list.extend(info_list)
    
    PE.info = info_list
    
    if args.verbose:
        print('Time %s: for PE-%d'%(current_timestamp,PE.ID),PE.info)
    

def f_L2_policy_write_to_csv(file_name, task):
    start_time = (task.start_time)
    ID = (task.ID)
    jobid = (task.jobID)
    PE_ID = (task.PE_ID)
    slack = (task.slack)
    min_freq = (task.min_freq)
    max_freq = (task.max_freq)
    avg_freq = (task.avg_freq)
    edp = (task.edp)
    runtime = (task.finish_time - task.start_time)
    create_header = False
    if not (os.path.exists(file_name)):
        # Create the CSV header
        create_header = True
    with open(file_name, 'a', newline='') as csvfile:
        dataset = csv.writer(csvfile, delimiter=',')
        if create_header == True:
            dataset.writerow(['Timestamp', 'Task Job ID', 'Task ID', 'PE ID', 'Slack', 'Minimum Frequency', 'Maximum Frequency', 'Average Frequency', 'Task EDP', 'Task run-time'])
        data_list = [start_time, jobid, ID, PE_ID, slack, min_freq, max_freq, avg_freq, edp, runtime]
        dataset.writerow(data_list)


def f_get_PE_state(timestamp, PE):
    # Retrieves the current PE state, this is used for the ML-based DVFS policy and should match the data that is used for training (check DTPM_train_model.py)
    # return [[timestamp, PE.utilization, PE.current_frequency, PE.current_voltage, PE.current_power, PE.energy_sampling_period]]
    return [[PE.utilization, PE.current_frequency, PE.current_voltage, PE.current_power, PE.energy_sampling_period]]


def get_task_attr_and_values(task_obj):
    task_attr = [attr for attr in dir(task_obj) if not callable(getattr(task_obj, attr)) and not attr.startswith("__")]
    task_values = [getattr(task_obj, member) for member in task_attr]
    return task_attr, task_values


def clear_policies():
    # Remove old policy files
    file_list = fnmatch.filter(os.listdir('.'), '*.pkl')
    for f in file_list:
        os.remove(f)

def clear_tmp_checkpoints():
    # Remove old checkpoint files
    file_list = fnmatch.filter(os.listdir('.'), '*-ckpt-*')
    for f in file_list:
        os.remove(f)
    file_list = fnmatch.filter(os.listdir('.'), '*-task-*')
    for f in file_list:
        os.remove(f)

def clear_all_checkpoints():
    file_list = fnmatch.filter(os.listdir('.'), '*checkpoint*')
    for f in file_list:
        os.remove(f)
