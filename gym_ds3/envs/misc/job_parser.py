import platform
import sys

import numpy as np

from gym_ds3.envs.core.env_storage import Applications, Tasks


def job_parse(jobs, file_name):
    """
	 In case of running platform is windows,opening and reading a file
    requires encoding = 'utf-8'
    In mac no need to specify encoding techique
    """
    try:
        current_platform = platform.system()                     # Find the platform
        if 'windows' in current_platform.lower():
            input_file = open(file_name, "r", encoding="utf-8")  # Read the configuration file
        elif 'darwin' in current_platform.lower():
            input_file = open(file_name, 'r')                    # Read the configuration file
        elif 'linux' in current_platform.lower():
            input_file = open(file_name, 'r')                    # Read the configuration file

    except IOError:
        # Print an error message, if the input file cannot be opened
        print("[E] Could not read configuration file that contains all tasks")
        print("[E] Please check if the file 'config_file.ini' has the correct file name")
        sys.exit()

    found_new_task = False      # The input lines do not correspond to a particular task
                                # unless found_new_task = = true;
    num_tasks_read = 0          # Initially none of the task are read from the file
    num_of_total_tasks = 0      # Initially there is no information about the number tasks
    
    # Instantiate the Applications object that contains all the information
    # about the next job                    
    new_job = Applications()
    
    # Now, the file is open. Read the input lines one by one
    for line in input_file:
        input_line = line.strip("\n\r ")                                        # Remove the end of line character
        current_line = input_line.split(" ")                                    # Split the input line into variables separated a space: " "

        if ( (len(input_line) == 0) or (current_line[0] == "#") or 
            '#' in current_line[0]):                                            # Ignore the lines that are empty or comments (start with #)
            continue

        if not(found_new_task):                                                 # new_task = common.Tasks()
            if current_line[0] == 'job_name':                               
                new_job.name = current_line[1]                                  # record new job's name and, 
                jobs.list.append(new_job)                                       # append the job list with the new job
                
            elif (current_line[0] == 'add_new_tasks'):                          # The key word "add_new_task" implies that the config file defines a new task
                num_of_total_tasks = int(current_line[1])
                
                new_job.comm_vol = np.zeros((num_of_total_tasks,
                                             num_of_total_tasks))     # Initialize the communication volume matrix
                # Set the flag to indicate that the following lines define the task parameters
                found_new_task = True
            
            elif current_line[0] == 'comm_vol':
                # The key word "comm_vol" implies that config file defines
                # an element of communication volume matrix
                new_job.comm_vol[int(current_line[1])][int(current_line[2])] = int(current_line[3])#*8.4
            
            else:
                print("[E] Cannot recognize the input line in task file: ", input_line)
                sys.exit() 
        
        # if not(found_new_task) (i.e., found a new task)
        else:
            # Check if this is the head (i.e., the leading task in this graph)
            if current_line[1] == 'HEAD':                               # Marked as the HEAD
                ind = new_job.task_list.index(new_job.task_list[-1])    # then take the id of the last added task and
                new_job.task_list[ind].head = True                      # change 'head' to True
                continue
            
            # Check if this is the tail (i.e., the last task in this graph)
            if current_line[1] == 'TAIL':                               # Marked as the TAIL
                ind = new_job.task_list.index(new_job.task_list[-1])    # then take the id of the last added task and
                new_job.task_list[ind].tail = True                      # change 'tail' to True
                continue
            
            if current_line[1] == 'earliest_start':                     # if 'earliest_start' in current line
                ind = new_job.task_list.index(new_job.task_list[-1])    # then take the id of the last added task and
                new_job.task_list[ind].est = current_line[2]            # add earliest start time (est), and
                new_job.task_list[ind].deadline = current_line[4]       # deadline for the task
                
                if (num_tasks_read == num_of_total_tasks):
                    # Reset these variables, since we completed reading the current resource
                    found_new_task = False
                    num_tasks_read = 0
                continue
            
            # Instantiate the Tasks object that contains all the information
            # about the next_task                    
            new_task = Tasks()
                        
            if num_tasks_read < num_of_total_tasks:
                new_task.name = current_line[0]
                new_task.ID = int(current_line[1])
                new_task.jobname = new_job.name
                
                # The rest of the inputs are predecessors (may be more than one)
                offset = 2
                # The first two inputs are name and ID
                for i in range(len(current_line)-offset):
                    new_task.predecessors.append(int(current_line[i+offset]))

                num_tasks_read += 1
                # Increment the number functionalities read so far
                new_job.task_list.append(new_task)
