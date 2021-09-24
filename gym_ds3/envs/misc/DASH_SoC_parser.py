import platform
import sys

import numpy as np

from gym_ds3.envs.core.env_storage import Resource


def resource_parse(args, resource_matrix, file_name):
    '''
	In case of running platform is windows,opening and reading a file
    requires encoding = 'utf-8'
	In mac no need to specify encoding techique.
    '''
    try:
        current_platform = platform.system()                                    # Find the platform
        if 'windows' in current_platform.lower():
            input_file = open(file_name, "r", encoding = "utf-8")               # Read the configuration file
        elif 'darwin' in current_platform.lower():
            input_file = open(file_name, 'r')                                   # Read the configuration file
        elif 'linux' in current_platform.lower():
            input_file = open(file_name, 'r')                                   # Read the configuration file


    except IOError:
        print("[E] Could not read configuration file that contains available resources in DASH-SoC")
        print("[E] Please check if the file 'config_file.ini' has the correct file name")
        sys.exit()                                                              # Print an error message, if the input file cannot be opened

    # Now, the file is open. Read the input lines one by one

    found_new_resource = False                                                  # The input lines do not correspond to a particular resource
                                                                                # unless found_new_resource = = true;
                                                                                # This variable shows the number of functionalities supported for a given resource
    num_functionality_read = 0                                                  # No new functionality has been read initially
    num_PEs_TRACE = 0
    resource_list = []
    DVFS_cfg_list = []
    
    for line in input_file:
        input_line = line.strip("\n\r ")                                        # Remove the end of line character
        current_line = input_line.split(" ")                                    # Split the input line into variables sepearated a space: " "
        if ( (len(input_line) == 0) or (current_line[0] == "#") or
            '#' in current_line[0]):                                            # Ignore the lines that are empty or comments (start with #)
            continue

        if not(found_new_resource):
            
            new_resource = Resource()
            if (current_line[0] == 'add_new_resource'):                         # The key word "add_new_resource" implies that the config file defines a new resource

                new_resource.type = current_line[1]
                new_resource.name = current_line[2]
                new_resource.ID = int(current_line[3])
                new_resource.capacity = int(current_line[4])
                new_resource.num_of_functionalities = int(current_line[5])      # Converting the input string to integer

                if args.generate_complete_trace == False:
                    new_resource.DVFS = current_line[6] # Obtain the DVFS mode for the given PE
                else:
                    if len(resource_list) < num_PEs_TRACE:
                        new_resource.DVFS = DVFS_cfg_list[len(resource_list)]
                    else:
                        new_resource.DVFS = current_line[6]

                resource_list.append(new_resource.ID)
                resource_matrix.comm_band = np.ones((len(resource_list), len(resource_list))) # Initialize the communication volume matrix
                
                found_new_resource = True # Set the flag to indicate that the following lines define the functionalities
                  
            elif current_line[0] == 'comm_band':
                # The key word "comm_band" implies that config file defines
                # an element of communication bandwidth matrix
                resource_matrix.comm_band[int(current_line[1]),int(current_line[2])] = int(current_line[3])
                resource_matrix.comm_band[int(current_line[2]),int(current_line[1])] = int(current_line[3])

            else:
                print("[E] Cannot recognize the input line in resource file:", input_line )
                sys.exit()

        # end of: if not(found_new_resource)

        # if not(found_new_resource) (i.e., found a new resource)
        else:
            if current_line[0] == 'opp':
                new_resource.OPP.append((int(current_line[1]), int(current_line[2])))

            elif (num_functionality_read < new_resource.num_of_functionalities):
                new_resource.supported_functionalities.append(current_line[0])
                new_resource.performance.append(float(current_line[1]))
                new_resource.power_consumption.append(float(current_line[2]))

                num_functionality_read += 1 # Increment the number functionalities read so far
                # print("number of functionality read: ", num_functionality_read)

                # Reset these variables, since we completed reading the current resource
                if (num_functionality_read == new_resource.num_of_functionalities):
                    found_new_resource = False
                    num_functionality_read = 0
                    new_resource.OPP = sorted(new_resource.OPP)
                    resource_matrix.list.append(new_resource)
        # end of else: # if not(found_new_resource)

    # Number of resources, ignore the memory
    num_PEs_TRACE = len(resource_list) - 1
