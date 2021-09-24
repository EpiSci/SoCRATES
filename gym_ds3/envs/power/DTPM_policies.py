import sys
import math

from gym_ds3.envs.power import DTPM_power_models as DTPM_power_models


def initialize_frequency(resource, current_PE):
    if current_PE.current_frequency == 0:
        if resource.DVFS == 'ondemand':
            current_PE.current_frequency = DTPM_power_models.get_max_freq(resource.OPP)
            current_PE.current_voltage = DTPM_power_models.get_max_voltage(resource.OPP)
        elif resource.DVFS == 'performance':
            current_PE.current_frequency = DTPM_power_models.get_max_freq(resource.OPP)
            current_PE.current_voltage = DTPM_power_models.get_max_voltage(resource.OPP)
        elif resource.DVFS == 'powersave':
            current_PE.current_frequency = DTPM_power_models.get_min_freq(resource.OPP)
            current_PE.current_voltage = DTPM_power_models.get_min_voltage(resource.OPP)
        elif str(resource.DVFS).startswith('constant'):
            DVFS_str_split = str(resource.DVFS).split("-")
            constantFrequency = int(DVFS_str_split[1])
            current_PE.current_frequency = constantFrequency
            current_PE.current_voltage = DTPM_power_models.get_voltage_constant_mode(resource.OPP, constantFrequency)
        elif resource.DVFS == 'imitation-learning':
            # current_PE.current_frequency = DVFS_utils.get_max_freq(resource.OPP)
            # current_PE.current_voltage = DVFS_utils.get_max_voltage(resource.OPP)
            middle_opp = math.floor(len(resource.OPP) / 2)
            current_PE.current_frequency  = resource.OPP[middle_opp][0]
            current_PE.current_voltage    = resource.OPP[middle_opp][1]
            # current_PE.current_frequency = DVFS_utils.get_min_freq(resource.OPP)
            # current_PE.current_voltage = DVFS_utils.get_min_voltage(resource.OPP)

def ondemand_policy(args, resource, current_PE, timestamp):
    # When using ondemand, evaluate the PE utilization and adjust the frequency accordingly
    utilization = current_PE.utilization

    if utilization <= args.util_high_threshold and utilization >= args.util_low_threshold:
        # Keep the current frequency
        DTPM_power_models.keep_frequency(resource, current_PE, timestamp)
    elif utilization > args.util_high_threshold:
        # Set the maximum frequency
        DTPM_power_models.set_max_frequency(resource, current_PE, timestamp)
    elif utilization < args.util_low_threshold:
        # Decrease the frequency
        DTPM_power_models.decrease_frequency(resource, current_PE, timestamp)
    else:
        print("[E] Error while evaluating the PE utilization in the DVFS module, all test cases must be previously covered")
        sys.exit()

def imitation_learning_policy_L1(args, resource, current_PE, timestamp):
    if current_PE.L1_policy == None:
        print("[E] Error while loading the L1 policy")
        sys.exit()

    if not args.CHECKPOINT_CREATE_TMP:
        if current_PE.utilization != 0:
            prediction = current_PE.L1_policy.predict(DASH_Sim_utils.f_get_PE_state(timestamp, current_PE))
            if (args.DEBUG_SIM):
                print('[D] Time %d: PE %s - The DVFS policy predicted: %d' % (timestamp, current_PE.name, prediction))
            if prediction == -1:
                # Decrease the frequency
                DTPM_power_models.decrease_frequency(resource, current_PE, timestamp)
            elif prediction == 0:
                # Keep the current frequency
                DTPM_power_models.keep_frequency(resource, current_PE, timestamp)
            elif prediction == 1:
                # Increase the frequency
                DTPM_power_models.increase_frequency(resource, current_PE, timestamp)
            else:
                print("[E] Error while evaluating the PE utilization in the DVFS module, all test cases must be previously covered")
                sys.exit()
        else:
            # Keep the current frequency
            DTPM_power_models.keep_frequency(resource, current_PE, timestamp)


def imitation_learning_policy_L2(args, resource, current_PE, task, timestamp):
    if current_PE.L2_policy == None:
        print("[E] Error while loading the L2 policy")
        sys.exit()
    if not args.CHECKPOINT_CREATE_TMP:
        task_features = DASH_Sim_utils.f_get_task_features(task)
        predicted_frequency = current_PE.L2_policy.predict(task_features)
        predicted_frequency = int(predicted_frequency[0])
        if args.L2_training:
            expert_frequency = DTPM_run_dagger.get_expert_feedback_L2(task, task_features, resource, current_PE)
            DTPM_run_dagger.apply_DAgger_L2(int(predicted_frequency), expert_frequency, task_features, current_PE)
        DTPM_power_models.set_L2_best_frequency(resource, current_PE, timestamp, int(predicted_frequency))
