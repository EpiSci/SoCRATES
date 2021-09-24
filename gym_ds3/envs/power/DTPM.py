'''
Description: This file contains the code for the DTPM module.
'''
import sys

from gym_ds3.envs.power import DTPM_policies as DTPM_policies
from gym_ds3.envs.power import DTPM_power_models as DTPM_power_models
from gym_ds3.envs.utils import DASH_Sim_utils


class DTPMmodule:
    '''
    The DTPM module is responsible for evaluating the PE utilization and changing the V/F according to the defined policy
    '''
    def __init__(self, args, env, env_storage, resource_matrix, PEs):
        '''
        env: Pointer to the current simulation environment
        resource_matrix: The data structure that defines power/performance
        PEs: The PEs available in the current SoC
        timestamp_last_update: Contains the timestamp from the last time each PE was evaluated
        '''
        self.args = args
        self.env = env
        self.env_storage = env_storage
        self.resource_matrix = resource_matrix
        self.PEs = PEs
        self.timestamp_last_update = [-1] * len(PEs)
        self.timestamp_sample_update = 0

        if args.verbose:
            print('[D] DVFS module was initialized')

    def evaluate_PE(self, resource, current_PE, timestamp):

        if self.timestamp_last_update[current_PE.ID] != timestamp and timestamp != self.args.initial_timestamp:

            self.timestamp_last_update[current_PE.ID] = timestamp

            DASH_Sim_utils.update_PE_utilization_and_info(self.args, self.env_storage, current_PE, timestamp)

            if self.args.verbose:
                print('%12s' % (''), 'Utilization for %s is %.2f' % (current_PE.name, current_PE.utilization))

            if resource.DVFS == 'ondemand' or resource.DVFS == 'powersave' or str(resource.DVFS).startswith("constant"):
                # The only DVFS mode that does not require OPPs is the performance one
                if len(resource.OPP) == 0:
                    print("[E] PEs using %s DVFS mode must have at least one OPP, please check the resource file" % resource.DVFS)
                    sys.exit()

            # Apply Monte Carlo method for the expert feedback, if enabled
            # if args.monte_carlo_simulations and str(resource.DVFS).startswith("constant") and args.CHECKPOINT_CREATE_TMP:
            #     if len(args.DVFS_cfg_monte_carlo[current_PE.ID]) > 1:
            #         random_walk_decision = args.DVFS_cfg_monte_carlo[current_PE.ID][args.sample_counter]
            #         if random_walk_decision == -1:
            #             DTPM_power_models.decrease_frequency(resource, current_PE, timestamp)
            #         elif random_walk_decision == 0:
            #             DTPM_power_models.keep_frequency(resource, current_PE, timestamp)
            #         elif random_walk_decision == 1:
            #             DTPM_power_models.increase_frequency(resource, current_PE, timestamp)
            #         else:
            #             print("[E] Monte carlo list could not be parsed correctly")
            #             sys.exit()
            #     else:
            #         DTPM_power_models.keep_frequency(resource, current_PE, timestamp)

            # Custom DVFS policies
            if resource.DVFS == 'ondemand':
                DTPM_policies.ondemand_policy(resource, current_PE, self.env.now)
            if resource.DVFS == 'imitation-learning':
                DTPM_policies.imitation_learning_policy_L1(resource, current_PE, self.env.now)

            # Update temperature
            if timestamp % self.args.sampling_rate_temperature == 0:
                current_PE.current_temperature_vector = DTPM_power_models.predict_temperature(current_PE)

            if resource.DVFS != 'none':
                # DASH_Sim_utils.f_trace_frequency(self.env.now, current_PE)
                if self.args.verbose:
                   print("[D] ", current_PE.ID, current_PE.current_temperature_vector)
                # DASH_Sim_utils.f_trace_PEs(self.env.now, current_PE)
                # DASH_Sim_utils.f_create_dataset_L1(self.env.now, current_PE)
                # DASH_Sim_utils.f_create_checkpoint(self.env.now, current_PE)

            # Reset the energy of the sampling period
            current_PE.energy_sampling_period = 0

        if self.timestamp_sample_update != timestamp:
            # Increment the sample counter
            # args.sample_counter += 1
            self.timestamp_sample_update = timestamp

    def evaluate_idle_PEs(self):
        '''
        Check all PEs and, for those that are idle, adjust the frequency and power accordingly.
        '''
        if self.args.verbose:
            print('[D] Time %s: DVFS module is evaluating PE utilization' % self.env.now)
        for i, resource in enumerate(self.resource_matrix.list):
            current_PE = self.PEs[i]
            # Only evaluate the PE if there is no process running, otherwise the PE itself will call the DVFS evaluation
            if current_PE.process.count == 0:
                self.evaluate_PE(resource, current_PE, self.env.now)
                # Update the power dissipation to be only the static power as the PE is currently idle
                current_PE.current_power = current_PE.current_leakage
