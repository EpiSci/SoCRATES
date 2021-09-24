import os
from os.path import dirname, abspath

from gym_ds3.envs.misc import DASH_SoC_parser


def str_to_list(x):
    # function to return a list based on a formatted string
    result = []
    for part in x.split(','):
        if 'txt' in part:
            result.append(part)
        elif '-' in part:
            a, b, c = part.split('-')
            a, b, c = int(a), int(b), int(c)
            result.extend(range(a, b, c))
        else:
            a = int(part)
            result.append(a)

    return result


def get_env():
    import gym
    import gym_ds3
    env = gym.make('Dsgym3-v0')
    gym.logger.set_level(40)  # gym logger
    return env


def get_scheduler(env, name):
    if name in ["MET", "met"]:
        from gym_ds3.schedulers.heuristic.simple_sched import MET
        return MET(env)
    elif name in ["EFT", "eft"]:
        from gym_ds3.schedulers.heuristic.simple_sched import EFT
        return EFT(env)
    elif name in ["ETF", "etf"]:
        from gym_ds3.schedulers.heuristic.simple_sched import ETF
        return ETF(env)
    elif name in ["HEFT_RT", "heft_rt", "heftrt", "HEFTRT"]:
        from gym_ds3.schedulers.heuristic.list_sched import HEFT_RT
        return HEFT_RT(env)
    elif name in ["stf" or "STF"]:
        from gym_ds3.schedulers.heuristic.simple_sched import STF
        return STF(env)
    elif name in ["DeepSoCS", "deepsocs"]:
        from gym_ds3.schedulers.deepsocs.deepsocs_scheduler import Deepsocs
        return Deepsocs(env)
    else:
        return Exception("No Scheduler Found")
    

NOOP = 3
def write_sched_stats(rt, no_op, env_storage):

    if rt.base_ID in env_storage.task_scheduling_stats:
        # valid op
        if not no_op:
            if rt.PE_ID in env_storage.task_scheduling_stats[rt.base_ID]:
                env_storage.task_scheduling_stats[rt.base_ID][rt.PE_ID] += 1
            else:
                env_storage.task_scheduling_stats[rt.base_ID][rt.PE_ID] = 1
        # no op
        else:
            if NOOP in env_storage.task_scheduling_stats[rt.base_ID]:
                env_storage.task_scheduling_stats[rt.base_ID][NOOP] += 1
            else:
                env_storage.task_scheduling_stats[rt.base_ID][NOOP] = {}
                env_storage.task_scheduling_stats[rt.base_ID][NOOP] = 1
    else:
        env_storage.task_scheduling_stats[rt.base_ID] = {}
        env_storage.task_scheduling_stats[rt.base_ID][rt.PE_ID] = 1


class Resourcelist:
        """
        Define the ResourceManager class to maintain
        the list of the resource in our DASH-SoC model.
        """
        def __init__(self):
            self.list = []
            self.comm_band = []


def num_pes(args):
    resource_matrix = Resourcelist()
    
    resource_file_list = str_to_list(
        os.path.join(dirname(dirname(dirname(abspath(__file__)))) + '/' + 'data' + '/' + args.resource_profile))
    for resource_file in resource_file_list:
        DASH_SoC_parser.resource_parse(args, resource_matrix, resource_file)
        
    return len(resource_matrix.list) - 1
