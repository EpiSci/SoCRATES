import numpy as np

from config import args
from gym_ds3.envs.utils.helper_envs import get_env


class HeuristicWorker(object):
    def __init__(self, args, _id):
        np.random.seed(args.seed + _id)
        
        self.args = args
        self.env = get_env()

    def run_episode(self, num_gradient_steps, scale):
        self.env.reset(args)
        
        from gym_ds3.envs.utils.helper_envs import get_scheduler
        scheduler = get_scheduler(self.env, args.scheduler_name)
        
        total_inj_jobs, total_comp_jobs, total_cum_exec_time, total_energy_consump, edp, avg_latency = \
            self.env.run(num_gradient_steps, scale, scheduler)

        stats = {}
        stats['episode'] = num_gradient_steps
        stats['latency'] = avg_latency
        stats['completed_jobs'] = total_comp_jobs
        stats['injected_jobs'] = total_inj_jobs
        stats['cumulative_execution_time'] = total_cum_exec_time
        stats['total_energy_consumption'] = total_energy_consump
        stats['edp'] = edp

        return stats
