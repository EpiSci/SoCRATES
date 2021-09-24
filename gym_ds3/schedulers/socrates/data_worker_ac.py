import numpy as np

import ray
import torch

from gym_ds3.schedulers.models.simple_model import SimpleModel
from gym_ds3.envs.utils.helper_envs import num_pes, get_env
from gym_ds3.envs.utils.helper_training import calculate_returns


@ray.remote
class ACWorker(object):
    def __init__(self, args, _id):
        self.args = args
        self.device = args.device
        self.seed = self.args.seed + _id
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.max_num_jobs = self.args.max_num_jobs
        self.num_tasks_in_jobs = self.args.num_tasks_in_job
        num_tasks = self.num_tasks_in_jobs * self.max_num_jobs
        num_actions = num_pes(args)
        
        self.model = SimpleModel(
            num_tasks=num_tasks, num_jobs=self.max_num_jobs, 
            num_actions=num_actions, device=self.device).to(self.device)

        self.gamma = self.args.gamma
        self.value_loss_coef = self.args.value_loss_coef
        self.entropy_coef = self.args.entropy_coef

    def compute_gradients(self, num_gradient_steps, weights, scale):

        self.env = get_env()
        self.env.reset(self.args)
            
        # synchronize weights
        if weights is not None:
            self.model.set_weights(weights)
        self.model.zero_grad()

        total_inj_jobs, total_comp_jobs, total_cum_exec_time, total_energy_consump, edp, avg_latency = \
            self.env.run(num_gradient_steps, scale, self.model)

        rewards = self.env.simulator.rewards_by_flops
        task_completed = self.env.simulator.task_completed

        returns = calculate_returns(rewards, self.gamma)

        stats = self.update(returns, task_completed)
        try:
            stats['episode'].append(num_gradient_steps)
            stats['latency'].append(avg_latency)
            stats['completed_jobs'].append(total_comp_jobs)
            stats['injected_jobs'].append(total_inj_jobs)
            stats['cumulative_execution_time'].append(total_cum_exec_time)
            stats['energy_consumption'].append(total_energy_consump)
            stats['edp'].append(edp)

            stats['Execution Time'].append(len(rewards))
            stats['total_reward'].append(np.sum(rewards))
        except:
            stats['episode'] = num_gradient_steps
            stats['latency'] = avg_latency
            stats['completed_jobs'] = total_comp_jobs
            stats['injected_jobs'] = total_inj_jobs
            stats['cumulative_execution_time'] = total_cum_exec_time
            stats['energy_consumption'] = total_energy_consump
            stats['edp'] = edp

            stats['Execution Time'] = len(rewards)
            stats['total_reward'] = np.sum(rewards)

        return self.model.get_gradients(), stats

    def update(self, returns, task_completed):
        summed_losses = {}

        for info in task_completed:  # Change for online updating maybe.
            losses = {}

            ti = info[0]
            _, log_prob, v = self.model.forward(ti.state_at_scheduling)
            G = returns[ti.timestep_of_scheduling]
            
            adv = G - v

            losses['actor'] = (-log_prob[ti.probs_idx][ti.action_at_scheduling] * adv)[0]
            losses['value'] = self.value_loss_coef * adv.pow(2)[0]
            losses['entropy'] = self.entropy_coef * \
                (torch.exp(log_prob)[ti.probs_idx] * log_prob[ti.probs_idx]).sum()

            combined_loss = torch.tensor(0., dtype=torch.float).to(self.device)
            for loss in losses.values():
                combined_loss += loss
            losses['combined'] = combined_loss

            for k, v in losses.items():
                if k not in summed_losses:
                    summed_losses[k] = [v, 1]
            for k, v in losses.items():
                summed_losses[k][0] += v
                summed_losses[k][1] += 1

            combined_loss.backward()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        averaged_losses = {k: v[0].detach().cpu().numpy() / v[1] for k, v in summed_losses.items()}

        return averaged_losses
