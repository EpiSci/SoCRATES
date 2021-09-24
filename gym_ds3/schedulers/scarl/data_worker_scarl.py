import numpy as np

import ray
import torch

from gym_ds3.schedulers.models.scarl_model import attention_net
from gym_ds3.envs.utils.helper_envs import get_env, num_pes
from gym_ds3.envs.utils.helper_training import calculate_returns


@ray.remote
class SCARLWorker(object):
    def __init__(self, args, _id):
        self.args = args
        self.device = args.device
        self.seed = args.seed +_id
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.max_num_jobs = args.max_num_jobs
        self.num_tasks_in_jobs = args.num_tasks_in_job
        num_tasks = self.num_tasks_in_jobs * self.max_num_jobs
        num_actions = num_pes(args)
        self.model = attention_net(
            num_tasks=num_tasks, num_jobs=self.max_num_jobs,
            num_actions=num_actions, device=self.device,
            job_dim=15, machine_dim=10, embedding_size=32, 
            wo_attention=False).to(self.device)

        self.gamma = self.args.gamma
        self.value_loss_coef = self.args.value_loss_coef
        self.entropy_coef = self.args.entropy_coef
        
    def compute_gradients(self, num_gradient_steps, weights, scale):
        
        env = get_env()
        env.reset(self.args)

        # synchronize weights
        if weights is not None:
            self.model.set_weights(weights)
        self.model.zero_grad()

        total_inj_jobs, total_comp_jobs, total_cum_exec_time, total_energy_consumption, edp, avg_latency = \
            env.run(num_gradient_steps, scale, self.model)

        rewards = env.simulator.rewards_by_flops
        task_completed = env.simulator.task_completed

        returns = calculate_returns(rewards, self.gamma)

        stats = self.update(returns, task_completed)
        try:
            stats['episode'].append(num_gradient_steps)
            stats['latency'].append(avg_latency)
            stats['completed_jobs'].append(total_comp_jobs)
            stats['injected_jobs'].append(total_inj_jobs)
            stats['cumulative_execution_time'].append(total_cum_exec_time)

            stats['Execution Time'].append(len(rewards))
            stats['total_reward'].append(np.sum(rewards))

            stats['total_energy_consumption'].append(total_energy_consumption)
            stats['edp'].append(edp) 
        except:
            stats['episode'] = num_gradient_steps
            stats['latency'] = avg_latency
            stats['completed_jobs'] = total_comp_jobs
            stats['injected_jobs'] = total_inj_jobs
            stats['cumulative_execution_time'] = total_cum_exec_time

            stats['Execution Time'] = len(rewards)
            stats['total_reward'] = np.sum(rewards)

            stats['total_energy_consumption'] = total_energy_consumption
            stats['edp'] = edp

        return self.model.get_gradients(), stats

    def update(self, returns, task_completed):
        summed_losses = {}

        for info in task_completed:  # Change for online updating maybe.
            losses = {}

            ti = info[0]
            _, _, log_prob = self.model.forward(ti.state_at_scheduling[0], ti.state_at_scheduling[1])
            G = returns[ti.timestep_of_scheduling]

            losses['loss'] = (-log_prob * G)

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
