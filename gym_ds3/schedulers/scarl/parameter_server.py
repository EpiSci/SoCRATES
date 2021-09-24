import numpy as np

import torch

from gym_ds3.envs.utils.helper_envs import num_pes
from gym_ds3.schedulers.models.scarl_model import attention_net


class ParameterServer(object):
    def __init__(self, args):
        self.device = args.device

        self.max_num_jobs = args.max_num_jobs
        self.num_tasks_in_jobs = args.num_tasks_in_job
        num_tasks = self.num_tasks_in_jobs * self.max_num_jobs
        num_actions = num_pes(args)
        
        self.model = attention_net(
            num_tasks=num_tasks, num_jobs=self.max_num_jobs,
            num_actions=num_actions, device=self.device,
            job_dim=15, machine_dim=10, embedding_size=32, 
            wo_attention=False).to(self.device)
        
        self.params = self.model.parameters()

        self.opt = torch.optim.Adam(self.params, 3e-4)

        self.apply_gradient_steps = 0

        self.max_grad_norm = args.max_grad_norm

    def get_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def get_model(self):
        return self.model

    def apply_gradients(self, gradients):
        summed_gradients = []
        for layer_grad in gradients[0]:
            summed_gradients.append(np.zeros_like(layer_grad))
        for worker_i, worker_grad in enumerate(gradients):
            for layer_i, lay_grad in enumerate(worker_grad):
                summed_gradients[layer_i] = np.add(summed_gradients[layer_i], lay_grad)
        
        self.opt.zero_grad()
        self.model.set_gradients(summed_gradients, self.device)
        torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.opt.step()
        self.apply_gradient_steps += 1
        return self.get_weights()
    