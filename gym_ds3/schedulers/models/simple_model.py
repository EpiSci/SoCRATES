import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self, num_tasks, num_jobs, num_actions, device):
        super(SimpleModel, self).__init__()
        self.num_tasks = num_tasks
        self.num_actions = num_actions
        self.device = device
        
        self.num_tasks_in_jobs = self.num_tasks // num_jobs
        
        self.hidden_size = 256

        self.enc_fc1 = nn.Linear(247, 512)  # 157, 247, 411
        self.enc_fc2 = nn.Linear(512, self.hidden_size)

        self.fc_act = nn.Linear(self.hidden_size, num_tasks * num_actions)
        self.fc_cri = nn.Linear(self.hidden_size, 1)

        self.train()

    def forward(self, inputs):
        x = F.relu(self.enc_fc1(torch.flatten(inputs.to(self.device))))
        feat = F.relu(self.enc_fc2(x))
        
        value = self.fc_cri(feat)
        probs = F.log_softmax(self.fc_act(feat).reshape(self.num_tasks, self.num_actions), dim=-1)

        return feat, probs, value

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients, device):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g).to(device)
