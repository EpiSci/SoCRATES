import os

import numpy as np
import ray
import torch

from gym_ds3.schedulers.socrates.data_worker_ac import ACWorker
from gym_ds3.schedulers.socrates.parameter_server import ParameterServer
from config import args
args.scheduler_name = 'socrates'


if args.use_tensorboard:
    from tensorboardX import SummaryWriter
    os.makedirs(args.data_dir, exist_ok=True)
    writer = SummaryWriter("%s/%s/" % (args.data_dir, args.exp_name))
if args.use_wandb:
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.exp_name)
    print('Model will be saved to: ', run.id + '/model.pth')
    wandb.config.update(args)


n_cpu = 1
ray.init(num_cpus=n_cpu)
workers = [ACWorker.remote(args=args, _id=i) for i in range(args.num_agents)]
ps = ParameterServer(args=args)
weight = ps.get_weights()
print("RAY initialized with [%d] cpus and [%d] workers." % (n_cpu, args.num_agents))

if args.saved_model:
    model = torch.load(args.saved_model, map_location=torch.device('cpu'))
    weight = {k: v.cpu() for k, v in model.state_dict().items()}

num_gradient_steps = 0


while True:
    if num_gradient_steps >= args.num_of_iterations:
        break

    ops = [worker.compute_gradients.remote(
        num_gradient_steps, weight, args.scale) for worker in workers]
    ops_vals = ray.get(ops)

    results = {}
    for k in ops_vals[0][1].keys():
        results.update({k: []})

    grads_results = []
    for val in ops_vals:
        {results[k].append(v) for k, v in val[1].items()}
        grads_results.append(val[0])
    
    norm_res = {k:np.mean(results[k]) for k,_ in results.items()}
    if args.use_wandb:
        wandb.log(norm_res)

    # Compute and apply gradients.
    weight = ps.apply_gradients(grads_results)
    
    if (num_gradient_steps + 1) % args.model_save_interval == 0:
        os.makedirs(args.exp_name, exist_ok=True)
        torch.save(ps.get_model(), os.path.join(args.exp_name, 'model.pth'))
    
    num_gradient_steps += 1

ray.shutdown()
