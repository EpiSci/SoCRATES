import os

import numpy as np
import ray
import tensorflow as tf

from gym_ds3.schedulers.deepsocs.data_worker_deepsocs import RayDeepSoCSWorker
from gym_ds3.schedulers.deepsocs.parameter_server import ParameterServer
from gym_ds3.envs.utils.helper_training import save_model
from gym_ds3.envs.utils.helper_deepsocs import decrease_var, aggregate_gradients
from config import args
args.scheduler_name = 'deepsocs'


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
tf.reset_default_graph()
ps = ParameterServer(args=args)
workers = [RayDeepSoCSWorker.remote(args=args, _id=i) for i in range(args.num_agents)]
print("RAY initialized with [%d] cpus and [%d] workers." % (n_cpu, args.num_agents))

num_gradient_steps = 0
entropy_weight = args.entropy_weight_init
if args.curriculum:
    scale = args.init_scale
else:
    scale = args.scale


while True:
    if num_gradient_steps >= args.num_of_iterations:
        break

    weights = ps.get_weights()
    _ = [worker.set_weights.remote(weights) for worker in workers]
    
    ops = [worker.rollout.remote(num_gradient_steps, scale) for worker in workers]
    ops_vals = ray.get(ops)
    
    # compute baselines
    results, adv = ps.compute_advantages(ops_vals)

    norm_results = {k: np.mean(results[k]) for k, _ in results.items()}
    if args.use_wandb:
        wandb.log(norm_results)
    
    # compute and apply gradients
    ops = [worker.get_gradients.remote(
        ops_vals[i][0], adv[i], entropy_weight) for i, worker in enumerate(workers)]
    grad_loss_vals = ray.get(ops)  # gradients, [pi loss, adv loss, ent loss, value loss]
    
    all_gradients = []
    all_pi_loss = []
    all_adv_loss = []
    all_entropy = []
    all_value_loss = []
    for i, val in enumerate(grad_loss_vals):
        all_gradients.append(val[0])
        all_pi_loss.append(val[1][0])
        all_adv_loss.append(val[1][1])
        all_entropy.append(-val[1][2] / float(len(ops_vals[i][0]['reward'])))
        all_value_loss.append(val[1][3])

    # apply gradient directly to update parameters
    ps.apply_gradients(aggregate_gradients(all_gradients))
    
    if (num_gradient_steps + 1) % args.model_save_interval == 0:
        save_model(args.model_path, ps, VERBOSE=False)
        
    num_gradient_steps += 1
    
    # entropy & scale decay
    entropy_weight = decrease_var(
        entropy_weight, args.entropy_weight_min, args.entropy_weight_decay)
    if args.curriculum:
        if scale > args.min_scale:
            scale -= args.decay_scale
