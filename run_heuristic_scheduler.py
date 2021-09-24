import os

from gym_ds3.schedulers.heuristic.data_worker_heuristic import HeuristicWorker
from config import args
args.scheduler_name = 'MET'

if args.use_tensorboard:
    from tensorboardX import SummaryWriter
    os.makedirs(args.data_dir, exist_ok=True)
    writer = SummaryWriter("%s/%s/" % (args.data_dir, args.exp_name))
if args.use_wandb:
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.exp_name)
    print('Model will be saved to: ', run.id + '/model.pth')
    wandb.config.update(args)


worker = HeuristicWorker(args=args, _id=0)

num_gradient_steps = 0

while True:
    if num_gradient_steps >= args.num_of_iterations:
        break

    results = worker.run_episode(num_gradient_steps, args.scale)

    if args.use_wandb:
        wandb.log(results)

    num_gradient_steps += 1
