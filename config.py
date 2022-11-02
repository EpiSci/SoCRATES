import argparse

import torch


parser = argparse.ArgumentParser(description='SoCRATES Framework')

# environment
parser.add_argument('--resource_profile', type=str, default='DASH.SoC.Top.txt', help='resource file')
parser.add_argument('--job_profile', type=str, default='job_Top.txt', help='job file')
parser.add_argument('--simulation_mode', type=str, default='performance', help='simulation mode (validation/performance)')
parser.add_argument('--shared_memory', type=bool, default=False, help='sharing memory mode (if FALSE, PE_to_PE mode)')
parser.add_argument('--verbose', type=bool, default=False, help='print debugging messages')
parser.add_argument('--resource_std', type=float, default=0., help='additional std to processing elements')
parser.add_argument('--scale', type=int, default=25, help='job injection frequecy')
parser.add_argument('--simulation_length', type=int, default=5000, help='simulation length')
parser.add_argument('--simulation_clk', type=int, default=1, help='simulation clock time')
parser.add_argument('--warmup_period', type=int, default=0, help='warm-up period')
parser.add_argument('--num_of_iterations', type=int, default=10000, help='number of episodes')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--scheduler_name', type=str, default='deepsocs', help='Scheduler name (etf/met/stf/heftrt/random/scarl/deepsocs/socrates)')
parser.add_argument('--max_num_jobs', type=int, default=3, help='length of job queue')
parser.add_argument('--num_tasks_in_job', type=int, default=10, help='number of tasks in job profile')


## Power management
parser.add_argument('--generate_complete_trace', type=bool, default=False, help='generate_complete_trace')
parser.add_argument('--initial_timestamp', type=int, default=0, help='initial_timestamp')
# Sampling rate for the DVFS mechanism
parser.add_argument('--sampling_rate', type=int, default=5, help='sampling rate for the DVFS mechanism')
parser.add_argument('--sampling_rate_temperature', type=int, default=10000, help='sampling_rate_temperature')
# High and low thresholds for the ondemand mode
parser.add_argument('--util_high_threshold', type=float, default=0.8, help='util_high_threshold')
parser.add_argument('--util_low_threshold', type=float, default=0.3, help='util_low_threshold')
# Machine learning-based DVFS policy: LR (Logistic Regression), MLP (Multi-Layer Perceptron), DT (Decision Tree)
parser.add_argument('--l2_inference', type=bool, default=False, help='L2_inference')
# Only enable L2_training if you are executing DTPM_train_model.py alone. DTPM_run_dagger.py already controls this flag.
parser.add_argument('--l2_training', type=bool, default=False, help='L2_training')
parser.add_argument('--dagger_iter', type=int, default=5, help='DAgger_iter')
parser.add_argument('--ml_algorithm', type=str, default='DT', help='ml_algorithm')
parser.add_argument('--run_expert', type=bool, default=False, help='run_expert')
parser.add_argument('--enable_monte_carlo_simulations', type=bool, default=False, help='enable_monte_carlo_simulations')
parser.add_argument('--num_monte_carlo_simulations', type=int, default=10, help='num_monte_carlo_simulations')
# Coefficients for the leakage power model (Odroid XU3 board)
parser.add_argument('--xu3_c1', type=float, default=0.002488, help='C1')
parser.add_argument('--xu3_c2', type=int, default=2660, help='C2')
parser.add_argument('--xu3_igate', type=float, default=0.000519, help='Igate')
parser.add_argument('--xu3_t_ambient', type=int, default=42, help='T ambient')


# RL common
parser.add_argument('--run_mode', type=str, default='run', help='running mode (run/step)')
parser.add_argument('--train', type=bool, default=True, help='train mode')
parser.add_argument('--pss', type=bool, default=True, help='pseudo steady state')
parser.add_argument('--num_agents', type=int, default=2, help='num_agents')
parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--learn_obj', type=str, default='compbonus', help='learning objective (duration/makespan/latency/compbonus)')


# SoCRATES
parser.add_argument('--value_loss_coef', type=float, default=0.5, help='value loss coefficient')
parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy coefficient')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max gradient norm')


# deepsocs
parser.add_argument('--node_input_dim', type=int, default=6,
                    help='node input dimensions to graph embedding (default: 5)')
parser.add_argument('--output_dim', type=int, default=8,
                    help='output dimensions throughout graph embedding (default: 8)')
parser.add_argument('--max_depth', type=int, default=2,
                    help='Maximum depth of root-leaf message passing (default: 2)')
parser.add_argument('--summ_levels', type=int, default=2,
                    help='Maximum depth of root-leaf message passing (default: 2)')
parser.add_argument('--job_input_dim', type=int, default=3,
                    help='job input dimensions to graph embedding (default: 3)')
parser.add_argument('--hid_dims', type=int, default=[16, 8], nargs='+',
                    help='hidden dimensions throughout graph embedding (default: [16, 8])')
parser.add_argument('--representation_map', type=str, default='gcn', help='Representation methods (gcn/mlp)')
parser.add_argument('--average_reward_storage_size', type=int, default=100000,
                    help='average_reward_storage_size')
parser.add_argument('--entropy_weight_init', type=float, default=1,
                    help='Initial exploration entropy weight (default: 1)')
parser.add_argument('--entropy_weight_min', type=float, default=0.0001,
                    help='Final minimum entropy weight (default: 0.0001)')
parser.add_argument('--entropy_weight_decay', type=float, default=1e-3,
                    help='Entropy weight decay rate (default: 1e-3)')
parser.add_argument('--reward_scale', type=float, default=10000.0,
                    help='scale the reward to some normal values (default: 100000.0)')
parser.add_argument('--pe_selection', type=str, default='eft', help='PE selection (eft/greedy)')
parser.add_argument('--curriculum', type=bool, default=False, help='curriculum learning')
parser.add_argument('--init_scale', type=int, default=1000, help='initial scale value')
parser.add_argument('--min_scale', type=int, default=25, help='minimum scale value')
parser.add_argument('--decay_scale', type=int, default=1, help='decay scale value')


# savings
parser.add_argument('--model_folder', type=str, default='./models/',
                    help='Model folder path (default: ./models)')
parser.add_argument('--saved_model', type=str, default=None,
                    help='Path to the saved tf model (default: None)')
parser.add_argument('--model_save_interval', type=int, default=50,
                    help='Interval for saving Tensorflow model (default: 1000)')


# visualization
parser.add_argument('--use_tensorboard', type=bool, default=False, help='tensorboardX')
parser.add_argument('--use_wandb', type=bool, default=False, help='wandb')
parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
parser.add_argument('--wandb_project', type=str, default='wandb_project', help='project for wandb visualization')
parser.add_argument('--wandb_entity', type=str, default='wandb_entity', help='entity for wandb visualization')


args = parser.parse_args()

args.device = 'cpu'
# if torch.cuda.is_available():
#     args.device = torch.device('cuda')
# else:
#     args.device = torch.device('cpu')
