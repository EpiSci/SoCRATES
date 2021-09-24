
import numpy as np
import tensorflow as tf

from gym_ds3.schedulers.deepsocs.average_reward import AveragePerStepReward
from gym_ds3.schedulers.deepsocs.compute_baselines import get_piecewise_linear_fit_baseline
from gym_ds3.schedulers.deepsocs.deepsocs_scheduler import Deepsocs
from gym_ds3.schedulers.models.deepsocs_model import create_deepsocs_model, create_deepsocs_graph
from gym_ds3.envs.utils.helper_deepsocs import suppress_tf_warning, discount



class ParameterServer(object):
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        suppress_tf_warning()  # suppress TF warnings

        # AAD model
        self.model, self.sess = create_deepsocs_model(args)
        self.graph = create_deepsocs_graph(args=args, model=self.model)

        # Deepsocs Scheduler
        self.deepsocs = Deepsocs(args, self.model, self.sess)
        
        self.avg_reward_calculator = AveragePerStepReward(size=100000)

        # Initialize model
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        self.sess.run(tf.global_variables_initializer())

        # Flag to initialize assign operations for 'set_weights()'
        self.FIRST_SET_FLAG = True

    def get_weights(self):
        weight_vals = self.sess.run(self.model['all_vars'])
        return weight_vals

    def set_weights(self, weight_vals):
        """
        Set weights without memory leakage
        """
        if self.FIRST_SET_FLAG:
            self.FIRST_SET_FLAG = False
            self.assign_placeholders = []
            self.assign_ops = []
            for w_idx, weight_tf_var in enumerate(self.model['all_vars']):
                a = weight_tf_var
                assign_placeholder = tf.placeholder(a.dtype, shape=a.get_shape())
                assign_op = a.assign(assign_placeholder)
                self.assign_placeholders.append(assign_placeholder)
                self.assign_ops.append(assign_op)
        for w_idx, weight_tf_var in enumerate(self.model['all_vars']):
            self.sess.run(self.assign_ops[w_idx],
                          {self.assign_placeholders[w_idx]: weight_vals[w_idx]})

    def apply_gradients(self, gradients):
        self.sess.run(self.graph['apply_grads'], feed_dict={
            i: d for i, d in zip(self.graph['gradients'], gradients)
        })
        
    def compute_advantages(self, ops_vals):
        # calculate advantages (input-dependent baselines)
        all_times, all_diff_times, all_rewards, last_returns = [], [], [], []

        results = {}
        for ops_val in ops_vals:
            rollout_val = ops_val[0]
            stat = ops_val[1]
            
            diff_time = np.array(rollout_val['wall_time'][1:]) - np.array(rollout_val['wall_time'][:-1])
            self.avg_reward_calculator.add_list_filter_zero(rollout_val['reward'], diff_time)

            all_diff_times.append(diff_time)
            all_times.append(rollout_val['wall_time'][1:])
            all_rewards.append(rollout_val['reward'])
            
            for k, v in stat.items():
                try:                        
                    results[k].append(v)
                except:
                    results.update({k: []})
                    results[k].append(v)

        adv, all_cum_reward = compute_advantage(
            self.args, self.avg_reward_calculator, all_rewards, all_diff_times, all_times)
        for cum_reward in all_cum_reward:
            last_returns.append(cum_reward[-1])
            
        return results, adv


def compute_advantage(args, reward_calculator, all_rewards, all_diff_times, all_times):
    # compute differential reward
    all_cum_reward = []
    avg_per_step_reward = reward_calculator.get_avg_per_step_reward()
    for i in range(args.num_agents):
        # differential reward mode on
        rewards = np.array([r - avg_per_step_reward * t for \
                            (r, t) in zip(all_rewards[i], all_diff_times[i])])

        cum_reward = discount(rewards, args.gamma)

        all_cum_reward.append(cum_reward)

    baselines = get_piecewise_linear_fit_baseline(all_cum_reward, all_times)

    # give worker back the advantage
    advs = []
    for i in range(args.num_agents):
        batch_adv = all_cum_reward[i] - baselines[i]
        batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])
        advs.append(batch_adv)

    return advs, all_cum_reward
