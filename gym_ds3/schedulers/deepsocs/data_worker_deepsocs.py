import numpy as np

import ray
import tensorflow as tf

from gym_ds3.schedulers.deepsocs.deepsocs_scheduler import Deepsocs
from gym_ds3.schedulers.models.deepsocs_model import create_deepsocs_model, create_deepsocs_graph
from gym_ds3.envs.utils.helper_deepsocs import suppress_tf_warning, truncate_experiences, \
    expand_sp_mat, merge_and_extend_sp_mat, aggregate_gradients
from gym_ds3.envs.utils.helper_envs import get_env
 
 


@ray.remote
class RayDeepSoCSWorker(object):
    def __init__(self, args, _id):
        self.args = args
        self.seed = (_id + args.seed)
        suppress_tf_warning()  # suppress TF warnings

        # AAD model
        self.model, self.sess = create_deepsocs_model(args)
        self.graph = create_deepsocs_graph(args=args, model=self.model)

        # Deepsocs Scheduler
        self.deepsocs = Deepsocs(args, self.model, self.sess)

        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        self.sess.run(tf.global_variables_initializer())

        # Flag to initialize assign operations for 'set_weights()'
        self.FIRST_SET_FLAG = True

    def set_weights(self, weight_vals):
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

    def rollout(self, episode, scale):
        env = get_env()
        env.reset(self.args)
        env.env_storage.buf.exp['wall_time'].append(env.now())
        
        self.deepsocs.reset()

        total_inj_jobs, total_comp_jobs, total_cum_exec_time, total_energy_consumption, edp, avg_latency = \
            env.run(num_gradient_steps=episode, scale=scale, scheduler=self.deepsocs)

        rollout_val = env.env_storage.buf.get()
        
        stats = {}
        try:
            stats['episode'].append(episode)
            stats['latency'].append(avg_latency)
            stats['completed_jobs'].append(total_comp_jobs)
            stats['injected_jobs'].append(total_inj_jobs)
            stats['cumulative_execution_time'].append(total_cum_exec_time)
            stats['energy_consumption'].append(total_energy_consumption)
            stats['edp'].append(edp)
        except:
            stats['episode'] = episode
            stats['latency'] = avg_latency
            stats['completed_jobs'] = total_comp_jobs
            stats['injected_jobs'] = total_inj_jobs
            stats['cumulative_execution_time'] = total_cum_exec_time
            stats['energy_consumption'] = total_energy_consumption
            stats['edp'] = edp

        return rollout_val, stats

    def get_gradients(self, buf_data, adv, entropy_weight=1.):
        # fetch batch size based on the job changing timestep
        batch_points = truncate_experiences(buf_data['obs_change'])

        all_gradients = []
        all_loss = [[], [], [], 0]

        for b in range(len(batch_points) - 1):
            # need to do different batches because the size of dags in state changes
            ba_start = batch_points[b]
            ba_end = batch_points[b + 1]

            # use a piece of experience
            batch_node_inputs = np.vstack(buf_data['node_inputs'][ba_start: ba_end])
            batch_node_valid_mask = np.vstack(buf_data['node_valid_mask'][ba_start: ba_end])
            batch_summ_mats = buf_data['summ_mats'][ba_start: ba_end]
            batch_running_dag_mats = buf_data['running_dag_mat'][ba_start: ba_end]
            batch_act1 = np.vstack(buf_data['a1'][ba_start:ba_end])

            gcn_mats = buf_data['gcn_mats'][b]
            gcn_masks = buf_data['gcn_masks'][b]
            summ_backward_map = buf_data['dag_summ_backward'][b]
            batch_adv = adv[ba_start:ba_end, :]

            # given an episode of experience (advantage computed from baseline)
            batch_size = ba_end - ba_start

            # expand sparse adj_mats
            extended_gcn_mats = expand_sp_mat(gcn_mats, batch_size)

            # extended masks (on the dimension according to extended adj_mat)
            extended_gcn_masks = [np.tile(m, (batch_size, 1)) for m in gcn_masks]

            # expand sparse summ_mats
            extended_summ_mats = merge_and_extend_sp_mat(batch_summ_mats)

            # expand sparse running_dag_mats
            extended_running_dag_mats = merge_and_extend_sp_mat(batch_running_dag_mats)

            # compute gradient
            feed_dict = {i: d for i, d in zip(
                [self.model['batch']] + [self.model['o_ph']] + self.model['adj_mats'] + self.model['masks'] +
                self.model['summ_mats'] + [self.model['dag_summ_backward']] +
                [self.model['node_valid_mask']] +
                [self.model['a1_ph']] + [self.model['adv_ph']] + [self.model['entropy_weight']],
                [batch_size] + [batch_node_inputs] + extended_gcn_mats + extended_gcn_masks +
                [extended_summ_mats, extended_running_dag_mats] + [summ_backward_map] +
                [batch_node_valid_mask] +
                [batch_act1] + [batch_adv] + [entropy_weight]
            )}

            losses = self.sess.run(
                [self.graph['adv_loss'], self.graph['entropy_loss'], self.graph['pi_loss']],
                feed_dict=feed_dict)

            grad = self.sess.run(
                self.graph['gradients'], feed_dict=feed_dict)

            all_gradients.append(grad)
            all_loss[0].append(losses[-1])
            all_loss[1].append(losses[0])
            all_loss[2].append(losses[1])

        all_loss[0] = np.sum(all_loss[0])  # pi loss
        all_loss[1] = np.sum(all_loss[1])  # adv_loss
        all_loss[2] = np.sum(all_loss[2])  # to get entropy
        all_loss[3] = np.sum(batch_adv ** 2)  # time based baseline loss

        # aggregate all gradients from the batches
        gradients = aggregate_gradients(all_gradients)

        return gradients, all_loss
