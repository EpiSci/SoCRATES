import tensorflow as tf

from gym_ds3.envs.utils.helper_deepsocs import get_vars, count_vars


def create_deepsocs_model(args, feat_hdims=[32,16], gnn_hdims=[16,8],
                     task_hdims=[32,16,8], pe_hdims=[64,32,16,8], output_actv=None):
    """
    Create DeepSoCS Model (compatible with Ray)
    """
    # import tensorflow as tf  # make it compatible with Ray actors

    def mlp(x, hdims, actv=tf.nn.leaky_relu, output_actv=None):
        for h in hdims[:-1]:
            x = tf.compat.v1.layers.dense(inputs=x, units=h, activation=actv)
        return tf.compat.v1.layers.dense(inputs=x, units=hdims[-1], activation=actv)

    def mlp_feature(x, hdims, actv=tf.nn.leaky_relu, output_actv=None):
        return mlp(x, hdims=hdims, actv=actv, output_actv=output_actv)

    def gcn_feature(inp, out, hdims, actv=tf.nn.relu):
        """
        Input: job_dag_list
        """
        gcn_out = gcn(inp, out, adj_mats, masks, hdims, actv=actv)

        return gcn_out

    def gsn_feature(inp, gcn_out, out, hdims, actv=tf.nn.relu):
        """
        Input: job_dag_list
        """
        gsn_out = gsn(tf.concat([inp, gcn_out], axis=1), out, summ_mats, hdims, actv=actv)
        return gsn_out

    def gcn(inp, out, adj_mats, masks, hdims, actv=tf.nn.leaky_relu):
        outdim = args.output_dim

        prep = mlp(inp, hdims=hdims + [outdim], actv=actv, output_actv=actv)

        for d in range(max_depth):
            with tf.compat.v1.variable_scope('check2', reuse=tf.compat.v1.AUTO_REUSE):
                proc = mlp(prep, hdims=hdims + [outdim], actv=actv, output_actv=actv)
            x = tf.sparse.sparse_dense_matmul(adj_mats[d], proc)
            with tf.compat.v1.variable_scope('check4', reuse=tf.compat.v1.AUTO_REUSE):
                agg = mlp(x, hdims=hdims + [outdim], actv=actv, output_actv=actv)
            x = agg * masks[d]

            x = prep + x

            return x

    def gsn(inp, out, summ_mats, hdims, actv=tf.nn.leaky_relu):
        outdim = args.output_dim  # out.shape.as_list()[-1]
        summaries = []

        dag = mlp(inp, hdims + [outdim], actv=actv, output_actv=actv)
        x1 = tf.sparse.sparse_dense_matmul(summ_mats[0], dag)
        summaries.append(x1)

        glob = mlp(x1, hdims + [outdim], actv=actv, output_actv=actv)
        x2 = tf.sparse.sparse_dense_matmul(summ_mats[1], glob)
        summaries.append(x2)

        return summaries

    def mlp_task_policy(batch_size, o, a, valid_mask,
                        hdims, actv=tf.nn.leaky_relu, output_actv=None):
        with tf.compat.v1.variable_scope('t_pi'):
            x = mlp(o, hdims=hdims, actv=actv, output_actv=actv)
            out = tf.compat.v1.layers.dense(inputs=x, units=1, activation=None)
            out = tf.reshape(out, [batch_size, -1])

            node_valid_mask = (valid_mask - 1) * 10000.0
            out = out + node_valid_mask
        return tf.nn.softmax(out, axis=-1)

    def preprocess_inp(batch_size, node_inp, gcn_out, gsn_out, dag_summ_backward):

        outdim = args.output_dim
        node_inp_dim = args.node_input_dim # node_inp.shape.as_list()[-1]

        # reshape node inputs to batch format
        node_inp_r = tf.reshape(node_inp, [batch_size, -1, node_inp_dim])  # [1, 110, 8]
        # reshape gcn outputs to batch format
        gcn_out_r = tf.reshape(gcn_out, [batch_size, -1, outdim])  # [1, 110, 8]
        # reshape gsn dag summary to batch format
        gsn_dag_summ, gsn_glob_summ = gsn_out
        gsn_dag_summ_r = tf.reshape(gsn_dag_summ, [batch_size, -1, outdim])  # [1, 11, 8]
        dag_summ_backward_e = tf.tile(tf.expand_dims(dag_summ_backward, axis=0), [batch_size, 1, 1])  # [1, 110, 12] -> [1, 110, 12]
        gsn_dag_summ_extend = tf.matmul(dag_summ_backward_e, gsn_dag_summ_r)

        gsn_glob_summ = tf.reshape(gsn_glob_summ, [batch_size, -1, outdim])
        gsn_glob_summ = tf.tile(gsn_glob_summ, [1, tf.shape(gsn_dag_summ_extend)[1], 1])

        merge_node = tf.concat([node_inp_r, gcn_out_r, gsn_dag_summ_extend, gsn_glob_summ], axis=2)

        merge_node = tf.compat.v1.layers.batch_normalization(merge_node, training=True)
        return merge_node

    def build_deepsocs(o, a1, dag_summ_backward, hdims_list, valid_masks):

        feat_hdims, task_hdims, pe_hdims = hdims_list
        node_valid_mask = valid_masks

        with tf.compat.v1.variable_scope('embed'):
            gcn_out = gcn_feature(o, o, gnn_hdims, actv=tf.nn.leaky_relu)
            gsn_out = gsn_feature(o, gcn_out, o, gnn_hdims, actv=tf.nn.leaky_relu)
            feature = preprocess_inp(batch, o, gcn_out, gsn_out, dag_summ_backward)

        with tf.compat.v1.variable_scope('actor'):
            t_pi = mlp_task_policy(
                batch, feature, a1, node_valid_mask, hdims=task_hdims,
                actv=tf.nn.leaky_relu, output_actv=None)

            logits = tf.math.log(t_pi)
            noise = tf.random.uniform(tf.shape(logits))
            _a1 = tf.argmax(logits - tf.math.log(-tf.math.log(noise)), 1)

        return _a1, t_pi, gcn_out, gsn_out

    # Have own session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # Placeholders of deepsocs
    max_depth, summ_levels = args.max_depth, args.summ_levels
    adj_mats = [tf.compat.v1.sparse_placeholder(tf.float32, [None, None], name='adj_mats')
                for _ in range(max_depth)]
    masks = [tf.compat.v1.placeholder(tf.float32, [None, 1], name='masks'+str(i)) for i in range(max_depth)]
    summ_mats = [tf.compat.v1.sparse_placeholder(tf.float32, [None, None], name='summ_mats'+str(i)) for i in range(summ_levels)]
    dag_summ_backward = tf.compat.v1.placeholder(tf.float32, [None, None], name='dag_summ_backward')
    batch = tf.compat.v1.placeholder(tf.int32, (), name='batch')
    node_valid_mask = tf.compat.v1.placeholder(tf.float32, [None, None], name='node_valid_mask')

    # Placeholders
    odim = args.node_input_dim  # env.observation_space.shape[0]
    adim = 1  # env.action_space.shape[0]
    o_ph = tf.compat.v1.placeholder(tf.float32, [None, odim], name='o_ph')
    a1_ph = tf.compat.v1.placeholder(tf.float32, [None, None, adim], name='a1_ph')
    adv_ph = tf.compat.v1.placeholder(tf.float32, [None, 1], name='adv_ph')

    # use entropy to promote exploration, this term decays over time
    entropy_weight = tf.compat.v1.placeholder(tf.float32, ())

    # Actor-critic model
    ac_kwargs = dict()
    ac_kwargs['hdims_list'] = (feat_hdims, task_hdims, pe_hdims)
    ac_kwargs['valid_masks'] = (node_valid_mask)

    with tf.compat.v1.variable_scope('main'):
        a1, t_pi, gcn_out, gsn_out = \
            build_deepsocs(o_ph, a1_ph, dag_summ_backward, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [o_ph, a1_ph, adv_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [a1, t_pi]

    # Get variables
    embed_vars, t_pi_vars, all_vars = \
        get_vars('main/embed'), get_vars('main/actor/t_pi'), get_vars('main')

    params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='main')

    # Count variables
    var_counts = tuple(count_vars(scope) for scope in [
        'main/embed', 'main/actor/t_pi', 'main/actor/pe_pi', 'main'])
    print('\nNumber of parameters: \t embedding: %d, task pi: %d, pe pi: %d, \t total: %d\n' % var_counts)

    # Accumulate model
    model = {
         'a1': a1, 'o_ph': o_ph, 'a1_ph': a1_ph,
         'adv_ph': adv_ph, 'entropy_weight': entropy_weight, 't_pi': t_pi,
         'all_phs': all_phs, 'get_t_action_ops': get_action_ops,
         'embed_vars': embed_vars, 't_pi_vars': t_pi_vars, 'all_vars': all_vars, 'params': params,
         'adj_mats': adj_mats, 'masks': masks, 'summ_mats': summ_mats, 'dag_summ_backward': dag_summ_backward,
         'batch': batch, 'node_valid_mask': node_valid_mask,
         'gcn_out': gcn_out, 'gsn_out': gsn_out
    }
    return model, sess


def create_deepsocs_graph(args, model):
    """
    Create DeepSoCS Graph
    """
    eps = tf.constant(1e-6, tf.float32)

    # actor loss due to advantage (negated)
    task_adv_loss = tf.reduce_sum(
        tf.multiply(tf.math.log(model['t_pi'] + eps), model['adv_ph']))
    adv_loss = task_adv_loss

    # entropy loss
    task_entropy = tf.reduce_mean(tf.multiply(model['t_pi'], tf.math.log(model['t_pi'] + eps)))
    entropy_loss = task_entropy
    # normalize entropy
    entropy_loss /= \
        (tf.math.log(tf.cast(tf.shape(model['t_pi'])[1], tf.float32)) + \
         tf.math.log(float(3)))  # 3 -> len(model.pe_levels)

    # define combined loss
    pi_loss = adv_loss + model['entropy_weight'] * entropy_loss

    # gradients
    gradients = tf.gradients(pi_loss, model['params'])

    # optimizer
    apply_grads = tf.compat.v1.train.AdamOptimizer(args.lr).apply_gradients(zip(gradients, model['params']))

    # Accumulate graph
    graph = {
        'gradients': gradients, 'pi_loss': pi_loss,
        'adv_loss': adv_loss, 'entropy_loss': entropy_loss,
        'apply_grads': apply_grads
    }

    return graph
