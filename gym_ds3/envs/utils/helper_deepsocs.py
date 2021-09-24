import os

import numpy as np
from scipy import signal

import tensorflow as tf


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))


def placeholders(*args):
    """
    Usage: a_ph,b_ph,c_ph = placeholders(adim,bdim,None)
    """
    return [placeholder(dim) for dim in args]


def get_vars(scope):
    return [x for x in tf.compat.v1.global_variables() if scope in x.name]


def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def decrease_var(var, min_var, decay_rate):
    if var - decay_rate >= min_var:
        var -= decay_rate
    else:
        var = min_var
    return var


def aggregate_gradients(gradients):

    ground_gradients = [np.zeros(g.shape) for g in gradients[0]]
    for gradient in gradients:
        for i in range(len(ground_gradients)):
            ground_gradients[i] += gradient[i]
    return ground_gradients


def suppress_tf_warning():
    import tensorflow as tf
    import os
    import logging
    from tensorflow.python.util import deprecation
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # tf.logging.set_verbosity(tf.logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.getLogger('tensorflow').disabled = True
    deprecation._PRINT_DEPRECATION_WARNINGS = False


def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sums of vectors.
    input: 
        vector x, [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def discount(x, gamma):
    out = np.zeros(x.shape)
    out[-1] = x[-1]
    for i in reversed(range(len(x) -1)):
        out[i] = x[i] + gamma * out[i+1]

    return out


def truncate_experiences(lst):
    batch_pts = [i for i, x in enumerate(lst) if x]
    batch_pts.append(len(lst))

    return batch_pts


def check_obs_change(prev_obs, curr_obs):
    if len(prev_obs) != len(curr_obs):
        return True
    return False


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x
    Args:
        x: An array containing samples of the scalar to produce statistics for.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n
    global_sum_sq = np.sum((x - mean) ** 2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std
    if with_min_and_max:
        global_min = (np.min(x) if len(x) > 0 else np.inf)
        global_max = (np.max(x) if len(x) > 0 else -np.inf)
        return mean, std, global_min, global_max
    return mean, std


## sparse_op
class SparseMat(object):
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        self.row = []
        self.col = []
        self.data = []

    def add(self, row, col, data):
        self.row.append(row)
        self.col.append(col)
        self.data.append(data)

    def get_col(self):
        return np.array(self.col)

    def get_row(self):
        return np.array(self.row)

    def get_data(self):
        return np.array(self.data)
    

def absorb_sp_mats(in_mats, depth):
    """
    Merge multiple sparse matrices to
    a giant one on its diagonal

    e.g.,

    [0, 1, 0]    [0, 1, 0]    [0, 0, 1]
    [1, 0, 0]    [0, 0, 1]    [0, 1, 0]
    [0, 0, 1]    [1, 0, 0]    [0, 1, 0]

    to

    [0, 1, 0]
    [1, 0, 0]   ..  ..    ..  ..
    [0, 0, 1]
              [0, 1, 0]
     ..  ..   [0, 0, 1]   ..  ..
              [1, 0, 0]
                        [0, 0, 1]
     ..  ..    ..  ..   [0, 1, 0]
                        [0, 1, 0]

    where ".." are all zeros

    depth is on the 3rd dimension,
    which is orthogonal to the planar
    operations above

    output SparseTensorValue from tensorflow
    """
    sp_mats = []

    for d in range(depth):
        row_idx = []
        col_idx = []
        data = []
        shape = 0
        base = 0
        for m in in_mats:
            row_idx.append(m[d].get_row() + base)
            col_idx.append(m[d].get_col() + base)
            data.append(m[d].get_data())
            shape += m[d].shape[0]
            base += m[d].shape[0]

        row_idx = np.hstack(row_idx)
        col_idx = np.hstack(col_idx)
        data = np.hstack(data)

        indices = np.mat([row_idx, col_idx]).transpose()
        sp_mats.append(tf.compat.v1.SparseTensorValue(
            indices, data, (shape, shape)))

    return sp_mats


def expand_sp_mat(sp, exp_step):
    """
    Make a stack of same sparse matrix to
    a giant one on its diagonal

    The input is tf.SparseTensorValue

    e.g., expand dimension 3

    [0, 1, 0]    [0, 1, 0]
    [1, 0, 0]    [1, 0, 0]  ..  ..   ..  ..
    [0, 0, 1]    [0, 0, 1]
                          [0, 1, 0]
              to  ..  ..  [1, 0, 0]  ..  ..
                          [0, 0, 1]
                                   [0, 1, 0]
                  ..  ..   ..  ..  [1, 0, 0]
                                   [0, 0, 1]

    where ".." are all zeros

    depth is on the 3rd dimension,
    which is orthogonal to the planar
    operations above

    output SparseTensorValue from tensorflow
    """

    extended_mat = []

    depth = len(sp)

    for d in range(depth):
        row_idx = []
        col_idx = []
        data = []
        shape = 0
        base = 0
        for i in range(exp_step):
            indices = sp[d].indices.transpose()
            row_idx.append(np.squeeze(np.asarray(indices[0, :]) + base))
            col_idx.append(np.squeeze(np.asarray(indices[1, :]) + base))
            data.append(sp[d].values)
            shape += sp[d].dense_shape[0]
            base += sp[d].dense_shape[0]

        row_idx = np.hstack(row_idx)
        col_idx = np.hstack(col_idx)
        data = np.hstack(data)

        indices = np.mat([row_idx, col_idx]).transpose()
        extended_mat.append(tf.compat.v1.SparseTensorValue(
            indices, data, (shape, shape)))

    return extended_mat


def merge_and_extend_sp_mat(sp):
    """
    Transform a stack of sparse matrix into a giant diagonal matrix
    These sparse matrices should have same shape

    e.g.,

    list of
    [1, 0, 1, 1] [0, 0, 0, 1]
    [1, 1, 1, 1] [0, 1, 1, 1]
    [0, 0, 1, 1] [1, 1, 1, 1]

    to

    [1, 0, 1, 1]
    [1, 1, 1, 1]    ..  ..
    [0, 0, 1, 1]
                 [0, 0, 0, 1]
       ..  ..    [0, 1, 1, 1]
                 [1, 1, 1, 1]
    """

    batch_size = len(sp)
    row_idx = []
    col_idx = []
    data = []
    shape = (sp[0].dense_shape[0] * batch_size, sp[0].dense_shape[1] * batch_size)

    row_base = 0
    col_base = 0
    for b in range(batch_size):
        indices = sp[b].indices.transpose()
        row_idx.append(np.squeeze(np.asarray(indices[0, :]) + row_base))
        col_idx.append(np.squeeze(np.asarray(indices[1, :]) + col_base))
        data.append(sp[b].values)
        row_base += sp[b].dense_shape[0]
        col_base += sp[b].dense_shape[1]

    row_idx = np.hstack(row_idx)
    col_idx = np.hstack(col_idx)
    data = np.hstack(data)

    indices = np.mat([row_idx, col_idx]).transpose()
    extended_mat = tf.compat.v1.SparseTensorValue(indices, data, shape)

    return extended_mat
