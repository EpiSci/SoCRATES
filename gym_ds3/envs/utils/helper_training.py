import os
import numpy as np

from numpy.polynomial.polynomial import polyval
from scipy.linalg import hankel

import torch


def state_to_tensor(state):
    return torch.from_numpy(state).float().unsqueeze(0)


def calculate_returns(rewards, gamma):
    return polyval(gamma, hankel(rewards))


def save_model(npz_path, R, VERBOSE=True):
    """
    Save model weights
    """
    # model
    tf_vars = R.model['all_vars'] 
    data2save, var_names, var_vals = dict(), [], []
    for v_idx, tf_var in enumerate(tf_vars):
        var_name, var_val = tf_var.name, R.sess.run(tf_var)
        var_names.append(var_name)
        var_vals.append(var_val)
        data2save[var_name] = var_val
        if VERBOSE:
            print ("[%02d]  var_name:[%s]  var_shape:%s"%
                (v_idx, var_name, var_val.shape)) 
        
    # Create folder if not exist
    dir_name = os.path.dirname(npz_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print("[%s] created." % (dir_name))
        
    # Save npz
    np.savez(npz_path, **data2save)
    print("[%s] saved." % (npz_path))


def restore_model(npz_path, R):
    """
    Restore model weights
    """
    # Load npz
    l = np.load(npz_path)
    print("[%s] loaded." % (npz_path))

    # Get values of model
    tf_vars = R.model['all_vars']
    var_vals = []
    for tf_var in tf_vars:
        var_vals.append(l[tf_var.name])

    # Assign weights of model
    R.set_weights(var_vals)
