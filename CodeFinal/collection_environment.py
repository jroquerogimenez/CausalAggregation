import numpy as np
import scipy as sp
import itertools


class CollectionEnvironment:

    def __init__(self, base_env):

        self.base_env = base_env
        self.env = {}
        self.true_function_y = base_env.true_function_y
        

    def add_synthetic_env(self, env_key, dict_interventions, n_samples = None):

        assert not env_key in list(self.env.keys())
        self.env[env_key] = self.base_env.generate_intervention(dict_interventions, n_samples)
    

    def reset_env(self):
        self.env = {}


    def stack_data_X_env(self, train=False, val_fraction=0.1):
        end_samples = {env_key: int(self.env[env_key]['X'].shape[1]*(1-val_fraction)) for env_key in list(self.env.keys())}
        if train:
            return np.hstack([self.env[env_key]['X'][:,:end_samples[env_key]] for env_key in list(self.env.keys())])
        else:
            return np.hstack([self.env[env_key]['X'][:,end_samples[env_key]:] for env_key in list(self.env.keys())])
        

    def stack_data_Y_env(self, train=False, val_fraction=0.1):
        end_samples = {env_key: int(len(self.env[env_key]['Y'])*(1-val_fraction)) for env_key in list(self.env.keys())}
        if train:
            return np.concatenate([self.env[env_key]['Y'][:end_samples[env_key]] for env_key in list(self.env.keys())])
        else:
            return np.concatenate([self.env[env_key]['Y'][end_samples[env_key]:] for env_key in list(self.env.keys())])


