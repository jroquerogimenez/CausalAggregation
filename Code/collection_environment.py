import numpy as np
import scipy as sp
import itertools


class CollectionEnvironment:

    def __init__(self, base_env):

        self.base_env = base_env
        self.env = {}
        self.true_function_y = base_env.true_function_y
        
    def add_env(self, env_key, dict_interventions, n_samples = None):

        assert not env_key in list(self.env.keys())
        self.env[env_key] = self.base_env.generate_intervention(dict_interventions)    
        if n_samples is not None:
            self.env[env_key].generate_samples(n_samples)

    def sort_env(self):
        list_randomized_index = [list(self.env[env_key].randomized_index) for env_key in list(self.env.keys())]
        dict_interactions_env = {}
        for randomized_index in list_randomized_index:
            dict_interactions_env.update({key:None for degree in np.arange(1,len(randomized_index)+1) for key in itertools.combinations(randomized_index, degree)})

        for key in dict_interactions_env.keys():
            dict_interactions_env[key] = [env_key for env_key in list(self.env.keys()) if set(key).subset(set(self.env[env_key].randomized_index))]

    def stack_data_env(self):
        return np.hstack([self.env[env_key].data['X'] for env_key in list(self.env.keys())])



