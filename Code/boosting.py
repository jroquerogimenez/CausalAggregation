import numpy as np
import scipy as sp
from tqdm import tqdm
import random


class Boosting(object):

    def __init__(self,
                 regression_method,
                 gap_convergence = 0.1,
                 max_n_iter = 100,
                 params_method = None,
                 ):

        self.regression_method = regression_method
        self.gap_convergence = gap_convergence
        self.max_n_iter = max_n_iter
        self.params_method = params_method


    def fit(self, collect_env):

        self.collect_env = collect_env
        self.list_env_keys = list(self.collect_env.env.keys())
        self.true_function_y = self.collect_env.true_function_y

        self.output_function_dict = {}
        self.output_function_merged = self.merge_output_function_dict()
        self.output_model_variation = {env_key: [] for env_key in self.list_env_keys}

        self.x_train_full = self.collect_env.stack_data_env()

        n_iter, improving, reconstruction_loss = 0, True, []
        pbar = tqdm(total=self.max_n_iter, leave=True)
        while improving and (n_iter<self.max_n_iter):
            n_iter += 1
            pbar.update(1)

            shuffled_list_env_keys = self.list_env_keys.copy()
            random.shuffle(shuffled_list_env_keys)
            for env_key in shuffled_list_env_keys:
                env = self.collect_env.env[env_key]
                randomized_covariates = env.data['X'][env.data['randomized_index'],:] 
                residual = env.data['Y'] - self.output_function_merged(env.data['X'])

                updated_model = self.regression_method(self.params_method, env_key, self.collect_env)
                updated_model.fit(randomized_covariates.T, residual)

                if env_key in self.output_function_dict:

                    self.output_model_variation[env_key].append(np.mean(np.square(
                        self.output_function_dict.get(env_key, lambda u: np.zeros(u.shape[0]))(self.x_train_full[env.data['randomized_index']].T) - 
                        updated_model.predict(self.x_train_full[env.data['randomized_index']].T)
                        ))
                    )
                
                def f(updated_model_tmp, output_function_dict_tmp): 
                    return lambda u: updated_model_tmp.predict(u) + output_function_dict_tmp(u)
                updated_function = f(updated_model, self.output_function_dict.get(env_key, lambda u: np.zeros(u.shape[0])))
                self.output_function_dict.update({env_key: updated_function})
                self.output_function_merged = self.merge_output_function_dict()

            reconstruction_loss.append(self.evaluate_gap())

        pbar.close()


    def merge_output_function_dict(self):

        def merged_function(x):

            output = np.zeros(np.array(x).shape[1]) 

            for env_key in self.output_function_dict.keys():
                env = self.collect_env.env[env_key]
                randomized_covariates = np.array(x)[env.data['randomized_index'], :]
                output += self.output_function_dict[env_key](randomized_covariates.T)

            return output

        return merged_function


    def evaluate_gap(self):
        x = self.x_train_full
        return np.mean(np.square(self.true_function_y(x) - self.output_function_merged(x)))
