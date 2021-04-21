import numpy as np
import scipy as sp
from tqdm import tqdm
import random


class Backfitting(object):

    def __init__(self,
                 regression_method,
                 update_method,
                 gap_convergence = 1e-3,
                 max_n_iter = 50,
                 params_method = None,
                 ):

        self.regression_method = regression_method
        self.update_method = update_method
        self.gap_convergence = gap_convergence
        self.max_n_iter = max_n_iter
        self.params_method = params_method


    def fit(self, collect_env):

        # Fit the backfitting/boosting procedure to the set of environments.

        self.collect_env = collect_env
        self.list_env_keys = list(self.collect_env.env.keys())
        self.true_function_y = self.collect_env.true_function_y

        # Store temp variables in the following dicts with keys env_key.
        self.output_model_variation = {env_key: [] for env_key in self.list_env_keys}
        self.output_model_dict = {} # Only used for backfitting. Stores environment-wise models.
        self.output_function_dict = {}
        self.output_function_merged = self.merge_output_function_dict()

        self.x_train_full = self.collect_env.stack_data_env()

        n_iter, improving, reconstruction_loss = 0, True, []
        pbar = tqdm(total=self.max_n_iter, leave=True)
        while improving and (n_iter<self.max_n_iter):
            n_iter += 1
            pbar.update(1)

            shuffled_list_env_keys = self.list_env_keys.copy()
            random.shuffle(shuffled_list_env_keys)
            variation_coefs = 0
            for env_key in shuffled_list_env_keys:
                # Loop over the environments. Identify the randomized covariates in each environment and
                # generate the residual: either full output_function_merge if boosting or partial if backfitting.
                env = self.collect_env.env[env_key]
                randomized_covariates = env.data['X'][env.data['randomized_index'],:] 
                if self.update_method=='boosting':
                    residual = env.data['Y'] - self.output_function_merged(env.data['X'])
                elif self.update_method=='backfitting':
                    residual = env.data['Y'] - self.output_function_partial(env.data['X'], env_key)

                # Fit a new model to the residual/randomized_covariates
                updated_model = self.regression_method(self.params_method, env_key, self.collect_env)
                updated_model.fit(randomized_covariates.T, residual)

                # We track changes in the backfitted function by comparing the Linf loss of coefficients.
                if self.update_method=='boosting':
                    variation_coefs += np.amax(np.abs(updated_model.coef_()))
                elif self.update_method=='backfitting' and (env_key in self.output_model_dict.keys()):
                    variation_coefs += np.amax(np.abs(updated_model.coef_() - self.output_model_dict[env_key].coef_()))
                elif self.update_method=='backfitting' and not (env_key in self.output_model_dict.keys()):
                    variation_coefs += np.amax(np.abs(updated_model.coef_()))

                # We estimate the L2 difference between previous and updated environment-wise function. 
                if env_key in self.output_function_dict:
                    if self.update_method=='backfitting':
                        self.output_model_variation[env_key].append(np.mean(np.square(
                            self.output_function_dict[env_key](self.x_train_full[env.data['randomized_index']].T) - 
                            updated_model.predict(self.x_train_full[env.data['randomized_index']].T)
                            ))
                        )
                    if self.update_method=='boosting':
                        self.output_model_variation[env_key].append(np.mean(np.square(
                            updated_model.predict(self.x_train_full[env.data['randomized_index']].T)
                            ))
                        )

                # The new function is either the fitted new model or the addition of the fitted new model to the previous one.
                def f_boosting(updated_model_tmp, output_function_dict_tmp): 
                    return lambda u: updated_model_tmp.predict(u) + output_function_dict_tmp(u)

                def f_backfitting(updated_model_tmp): 
                    return lambda u: updated_model_tmp.predict(u)

                if self.update_method=='boosting':
                    updated_function = f_boosting(updated_model, self.output_function_dict.get(env_key, lambda u: np.zeros(u.shape[0])))
                elif self.update_method=='backfitting':
                    updated_function = updated_model.predict 
                    self.output_model_dict[env_key] = updated_model

                self.output_function_dict.update({env_key: updated_function})
                self.output_function_merged = self.merge_output_function_dict()

            reconstruction_loss.append(self.evaluate_gap())
            improving = (variation_coefs/len(shuffled_list_env_keys)>self.gap_convergence) or n_iter<3

        pbar.close()


    def output_function_partial(self, covariates, excluded_env_key):

        output_values_dict = {}
        list_env_keys_wo_excluded = self.list_env_keys.copy()
        list_env_keys_wo_excluded.remove(excluded_env_key)

        for env_key in list_env_keys_wo_excluded:
            env = self.collect_env.env[env_key]
            randomized_covariates = covariates[env.data['randomized_index'],:] 

            output_values_dict[env_key] = self.output_function_dict.get(env_key, lambda u: np.zeros(u.shape[0]))(randomized_covariates.T)
            assert len(output_values_dict[env_key]) == randomized_covariates.shape[1]

        output_values = np.sum(
            np.stack([output_values_dict[env_key] for env_key in output_values_dict.keys()]),
            axis=0
        )

        return output_values
            

    def merge_output_function_dict(self):

        def merged_function(x):

            output = np.zeros(np.array(x).shape[1]) 

            for env_key in self.output_function_dict.keys():
                env = self.collect_env.env[env_key]
                randomized_covariates = np.array(x)[env.data['randomized_index'], :]
                output += self.output_function_dict[env_key](randomized_covariates.T)

            return output

        return merged_function


    def evaluate_gap(self, x=None):
        if x is None:
            x = self.x_train_full
        return np.mean(np.square(self.true_function_y(x) - self.output_function_merged(x)))