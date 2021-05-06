import numpy as np
import scipy as sp
import cvxpy as cp
from tqdm import tqdm
import matplotlib.pyplot as plt
import random, time


class Backfitting(object):

    def __init__(self,
                 regression_method,
                 update_method,
                 gap_convergence = 1e-3,
                 max_n_iter = 50,
                 warm_start = True,
                 params_method = None,
                 true_y_coeff = None,
                 reweighting_candidates = False,
                 training_fraction = 0.8,
                 update_within_loop = False,
                 ):

        self.regression_method = regression_method
        self.update_method = update_method
        self.gap_convergence = gap_convergence
        self.max_n_iter = max_n_iter
        self.params_method = params_method
        self.warm_start = warm_start
        self.reweighting_candidates = reweighting_candidates
        self.training_fraction = training_fraction 
        self.update_within_loop = update_within_loop

        self.true_y_coeff = true_y_coeff



    def fitCV(self, collect_env, range_min=-3, range_max=0.5, range_step=0.5, print_selected=False):

        validation_list = []
        
        hyperparam_range = np.exp(np.log(10)*np.arange(range_min, range_max, range_step))

        for hyperparam in hyperparam_range:
            self.params_method['min_impurity_decrease'] = hyperparam
            validation_loss, oracle_loss = self.fit(collect_env)
            validation_list.append(validation_loss)
        
        if print_selected:
            print(np.argmin(validation_list), hyperparam_range[np.argmin(validation_list)])
        self.params_method['min_impurity_decrease'] = hyperparam_range[np.argmin(validation_list)]

        return self.fit(collect_env)



    def fit(self, collect_env):

        # Fit the backfitting/boosting procedure to the set of environments.

        self.collect_env = collect_env
        self.list_env_keys = list(self.collect_env.env.keys())
        self.true_function_y = self.collect_env.true_function_y

        # Store temp variables in the following dicts with keys env_key.
        self.output_model_variation = {env_key: [] for env_key in self.list_env_keys}
        self.output_model_dict = {} # Only used for backfitting. Stores environment-wise models.
        self.output_function_dict = {}

        self.x_train_full = self.collect_env.stack_data_X_env(train=True, val_fraction=(1-self.training_fraction))
        self.y_train_full = self.collect_env.stack_data_Y_env(train=True, val_fraction=(1-self.training_fraction))
        self.x_val_full = self.collect_env.stack_data_X_env(train=False, val_fraction=(1-self.training_fraction))
        self.y_val_full = self.collect_env.stack_data_Y_env(train=False, val_fraction=(1-self.training_fraction))
        self.y_val_true = self.true_function_y(self.x_val_full)

        # Initialize the models.
        if self.warm_start:
            all_covariates = np.concatenate([self.params_method['selected_features'][env_key] for env_key in self.list_env_keys])
            max_degree = np.amax([self.params_method['power_features'][env_key] for env_key in self.list_env_keys]) 
            initial_model = self.regression_method({
                'regression_method':self.params_method['regression_method'],
                'power_features':max_degree,
                'selected_features':all_covariates,
                },
                self.collect_env
                )
            initial_model.fit(self.x_train_full.T, self.y_train_full)
 
            # Need to create individual functions in self.output_function_dict.
            for env_key in self.list_env_keys:
                env = self.collect_env.env[env_key]
                randomized_covariates = env.data['X'][env.data['randomized_index'],:int(env.data['X'].shape[1]*self.training_fraction)] 
 
                initial_env_model = self.regression_method(self.params_method, self.collect_env, env_key)
                initial_env_model.fit(randomized_covariates.T, env.data['Y'][:int(len(env.data['Y'])*self.training_fraction)])
                initial_env_model.lr.coef_ = np.array([initial_model.coef_()[initial_model.get_feature_names().index(coef_env_name)] for coef_env_name in initial_env_model.get_feature_names()])
 
                self.output_function_dict.update({env_key: initial_env_model.predict})

        # Create first merged function after initialization.
        self.output_function_merged = self.merge_output_function_dict()

        # Start iterative procedure.
        n_iter, validation_loss, oracle_loss = 0, [], []
        shuffled_list_env_keys = self.list_env_keys.copy()
        while (n_iter<self.max_n_iter):
            n_iter += 1

            random.shuffle(shuffled_list_env_keys)
            new_output_function_dict = {}
            for env_key in shuffled_list_env_keys:

                # Loop over the environments. Identify the randomized covariates in each environment and
                # generate the residual: either full output_function_merge if boosting or partial if backfitting.
                env = self.collect_env.env[env_key]
                randomized_covariates = env.data['X'][env.data['randomized_index'],:int(env.data['X'].shape[1]*self.training_fraction)] 
                if self.update_method=='boosting':
                    residual = env.data['Y'][:int(len(env.data['Y'])*self.training_fraction)] - self.output_function_merged(env.data['X'][:,:int(env.data['X'].shape[1]*self.training_fraction)])
                elif self.update_method=='backfitting':
                    residual = env.data['Y'][:int(len(env.data['Y'])*self.training_fraction)] - self.output_function_partial(env.data['X'][:,:int(env.data['X'].shape[1]*self.training_fraction)], env_key)

                # Fit a new model to the residual/randomized_covariates
                updated_model = self.regression_method(self.params_method, self.collect_env, env_key)
                updated_model.fit(randomized_covariates.T, residual)

                # The new function is either the fitted new model or the addition of the fitted new model to the previous one.
                new_output_function_dict.update({env_key: updated_model.predict})

                self.output_model_dict.update({env_key: updated_model})


                if self.update_within_loop:
                    def f_reweighted_boosting(weight, candidate_function, previous_function):
                        return lambda u: weight*candidate_function(u) + previous_function(u)
                    def f_reweighted_backfitting(weight, candidate_function):
                        return lambda u: weight*candidate_function(u)
                    
                    if self.update_method=='boosting':
                        updated_function = f_reweighted_boosting(
                            1,
                            new_output_function_dict[env_key],
                            self.output_function_dict.get(env_key, lambda u: np.zeros(u.shape[0])),
                            )
                    elif self.update_method=='backfitting':
                        updated_function = f_reweighted_backfitting(
                            1,
                            new_output_function_dict[env_key],
                            )
                    self.output_function_dict.update({env_key: updated_function})
                    self.output_function_merged = self.merge_output_function_dict()

            # Reweighting all candidates for optimal prediciton.
            if not self.update_within_loop:
                self.update_function_dict(new_output_function_dict)
                self.output_function_merged = self.merge_output_function_dict()

            oracle_loss.append(self.evaluate_oracle())
            validation_loss.append(self.evaluate_validation())

        return validation_loss[-1], oracle_loss[-1]


    def update_function_dict(self, new_output_function_dict):

        alphas = cp.Variable(len(self.list_env_keys))
        objective = 0 

        if self.reweighting_candidates:
            # Loop over the environments to define the weighted quadratic loss to minimize in order to get the weights.
            for env_key in self.list_env_keys: 
 
                # For each env, get the predictor for each env function, and weight it by alphas.
                # The residual depends on the update_method.
                env = self.collect_env.env[env_key]
                if self.update_method=='boosting':
                    residual = env.data['Y'][:int(len(env.data['Y'])*self.training_fraction)] - self.output_function_merged(env.data['X'][:,:int(env.data['X'].shape[1]*self.training_fraction)])
                elif self.update_method=='backfitting':
                    residual = env.data['Y'][:int(len(env.data['Y'])*self.training_fraction)] - self.output_function_partial(env.data['X'][:,:int(env.data['X'].shape[1]*self.training_fraction)], env_key)
 
                predictor = np.zeros((len(residual), 0))
                normalized_predictor = np.zeros((len(residual), 0))
                for env_key_2 in self.list_env_keys:
                    # We use the same data from the env_key env. But we take the randomized indices to be those of env_2.
                    env_2  = self.collect_env.env[env_key_2]
                    randomized_covariates_2 = env.data['X'][env_2.data['randomized_index'],:int(env.data['X'].shape[1]*self.training_fraction)]
                    env_predictor = new_output_function_dict[env_key_2](randomized_covariates_2.T).reshape(-1,1)
                    if np.amin(np.std(env_predictor, axis=1).reshape(-1,1))>0:
                        normalized_env_predictor = env_predictor/np.std(env_predictor, axis=1).reshape(-1,1)
                    else:
                        normalized_env_predictor = env_predictor
                    predictor = np.hstack([predictor, env_predictor])
                    normalized_predictor = np.hstack([normalized_predictor, normalized_env_predictor])
 
                objective = objective + cp.abs((residual - predictor @ alphas) @normalized_predictor.T[self.list_env_keys.index(env_key)])
            
            objective = cp.max(objective) + cp.norm(alphas, 1)
 
            # Once loss is evaluated over all envs, solve cvx problem.
            problem = cp.Problem(cp.Minimize(objective), [])
            problem.solve()
 
            solution = problem.variables()[0].value
        else:
            solution = np.ones(len(self.list_env_keys))

        # Once the weights are determined, update the output_function_dict for all envs.
        for env_key in self.list_env_keys:
            def f_reweighted_boosting(weight, candidate_function, previous_function):
                return lambda u: weight*candidate_function(u) + previous_function(u)
            def f_reweighted_backfitting(weight, candidate_function):
                return lambda u: weight*candidate_function(u)

            if self.update_method=='boosting':
                updated_function = f_reweighted_boosting(
                    solution[self.list_env_keys.index(env_key)],
                    new_output_function_dict[env_key],
                    self.output_function_dict.get(env_key, lambda u: np.zeros(u.shape[0])),
                    )
            elif self.update_method=='backfitting':
                updated_function = f_reweighted_backfitting(
                    solution[self.list_env_keys.index(env_key)],
                    new_output_function_dict[env_key],
                    )
            self.output_function_dict.update({env_key: updated_function})



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


    def evaluate_oracle(self, x=None):
        if x is None:
            x = self.x_val_full
        return np.mean(np.square(self.true_function_y(x) - self.output_function_merged(x)))


    def evaluate_validation(self):
        return np.mean(np.square(self.y_val_full - self.output_function_merged(self.x_val_full)))


    def evaluate_l1_loss_parameter(self):
        dict_coeffs = {}
        for model in self.output_model_dict.values():
            dict_coeffs.update({name : model.coef_()[i] for (i, name) in enumerate(model.get_feature_names())})
        dict_diff_coeffs = {key: dict_coeffs[key] - self.true_y_coeff.get(key, 0) for key in dict_coeffs.keys()}
        l1_loss = np.sum([np.abs(diff_coef) for diff_coef in dict_diff_coeffs.values()])
        return l1_loss


    def return_results(self, x=None):
        if x is None:
            x = self.x_val_full
        return  np.vstack([self.x_val_full, self.true_function_y(x), self.y_val_full, self.output_function_merged(x)]).T


    def fit_naive(self, collect_env):

        self.collect_env = collect_env
        self.list_env_keys = list(self.collect_env.env.keys())
        self.true_function_y = self.collect_env.true_function_y

        self.x_train_full = self.collect_env.stack_data_X_env(train=True, val_fraction=(1-self.training_fraction))
        self.y_train_full = self.collect_env.stack_data_Y_env(train=True, val_fraction=(1-self.training_fraction))
        self.x_val_full = self.collect_env.stack_data_X_env(train=False, val_fraction=(1-self.training_fraction))
        self.y_val_full = self.collect_env.stack_data_Y_env(train=False, val_fraction=(1-self.training_fraction))

        self.naive_model = self.regression_method(self.params_method, self.collect_env, None)
        self.naive_model.fit(self.x_train_full.T, self.y_train_full)

        oracle_loss = np.mean(np.square(self.true_function_y(self.x_val_full) - self.naive_model.predict(self.x_val_full.T)))
        validation_loss = np.mean(np.square(self.y_val_full - self.naive_model.predict(self.x_val_full.T))) 

        return validation_loss, oracle_loss


    def fit_naiveCV(self, collect_env, range_min=-3, range_max=0.5, range_step=0.5):

        validation_list = []
        
        hyperparam_range = np.exp(np.log(10)*np.arange(range_min, range_max, range_step))

        for hyperparam in hyperparam_range:
            self.params_method['min_impurity_decrease'] = hyperparam
            validation_loss, oracle_loss = self.fit_naive(collect_env)
            validation_list.append(validation_loss)
        
        self.params_method['min_impurity_decrease'] = hyperparam_range[np.argmin(validation_list)]

        return self.fit_naive(collect_env)