import numpy as np
import cvxpy as cp
import random


class CausalBoosting(object):

    def __init__(self,
                 regression_method,
                 gap_convergence = 1e-3,
                 max_n_iter = 50,
                 training_fraction = 0.8,
                 learning_rate=0.5,
                 penalty_weight=1.,
                 ):

        self.regression_method = regression_method
        self.gap_convergence = gap_convergence
        self.max_n_iter = max_n_iter
        self.training_fraction = training_fraction 
        self.learning_rate=learning_rate
        self.penalty_weight=penalty_weight


    def fit(self, collect_env, params_method={}):

        self.collect_env = collect_env
        self.list_env_keys = list(self.collect_env.env.keys())

        # Store temp variables in the following dicts with keys env_key.
        self.output_function_dict = {}
        self.output_function_merged = self._merge_output_function_dict()

        # Start iterative procedure.
        n_iter, gap = 0, 1e10
        shuffled_list_env_keys = self.list_env_keys.copy()
        while (n_iter<self.max_n_iter) and (gap>self.gap_convergence):
            n_iter += 1
            random.shuffle(shuffled_list_env_keys)
            new_output_function_dict = {}
            for env_key in shuffled_list_env_keys:

                # Loop over the environments. Identify the randomized covariates in each environment and
                # generate the residual: either full output_function_merged if boosting or partial if backfitting.
                env = self.collect_env.env[env_key]
                randomized_covariates = env['X'][env['randomized_index'],:int(env['X'].shape[1]*self.training_fraction)] 
                residual = env['Y'][:int(len(env['Y'])*self.training_fraction)] - self.output_function_merged(env['X'][:,:int(env['X'].shape[1]*self.training_fraction)])

                # Fit a new model to the residual/randomized_covariates
                updated_model = self.regression_method(params_method, self.collect_env, env_key)
                updated_model.fit(randomized_covariates.T, residual)

                new_output_function_dict.update({env_key: updated_model.predict})

            # Reweighting all candidates for optimal prediciton.
            gap = self._update_function_dict(new_output_function_dict)


    def _update_function_dict(self, new_output_function_dict):

        alphas = cp.Variable(len(self.list_env_keys))
        objective = 0 

        def f_reweighted(weight, candidate_function, previous_function):
            return lambda u: self.learning_rate*weight*candidate_function(u) + previous_function(u)

        # Loop over the environments to define the weighted quadratic loss to minimize in order to get the weights.
        for env_key in self.list_env_keys: 
 
            # For each env, get the predictor for each env function, and weight it by alphas.
            # The residual depends on the update_method.
            env = self.collect_env.env[env_key]
            residual = env['Y'][:int(len(env['Y'])*self.training_fraction)] - self.output_function_merged(env['X'][:,:int(env['X'].shape[1]*self.training_fraction)])
 
            predictor = np.zeros((len(residual), 0))
            normalized_predictor = np.zeros((len(residual), 0))
            for env_key_2 in self.list_env_keys:
                # We use the same data from the env_key env. But we take the randomized indices to be those of env_2.
                env_2  = self.collect_env.env[env_key_2]
                randomized_covariates_2 = env['X'][env_2['randomized_index'],:int(env['X'].shape[1]*self.training_fraction)]
                env_predictor = new_output_function_dict[env_key_2](randomized_covariates_2.T).reshape(-1,1)
                if np.amin(np.std(env_predictor, axis=1).reshape(-1,1))>0:
                    normalized_env_predictor = env_predictor/np.std(env_predictor, axis=1).reshape(-1,1)
                else:
                    normalized_env_predictor = env_predictor
                predictor = np.hstack([predictor, env_predictor])
                normalized_predictor = np.hstack([normalized_predictor, normalized_env_predictor])
 
            objective = objective + cp.abs((residual - predictor @ alphas) @normalized_predictor.T[self.list_env_keys.index(env_key)])
        
        objective = cp.max(objective) + self.penalty_weight*cp.norm(alphas, 2)
 
        # Once loss is evaluated over all envs, solve cvx problem.
        problem = cp.Problem(cp.Minimize(objective), [])
        problem.solve()
        solution = problem.variables()[0].value

        # Evaluate variation due to the updated model.
        x_train_full = self.collect_env.stack_data_X_env(train=True, val_fraction=(1-self.training_fraction))
        previous_response = self.output_function_merged(x_train_full)

        # Once the weights are determined, update the output_function_dict for all envs.
        self.output_function_dict.update({env_key: f_reweighted(
            solution[self.list_env_keys.index(env_key)],
            new_output_function_dict[env_key],
            self.output_function_dict.get(env_key, lambda u: np.zeros(u.shape[0])),
            )
            for env_key in self.list_env_keys
            })

        # And update the final merged function.
        self.output_function_merged = self._merge_output_function_dict()

        # Evaluate the new gap:
        return np.amax(np.abs(previous_response - self.output_function_merged(x_train_full)))


    def _merge_output_function_dict(self):

        def merged_function(x):
            output = np.zeros(np.array(x).shape[1]) 
            for env_key in self.output_function_dict.keys():
                env = self.collect_env.env[env_key]
                randomized_covariates = np.array(x)[env['randomized_index'], :]
                output += self.output_function_dict[env_key](randomized_covariates.T)
            return output

        return merged_function


    def predict(self, x):
        return self.output_function_merged(x)

    # This is the only part where we make use of the true structural equation.
    # Only methods with 'oracle' name use this oracle knowledge.
    def oracle(self, x):
        return self.collect_env.true_function_y(x)


    def validation_loss(self):
        x_val_full = self.collect_env.stack_data_X_env(train=False, val_fraction=(1-self.training_fraction))
        y_val_full = self.collect_env.stack_data_Y_env(train=False, val_fraction=(1-self.training_fraction))
        return np.mean(np.square(y_val_full - self.predict(x_val_full)))


    def oracle_loss(self, x=None):
        if x is None:
            x = self.collect_env.stack_data_X_env(train=False, val_fraction=(1-self.training_fraction))
        return np.mean(np.square(self.oracle(x) - self.predict(x)))



# Add section with naive methods.

class CausalNaive(object):

    def __init__(self,
                 regression_method,
                 training_fraction = 0.8,
                 ):

        self.regression_method = regression_method
        self.training_fraction = training_fraction 


    def fit(self, collect_env, params_method={}):

        self.collect_env = collect_env

        x_train_full = self.collect_env.stack_data_X_env(train=True, val_fraction=(1-self.training_fraction))
        y_train_full = self.collect_env.stack_data_Y_env(train=True, val_fraction=(1-self.training_fraction))

        self.naive_model = self.regression_method(params_method, self.collect_env, None)
        self.naive_model.fit(x_train_full.T, y_train_full)


    def predict(self, x):
        return self.naive_model.predict(x.T)


    def validation_loss(self):
        x_val_full = self.collect_env.stack_data_X_env(train=False, val_fraction=(1-self.training_fraction))
        y_val_full = self.collect_env.stack_data_Y_env(train=False, val_fraction=(1-self.training_fraction))
        return np.mean(np.square(y_val_full - self.predict(x_val_full)))

    # This is the only part where we make use of the true structural equation.
    # Only methods with 'oracle' name use this oracle knowledge.
    def oracle(self, x):
        return self.collect_env.true_function_y(x)

    def oracle_loss(self, x=None):
        if x is None:
            x = self.collect_env.stack_data_X_env(train=False, val_fraction=(1-self.training_fraction))
        return np.mean(np.square(self.oracle(x) - self.predict(x)))