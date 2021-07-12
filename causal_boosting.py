import numpy as np
import scipy as sp
import cvxpy as cp


class CausalBoosting(object):

    def __init__(self,
                 regression_method,
                 max_n_iter = 50,
                 params_method = None,
                 training_fraction = 0.8,
                 ):

        self.regression_method = regression_method
        self.max_n_iter = max_n_iter
        self.params_method = params_method
        self.training_fraction = training_fraction 

        self.learning_rate=0.5


    def fitCV(self, collect_env, range_min=-3, range_max=0.5, range_step=0.5):

        validation_list = []
        hyperparam_range = np.exp(np.log(10)*np.arange(range_min, range_max, range_step))

        for hyperparam in hyperparam_range:
            self.params_method['min_impurity_decrease'] = hyperparam
            self.fit(collect_env)
            validation_list.append(self.evaluate_validation())
        
        self.params_method['min_impurity_decrease'] = hyperparam_range[np.argmin(validation_list)]
        self.fit(collect_env)



    def fit(self, collect_env):

        # Fit the boosting procedure to the set of environments.
        self.collect_env = collect_env
        self.list_env_keys = list(self.collect_env.env.keys())
        self.true_function_y = self.collect_env.true_function_y

        # Store temp variables in the following dicts with keys env_key.
        self.output_function_dict = {}
        self.output_function_merged = self.merge_output_function_dict()

        # Start iterative procedure.
        for _ in np.arange(self.max_n_iter):

            new_output_function_dict = {}
            for env_key in self.list_env_keys:

                # Loop over the environments. Identify the randomized covariates in each environment and
                # generate the residual: either full output_function_merge if boosting or partial if backfitting.
                env = self.collect_env.env[env_key]
                randomized_covariates = env.data['X'][env.data['randomized_index'],:int(env.data['X'].shape[1]*self.training_fraction)] 
                residual = (env.data['Y'][:int(len(env.data['Y'])*self.training_fraction)] - 
                            self.output_function_merged(env.data['X'][:,:int(env.data['X'].shape[1]*self.training_fraction)])
                )

                # Fit a new model to the residual/randomized_covariates
                updated_model = self.regression_method(self.params_method, self.collect_env, env_key)
                updated_model.fit(randomized_covariates.T, residual)

                # The new function is either the fitted new model or the addition of the fitted new model to the previous one.
                new_output_function_dict.update({env_key: updated_model.predict})

            # Reweighting all candidates for optimal prediciton.
            solution = self.find_weights(new_output_function_dict)
            # Once the weights are determined, update the output_function_dict for all envs.
            for env_key in self.list_env_keys:
                def f_reweighted_boosting(weight, candidate_function, previous_function):
                    return lambda u: self.learning_rate*weight*candidate_function(u) + previous_function(u)
            
                self.output_function_dict.update({env_key: 
                    f_reweighted_boosting(
                        solution[self.list_env_keys.index(env_key)],
                        new_output_function_dict[env_key],
                        self.output_function_dict.get(env_key, lambda u: np.zeros(u.shape[0])),
                    )
                })
            self.output_function_merged = self.merge_output_function_dict()


    def find_weights(self, new_output_function_dict):

        alphas = cp.Variable(len(self.list_env_keys))
        objective = 0 

        for env_key in self.list_env_keys: 
 
            # For each env, get the predictor for each env function, and weight it by alphas.
            # The residual depends on the update_method.
            env = self.collect_env.env[env_key]

            residual = (env.data['Y'][:int(len(env.data['Y'])*self.training_fraction)] -
                        self.output_function_merged(env.data['X'][:,:int(env.data['X'].shape[1]*self.training_fraction)])
            )
            predictor = np.zeros((len(residual), 0))
            normalized_predictor = np.zeros((len(residual), 0))

            for env_key_2 in self.list_env_keys:
                # We use the same data from the env_key env. But we take the randomized indices to be those of env_2.
                env_2  = self.collect_env.env[env_key_2]
                randomized_covariates_2 = env.data['X'][env_2.data['randomized_index'],:int(env.data['X'].shape[1]*self.training_fraction)]
                env_predictor = new_output_function_dict[env_key_2](randomized_covariates_2.T).reshape(-1,1)

                if np.amin(np.std(env_predictor, axis=1))>0:
                    normalized_env_predictor = env_predictor/np.std(env_predictor, axis=1).reshape(-1,1)
                else:
                    normalized_env_predictor = env_predictor
                predictor = np.hstack([predictor, env_predictor])
                normalized_predictor = np.hstack([normalized_predictor, normalized_env_predictor])
 
            objective = objective + cp.abs((residual - predictor @ alphas) @normalized_predictor.T[self.list_env_keys.index(env_key)])
        
        objective = cp.max(objective) + cp.norm(alphas, 2)
 
        # Once loss is evaluated over all envs, solve cvx problem.
        problem = cp.Problem(cp.Minimize(objective), [])
        #problem.solve(verbose=False, solver='SCS')
        problem.solve(verbose=False, solver='MOSEK')
 
        return problem.variables()[0].value



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
            x = self.collect_env.stack_data_X_env(train=False, val_fraction=(1-self.training_fraction))
        return np.mean(np.square(self.true_function_y(x) - self.output_function_merged(x)))


    def evaluate_validation(self):
        self.x_val_full = self.collect_env.stack_data_X_env(train=False, val_fraction=(1-self.training_fraction))
        self.y_val_full = self.collect_env.stack_data_Y_env(train=False, val_fraction=(1-self.training_fraction))
        return np.mean(np.square(self.y_val_full - self.output_function_merged(self.x_val_full)))

