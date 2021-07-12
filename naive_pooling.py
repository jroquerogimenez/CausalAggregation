import numpy as np
import scipy as sp



class NaivePooling(object):

    def __init__(self,
                 regression_method,
                 params_method = None,
                 training_fraction = 0.8,
                 ):

        self.regression_method = regression_method
        self.params_method = params_method
        self.training_fraction = training_fraction 

    def fit(self, collect_env):

        self.collect_env = collect_env
        self.list_env_keys = list(self.collect_env.env.keys())
        self.true_function_y = self.collect_env.true_function_y

        self.x_train_full = self.collect_env.stack_data_X_env(train=True, val_fraction=(1-self.training_fraction))
        self.y_train_full = self.collect_env.stack_data_Y_env(train=True, val_fraction=(1-self.training_fraction))

        self.naive_model = self.regression_method(self.params_method, self.collect_env, None)
        self.naive_model.fit(self.x_train_full.T, self.y_train_full)


    def fitCV(self, collect_env, range_min=-3, range_max=0.5, range_step=0.5):

        validation_list = []
        
        hyperparam_range = np.exp(np.log(10)*np.arange(range_min, range_max, range_step))

        for hyperparam in hyperparam_range:
            self.params_method['min_impurity_decrease'] = hyperparam
            self.fit(collect_env)
            validation_list.append(self.evaluate_validation())
        
        self.params_method['min_impurity_decrease'] = hyperparam_range[np.argmin(validation_list)]
        self.fit(collect_env)


    def evaluate_oracle(self, x=None):
        if x is None:
            x = self.collect_env.stack_data_X_env(train=False, val_fraction=(1-self.training_fraction))
        return np.mean(np.square(self.true_function_y(x) - self.naive_model.predict(x.T)))


    def evaluate_validation(self):
        self.x_val_full = self.collect_env.stack_data_X_env(train=False, val_fraction=(1-self.training_fraction))
        self.y_val_full = self.collect_env.stack_data_Y_env(train=False, val_fraction=(1-self.training_fraction))
        return np.mean(np.square(self.y_val_full - self.naive_model.predict(self.x_val_full.T)))
