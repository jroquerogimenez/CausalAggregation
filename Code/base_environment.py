import numpy as np
import scipy as sp
import cvxpy as cp
from sklearn.linear_model import LinearRegression, LassoCV

'''
topo_order: list of str containing the keys with each node name.
structural_equation_dict: key=node name, val= dict {'input_keys':parent node name, 'structural_eq': function}
disturbance_sampler_dict: key=node name, val= function: n_samples -> output eps
y_key: str, target node name
x_key: list of str, observed node name
'''
class BaseEnvironment(object):

    def __init__(self, structural_equation_dict, disturbance_sampler_dict, topo_order, y_key, x_key=None):

        self.structural_equation_dict = structural_equation_dict
        self.disturbance_sampler_dict = disturbance_sampler_dict
        self.topo_order = topo_order
        self.y_key = y_key
        if x_key is None: 
            self.x_key = list(self.topo_order)
            self.x_key.remove(self.y_key)
        else:
            self.x_key = self.sort_topo_order(x_key, self.topo_order)

        assert set(self.topo_order)==set(self.structural_equation_dict.keys())
        assert set(self.topo_order)==set(self.disturbance_sampler_dict.keys())
        assert (self.y_key in self.topo_order) and not (self.y_key in self.x_key)
        assert np.array([(x_k in self.topo_order) for x_k in self.x_key]).all()

        for var_key, structural_equation in self.structural_equation_dict.items():
            assert np.array([
                self.topo_order.index(input_key)<self.topo_order.index(var_key)
                for input_key in structural_equation['input_keys']
            ]).all()
            assert np.array([
                structural_equation['input_keys'] == self.sort_topo_order(structural_equation['input_keys'], self.topo_order)
            ]).all()
        
        self.true_structural_eq_y = self.structural_equation_dict[self.y_key]['structural_eq']
        self.true_input_indices_for_y = np.array([self.structural_equation_dict[self.y_key]['input_keys'].index(key) for key in self.structural_equation_dict[self.y_key]['input_keys'] if key in self.x_key])
        self.true_input_indices_for_x = np.array([self.x_key.index(key) for key in self.structural_equation_dict[self.y_key]['input_keys'] if key in self.x_key])

        def true_function_y(x):
            x = np.array(x)
            arg_y = np.zeros((len(self.structural_equation_dict[self.y_key]['input_keys']) + 1, x.shape[1]))
            arg_y[self.true_input_indices_for_y,:] = x[self.true_input_indices_for_x,:]

            return self.true_structural_eq_y(arg_y)

        self.true_function_y = true_function_y


    def sort_topo_order(self, list_keys, topo_order):
        indices = np.sort([topo_order.index(key) for key in list_keys])
        return [topo_order[index] for index in indices]


    def generate_intervention(self, dict_interventions):

        structural_equation_dict_int = self.structural_equation_dict.copy()
        disturbance_sampler_dict_int = self.disturbance_sampler_dict.copy()
        randomized_key = []

        # Update the connectivity matrix according to the dict interventions.
        for int_var_key, intervention in dict_interventions.items():
            assert int_var_key in self.x_key

            if intervention['type']=='independent':
                randomized_key.append(int_var_key)
                del structural_equation_dict_int[int_var_key]
                structural_equation_dict_int[int_var_key] = {
                   'input_keys' : np.array([]),
                   'structural_eq' : (lambda u: u) 
                }               

            if intervention['type']=='do-zero':
                del structural_equation_dict_int[int_var_key]
                structural_equation_dict_int[int_var_key]['input_keys'] = np.array([])
                structural_equation_dict_int[int_var_key]['structural_eq'] = (lambda u: u)

        environment = Environment(
            structural_equation_dict_int,
            disturbance_sampler_dict_int,
            self.topo_order,
            self.x_key,
            self.y_key,
            self.sort_topo_order(randomized_key, self.topo_order))

        return environment




class Environment(object):

    def __init__(
        self,
        structural_equation_dict,
        disturbance_sampler_dict, 
        topo_order, 
        x_key,
        y_key,
        randomized_key = None,
    ):
    

        self.structural_equation_dict = structural_equation_dict
        self.disturbance_sampler_dict = disturbance_sampler_dict
        self.topo_order = topo_order
        self.x_key = x_key
        self.y_key = y_key
        if randomized_key is not None:
            self.randomized_index = np.array([self.x_key.index(rand_key) for rand_key in randomized_key])
        else:
            self.randomized_index = np.array([])

        assert set(self.topo_order)==set(self.structural_equation_dict.keys())
        assert set(self.topo_order)==set(self.disturbance_sampler_dict.keys())
        assert self.y_key in self.topo_order

        self.structural_equation_dict = {key: self.structural_equation_dict[key] for key in self.topo_order}

        for var_key in self.topo_order:
            structural_equation = self.structural_equation_dict[var_key]
            assert np.array([
                self.topo_order.index(input_key)<self.topo_order.index(var_key)
                for input_key in structural_equation['input_keys']
                ]).all()
        

    def generate_samples(self, n_samples):

        self.disturbance_samples_dict = {
            var_key: sampler(n_samples) 
            for var_key, sampler in self.disturbance_sampler_dict.items()}
        self.samples_dict = {}

        for var_key in self.topo_order:
            structural_equation = self.structural_equation_dict[var_key]
            if len(structural_equation['input_keys'])>0:
                input_samples = np.stack(
                    [self.samples_dict[input_key]
                    for input_key in structural_equation['input_keys']])
                input_samples = np.vstack([input_samples, self.disturbance_samples_dict[var_key]])
            else:
                input_samples = self.disturbance_samples_dict[var_key]

            self.samples_dict[var_key] = structural_equation['structural_eq'](input_samples)
            

        # From now on, the var_keys are no longer used: we output a covariate matrix X and a response Y,
        # and the indices in X that are randomized. 
        self.data = {
            'X': np.stack([self.samples_dict[x_k] for x_k in self.x_key]),
            'Y': self.samples_dict[self.y_key],
            'randomized_index': self.randomized_index,
            }

        orth_covariates = self.data['X'][self.randomized_index].dot(self.data['X'].T)/self.data['X'].shape[1]
        orth_response = self.data['X'][self.randomized_index].dot(self.data['Y'])/self.data['X'].shape[1]

        self.data.update({'constraints': np.hstack([orth_covariates, orth_response.reshape(-1,1)])})




    def compute_residual_variance(self, environment, beta_hat):

        # Compute the residual variance within an environment of a linear fit with
        # a regression coefficient beta_hat.
        dataset = environment['dataset']

        environment['residual_var'] = np.sum(np.square(beta_hat.dot(dataset[self.x_indices,:]) -
            dataset[self.y_index,:]))/dataset.shape[1]


    def estimator_LR(self, environment):

        # Compute LS estimator based on an environment.
        samples = environment['dataset']
        regr=LinearRegression()
        regr.fit(samples[self.x_indices,:].T, samples[self.y_index,:])
        return regr.coef_
        

    def estimator_Lasso(self, environment):

        # Compute Lasso estimator based on an environment.
        samples = environment['dataset']
        regr=LassoCV()
        regr.fit(samples[self.x_indices,:].T, samples[self.y_index,:])
        return regr.coef_



