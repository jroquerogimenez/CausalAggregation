import numpy as np
import scipy as sp
import cvxpy as cp
from sklearn.linear_model import LinearRegression, LassoCV


class GenerateEnvironment(object):

    def __init__(self, obs_connectivity, x_indices, y_index):

        self.obs_connectivity = obs_connectivity
        self.x_indices = x_indices
        self.y_index = y_index
        self.n_dim = self.obs_connectivity.shape[0]
        self.x_dim = len(self.x_indices)
        self.instrument_id = np.where(np.sum(np.abs(self.obs_connectivity), axis=1)==0)[0]
        self.beta=obs_connectivity[self.y_index,self.x_indices]


    def generate_intervention(self, n_samples, dict_interventions):

        environment = {}

        # Create a copy of the connectivity matrix.
        int_connectivity = np.copy(self.obs_connectivity)

        # Update the connectivity matrix according to the dict interventions.
        for index, intervention in dict_interventions.items():
            assert index in self.x_indices, 'Invalid intervention'
            if intervention['type']=='independent':
                int_connectivity[index,:] = 0
            if intervention['type']=='iv':
                int_connectivity[index, intervention['iv_index']]=1
            if intervention['type']=='parental':
                pass

        # Generate dataset.
        eps = self.generate_epsilon(n_samples)
        dataset = np.linalg.inv(np.eye(self.n_dim) - int_connectivity).dot(eps)

        # dataset shape: (n_dim, n_samples)
        environment['dataset'] = dataset
        environment['dict_interventions'] = dict_interventions
        environment['x_indices'] = self.x_indices
        environment['y_index'] = self.y_index
        environment['n_samples'] = dataset.shape[1]

        generate_constraints(environment)

        return environment




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


    def generate_epsilon(self, n_samples):
        return np.random.normal(size=(self.n_dim, n_samples))


def generate_constraints(environment):

    # Build orthogonality vectors.
    array_constraints = np.empty((environment['dataset'].shape[0],0))
    variance_constraints = []
    dataset = environment['dataset']
    for index, intervention in environment['dict_interventions'].items():
        if intervention['type']=='independent':
            orth_var=dataset[index,:].reshape(-1,1)
        if intervention['type']=='iv':
            orth_var=dataset[intervention['iv_index'],:].reshape(-1,1) 
        if intervention['type']=='parental':
            lr = LinearRegression()
            lr.fit(dataset[intervention['parental_index'],:].T, dataset[index,:])
            orth_var=(dataset[index,:]-dataset[intervention['parental_index'],:].T.dot(lr.coef_)).reshape(-1,1)
          
        new_constraint = dataset.dot(orth_var)/dataset.shape[1]
        array_constraints = np.hstack([array_constraints, new_constraint])
        variance_constraints.append(np.std(orth_var)**2)
            
    # constraint shape: (n_dim, n_constraints)
    environment['constraints'] = array_constraints
    environment['variance_constraints'] = variance_constraints


def compute_residual_variance(environment, beta_hat):

    # Compute the residual variance within an environment of a linear fit with
    # a regression coefficient beta_hat.
    dataset = environment['dataset']

    return np.sum(np.square(beta_hat.dot(dataset[environment['x_indices'],:]) -
        dataset[environment['y_index'],:]))/dataset.shape[1]
