import copy
import numpy as np
import scipy as sp
import cvxpy as cp
from sklearn.linear_model import LinearRegression
from generateEnvironment import GenerateEnvironment, generate_constraints, compute_residual_variance


class SolveProblem(GenerateEnvironment):

    def __init__(self, obs_connectivity, x_indices, y_index):
        super().__init__(obs_connectivity, x_indices, y_index)


    def combine_constraints(self, list_environments, sample_size_rescaled=False):
        
        stacked_constraints = np.empty((self.n_dim, 0))
        stacked_constraints = np.hstack([env['constraints'] for env in list_environments])

        observable_indices = np.array(list(self.x_indices)+[self.y_index])
        # M shape: (obs_dim, n_constraints)
        M = stacked_constraints[observable_indices,:]
        # proj_M shape: (obs_dim, obs_dim)
        proj_M = M.dot(np.linalg.inv(M.T.dot(M)).dot(M.T))

        return M, proj_M


    def compute_wass_radius(self, obs_environment, M, interventional_indices):

        obs_dataset = obs_environment['dataset']
        return np.sum(np.square(M.dot(obs_dataset[interventional_indices, :])))/obs_dataset.shape[1]


    def robust_estimator(self, lmbda, list_environments, obs_environment):

        _, proj_M = self.combine_constraints(list_environments)

        beta = cp.Variable(shape=self.x_dim, name='beta')

        obs_dataset = obs_environment['dataset']

        empirical_loss = cp.norm(obs_dataset[self.x_indices,:].T @ beta - obs_dataset[self.y_index, :], 2)/np.sqrt(obs_dataset.shape[1])
        regularization_loss = np.sqrt(lmbda)*cp.norm(proj_M[:,:-1] @ beta - proj_M[:,-1], 2)
        
        objective = cp.Minimize(empirical_loss + regularization_loss)

        problem = cp.Problem(objective)
        problem.solve(solver='MOSEK')

        return problem

