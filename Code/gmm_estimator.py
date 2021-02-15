import copy
import numpy as np
import scipy as sp
import cvxpy as cp
from sklearn.linear_model import LinearRegression
from generateEnvironment import GenerateEnvironment, generate_constraints, compute_residual_variance


class SolveProblem(GenerateEnvironment):

    def __init__(self, obs_connectivity, x_indices, y_index):
        super().__init__(obs_connectivity, x_indices, y_index)

    def combine_constraints(self, list_environments):
        
        stacked_constraints = np.empty((self.n_dim, 0))

        stacked_constraints = np.hstack([env['constraints'] for env in list_environments])

        n_samples_prop = np.hstack([[env['n_samples']]*env['constraints'].shape[1] for env in list_environments])
        n_samples_prop = n_samples_prop/np.sum([env['n_samples'] for env in list_environments])

        # M shape: (x_dim, n_constraints)
        M = stacked_constraints[self.x_indices,:]
        # Z shape: n_constraints,
        Z = stacked_constraints[self.y_index,:]

        # Multiply by the sample proportions.
        M = M.dot(np.diag(n_samples_prop))
        Z = Z.dot(np.diag(n_samples_prop))

        return M, Z


    def compute_beta(self, list_environments, reweighting_matrix):

        M, Z = self.combine_constraints(list_environments)
        
        beta = np.linalg.inv(M.dot(reweighting_matrix).dot(M.T)).dot(M.dot(reweighting_matrix).dot(Z))

        return beta


    def compute_sample_residual_cov(self, list_environments, beta_hat):

        for env in list_environments:
            self.compute_residual_variance(env, beta_hat)

        residual_var = np.hstack([[env['residual_var']]*env['constraints'].shape[1] for env in list_environments])
        constraints_var = np.hstack([np.array(env['variance_constraints']) for env in list_environments])
        n_samples_prop = np.hstack([[env['n_samples']]*env['constraints'].shape[1] for env in list_environments])
        n_samples_prop = n_samples_prop/np.sum([env['n_samples'] for env in list_environments])

        S = constraints_var*residual_var*n_samples_prop

        return np.diag(S)


    def compute_aCov(self, list_environments, reweighting_matrix):

        M, Z = self.combine_constraints(list_environments)
        beta_hat = self.compute_beta(list_environments, reweighting_matrix=reweighting_matrix)

        S = self.compute_sample_residual_cov(list_environments, beta_hat)

        aCov = np.linalg.inv(M.dot(np.linalg.inv(S)).dot(M.T))

        return aCov


    def compute_beta_GMM(self, list_environments, return_first=False):

        n_constraints_tot = np.sum([env['constraints'].shape[1] for env in list_environments])

        # Compute first-step estimator:
        beta_first = self.compute_beta(list_environments, np.eye(n_constraints_tot))
        sample_residual_cov = self.compute_sample_residual_cov(list_environments, beta_first)

        # Compute second-step estimator:
        beta_second = self.compute_beta(list_environments, 
								 np.linalg.inv(sample_residual_cov))
        aCov = self.compute_aCov(list_environments, 
								 np.linalg.inv(sample_residual_cov))
        if return_first:
            return beta_second, aCov, beta_first
        else:
            return beta_second, aCov


    def compute_CI(self, beta_hat, aCov, n_samples_tot, alpha):

        v = sp.stats.norm.ppf(1-alpha/2)
        CI = np.vstack([beta_hat - np.sqrt(np.diag(aCov))*v/np.sqrt(n_samples_tot),
					   	beta_hat + np.sqrt(np.diag(aCov))*v/np.sqrt(n_samples_tot)])

        return CI


