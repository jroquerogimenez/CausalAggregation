import numpy as np
import scipy as sp
import cvxpy as cp
from sklearn.linear_model import LinearRegression
from generateEnvironment import GenerateEnvironment


class SolveProblem(GenerateEnvironment):

    def __init__(self, obs_connectivity, x_indices, y_index):
        super().__init__(obs_connectivity, x_indices, y_index)


    def combine_constraints(self, list_environments):
        
        stacked_constraints = np.empty((self.n_dim, 0))

        stacked_constraints = np.hstack([env['constraints'] for env in list_environments])

        # M shape: (x_dim, n_constraints)
        M = stacked_constraints[self.x_indices,:]
        # Z shape: n_constraints,
        Z = stacked_constraints[self.y_index,:]
        # proj_M shape: (x_dim, x_dim)
        proj_M = np.eye(M.shape[0]) - M.dot(np.linalg.inv(M.T.dot(M)).dot(M.T))

        return M, Z, proj_M


    def compute_beta(self, M, Z):

        # If as many constraints as covariates, compute beta.
        assert M.shape[0]==M.shape[1]
        return np.linalg.inv(M.T).dot(Z)


    def compute_asymptotic_cov(self, list_environments, alpha):

        v=sp.stats.norm.ppf(1-alpha/2)

        M, Z, _ = self.combine_constraints(list_environments)
        M_T_inv = np.linalg.inv(M.T)

        beta_hat = self.compute_beta(M, Z)

        for env in list_environments:
            self.compute_residual_variance(env, beta_hat)

        residual_var = np.hstack([[env['residual_var']]*env['constraints'].shape[1] for env in list_environments])

        D = np.hstack([np.array(env['variance_constraints'])/env['dataset'].shape[1] for env in list_environments])
        D = D*residual_var

        n_samples_tot = np.sum([env['dataset'].shape[1] for env in list_environments])
        asymptotic_cov=n_samples_tot*M_T_inv.dot(np.diag(D)).dot(M_T_inv.T)

        CI = np.vstack([beta_hat - np.sqrt(np.diag(asymptotic_cov))*v/np.sqrt(n_samples_tot),
					   	beta_hat + np.sqrt(np.diag(asymptotic_cov))*v/np.sqrt(n_samples_tot)])

        return asymptotic_cov, CI


    def solve_problem(self, lmbda, list_environments, n_samples):

        M, Z, proj_M = self.combine_constraints(list_environments)

        beta = cp.Variable(shape=self.x_dim, name='beta')

        obs_dataset = self.generate_intervention(n_samples, {})['dataset']
        projected_obs_residual = proj_M @ obs_dataset[self.x_indices, :].dot(obs_dataset.T)/obs_dataset.shape[1]
        
        constraint_list = [cp.norm(beta @ M - Z, 'inf') <= lmbda]

        constraint_list.append(cp.norm(projected_obs_residual[:,self.x_indices] @ beta - projected_obs_residual[:,self.y_index], 'inf') <= lmbda)

        objective_function = cp.norm(beta,1)
        objective = cp.Minimize(objective_function)

        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver='MOSEK')

        return problem


