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

        stacked_sample_prop = np.diag(np.hstack([[env['n_samples']]*env['constraints'].shape[1]
											   	for env in list_environments]))

        # M shape: (x_dim, n_constraints)
        M = stacked_constraints[self.x_indices,:]
        # Z shape: n_constraints,
        Z = stacked_constraints[self.y_index,:]
        # proj_M shape: (x_dim, x_dim)
#        if not sample_size_rescaled:
#            proj_M = np.eye(M.shape[0]) - M.dot(np.linalg.inv(M.T.dot(M)).dot(M.T))
#        else:
#            proj_M=None
        proj_M=None

        # Multiply by the sample proportions.
        if sample_size_rescaled:
            M = M.dot(stacked_sample_prop)/np.sum(stacked_sample_prop)
            Z = Z.dot(stacked_sample_prop)/np.sum(stacked_sample_prop)

        return M, Z, proj_M


    def compute_beta_GMM(self, list_environments):

        # Compute first-step estimator:
        sample_residual_cov = self.compute_sample_residual_cov(list_environments,
								  np.eye(np.sum([env['constraints'].shape[1] for env in list_environments])))

        beta = self.compute_beta(list_environments, 
								 np.linalg.inv(sample_residual_cov))

        aCov = self.compute_aCov(list_environments, 
								 np.linalg.inv(sample_residual_cov))

        return beta, aCov

    def compute_beta(self, list_environments, reweighting_matrix = None):

        if reweighting_matrix is None:
            M, Z, _ = self.combine_constraints(list_environments)
        else:
            M, Z, _ = self.combine_constraints(list_environments, sample_size_rescaled=True)
        
        # If as many constraints as covariates, compute beta.
        if reweighting_matrix is None:
            assert M.shape[0]==M.shape[1]
            beta = np.linalg.inv(M.T).dot(Z)
        # Otherwise has more constraints than covariates and a reweighting matrix.
        else:
            beta = np.linalg.inv(M.dot(reweighting_matrix).dot(M.T)).dot(M.dot(reweighting_matrix).dot(Z))

        return beta


    def compute_sample_residual_cov(self, list_environments, reweighting_matrix = None):

        beta_hat = self.compute_beta(list_environments, reweighting_matrix=reweighting_matrix)

        for env in list_environments:
            self.compute_residual_variance(env, beta_hat)

        residual_var = np.hstack([[env['residual_var']]*env['constraints'].shape[1] for env in list_environments])
        constraints_var = np.hstack([np.array(env['variance_constraints'])/env['dataset'].shape[1] for env in list_environments])
        n_samples_tot = np.sum([env['dataset'].shape[1] for env in list_environments])

        S = constraints_var*residual_var*n_samples_tot

        return np.diag(S)


    def compute_aCov(self, list_environments, reweighting_matrix = None):

#        if reweighting_matrix is None:
#            M, Z, _ = self.combine_constraints(list_environments)
#        else:
#            M, Z, _ = self.combine_constraints(list_environments, sample_size_rescaled=True)

        M, Z, _ = self.combine_constraints(list_environments)
        S = self.compute_sample_residual_cov(list_environments, reweighting_matrix=reweighting_matrix)

        aCov = np.linalg.inv(M.dot(np.linalg.inv(S)).dot(M.T))

        return aCov


    def compute_CI(self, beta_hat, aCov, n_samples_tot, alpha):

        v = sp.stats.norm.ppf(1-alpha/2)
        CI = np.vstack([beta_hat - np.sqrt(np.diag(aCov))*v/np.sqrt(n_samples_tot),
					   	beta_hat + np.sqrt(np.diag(aCov))*v/np.sqrt(n_samples_tot)])

        return CI


    def solve_problem(self, lmbda, list_environments, obs_environment):

        M, Z, proj_M = self.combine_constraints(list_environments)

        beta = cp.Variable(shape=self.x_dim, name='beta')

        obs_dataset = obs_environment['dataset']
        projected_obs_residual = proj_M @ obs_dataset[self.x_indices, :].dot(obs_dataset.T)/obs_dataset.shape[1]
        projected_obs_residual = projected_obs_residual[:,self.x_indices] @ beta - projected_obs_residual[:,self.y_index]
        
        objective = cp.Minimize(cp.norm(beta,1))
        constraint_list = [cp.norm(beta @ M - Z, 'inf') <= lmbda, cp.norm(projected_obs_residual, 'inf') <= lmbda]

        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver='MOSEK')

        return problem


    def solve_problem_CV(self, list_environments, obs_environment, n_splits=4):

        list_lmbdas = np.exp(np.log(10)*np.arange(-3,2,0.2))
        val = []

        for lmbda in list_lmbdas:
            val.append(self.solve_problem_lmbda(lmbda, list_environments, obs_environment))

        self.selected_lmbda = list_lmbdas[np.argmin(val)]

        return self.solve_problem(self.selected_lmbda, list_environments, obs_environment)


    def solve_problem_lmbda(self, lmbda, list_environments, obs_environment, n_splits=4):

        loss = 0
        for k in np.arange(n_splits):
            list_environments_split = []
            for environment in list_environments:
                n_samples = environment['dataset'].shape[1]
                split_indices = np.arange((n_samples//n_splits)*k, (n_samples//n_splits)*(k+1))

                environment_split = copy.deepcopy(environment)
                environment_split['dataset'] = environment['dataset'][:,split_indices]

                generate_constraints(environment_split)
                list_environments_split.append(environment_split)

            obs_environment_split = copy.deepcopy(obs_environment)
            n_samples = obs_environment['dataset'].shape[1]
            split_indices = np.arange((n_samples//n_splits)*k, (n_samples//n_splits)*(k+1))

            obs_environment_split['dataset'] = obs_environment['dataset'][:,split_indices]


            problem = self.solve_problem(lmbda, list_environments_split, obs_environment_split)
            loss += np.mean([compute_residual_variance(environment, problem.variables()[0].value)
						   	for environment in list_environments_split])

        return loss/n_splits





