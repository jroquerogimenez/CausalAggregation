import numpy as np
import scipy as sp
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures





#class RegressionMethod(LinearRegression):
#
#    def __init__(self, **params_method):
#        if params_method['method_name'] == 'PolynomialRegression'
#            
#
#    def fit(X, Y):
#    
#    def predict(X):





class PolynomialRegression(LinearRegression, Ridge):

    def __init__(self, params_method, env_key, collect_env):
        self.env = collect_env.env[env_key]
        self.params_method = params_method
        self.env_key = env_key

        regression_method = params_method.get('regression_method')
        if regression_method=='LinearRegression':
            self.lr = LinearRegression()
        elif regression_method=='Ridge':
            self.lr = Ridge()
        elif regression_method=='HuberRegressor':
            self.lr = HuberRegressor()

        self.power_features = self.params_method.get('power_features')[self.env_key]
        self.interaction_only = self.params_method.get('interactions_env')[self.env_key]=='int'
        self.include_bias = self.params_method.get('include_bias')[self.env_key]

        self.poly = PolynomialFeatures(
            degree=self.power_features,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only,
            )
        

    def fit(self, X, Y):

        self.truncate_column = int(self.interaction_only*X.shape[1] + self.include_bias*1.)
        X_extended = self.poly.fit_transform(X)[:,self.truncate_column:]
        self.lr.fit(X_extended, Y)


    def predict(self, X):

        X_extended = self.poly.fit_transform(X)[:,self.truncate_column:]

        return self.lr.predict(X_extended)



    
    def coef_(self):
        return self.lr.coef_

    def get_feature_names(self):
        name_list = self.poly.get_feature_names()[self.truncate_column:]
        final_list = []
        for elem in name_list:
            list_vars_deg = [(self.env.data['randomized_index'][int(var[1])], (len(var)>2)*int(var[-1])) for var in elem.split(' ')]
            final_list.append(''.join(['x{}^{}'.format(elem[0]+1, elem[1]) for elem in list_vars_deg]))

        return final_list



class RandomForestRegression:

    def __init__(self, params_method, env_key, collect_env):

        self.lr = RandomForestRegressor(
            n_estimators=params_method.get('n_estimators')
        )

    def fit(self, X, Y):

        self.lr.fit(X, Y)

    def predict(self, X):

        return self.lr.predict(X)

