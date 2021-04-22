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
            self.lr = LinearRegression(fit_intercept=False)
        elif regression_method=='Ridge':
            self.lr = Ridge(fit_intercept=False)
        elif regression_method=='HuberRegressor':
            self.lr = HuberRegressor(fit_intercept=False)

        self.poly = PolynomialFeatures(
            degree=self.params_method.get('power_features')[self.env_key]
            )
        

    def fit(self, X, Y):
        _ = self.poly.fit(X)
        self.set_feature_choices(self.params_method.get('selected_features')[self.env_key])
        X_extended = self.poly.transform(X)[:,self.selected_indices]
        self.lr.fit(X_extended, Y)


    def predict(self, X):
        X_extended = self.poly.fit_transform(X)[:,self.selected_indices]
        return self.lr.predict(X_extended)



    
    def coef_(self):
        return self.lr.coef_

    def set_feature_choices(self, selected_features):
        # A list of selected features contains all those interactions that we want to keep
        # in the polynomial extension. eg. 'x1^0x2^2' refers to one term in a polynomial extension
        # of degree at least 3, where x1 and x2 are input variables (i.e. randomized covariates)
        name_list = self.poly.get_feature_names()
        self.feature_names = []

        for elem in name_list:
            if elem=='1':
                self.feature_names.append('1')
            else:
                list_vars_deg = [
                    (self.env.data['randomized_index'][int(var[1])] + 1,
                    (len(var)>2)*int(var[-1])) for var in elem.split(' ')
                    ]
                self.feature_names.append(''.join(['x{}^{}'.format(elem[0], elem[1]) for elem in list_vars_deg]))

        self.selected_indices = [self.feature_names.index(name) for name in selected_features]

    def get_feature_names(self):
        return [self.feature_names[index] for index in self.selected_indices]


class RandomForestRegression:

    def __init__(self, params_method, env_key, collect_env):

        self.lr = RandomForestRegressor(
            n_estimators=params_method.get('n_estimators')
        )

    def fit(self, X, Y):

        self.lr.fit(X, Y)

    def predict(self, X):

        return self.lr.predict(X)

