import numpy as np
import scipy as sp
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor


class PolynomialRegression(LinearRegression, Ridge):

    def __init__(self, params_method, collect_env, env_key=None):
        '''
        - params_method contains keys 'power_features' and 'selected_features'
        - If env_key is None, these are an int and a list of features. Otherwise a dict with 
        env_key as key and values same as above. In that case, then also get self.randomized_index
        for randomized covs in corresponding env. 
        '''
        
        self.params_method = params_method
        self.env_key = env_key

        # Define randomized indices and self.poly preprocessing.
        if self.env_key is not None:
            self.randomized_index = collect_env.env[env_key].data['randomized_index']
            self.poly = PolynomialFeatures(
                degree=self.params_method.get('power_features')[self.env_key]
                )
        else:
            self.randomized_index = np.arange(collect_env.stack_data_X_env().shape[1])
            self.poly = PolynomialFeatures(
                degree=self.params_method.get('power_features')
                )

        # Define the regressor. Intercept manually added with self.poly.
        regression_method = params_method['regression_method']
        if regression_method=='LinearRegression':
            self.lr = LinearRegression(fit_intercept=False)
        if regression_method=='LassoCV':
            self.lr = LassoCV(fit_intercept=False)
        elif regression_method=='Ridge':
            self.lr = Ridge(fit_intercept=False)
        elif regression_method=='HuberRegressor':
            self.lr = HuberRegressor(fit_intercept=False, max_iter=5000)
        

    def fit(self, X, Y):
        _ = self.poly.fit(X)
        self.set_feature_choices()
        X_extended = self.poly.transform(X)[:,self.selected_indices]
        self.lr.fit(X_extended, Y)


    def predict(self, X):
        X_extended = self.poly.fit_transform(X)[:,self.selected_indices]
        return self.lr.predict(X_extended)


    def coef_(self):
        return self.lr.coef_


    def set_feature_choices(self):
        # A list of selected features contains all those interactions that we want to keep
        # in the polynomial extension. eg. 'x1^0x2^2' refers to one term in a polynomial extension
        # of degree at least 3, where x1 and x2 are input variables (i.e. randomized covariates)
        name_list = self.poly.get_feature_names() # All interactions in self.poly.
        self.feature_names = []
        for elem in name_list:
            if elem=='1':
                self.feature_names.append('1')
            else:
                list_vars_deg = [
                    (self.randomized_index[int(var[1])] + 1,
                    (len(var)>2)*int(var[-1])) for var in elem.split(' ')
                    ]
                self.feature_names.append(''.join(['x{}^{}'.format(elem[0], elem[1]) for elem in list_vars_deg]))

        # We keep only indices of features that were selected in the params_method dict.
        if self.env_key is not None:
            selected_features = self.params_method.get('selected_features')[self.env_key] 
        else:
            selected_features = self.params_method.get('selected_features')
        self.selected_indices = [self.feature_names.index(name) for name in selected_features]


    def get_feature_names(self):
        return [self.feature_names[index] for index in self.selected_indices]


class RandomForestRegression:

    def __init__(self, params_method, collect_env, env_key=None):

        self.lr = RandomForestRegressor(
            n_estimators=params_method.get('n_estimators')
        )

    def fit(self, X, Y):

        self.lr.fit(X, Y)

    def predict(self, X):

        return self.lr.predict(X)


    def coef_(self):
        return 0




   
class DecisionTreeRegression:

    def __init__(self, params_method, collect_env, env_key=None):

        self.lr = DecisionTreeRegressor(
            min_impurity_decrease=params_method.get('min_impurity_decrease', 0.2)
        )

    def fit(self, X, Y):

        self.lr.fit(X, Y)

    def predict(self, X):

        return self.lr.predict(X)


    def coef_(self):
        return 0