#!/usr/bin/python3
#
# Copyright (C) 2019--2021 Richard Preen <rpreen@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

'''Models for surrogate-assisted evolution.'''

import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from joblib import Parallel, delayed
from constants import Constants as cons

def acquisition(mu_sample_opt, mu, std):
    '''Applies the acquisition function to predicted samples.'''
    if cons.ACQUISITION == 'ei': # expected improvement
        XI = 0.01
        imp = mu - mu_sample_opt - XI
        Z = imp / (std + 1E-9)
        return imp * norm.cdf(Z) + (std + 1E-9) * norm.pdf(Z)
    if cons.ACQUISITION == 'uc': # upper confidence
        return mu + std
    if cons.ACQUISITION == 'pi': # probability of improvement
        return norm.cdf((mu - mu_sample_opt) / (std + 1E-9))
    return mu # mean

def model_gp(seed):
    '''Gaussian Process Regressor'''
    kernel = RBF(length_scale=1)
    return GaussianProcessRegressor(kernel=kernel,
        random_state=seed, normalize_y=False, copy_X_train=False)

def model_mlp(seed):
    '''MLP Regressor'''
    return MLPRegressor(hidden_layer_sizes=(cons.H,), activation='relu',
        solver='lbfgs', alpha=0.001, batch_size='auto', learning_rate='constant',
        learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=seed, tol=0.0001, verbose=False, warm_start=False,
        momentum=0.9, nesterovs_momentum=True, early_stopping=True,
        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

def model_svr():
    '''Support Vector Regression'''
    return SVR(kernel='rbf')

def model_linear():
    '''Linear Regression'''
    return LinearRegression()

def model_gradient(seed):
    '''Gradient Boosting Regressor'''
    return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
        max_depth=3, random_state=seed, loss='ls')

def model_tree(seed):
    '''Decision Tree Regressor'''
    return DecisionTreeRegressor(random_state=seed)

def fit_model(X, y, seed=None):
    '''Trains a surrogate model.'''
    if cons.MODEL == 'gp':
        model = model_gp(seed)
    elif cons.MODEL == 'mlp':
        model = model_mlp(seed)
    elif cons.MODEL == 'svr':
        model = model_svr()
    elif cons.MODEL == 'tree':
        model = model_tree(seed)
    elif cons.MODEL == 'linear':
        model = model_linear()
    elif cons.MODEL == 'gradient':
        model = model_gradient(seed)
    else:
        print('unsupported surrogate model')
        sys.exit()
    return model.fit(X, y)

class Model:
    '''A surrogate model for the NKCS EA.'''

    def __init__(self):
        '''Initialises a surrogate model.'''
        self.output_scaler = StandardScaler()
        self.models = []
        self.mu_sample_opt = 0

    def fit(self, X, y):
        '''Trains a surrogate model using the evaluated genomes and fitnesses.'''
        # normalise training data (zero mean and unit variance)
        y = np.asarray(y).reshape(-1, 1)
        self.output_scaler.fit(y)
        y_train = self.output_scaler.transform(y).ravel()
        self.mu_sample_opt = np.max(y_train)
        X_train = X # unscaled binary inputs
        # fit models
        if cons.MODEL == 'gp':
            self.models.append(fit_model(X_train, y_train))
        else:
            self.models = Parallel(n_jobs=cons.NUM_THREADS)(delayed
                (fit_model)(X_train, y_train) for _ in range(cons.N_MODELS))

    def predict(self, X):
        '''Uses the surrogate model to predict the fitnesses of candidate genomes.'''
        if cons.MODEL == 'gp': # only one GP model
            mu, std = self.models[0].predict(X, return_std=True)
        else: # model prediction(s)
            n_models = len(self.models)
            n_samples = len(X)
            p = np.zeros((n_models, n_samples))
            for i in range(n_models):
                p[i] = self.models[i].predict(X)
            if n_models > 1:
                mu = np.mean(p, axis=0)
                std = np.std(p, axis=0)
            else:
                mu = p[0]
                std = 0
        return acquisition(self.mu_sample_opt, mu, std)

    def print_score(self, X, y):
        '''Prints the R squared model scores.'''
        for model in self.models:
            print('rsquared = ' + str(model.score(X, y)))
