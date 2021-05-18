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

def expected_improvement(mu_sample_opt, mu, std):
    '''Returns expected improvement.'''
    XI = 0.01
    ei = 0
    if std != 0:
        imp = mu - mu_sample_opt - XI
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
    return ei

def get_fitness(model, mu_sample_opt, child):
    '''Returns predicted offspring fitness.'''
    mu, std = model.predict(child.genome)
    if cons.ALGORITHM == 'ei':
        return expected_improvement(mu_sample_opt, mu, std)
    elif cons.ALGORITHM == 'uc':
        return mu + std # upper confidence
    return mu # mean

def model_gp(seed):
    '''Gaussian Process Regressor'''
    kernel = RBF(length_scale=1)
    model = GaussianProcessRegressor(kernel=kernel,
        random_state=seed, normalize_y=False, copy_X_train=False)
    return model

def model_mlp(seed):
    '''MLP Regressor'''
    model = MLPRegressor(hidden_layer_sizes=(cons.H,), activation='relu',
        solver='lbfgs', alpha=0.001, batch_size='auto', learning_rate='constant',
        learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=seed, tol=0.0001, verbose=False, warm_start=False,
        momentum=0.9, nesterovs_momentum=True, early_stopping=True,
        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return model

def model_svr():
    '''Support Vector Regression'''
    model = SVR(kernel='rbf')
    return model

def model_linear():
    '''Linear Regression'''
    model = LinearRegression()
    return model

def model_gradient(seed):
    '''Gradient Boosting Regressor'''
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
        max_depth=3, random_state=seed, loss='ls')
    return model

def model_tree(seed):
    '''Decision Tree Regressor'''
    model = DecisionTreeRegressor(random_state=seed)
    return model

def train_model(X, y, seed=None):
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

def test_model(model, X):
    '''Returns the predictions using a surrogate model.'''
    return model.predict(X.reshape(1,-1))[0]

class Model:
    '''A surrogate model for the NKCS EA.'''

    def __init__(self):
        '''Initialises a surrogate model.'''
        self.output_scaler = StandardScaler()
        self.models = []

    def train(self, X, y):
        '''Trains a surrogate model using the evaluated genomes and fitnesses.'''
        # normalise training data (zero mean and unit variance)
        y = np.asarray(y).reshape(-1, 1)
        self.output_scaler.fit(y)
        y_train = self.output_scaler.transform(y).ravel()
        X_train = X # unscaled binary inputs
        # fit models
        if cons.MODEL == 'gp':
            self.models.append(train_model(X_train, y_train))
        else:
            self.models = Parallel(n_jobs=cons.NUM_THREADS)(delayed
                (train_model)(X_train, y_train) for _ in range(cons.N_MODELS))

    def predict(self, genome):
        '''Uses the surrogate model to predict the fitness of a genome.'''
        inputs = np.asarray(genome).reshape(1, -1)
        # only one GP model
        if cons.MODEL == 'gp': # only one GP model
            fit, std = self.models[0].predict(inputs, return_std=True)
            fit = fit[0] # single sample
            std = std[0]
        # model prediction(s)
        else:
            N = len(self.models)
            p = np.zeros(N)
            for i in range(N):
                p[i] = self.models[i].predict(inputs)[0]
            if N > 1:
                fit = np.mean(p)
                std = np.std(p)
            else:
                fit = p[0]
                std = 0
        return fit, std

    def print_score(self, X, y):
        '''Prints the R squared model scores.'''
        for model in self.models:
            print('rsquared = ' + str(model.score(X, y)))
