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

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from joblib import Parallel, delayed
from constants import Constants as cons

def train_model(X, y, seed):
    '''Trains a surrogate model.'''
    model = MLPRegressor(hidden_layer_sizes=(cons.H,), activation='relu',
        solver='lbfgs', alpha=0.001, batch_size='auto', learning_rate='adaptive',
        learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=seed, tol=0.0001, verbose=False, warm_start=False,
        momentum=0.9, nesterovs_momentum=True, early_stopping=False,
        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #model = SVR(kernel='rbf')
    return model.fit(X, y)

def test_model(model, X):
    '''Returns the predictions using a surrogate model.'''
    return model.predict(X.reshape(1,-1))[0]

class Model:
    '''A surrogate model for the NKCS EA.'''

    def __init__(self):
        '''Initialises a surrogate model.'''
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

#        self.models = []

#        self.model = MLPRegressor(hidden_layer_sizes=(cons.H,), activation='tanh',
#            solver='lbfgs', alpha=0.001, batch_size='auto', learning_rate='adaptive',
#            learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
#            random_state=None, tol=0.0001, verbose=False, warm_start=False,
#            momentum=0.9, nesterovs_momentum=True, early_stopping=False,
#            validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#        self.model = MLPRegressor(hidden_layer_sizes=(cons.H,), activation='relu',
#            solver='lbfgs', alpha=0.001, batch_size='auto', learning_rate='constant',
#            learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
#            random_state=None, tol=0.0001, verbose=False, warm_start=False,
#            momentum=0.9, nesterovs_momentum=True, early_stopping=False,
#            validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#        kernel = ConstantKernel(1, constant_value_bounds='fixed') * RBF(length_scale=1)
#        self.model = GaussianProcessRegressor(kernel=kernel, alpha=1e-10,
#            optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=True,
#            copy_X_train=False, random_state=None)
#
        # THIS ONE!
        rbf = RBF(length_scale=1)
        self.model = GaussianProcessRegressor(kernel=rbf, normalize_y=True, copy_X_train=False)

#        self.model = SVR(kernel='rbf')
#
#        self.model = DecisionTreeRegressor()

#        self.model = LinearRegression()

#        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
#            max_depth=3, random_state=None, loss='ls')

    def train(self, X, y):
        '''Trains a surrogate model using the evaluated genomes and fitnesses.'''
        # normalise training data (zero mean and unit variance)
        self.input_scaler.fit(X)
        y = np.asarray(y).reshape(-1, 1)
        self.output_scaler.fit(y)
        X_train = self.input_scaler.transform(X)
        y_train = y # self.output_scaler.transform(y)
        # fit the model
        self.model.fit(X_train, y_train.ravel())
#        print('rsquared = ' + str(self.model.score(X_train, y_train.ravel())))

#        # returns fitted models
#        self.models = Parallel(n_jobs=cons.NUM_THREADS)(
#            delayed (train_model)(X, y, seed) for seed in range(cons.N_MODELS))

    def predict(self, genome):
        '''Uses the surrogate model to predict the fitness of a genome.'''
        rstd = True # True for GP
        # normalise input
        genome = np.asarray(genome).reshape(1, -1)
        inputs = self.input_scaler.transform(genome)

#        # NEW!
#        p = np.zeros(cons.N_MODELS)
#        for i in range(cons.N_MODELS):
#            p[i] = self.models[i].predict(inputs.reshape(1,-1))[0]
#        fit = np.mean(p)
#        std = np.std(p)
#        return fit, std

        # model prediction
        if rstd:
            fit, std = self.model.predict(inputs, return_std=rstd)
            return fit, std
        else:
            fit = self.model.predict(inputs)[0]
            return fit, 0
