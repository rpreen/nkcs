#!/usr/bin/python3
#
# Copyright (C) 2019--2024 Richard Preen <rpreen@gmail.com>
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

"""Models for surrogate-assisted evolution."""

from typing import List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .constants import Constants as Cons

np.set_printoptions(suppress=True)


def acquisition(
    mu_sample_opt: float, mu: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply the acquisition function to predicted samples."""
    if Cons.ACQUISITION == "ei":  # expected improvement
        xi = 0.01
        imp = mu - mu_sample_opt - xi
        z = imp / (std + 1e-9)
        return imp * norm.cdf(z) + (std + 1e-9) * norm.pdf(z)
    if Cons.ACQUISITION == "uc":  # upper confidence
        return mu + std
    if Cons.ACQUISITION == "pi":  # probability of improvement
        return norm.cdf((mu - mu_sample_opt) / (std + 1e-9))
    if Cons.ACQUISITION == "mean":  # mean
        return mu
    raise ValueError("unknown acquisition method: %s", Cons.ACQUISITION)


def model_gp(seed: Optional[int]) -> GaussianProcessRegressor:
    """Gaussian Process Regressor."""
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    return GaussianProcessRegressor(
        kernel=kernel,  # n_restarts_optimizer=10,
        random_state=seed,
        normalize_y=False,
        copy_X_train=False,
    )


def model_mlp(seed: Optional[int]) -> MLPRegressor:
    """MLP Regressor."""
    return MLPRegressor(
        hidden_layer_sizes=(Cons.H,),
        activation="relu",
        solver="lbfgs",
        alpha=0.001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.01,
        power_t=0.5,
        max_iter=1000,
        shuffle=True,
        random_state=seed,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=True,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
    )


def fit_model(
    x: np.ndarray, y: np.ndarray, seed: int = None
) -> Union[GaussianProcessRegressor, MLPRegressor]:
    """Train a surrogate model."""
    if Cons.MODEL == "gp":
        model = model_gp(seed)
    elif Cons.MODEL == "mlp":
        model = model_mlp(seed)
    else:
        raise ValueError("unsupported surrogate model")
    return model.fit(x, y)


class Model:
    """A surrogate model for the NKCS EA."""

    def __init__(self) -> None:
        """Initialise a surrogate model."""
        self.output_scaler: StandardScaler = StandardScaler()
        self.input_scaler: StandardScaler = StandardScaler()
        self.models: List[Union[GaussianProcessRegressor, MLPRegressor]] = []
        self.mu_sample_opt: float = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train a surrogate model using the evaluated genomes and fitnesses."""
        y_train = self.output_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        X_train = X
        if Cons.MODEL == "gp":
            self.models.append(fit_model(X_train, y_train))
        else:
            self.models = Parallel(n_jobs=Cons.NUM_THREADS)(
                delayed(fit_model)(X_train, y_train)
                for _ in range(Cons.N_MODELS)
            )
        self.mu_sample_opt = np.max(y_train)

    def __predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict fitnesses of candidate genomes with surrogate model."""
        X_predict = X
        if Cons.MODEL == "gp":  # only one GP model
            mu, std = self.models[0].predict(X_predict, return_std=True)
        else:  # model prediction(s)
            n_models = len(self.models)
            n_samples = len(X_predict)
            p = np.zeros((n_models, n_samples))
            for i in range(n_models):
                p[i] = self.models[i].predict(X_predict)
            if n_models > 1:
                mu = np.mean(p, axis=0)
                std = np.std(p, axis=0)
            else:
                mu = p[0]
                std = 0
        return mu, std

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return the utility scores of candidate genomes."""
        mu, std = self.__predict(X)
        return acquisition(self.mu_sample_opt, mu, std)
