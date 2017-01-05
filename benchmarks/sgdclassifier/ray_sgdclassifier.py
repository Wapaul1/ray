from sklearn import linear_model
from sklearn.linear_model.stochastic_gradient import fit_binary
import numpy as np
import ray
import copy
import time
import IPython


@ray.remote
def ray_fit_binary_nocopy(est, i, X, y, alpha, C, learning_rate, n_iter,
               pos_weight, neg_weight, sample_weight):
  
  before = time.time()
  # X = copy.deepcopy(X)
  # y = copy.deepcopy(y)
  after = time.time() - before
  result =  fit_binary(est, i, X, y, alpha, C, learning_rate, n_iter,
               pos_weight, neg_weight, sample_weight)
  return after, result

@ray.remote
def ray_fit_binary_copy(est, i, X, y, alpha, C, learning_rate, n_iter,
               pos_weight, neg_weight, sample_weight):
  
  before = time.time()
  X = copy.deepcopy(X)
  y = copy.deepcopy(y)
  after = time.time() - before
  result =  fit_binary(est, i, X, y, alpha, C, learning_rate, n_iter,
               pos_weight, neg_weight, sample_weight)
  return after, result


class RaySGDClassifier(linear_model.SGDClassifier):
  def _fit_multiclass(self, X, y, alpha, C, learning_rate,
                        sample_weight, n_iter):
        """Fit a multi-class classifier by combining binary classifiers
        Each binary classifier predicts one class versus all others. This
        strategy is called OVA: One Versus All.
        """
        # Use joblib to fit OvA in parallel.
        Xdata = ray.put(X)
        ydata = ray.put(y)
        sum_time = 0
        self.times = []
        result = ray.get([ray_fit_binary_nocopy.remote(self, i, Xdata, ydata, alpha, C, learning_rate,
                                n_iter, self._expanded_class_weight[i], 1.,
                                sample_weight) for i in range(len(self.classes_))])
        for i, (after, (weight, intercept)) in enumerate(result):
            self.times.append(after)
            self.coef_[i] = weight
            self.intercept_[i] = intercept
        self.t_ += n_iter * X.shape[0]
        if self.average > 0:
            if self.average <= self.t_ - 1.0:
                self.coef_ = self.average_coef_
                self.intercept_ = self.average_intercept_
            else:
                self.coef_ = self.standard_coef_
                self.standard_intercept_ = np.atleast_1d(self.intercept_)
                self.intercept_ = self.standard_intercept_

class RaySGDClassifier2(linear_model.SGDClassifier):
  def _fit_multiclass(self, X, y, alpha, C, learning_rate,
                        sample_weight, n_iter):
        """Fit a multi-class classifier by combining binary classifiers
        Each binary classifier predicts one class versus all others. This
        strategy is called OVA: One Versus All.
        """
        # Use joblib to fit OvA in parallel.
        Xdata = ray.put(X)
        ydata = ray.put(y)
        sum_time = 0
        self.times = []
        result = ray.get([ray_fit_binary_copy.remote(self, i, Xdata, ydata, alpha, C, learning_rate,
                                n_iter, self._expanded_class_weight[i], 1.,
                                sample_weight) for i in range(len(self.classes_))])
        for i, (after, (weight, intercept)) in enumerate(result):
            self.times.append(after)
            self.coef_[i] = weight
            self.intercept_[i] = intercept
        self.t_ += n_iter * X.shape[0]
        if self.average > 0:
            if self.average <= self.t_ - 1.0:
                self.coef_ = self.average_coef_
                self.intercept_ = self.average_intercept_
            else:
                self.coef_ = self.standard_coef_
                self.standard_intercept_ = np.atleast_1d(self.intercept_)
                self.intercept_ = self.standard_intercept_


ray.register_class(RaySGDClassifier)
ray.register_class(RaySGDClassifier2)
ray.register_class(linear_model.sgd_fast.Hinge, pickle=True)
