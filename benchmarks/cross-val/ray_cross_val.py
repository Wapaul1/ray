import sklearn.model_selection._validation as val
import sklearn.model_selection as model
from sklearn.utils import indexable
import ray
from sklearn.base import is_classifier, clone
from sklearn.utils import indexable
from sklearn.model_selection._split import check_cv
from sklearn.metrics.scorer import check_scoring
import numpy as np

@ray.remote
def ray_fit_and_score(estimator, X, y, scorer, train, test, verbose,
                      parameters, fit_params, return_train_score=False,
                      return_parameters=False, return_n_test_samples=False,
                      return_times=False, error_score='raise'):
  return val._fit_and_score(estimator, X, y, scorer, train, test, verbose,
                            parameters, fit_params, return_train_score,
                            return_parameters, return_n_test_samples,
                            return_times, error_score)

def ray_cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None,
                    n_jobs=1, verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs'):  
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    scores = [ray_fit_and_score.remote(clone(estimator), X, y, scorer,
                                              train, test, verbose, None,
                                              fit_params)
                      for train, test in cv.split(X, y, groups)]
    return np.array(ray.get(scores))
