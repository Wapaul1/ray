from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.forest import _parallel_build_trees
from sklearn.ensemble.base import _partition_estimators
from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.utils.validation import check_is_fitted
from sklearn.tree._tree import DTYPE, DOUBLE, Tree
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from scipy.sparse import issparse
import warnings
from warnings import warn
import ray
import time
import IPython

ray.register_class(DecisionTreeClassifier)
ray.register_class(Tree, pickle=True)

def _generate_sample_indices(random_state, n_samples):
    """Private function used to _parallel_build_trees function."""
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)

    return sample_indices

@ray.remote
def ray_parallel_build_trees(tree, forest, X, y, sample_weight, tree_idx, n_trees,
                            verbose=0, class_weight=None):
    """Private function used to fit a single tree in parallel."""
    X = X.copy()
    y = y.copy()
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))
    
    if forest.bootstrap:
        n_samples = X.shape[0]
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
        else:
            curr_sample_weight = sample_weight.copy()

        indices = _generate_sample_indices(tree.random_state, n_samples)
        sample_counts = bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        if class_weight == 'subsample':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                curr_sample_weight *= compute_sample_weight('auto', y, indices)
        elif class_weight == 'balanced_subsample':
            curr_sample_weight *= compute_sample_weight('balanced', y, indices)
        tree.fit(X, y, sample_weight=curr_sample_weight, check_input=False)
    else:
        tree.fit(X, y, sample_weight=sample_weight, check_input=False)

    return tree

@ray.remote
def ray_predict_proba(estimator, X, check_input):
  return estimator.predict_proba(X, check_input)

class RayRandomForestClassifier(RandomForestClassifier):

  def fit(self, X, y, sample_weight=None):
    """Build a forest of trees from the training set (X, y).
    Parameters
    ----------
    X : array-like or sparse matrix of shape = [n_samples, n_features]
        The training input samples. Internally, its dtype will be converted to
        ``dtype=np.float32``. If a sparse matrix is provided, it will be
        converted into a sparse ``csc_matrix``.
    y : array-like, shape = [n_samples] or [n_samples, n_outputs]
        The target values (class labels in classification, real numbers in
        regression).
    sample_weight : array-like, shape = [n_samples] or None
        Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node. In the case of
    classification, splits are also ignored if they would result in any
    single class carrying a negative weight in either child node.
    Returns
    -------
    self : object
      Returns self.
    """
    # Validate or convert input data
    X = check_array(X, accept_sparse="csc", dtype=DTYPE)
    y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
    if sample_weight is not None:
      sample_weight = check_array(sample_weight, ensure_2d=False)
    if issparse(X):
      # Pre-sort indices to avoid that each individual tree of the
      # ensemble sorts the indices.
      X.sort_indices()

      # Remap output
    n_samples, self.n_features_ = X.shape

    y = np.atleast_1d(y)
    if y.ndim == 2 and y.shape[1] == 1:
      warn("A column-vector y was passed when a 1d array was"
	   " expected. Please change the shape of y to "
	   "(n_samples,), for example using ravel().",
	   DataConversionWarning, stacklevel=2)

    if y.ndim == 1:
      # reshape is necessary to preserve the data contiguity against vs
      # [:, np.newaxis] that does not.
      y = np.reshape(y, (-1, 1))

    self.n_outputs_ = y.shape[1]

    y, expanded_class_weight = self._validate_y_class_weight(y)

    if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
      y = np.ascontiguousarray(y, dtype=DOUBLE)

    if expanded_class_weight is not None:
      if sample_weight is not None:
	sample_weight = sample_weight * expanded_class_weight
      else:
	sample_weight = expanded_class_weight

    # Check parameters
    self._validate_estimator()

    if not self.bootstrap and self.oob_score:
      raise ValueError("Out of bag estimation only available"
			 " if bootstrap=True")

    random_state = check_random_state(self.random_state)

    if not self.warm_start or not hasattr(self, "estimators_"):
      # Free allocated memory, if any
      self.estimators_ = []

    n_more_estimators = self.n_estimators - len(self.estimators_)

    if n_more_estimators < 0:
      raise ValueError('n_estimators=%d must be larger or equal to '
		       'len(estimators_)=%d when warm_start==True'
		       % (self.n_estimators, len(self.estimators_)))

    elif n_more_estimators == 0:
      warn("Warm-start fitting without increasing n_estimators does not "
	   "fit new trees.")
    else:
      if self.warm_start and len(self.estimators_) > 0:
	# We draw from the random state to get the random state we
	# would have got if we hadn't used a warm_start.
	random_state.randint(MAX_INT, size=len(self.estimators_))

      trees = []
      for i in range(n_more_estimators):
	tree = self._make_estimator(append=False,
				    random_state=random_state)
	trees.append(tree)

      # Parallel loop: we use the threading backend as the Cython code
      # for fitting the trees is internally releasing the Python GIL
      # making threading always more efficient than multiprocessing in
      # that case 
      Xdata = ray.put(X)
      Ydata = ray.put(y)
      IPython.embed()
      trees = ray.get([ray_parallel_build_trees.remote(t, self, Xdata, Ydata, sample_weight, i, len(trees),
		      verbose=self.verbose, class_weight=self.class_weight) for i, t in enumerate(trees)])
      # Collect newly grown trees
      self.estimators_.extend(trees)

    if self.oob_score:
      self._set_oob_score(X, y)

    # Decapsulate classes_ attributes
    if hasattr(self, "classes_") and self.n_outputs_ == 1:
      self.n_classes_ = self.n_classes_[0]
      self.classes_ = self.classes_[0]

    return self

  def predict_proba(self, X):
    """Predict class probabilities for X.
    The predicted class probabilities of an input sample are computed as
    the mean predicted class probabilities of the trees in the forest. The
    class probability of a single tree is the fraction of samples of the same
    class in a leaf.
    Parameters
    ----------
    X : array-like or sparse matrix of shape = [n_samples, n_features]
	The input samples. Internally, its dtype will be converted to
	``dtype=np.float32``. If a sparse matrix is provided, it will be
	converted into a sparse ``csr_matrix``.
    Returns
    -------
    p : array of shape = [n_samples, n_classes], or a list of n_outputs
	such arrays if n_outputs > 1.
	The class probabilities of the input samples. The order of the
	classes corresponds to that in the attribute `classes_`.
    """
    check_is_fitted(self, 'estimators_')
    # Check data
    X = self._validate_X_predict(X)

    # Assign chunk of trees to jobs
    n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

    # Parallel loop
    all_proba = ray.get([ray_predict_proba.remote(e, X, False) for e in self.estimators_])

    # Reduce
    proba = all_proba[0]

    if self.n_outputs_ == 1:
      for j in range(1, len(all_proba)):
	proba += all_proba[j]

      proba /= len(self.estimators_)

    else:
      for j in range(1, len(all_proba)):
	for k in range(self.n_outputs_):
	  proba[k] += all_proba[j][k]

	for k in range(self.n_outputs_):
	  proba[k] /= self.n_estimators

    return proba

ray.register_class(RayRandomForestClassifier)
