import ray
import numpy as np
import copy
import time

@ray.remote
def test_nocopy(X):
  before = time.time()
  result = (2*X).max()
  after = time.time() - before
  return after

@ray.remote
def test_copy(X):
  X = copy.deepcopy(X)
  before = time.time()
  result = (2*X).max()
  after = time.time() - before
  return after

ray.init(start_ray_local=True)

array = np.random.randint(0, 100, (10000, 784))
print np.mean(ray.get([test_nocopy.remote(array) for _ in range(5)]))
print np.mean(ray.get([test_copy.remote(array) for _ in range(5)]))
