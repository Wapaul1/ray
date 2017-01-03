import ray
import time
import numpy as np

def test_func(X):
  before = time.time()
  result = X.max()
  after = time.time() - before
  return after

ray.init(start_ray_local=True)

b = np.random.randint(0,100, (100000,784))
b_copy = ray.get(ray.put(b))
print test_func(b), test_func(b_copy)

c = np.random.randint(0,100,(100000,784))
c_copy = ray.get(ray.put(c))
print np.mean([test_func(c) for _ in range(50)]), np.mean([test_func(c_copy) for _ in range(50)])
