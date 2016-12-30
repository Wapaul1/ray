from tensorflow.examples.tutorials.mnist import input_data
import ray
from sklearn import svm
import ray_cross_val
import time
from sklearn.model_selection import cross_val_score

times = []
mnist = input_data.read_data_sets("MNIST_data/")
before2 = time.time()
result2 = cross_val_score(svm.SVC(), mnist.test.images, mnist.test.labels, cv=20, n_jobs=-1)
after = time.time() - before2
ray.init(start_ray_local=True, num_workers=10, num_local_schedulers=1)
before = time.time()
result = ray_cross_val.ray_cross_val_score(svm.SVC(), mnist.test.images, mnist.test.labels, cv=20)
times.append(time.time() - before)
print times[-1], after, result, result2
