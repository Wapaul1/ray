from tensorflow.examples.tutorials.mnist import input_data
import ray
from sklearn import svm
import ray_cross_val
import time
from sklearn.model_selection import cross_val_score
import os

worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../lib/python/ray/workers/default_worker.py")
times_ray_workers = []
times_ray_folds = []
times_skl_folds = []
folds = [2*i for i in range(1,11)]
workers = [2*i for i in range(1,9)]
mnist = input_data.read_data_sets("MNIST_data/")
for fold in folds:
  before = time.time()
 # result = cross_val_score(svm.SVC(), mnist.test.images, mnist.test.labels, cv=fold, n_jobs=-1)
  after = time.time() - before
  times_skl_folds.append(after)
  print "SKL_folds ", times_skl_folds
data = ray.init(start_ray_local=True, num_workers=2, num_local_schedulers=1)
for i in workers:
  before = time.time()
#  result = ray_cross_val.ray_cross_val_score(svm.SVC(), mnist.test.images, mnist.test.labels, cv=16)
  times_ray_workers.append(time.time() - before)
  if i == 10:
    for fold in folds:
      before = time.time()
      result = ray_cross_val.ray_cross_val_score(svm.SVC(), mnist.test.images, mnist.test.labels, cv=fold)
      times_ray_folds.append(time.time() - before)
      print "Ray_folds ", times_ray_folds
  ray.services.start_worker(data["node_ip_address"], data["store_socket_name"], data["manager_socket_name"], data["local_scheduler_socket_name"], data["redis_address"], worker_path, True, False)
  ray.services.start_worker(data["node_ip_address"], data["store_socket_name"], data["manager_socket_name"], data["local_scheduler_socket_name"], data["redis_address"], worker_path, True, False)
  print "Ray_workers ", times_ray_workers

with open("results.txt", "w") as resultfile:
  resultfile.write(" ".join(map(str, folds)) + "\n")
  resultfile.write(" ".join(map(str, times_skl_folds)) + "\n")
  resultfile.write(" ".join(map(str, times_ray_folds)) + "\n")
  resultfile.write(" ".join(map(str, workers)) + "\n")
  resultfile.write(" ".join(map(str, times_ray_workers)))
