from tensorflow.examples.tutorials.mnist import input_data
import ray
import ray_random_forest
import time
import os
from sklearn.ensemble import RandomForestClassifier

times_ray_workers = []
times_ray_folds = []
times_skl_folds = []
trees = [2*i for i in range(5,11)]
mnist = input_data.read_data_sets("MNIST_data/")
for tree in trees:
  before = time.time()
  forest = RandomForestClassifier(n_estimators = tree, n_jobs=-1).fit(mnist.train.images, mnist.train.labels)
  after = time.time() - before
  result = forest.score(mnist.test.images, mnist.test.labels)
  times_skl_folds.append(after)
  print "SKL_folds ", times_skl_folds, " Score:", result
data = ray.init(start_ray_local=True, num_workers=16)
for tree in trees:
  before = time.time()
  forest = ray_random_forest.RayRandomForestClassifier(n_estimators = tree).fit(mnist.train.images, mnist.train.labels)
  times_ray_folds.append(time.time() - before)
  result = forest.score(mnist.test.images, mnist.test.labels)
  print "Ray_folds ", times_ray_folds, " Score:", result

with open("results.txt", "w") as resultfile:
  resultfile.write(" ".join(map(str, trees)) + "\n")
  resultfile.write(" ".join(map(str, times_skl_folds)) + "\n")
  resultfile.write(" ".join(map(str, times_ray_folds)))
