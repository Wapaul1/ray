from tensorflow.examples.tutorials.mnist import input_data
import ray
import ray_sgdclassifier
import time
import os
from sklearn.linear_model import SGDClassifier
import numpy as np

times_ray_workers = []
times_ray_folds = []
times_skl_folds = []
epochs = [5*i for i in range(1,9)]
mnist = input_data.read_data_sets("MNIST_data/")
for epoch in epochs:
  before = time.time()
  classifier = SGDClassifier(n_iter = epoch, n_jobs=-1).fit(mnist.train.images, mnist.train.labels)
  print "Intercepts:", classifier.intercept_
  after = time.time() - before
  result = classifier.score(mnist.test.images, mnist.test.labels)
  times_skl_folds.append(after)
  print "SKL_Epochs ", times_skl_folds, " Score:", result
data = ray.init(start_ray_local=True, num_workers=16)
for epoch in epochs:
  before = time.time()
  classifier = ray_sgdclassifier.RaySGDClassifier(n_iter = epoch).fit(mnist.train.images, mnist.train.labels)
  times_ray_folds.append(time.time() - before)
  print "Intercepts:", classifier.intercept_
  result = classifier.score(mnist.test.images, mnist.test.labels)
  print "Ray_Epochs ", times_ray_folds, " Score:", result

#with open("results.txt", "w") as resultfile:
#  resultfile.write(" ".join(map(str, epochs)) + "\n")
#  resultfile.write(" ".join(map(str, times_skl_folds)) + "\n")
#  resultfile.write(" ".join(map(str, times_ray_folds)))
