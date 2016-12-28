from tensorflow.examples.tutorials.mnist import input_data
import ray
from sklearn import svm
import ray_cross_val

mnist = input_data.read_data_sets("MNIST_data/")

ray.init(start_ray_local=True, num_workers=10)

print ray_cross_val.ray_cross_val_score(svm.SVC(), mnist.test.images, mnist.test.labels)
