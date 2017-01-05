from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray

import numpy as np
import scipy.optimize
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
  #ray.init(start_ray_local=True, num_workers=10)
  ray.init(redis_address="172.31.1.135:61773")
  # Define the dimensions of the data and of the model.
  image_dimension = 784
  label_dimension = 10
  w_shape = [image_dimension, label_dimension]
  w_size = np.prod(w_shape)
  b_shape = [label_dimension]
  b_size = np.prod(b_shape)
  dim = w_size + b_size

  # Define a function for initializing the network. Note that this code does not
  # call initialize the network weights. If it did, the weights would be
  # randomly initialized on each worker and would differ from worker to worker.
  # We pass the weights into the remote functions loss and grad so that the
  # weights are the same on each worker.
  class LinearModel(object):
    def __init__(self, shape):
      image_dimension = shape[0]
      label_dimension = shape[1]
      x = tf.placeholder(tf.float32, [None, image_dimension])
      w = tf.Variable(tf.zeros(shape))
      b = tf.Variable(tf.zeros(shape[1]))
      self.x = x
      self.w = w
      self.b = b
      y = tf.nn.softmax(tf.matmul(x, w) + b)
      y_ = tf.placeholder(tf.float32, [None, label_dimension])
      self.y = y
      self.y_ = y_
      cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
      self.cross_entropy = cross_entropy
      self.cross_entropy_grads = tf.gradients(cross_entropy, [w, b])
      self.variables = [w, b]
      self.sess = tf.Session()

      # In order to set the weights of the TensorFlow graph on a worker, we add
      # assignment nodes. To get the network weights (as a list of numpy arrays)
      # and to set the network weights (from a list of numpy arrays), use the
      # methods get_weights and set_weights. This can be done from within a remote
      # function or on the driver.
      #def get_and_set_weights_methods():
      assignment_placeholders = []
      assignment_nodes = []
      for var in [self.w, self.b]:
        assignment_placeholders.append(tf.placeholder(var.value().dtype, var.get_shape().as_list()))
        assignment_nodes.append(var.assign(assignment_placeholders[-1]))

      self.assignment_placeholders = assignment_placeholders
      self.assignment_nodes = assignment_nodes

    def get_weights(self):
      return [v.eval(session=self.sess) for v in self.variables]

    def set_weights(self, new_weights):
      self.sess.run(self.assignment_nodes, feed_dict={p: w for p, w in zip(self.assignment_placeholders, new_weights)})
    
    def run_loss(self, xs, ys):
      return float(self.sess.run(self.cross_entropy, feed_dict={self.x: xs, self.y_: ys}))
  
    def run_grad(self, xs, ys):
      return self.sess.run(self.cross_entropy_grads, feed_dict={self.x: xs, self.y_: ys})

  def net_initialization():
    net = LinearModel(w_shape)
    return net
  # By default, when a reusable variable is used by a remote function, the
  # initialization code will be rerun at the end of the remote task to ensure
  # that the state of the variable is not changed by the remote task. However,
  # the initialization code may be expensive. This case is one example, because
  # a TensorFlow network is constructed. In this case, we pass in a special
  # reinitialization function which gets run instead of the original
  # initialization code. As users, if we pass in custom reinitialization code,
  # we must ensure that no state is leaked between tasks.
  def net_reinitialization(net_vars):
    return net_vars

  # Create a reusable variable for the network.
  ray.register_class(LinearModel)
  ray.reusables.net_vars = ray.Reusable(net_initialization, net_reinitialization)

  # Load the weights into the network.
  def load_weights(theta):
    net = ray.reusables.net_vars
    net.set_weights([theta[:w_size].reshape(w_shape), theta[w_size:].reshape(b_shape)])

  # Compute the loss on a batch of data.
  @ray.remote
  def loss(theta, xs, ys):
    net = ray.reusables.net_vars
    load_weights(theta)
    return net.run_loss(xs,ys)

  # Compute the gradient of the loss on a batch of data.
  @ray.remote
  def grad(theta, xs, ys):
    net = ray.reusables.net_vars
    load_weights(theta)
    gradients = net.run_grad(xs, ys)
    return np.concatenate([g.flatten() for g in gradients])

  # Compute the loss on the entire dataset.
  def full_loss(theta):
    theta_id = ray.put(theta)
    loss_ids = [loss.remote(theta_id, xs_id, ys_id) for (xs_id, ys_id) in batch_ids]
   # loss_ids = [loss.executor([ray.get(theta_id), ray.get(xs_id), ray.get(ys_id)]) for (xs_id, ys_id) in batch_ids]
    return sum(ray.get(loss_ids))

  # Compute the gradient of the loss on the entire dataset.
  def full_grad(theta):
    theta_id = ray.put(theta)
   # grad_ids = [grad.remote(theta_id, xs_id, ys_id) for (xs_id, ys_id) in batch_ids]
    grad_ids = [grad.executor([ray.get(theta_id), ray.get(xs_id), ray.get(ys_id)]) for (xs_id, ys_id) in batch_ids]
    return sum(grad_ids).astype("float64") # This conversion is necessary for use with fmin_l_bfgs_b.

  # From the perspective of scipy.optimize.fmin_l_bfgs_b, full_loss is simply a
  # function which takes some parameters theta, and computes a loss. Similarly,
  # full_grad is a function which takes some parameters theta, and computes the
  # gradient of the loss. Internally, these functions use Ray to distribute the
  # computation of the loss and the gradient over the data that is represented
  # by the remote object IDs x_batches and y_batches and which is potentially
  # distributed over a cluster. However, these details are hidden from
  # scipy.optimize.fmin_l_bfgs_b, which simply uses it to run the L-BFGS
  # algorithm.

  # Load the mnist data and turn the data into remote objects.
  print("Downloading the MNIST dataset. This may take a minute.")
  mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
  batch_size = 100
  num_batches = mnist.train.num_examples // batch_size
  batches = [mnist.train.next_batch(batch_size) for _ in range(num_batches)]
  print("Putting MNIST in the object store.")
  batch_ids = [(ray.put(xs), ray.put(ys)) for (xs, ys) in batches]

  # Initialize the weights for the network to the vector of all zeros.
  theta_init = 1e-2 * np.random.normal(size=dim)
  # Use L-BFGS to minimize the loss function.
  print("Running L-BFGS.")
  result = scipy.optimize.fmin_l_bfgs_b(full_loss, theta_init, maxiter=10, fprime=full_grad, disp=True)
