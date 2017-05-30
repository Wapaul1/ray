from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import ray
from ray.test.multi_node_tests import (_wait_for_nodes_to_join,
                                       _broadcast_event,
                                       _wait_for_event)

# This test should be run with 5 nodes, which have 0, 2, 4, 8, and 16 GPUs for
# a total of 61 GPUs. It should be run with a large number of drivers (e.g.,
# 100). At most 10 drivers will run at a time, and each driver will use at most
# 5 GPUs (this is ceil(61 / 15), which guarantees that we will always be able
# to make progress).
total_num_nodes = 5

@ray.remote(num_gpus=1)
def use_one_gpus(ti):
  time_started = time.time() - ti
  assert len(ray.get_gpu_ids()) == 1
  time.sleep(4)
  return time_started, ray.get_gpu_ids()


@ray.remote(num_gpus=2)
def use_two_gpus():
  assert len(ray.get_gpu_ids()) == 2
  time.sleep(4)
  return ray.get_gpu_ids()

@ray.remote(num_gpus=2)
def use_two_gpus_long():
  assert len(ray.get_gpu_ids()) == 2
  time.sleep(9)
  return ray.get_gpu_ids()

@ray.remote(num_gpus=1)
class Actor1(object):
  def __init__(self):
    assert len(ray.get_gpu_ids()) == 1

  def check_ids(self):
    assert len(ray.get_gpu_ids()) == 1
    return ray.get_gpu_ids()

@ray.remote(num_gpus=4)
class Actor4(object):
  def __init__(self):
    assert len(ray.get_gpu_ids()) == 4

  def check_ids(self):
    assert len(ray.get_gpu_ids()) == 4
    return ray.get_gpu_ids()

def driver(redis_address):
  """The script for driver 0.

  This driver should create five actors that each use one GPU and some actors
  that use no GPUs. After a while, it should exit.
  """
  ray.init(redis_address=redis_address)

  # Wait for all the nodes to join the cluster.
  _wait_for_nodes_to_join(total_num_nodes)

  # Test that all gpus can be used at the same time using single gpu tasks.
  gpus = [use_one_gpus.remote(time.time()) for _ in range(30)]
  print(ray.get(gpus))
  assert sum([ele for l in ray.get(gpus) for ele in l]) == 150

  # Test that all gpus can be used at the same time using double gpu tasks.
  gpus = [use_two_gpus.remote() for _ in range(15)]
  assert sum([ele for l in ray.get(gpus) for ele in l]) == 150


  # Test that all gpus can be used at the same time using single and double gpu tasks.
  gpus = [use_one_gpus.remote() for _ in range(14)] + [use_two_gpus.remote() for _ in range(8)] 
  assert sum([ele for l in ray.get(gpus) for ele in l]) == 150

  actors = [Actor4.remote() for _ in range(2)] + [Actor1.remote() for _ in range(4)]
  actor_gpus = [ele for l in ray.get([actor.check_ids() for actor in actors]) for ele in l]
  gpus = ray.get([use_one_gpus.remote() for _ in range(18)])
  assert sum([ele for l in gpus for ele in l]) + sum(actor_gpus) == 150

  # Check that task gpus do not intersect with actor gpus.
  gpus = ray.get([use_one_gpus.remote() for _ in range(100)])
  assert len(set([ele for l in gpus for ele in l]).intersection(set(actor_gpus))) == 0

  # Check that multiple tasks do not intersect with other tasks.
  task_gpus = use_two_gpus_long.remote()
  gpus = ray.get([use_one_gpus.remote() for _ in range(28)])
  assert len(set([ele for l in gpus for ele in l]).intersection(set(task_gpus))) == 0

  # Check that actors have unique ids.
  actors_2 = [Actor4.remote() for _ in range(4)] + [Actor1.remote() for _ in range(2)]
  actor_gpus_2 = ray.get([actor.check_ids() for actor in actors_2])
  gpus = sum(actor_gpus) + sum([sum(gpus) for gpus in actor_gpus_2])
  assert gpus == 150

if __name__ == "__main__":
  redis_address = os.environ["RAY_REDIS_ADDRESS"]

  driver(redis_address)
