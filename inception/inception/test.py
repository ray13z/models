from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from inception import inception_eval
from inception.flowers_data import FlowersData

"""
Just a test script to check out bazel builds.

Steps:
1. create python script: tensorflow/models/inception/inception/test.py
2. Edit bazel BUILD file (tensorflow/models/inception/inception/BUILD):
    py_binary(
        name = "test",
        srcs = [
            "test.py",
        ],
        deps = [
            ":flowers_data",
            ":inception_eval",
        ],
    )
3. Build: $ bazel build inception/test
4. Run: bazel-bin/inception/test
"""

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=None):
    print("running now...")


if __name__ == '__main__':
    tf.app.run()
