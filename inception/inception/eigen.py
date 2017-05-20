"""This models the architcture in Eigen depth."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
# import numpy as np
import tensorflow as tf

# import inception.slim as slim
from inception.slim.scopes import arg_scope
from inception.slim.ops import conv2d, max_pool, fc

tf.logging.set_verbosity(tf.logging.INFO)


# TODO
# Application logic will be added here
def coarse_stack(features, predictions):
    """Model function for coarse stack."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 240, 320, 3])
    print("input_layer: ", tf.shape(input_layer))
    with arg_scope([conv2d, fc], stddev=0.01,
                   weight_decay=0.0005, activation=tf.nn.relu):
        # See https://github.com/tensorflow/models/blob/master/inception/
        # inception/slim/ops.py for more details.
        #   net = conv2d(inputs=[batch_size, height, width, channels],
        #       num_filters_out, kernel_size=[kernel_height, kernel_width],
        #       stride=int|[stride_height, stride_width],
        #       padding='VALID'|'SAME',
        #       scope='conv1')
        #   net = max_pool(inputs,
        #                           kernel_size=[kernel_height, kernel_width],
        #                           stride=int|[stride_height, stride_width],
        #                           scope='pool1')
        net = conv2d(input_layer, 96, kernel_size=[11, 11],
                     stride=4, padding='VALID', scope='conv1')
        print("conv1: ", tf.shape(net))
        net = max_pool(net, kernel_size=[3, 3], stride=[2, 2],
                       scope='pool1')
        print("pool1: ", tf.shape(net))
        net = conv2d(net, 256, kernel_size=[5, 5], stride=1,
                     padding='SAME', scope='conv2')
        print("conv2: ", tf.shape(net))
        net = max_pool(net, kernel_size=[3, 3], stride=[2, 2],
                       scope='pool2')
        print("pool2: ", tf.shape(net))
        net = conv2d(net, 384, kernel_size=[3, 3], stride=1,
                     padding='SAME', scope='conv3')
        print("conv3: ", tf.shape(net))
        net = conv2d(net, 384, kernel_size=[3, 3], stride=1,
                     padding='SAME', scope='conv4')
        print("conv4: ", tf.shape(net))
        net = conv2d(net, 256, kernel_size=[3, 3], stride=1,
                     padding='SAME', scope='conv5')
        print("conv5: ", tf.shape(net))
        net = max_pool(net, kernel_size=[3, 3], stride=[2, 2],
                       scope='pool5')
        print("pool5: ", tf.shape(net))

        # net = flatten(net, scope='flatten5')
        # net = fc(net, 4096, scope='fc6')
        # net = dropout(net, 0.5, scope='dropout6')
        # net = fc(net, 4096, scope='fc7')
        # net = dropout(net, 0.5, scope='dropout7')
        # net = fc(net, 1000, activation=None, scope='fc8')


def main(unused_argv):
    """Main."""
    # TODO
    images = tf.Variable(tf.random_normal([1, 240, 320, 3], stddev=0.35),
                         name="images")
    depths = tf.Variable(tf.random_normal([1, 55, 74], stddev=0.35),
                         name="depths")
    model = coarse_stack(images, depths)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, save the
    # variables to disk.
    with tf.Session() as sess:
        sess.run(init_op)
        # Do some work with the model.

        # Save the variables to disk.
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)

    print("In main...")


if __name__ == "__main__":
    print("Before run...")
    tf.app.run()
