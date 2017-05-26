"""This models the architcture in Eigen depth."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
# import numpy as np
import tensorflow as tf
import h5py
import numpy as np

from matplotlib import pyplot as plt

# import inception.slim as slim
# import inception.slim as slim
from inception.slim.scopes import arg_scope
from inception.slim.ops import conv2d, max_pool, fc, flatten

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = None

# TODO
# Application logic will be added here
def coarse_stack(features):
    """Model function for coarse stack."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 240, 320, 3])
    print("input_layer: ", input_layer.get_shape())

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
        print("conv1: ", net.get_shape())

        net = max_pool(net, kernel_size=[3, 3], stride=[2, 2],
                       scope='pool1')
        print("pool1: ", net.get_shape())

        net = conv2d(net, 256, kernel_size=[5, 5], stride=1,
                     padding='SAME', scope='conv2')
        print("conv2: ", net.get_shape())

        net = max_pool(net, kernel_size=[3, 3], stride=[2, 2],
                       scope='pool2')
        print("pool2: ", net.get_shape())

        net = conv2d(net, 384, kernel_size=[3, 3], stride=1,
                     padding='SAME', scope='conv3')
        print("conv3: ", net.get_shape())

        net = conv2d(net, 384, kernel_size=[3, 3], stride=1,
                     padding='SAME', scope='conv4')
        print("conv4: ", net.get_shape())

        net = conv2d(net, 256, kernel_size=[3, 3], stride=1,
                     padding='SAME', scope='conv5')
        print("conv5: ", net.get_shape())

        net = max_pool(net, kernel_size=[3, 3], stride=[2, 2],
                       scope='pool5')
        print("pool5: ", net.get_shape())

        # net = flatten(net, scope='flatten5')
        # net = fc(net, 4096, scope='fc6')
        # net = dropout(net, 0.5, scope='dropout6')
        # net = fc(net, 4096, scope='fc7')
        # net = dropout(net, 0.5, scope='dropout7')
        # net = fc(net, 1000, activation=None, scope='fc8')

def train():
    print('Training...')

    # load labeled NYU Depth v2 dataset
    f = h5py.File('/home/rayner/Downloads/nyu_depth_v2_labeled.mat')

    images = f['images']
    depths = f['depths']

    del(f) # Free up some memory

    N = images.shape[0]
    Ntest = int(0.2* N)
    Ntrain = int(0.8 * (N-Ntest))
    Nvalid = int(0.2 * (N-Ntest))
    shuffler = np.random.permutation(N)

    Xtrain = images[shuffler[:Ntrain],]
    ytrain = depths[shuffler[:Ntrain],]
    Xvalid = images[shuffler[Ntrain:Ntrain + Nvalid],]
    yvalid = depths[shuffler[Ntrain:Ntrain + Nvalid],]
    Xtest = images[shuffler[Ntrain + Nvalid:],]
    ytest = depths[shuffler[Ntrain + Nvalid:],]

    # images = tf.Variable(tf.random_normal([1, 240, 320, 3], stddev=0.35),
    #                      name="images")
    # depths = tf.Variable(tf.random_normal([1, 55, 74], stddev=0.35),
    #                      name="depths")
    # model = coarse_stack(images, depths)
    #
    # # Add an op to initialize the variables.
    # init_op = tf.global_variables_initializer()
    #
    # # Add ops to save and restore all the variables.
    # saver = tf.train.Saver()
    #
    # # Later, launch the model, initialize the variables, do some work, save the
    # # variables to disk.
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     # Do some work with the model.
    #
    #     # Save the variables to disk.
    #     save_path = saver.save(sess, "/tmp/model.ckpt")
    #     print("Model saved in file: %s" % save_path)

def main(unused_argv):
    """Main."""
    # TODO
    print("In main...")
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == "__main__":
    print("Before run...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                      help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
